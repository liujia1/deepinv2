# -*- coding: utf-8 -*-
"""
实验8.3-1 变分推断与真实后验对比
对应章节：8.3（变分推断作为优化问题）
素材来源：🆕 新写
★ 原创设计：CAVI闭式实现 + 平均场局限与变分族容量局限的分层对比

知识点：
  - 变分推断 = 在变分族约束下最大化ELBO
  - 平均场近似：q(z,c) = q(z)·q(c)（假设潜变量相互独立）
  - CAVI算法：坐标上升变分推断，每个因子有闭式更新
  - 变分间隙：log p(x) - ELBO = KL(q || p(z,c|x))，衡量近似质量
  - 间隙有两个不同来源，必须分开：
      (1) 平均场独立性约束（本例 q(z)⊥q(c)）——结构约束
      (2) 变分族参数容量（如用单个高斯）——表达能力约束

实验内容：
  步骤1：CAVI闭式实现（含离散分配变量c的真正平均场），多初始化展示对称破缺与局部最优
  步骤2：平均场近似 vs 真实后验——独立性间隙（mode-seeking，锁定单峰）
  步骤3：变分族容量对比——高斯族的居中摊平 vs 混合族精确表示

模型：
  c ~ Cat([0.5, 0.5])
  z | c=k ~ N(mu_k, tau_k^2),  mu = [-2, 2], tau = [1, 1]
  x | z   ~ N(z, sigma_obs^2), sigma_obs = 1, 观测 x = 0
  => 真实后验 p(z|x) = 0.5 N(-1, 0.707^2) + 0.5 N(+1, 0.707^2)（对称双峰）

运行前提：纯NumPy/SciPy CPU即可
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.integrate import quad
import os
import sys
import io
import json

# 设置控制台输出为 UTF-8 (Windows 下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验8.3-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(_chinese_path, exist_ok=True)

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)

# ============================================================
# 模型设定：1D高斯混合先验 + 高斯观测
# 对称设置 + 观测值 x=0 恰在两峰中间 => 后验两个分量严格等权，
# 是展示平均场"对称破缺"与单高斯"居中摊平"两种病理的最干净场景
# ============================================================
prior_weights = [0.5, 0.5]
prior_means = [-2.0, 2.0]
prior_stds = [1.0, 1.0]
sigma_obs = 1.0
x_obs = 0.0
K = len(prior_weights)
_pw = np.array(prior_weights)
_pm = np.array(prior_means)
_pt = np.array(prior_stds)
_pt2 = _pt ** 2

print("=" * 64)
print("实验8.3-1：变分推断与真实后验对比")
print("=" * 64)
print(f"\n生成模型：")
print(f"  c ~ Cat({prior_weights})")
print(f"  z | c=k ~ N(mu_k, tau_k^2), mu={prior_means}, tau={prior_stds}")
print(f"  x | z ~ N(z, {sigma_obs}^2), 观测 x = {x_obs}")


# ============================================================
# 辅助函数：真实后验、边际似然、高斯混合PDF、对数高斯密度
# ============================================================

def log_gauss(t, m, s):
    """log N(t; m, s^2)，支持广播"""
    return -0.5 * np.log(2 * np.pi) - np.log(s) - 0.5 * ((t - m) / s) ** 2


def compute_posterior(x, weights, means, stds, sigma_obs):
    """解析计算高斯混合先验+高斯似然的后验（仍是同分量数的高斯混合）"""
    post_weights, post_means, post_stds = [], [], []
    for w, mu, tau in zip(weights, means, stds):
        s2 = 1.0 / (1.0 / sigma_obs**2 + 1.0 / tau**2)
        m = s2 * (x / sigma_obs**2 + mu / tau**2)
        ms = np.sqrt(sigma_obs**2 + tau**2)
        ml = -0.5 * ((x - mu) / ms)**2 - np.log(ms)  # 略去公共常数 -0.5log(2pi)
        post_weights.append(w * np.exp(ml))
        post_means.append(m)
        post_stds.append(np.sqrt(s2))
    total = sum(post_weights)
    return [pw / total for pw in post_weights], post_means, post_stds


def log_marginal(x, weights, means, stds, sigma_obs):
    """log p(x) = log Σₖ wₖ·N(x; μₖ, σ_obs²+τₖ²)"""
    terms = []
    for w, mu, tau in zip(weights, means, stds):
        ms = np.sqrt(sigma_obs**2 + tau**2)
        terms.append(np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(ms)
                     - 0.5 * ((x - mu) / ms)**2)
    mx = max(terms)
    return mx + np.log(sum(np.exp(t - mx) for t in terms))


def gm_pdf(x, weights, means, stds):
    """高斯混合分布的PDF"""
    pdf = np.zeros_like(x, dtype=float)
    for w, m, s in zip(weights, means, stds):
        pdf += w * np.exp(-0.5 * ((x - m) / s)**2) / (s * np.sqrt(2 * np.pi))
    return pdf


post_w, post_m, post_s = compute_posterior(x_obs, prior_weights, prior_means,
                                           prior_stds, sigma_obs)
log_px = log_marginal(x_obs, prior_weights, prior_means, prior_stds, sigma_obs)

print(f"\n真实后验 p(z,c|x={x_obs})：")
print(f"  p(z|x) = ", end="")
print(" + ".join(f"{w:.4f}·N({m:.4f}, {s:.4f}²)"
                 for w, m, s in zip(post_w, post_m, post_s)))
print(f"  （双峰、且 z 与分量分配 c 强相关：z<0 几乎必有 c=1）")
print(f"log p(x={x_obs}) = {log_px:.6f}  （ELBO 不可逾越的上界）")


# ============================================================
# 步骤1：CAVI 闭式实现（真正的平均场）
# ----------------------------------------------------------------
# 平均场族：q(z,c) = q(z) q(c)，q(z)=N(m,s²)，q(c)=Cat(r)
# 坐标上升（每步都是闭式最优）：
#   1/s²_new = 1/σ² + Σₖ rₖ/τₖ²
#   m_new    = s²_new · (x/σ² + Σₖ rₖ μₖ/τₖ²)
#   log rₖ ∝ log πₖ − 0.5 log(2πτₖ²) − [(m−μₖ)²+s²]/(2τₖ²)   (softmax归一化)
# 对应 Bishop PRML §10.2 / GMM 平均场更新。
# ============================================================
print("\n" + "=" * 64)
print("步骤1：CAVI 闭式实现（平均场 q(z)·q(c)）")
print("=" * 64)
print("""
[核心思想]
  变分推断 = 在变分族 Q 约束下最大化 ELBO
  平均场：q(z,c) = q(z)·q(c)，强制 z ⊥ c（丢失后验中的 z-c 相关结构）
  CAVI：固定其他因子，把每个因子更新为其闭式最优，ELBO 逐轮不降
""")


def cavi_elbo(m, s, r):
    """平均场 ELBO（全部闭式期望）
    ELBO = E_q[log p(x|z)] + E_q[log p(z,c)] + H[q(z)] + H[q(c)]
    """
    s2 = s ** 2
    e_log_pxz = (-0.5 * np.log(2 * np.pi * sigma_obs**2)
                 - 0.5 * ((x_obs - m)**2 + s2) / sigma_obs**2)
    e_log_pzc = 0.0
    for k in range(K):
        e_dist2 = (m - _pm[k])**2 + s2
        e_log_pzc += r[k] * (np.log(_pw[k])
                             - 0.5 * np.log(2 * np.pi * _pt2[k])
                             - 0.5 * e_dist2 / _pt2[k])
    h_z = 0.5 * np.log(2 * np.pi * np.e * s2)
    h_c = -np.sum(r * np.log(r))
    return e_log_pxz + e_log_pzc + h_z + h_c


def run_cavi(r_init, max_iter=200, tol=1e-10, verbose=False):
    """从 q(c)=Cat(r_init) 出发做闭式坐标上升，返回 m,s,r,ELBO历史"""
    r = np.array(r_init, dtype=float)
    r /= r.sum()
    hist = []
    for it in range(max_iter):
        # --- q(z) 更新（固定 r）---
        prec = 1.0 / sigma_obs**2 + np.sum(r / _pt2)
        s2 = 1.0 / prec
        m = s2 * (x_obs / sigma_obs**2 + np.sum(r * _pm / _pt2))
        s = np.sqrt(s2)
        # --- q(c) 更新（固定新的 q(z)）---
        log_r = np.zeros(K)
        for k in range(K):
            log_r[k] = (np.log(_pw[k]) - 0.5 * np.log(2 * np.pi * _pt2[k])
                        - 0.5 * ((m - _pm[k])**2 + s2) / _pt2[k])
        log_r -= np.max(log_r)  # 数值稳定
        r = np.exp(log_r)
        r /= r.sum()
        # --- 记录 ELBO ---
        elbo = cavi_elbo(m, s, r)
        hist.append(elbo)
        if verbose:
            print(f"    iter {it:2d}: m={m:+.4f}, s={s:.4f}, "
                  f"r=[{r[0]:.4f},{r[1]:.4f}], ELBO={elbo:.6f}, "
                  f"gap={log_px - elbo:.6f}")
        if len(hist) >= 2 and abs(hist[-1] - hist[-2]) < tol:
            break
    return m, s, r, np.array(hist)


def kl_mf_quad(m, s, r):
    """用一维数值积分独立核对 KL(q(z)q(c) || p(z,c|x))，应等于 log p(x)-ELBO"""
    def integrand(z, k):
        qz = np.exp(log_gauss(z, m, s))
        log_joint_unnorm = (log_gauss(x_obs, z, sigma_obs)
                            + np.log(_pw[k]) + log_gauss(z, _pm[k], _pt[k]))
        # KL 项：q(z) r_k [log q(z) + log r_k - (log p(x,z,c=k) - log p(x))]
        return qz * r[k] * (np.log(qz) + np.log(r[k])
                            - log_joint_unnorm + log_px)
    total = 0.0
    for k in range(K):
        val, _ = quad(integrand, -12, 12, args=(k,), epsabs=1e-12,
                      epsrel=1e-12, limit=200)
        total += val
    return total


cavi_runs = {}
# 三组初始化：精确对称（不稳定驻点）、微扰（缓慢滑离鞍点→左峰）、
# 强偏置（几步即收敛到右峰）。后两者终点 ELBO 相同、锁定的峰相反，
# 且轨迹形状不同，可直观展示初始化依赖与"异途同归（镜像等优）"。
run_specs = [
    ("symmetric",        [0.5, 0.5],  "对称初始化 r=(0.5,0.5)"),
    ("perturb_left",     [0.501, 0.499], "微扰 r=(0.501,0.499)"),
    ("biased_right",     [0.1, 0.9],  "强偏置 r=(0.1,0.9)"),
]

for key, r0, desc in run_specs:
    print(f"\n  ── {desc} ──")
    m, s, r, hist = run_cavi(r0, verbose=True)
    mono = bool(np.all(np.diff(hist) >= -1e-9))
    gap = log_px - hist[-1]
    kl_num = kl_mf_quad(m, s, r)
    cavi_runs[key] = dict(r_init=[float(v) for v in r0],
                          m=float(m), s=float(s), r=r.tolist(),
                          elbo=float(hist[-1]), gap=float(gap),
                          kl_quad=float(kl_num), n_iter=len(hist),
                          monotone=mono, history=hist.tolist())
    print(f"    收敛：m={m:+.4f}, s={s:.4f}, r=[{r[0]:.4f},{r[1]:.4f}]")
    print(f"    ELBO={hist[-1]:.6f}, gap={gap:.6f}；"
          f"独立quadrature KL={kl_num:.6f}（应=gap，互核）")
    print(f"    ELBO 逐轮非降：{'是 ✓' if mono else '否 ✗'}")

mf = cavi_runs["perturb_left"]                     # 代表性平均场解（锁左峰）
mf_right = cavi_runs["biased_right"]               # 镜像等优解（锁右峰）
mf_sym = cavi_runs["symmetric"]

print(f"""
[CAVI 现象解读]
  1) 三种初始化下 ELBO 均逐轮非降——CAVI 坐标上升的基本性质 ✓
  2) 严格对称的初始化停在 m=0 的驻点（gap={mf_sym['gap']:.4f}；该值等于
     q(z)=N(0,0.5) 到两个等权后验分量 N(±1,0.5) 的平均高斯KL = 1²/(2·0.5)=1，
     并非 log2），但它是【不稳定鞍点】：把 r 微扰 0.001，迭代立刻自发滑走
  3) 微扰/偏置后发生【对称破缺】：q(z) 只锁定左峰或右峰之一，
     两个镜像解等优（ELBO 均={mf['elbo']:.6f}），取哪个完全由初始化决定
     ——这就是平均场 VI 局部最优、初始化敏感的经典来源
""")


# ============================================================
# 步骤2：平均场近似 vs 真实后验 —— 独立性间隙
# ============================================================
print("=" * 64)
print("步骤2：平均场近似 vs 真实后验（独立性间隙）")
print("=" * 64)

z_grid = np.linspace(-5, 5, 500)
posterior_pdf = gm_pdf(z_grid, post_w, post_m, post_s)
prior_pdf = gm_pdf(z_grid, prior_weights, prior_means, prior_stds)
mf_pdf_left = np.exp(log_gauss(z_grid, mf["m"], mf["s"]))
mf_pdf_right = np.exp(log_gauss(z_grid, mf_right["m"], mf_right["s"]))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：真实后验 vs 平均场 q*(z)
axes[0].fill_between(z_grid, posterior_pdf, alpha=0.25, color='blue',
                     label=r'True posterior $p(z|x)$ (bimodal)')
axes[0].plot(z_grid, posterior_pdf, 'b-', lw=2)
axes[0].plot(z_grid, mf_pdf_left, 'g-', lw=2,
             label=rf'Mean-field $q^*(z)=\mathcal{{N}}({mf["m"]:.2f},{mf["s"]:.2f}^2)$, '
                   rf'lock left, gap={mf["gap"]:.3f}')
axes[0].plot(z_grid, mf_pdf_right, 'g--', lw=1.5, alpha=0.7,
             label=r'mirror solution (lock right, same ELBO)')
axes[0].plot(z_grid, prior_pdf, 'k:', lw=1, alpha=0.5, label=r'Prior $p(z)$')
axes[0].set_xlabel(r'$z$')
axes[0].set_ylabel(r'density')
axes[0].set_title('Mean-field CAVI vs true posterior:\n'
                  'independence gap forces mode-seeking')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# 右图：三种初始化的 ELBO 收敛轨迹
colors = {'symmetric': 'gray', 'perturb_left': 'green', 'biased_right': 'orange'}
labels = {
    'symmetric': r'symmetric init: unstable stationary point, gap=1.000',
    'perturb_left': r'tiny perturbation $\to$ slowly break to left peak',
    'biased_right': r'strong bias $\to$ fast converge to right peak',
}
for key, _, _ in run_specs:
    h = np.array(cavi_runs[key]["history"])
    if key == 'symmetric':
        # 对称轨迹仅2点且与微扰轨迹前段几乎重合，用方块置顶以可见
        axes[1].plot(np.arange(len(h)), h, linestyle='None', marker='s',
                     markersize=8, markerfacecolor='none', markeredgewidth=1.8,
                     color=colors[key], label=labels[key], zorder=5)
    else:
        axes[1].plot(np.arange(len(h)), h, '-o', markersize=3.5, lw=1.8,
                     color=colors[key], label=labels[key])
axes[1].axhline(log_px, color='r', linestyle='--', lw=2,
                label=rf'$\log p(x)={log_px:.4f}$ (upper bound)')
axes[1].set_xlabel(r'CAVI iteration')
axes[1].set_ylabel(r'ELBO')
axes[1].set_title('CAVI convergence: monotone ELBO,\nsymmetry breaking and local optima')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
fig2_path = os.path.join(SAVE_DIR, '步骤2_CAVI平均场vs真实后验.png')
plt.savefig(fig2_path, dpi=120)
plt.close()
print(f"图表已保存: {os.path.basename(fig2_path)}")

print(f"""
[关键点密度对比]——平均场丢失 z-c 相关结构，zero-avoiding：
  {'z':>5} | {'p(z|x)真实':>10} | {'平均场q*(z)锁左峰':>16} | 说明
  ----------------------------------------------------------------""")
for z_val, note in [(-1.0, "左峰：q*过度集中"),
                    (0.0, "两峰中间"),
                    (1.0, "右峰：q*几乎置零（右峰被整体放弃）")]:
    p_val = gm_pdf(np.array([z_val]), post_w, post_m, post_s)[0]
    q_val = np.exp(log_gauss(z_val, mf["m"], mf["s"]))
    print(f"  {z_val:+5.1f} | {p_val:10.4f} | {q_val:16.4f} | {note}")
print(f"""
平均场间隙 = {mf['gap']:.4f}
  来源：q(z)⊥q(c) 抹掉了"z 与所属分量 c 的后验相关性"（对 c 的 Jensen 界），
  单峰的 q(z) 在逆向 KL 下只能保住一个峰，另一峰的概率质量被整体放弃。
""")


# ============================================================
# 步骤3：变分族容量对比 —— 高斯族的居中摊平 vs 混合族精确表示
# ----------------------------------------------------------------
# 注意区分：步骤1/2 的间隙来自【独立性结构约束】；
# 这里把 c 积掉、只对单一标量 z 取变分族，没有任何独立性约束可言
# （一个变量无所谓独立），剩余间隙纯粹来自【参数族容量】。
# 一维光滑目标用 Gauss-Hermite 求积算 ELBO，近机器精度、无 MC 噪声；
# 高维情形换重参数化 MC 即可（注意用公共随机数为优化器降噪）。
# ============================================================
print("=" * 64)
print("步骤3：变分族容量对比（族容量间隙，与独立性无关）")
print("=" * 64)

gh_x, gh_w = hermgauss(128)  # ∫ e^{-t²} f(t) dt ≈ Σ w_i f(t_i)，128节点求积误差<1e-9
_SQRT_PI = np.sqrt(np.pi)


def gaussian_family_elbo(m, s):
    """q(z)=N(m,s²)（无 c、无因子分解）的精确 ELBO，Gauss-Hermite 求积"""
    z = m + np.sqrt(2.0) * s * gh_x
    e_log_pxz = np.sum(gh_w * log_gauss(x_obs, z, sigma_obs)) / _SQRT_PI
    log_pz = np.full_like(z, -np.inf)
    for k in range(K):
        log_pz = np.logaddexp(log_pz, np.log(_pw[k]) + log_gauss(z, _pm[k], _pt[k]))
    e_log_pz = np.sum(gh_w * log_pz) / _SQRT_PI
    entropy = 0.5 * np.log(2 * np.pi * np.e * s**2)
    return e_log_pxz + e_log_pz + entropy


def kl_gaussian_quad(m, s):
    """独立 quadrature 核对 KL(N(m,s²) || p(z|x))"""
    def integrand(z):
        qz = np.exp(log_gauss(z, m, s))
        pz = gm_pdf(np.array([z]), post_w, post_m, post_s)[0]
        return qz * (np.log(qz) - np.log(pz))
    val, _ = quad(integrand, -12, 12, epsabs=1e-12, epsrel=1e-12, limit=200)
    return val


# 多起点优化（验证该 ELBO 地形单峰、居中解为全局最优）
opt_results = []
for m0, ls0 in [(0.0, 0.0), (-1.0, np.log(0.7)), (1.0, np.log(0.7)),
                (-0.5, np.log(1.3)), (0.5, np.log(1.3))]:
    res = minimize(lambda p: -gaussian_family_elbo(p[0], np.exp(p[1])),
                   [m0, ls0], method='Nelder-Mead',
                   options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-10})
    opt_results.append((-res.fun, res.x[0], np.exp(res.x[1])))
best_g = max(opt_results, key=lambda t: t[0])
g_elbo, g_mu, g_sigma = best_g
g_mu = 0.0 if abs(g_mu) < 5e-6 else g_mu   # 优化残差（~1e-8）在显示与对称性上归零
g_gap = log_px - g_elbo
g_kl = kl_gaussian_quad(g_mu, g_sigma)
all_centered = all(abs(o[1]) < 1e-3 and abs(o[2] - g_sigma) < 1e-3
                   for o in opt_results)

print(f"\n  单高斯自由变分族 Q={{N(μ,σ²)}}（只有标量 z，无独立性约束）：")
print(f"    5 个起点全部收敛到同一解：{'是 ✓（族内全局最优，无局部最优问题）' if all_centered else '否'}")
print(f"    q*(z) = N({g_mu:+.4f}, {g_sigma:.4f}²)")
print(f"    ELBO = {g_elbo:.6f}，gap = {g_gap:.6f}；"
      f"独立 quadrature KL = {g_kl:.6f}（互核）")

# 混合变分族：真实后验本身就是2分量混合 => q=p(z|x) ∈ Q => gap 恒为 0
def elbo_at_true_posterior():
    """q=p(z|x) 时 ELBO=log p(x)（KL=0 恒等），用 quadrature 数值核对"""
    def integrand(z):
        pz = gm_pdf(np.array([z]), post_w, post_m, post_s)[0]
        log_prior = np.log(gm_pdf(np.array([z]), prior_weights, prior_means,
                                  prior_stds)[0])
        return pz * (log_gauss(x_obs, z, sigma_obs) + log_prior
                     - np.log(pz))
    val, _ = quad(integrand, -12, 12, epsabs=1e-12, epsrel=1e-12, limit=200)
    return val

mix_elbo = elbo_at_true_posterior()
print(f"\n  2分量混合变分族 Q={{π·N(a₁,s₁²)+(1−π)·N(a₂,s₂²)}}：")
print(f"    真实后验本身就是该族成员，取 q*=p(z|x) 即最优，"
          f"gap 恒为 0")
print(f"    quadrature 核对 ELBO(q=p(z|x)) = {mix_elbo:.6f} vs "
      f"log p(x) = {log_px:.6f}，偏差 = {abs(mix_elbo - log_px):.2e}")

# 图3
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

gauss_pdf = np.exp(log_gauss(z_grid, g_mu, g_sigma))
axes[0].fill_between(z_grid, posterior_pdf, alpha=0.2, color='blue')
axes[0].plot(z_grid, posterior_pdf, 'b-', lw=2,
             label=r'True posterior $p(z|x)$ = mixture-family $q^*$ (gap=0)')
axes[0].plot(z_grid, gauss_pdf, 'r--', lw=2,
             label=rf'Best single Gaussian $\mathcal{{N}}({g_mu:.2f},{g_sigma:.2f}^2)$, '
                   rf'gap={g_gap:.3f}')
axes[0].plot(z_grid, mf_pdf_left, 'g:', lw=2,
             label=rf'Mean-field CAVI $q(z)$, gap={mf["gap"]:.3f}')
axes[0].set_xlabel(r'$z$')
axes[0].set_ylabel(r'density')
axes[0].set_title('Two distinct sources of the variational gap')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

bar_labels = ['Mean-field\n(saddle, unstable)', 'Mean-field CAVI\n(broken symmetry)',
              'Single Gaussian\n(free family)', '2-comp. Mixture\n(contains truth)']
bar_gaps = [mf_sym['gap'], mf['gap'], g_gap, 0.0]
bar_colors = ['gray', 'green', 'red', 'blue']
bars = axes[1].bar(bar_labels, bar_gaps, color=bar_colors, alpha=0.75)
for b, v in zip(bars, bar_gaps):
    axes[1].text(b.get_x() + b.get_width() / 2, v + 0.015, f'{v:.4f}',
                 ha='center', fontsize=10)
axes[1].set_ylabel(r'gap $= \log p(x) - \mathrm{ELBO} = \mathrm{KL}(q\|p(\cdot|x))$')
axes[1].set_title('Gap decomposition: independence structure\nvs family capacity')
axes[1].set_ylim(0, 1.15)
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
fig3_path = os.path.join(SAVE_DIR, '步骤3_变分族容量对比.png')
plt.savefig(fig3_path, dpi=120)
plt.close()
print(f"\n图表已保存: {os.path.basename(fig3_path)}")

print(f"""
[步骤3 解读]
  - 单高斯自由族（无 c、无独立性假设）仍有 gap={g_gap:.4f}：
    它居中摊开、把两峰间凹陷抬高，这是【高斯族容量不足】，与平均场无关。
  - 换成 2 分量混合族（依旧只是标量 z 的分布、无任何独立性可放松），
    q*=p(z|x)，gap 立即归 0。
  - 对比步骤2：平均场 q(z)⊥q(c) 的 gap={mf['gap']:.4f} 远大于高斯族容量 gap，
    且表现为"锁单峰"而非"摊平"——两种间隙的来源和几何行为都不同，不可混称。
""")


# ============================================================
# 结果落盘（与实验8.1-1 的 results_summary.json 惯例一致）
# ============================================================
results = {
    "model": {
        "prior_weights": prior_weights,
        "prior_means": prior_means,
        "prior_stds": prior_stds,
        "sigma_obs": sigma_obs,
        "x_obs": x_obs,
        "latent_variables": "c ~ Cat(pi); z | c=k ~ N(mu_k,tau_k^2); x | z ~ N(z,sigma_obs^2)"
    },
    "true_posterior": {
        "weights": post_w,
        "means": post_m,
        "stds": post_s,
        "log_marginal_log_px": log_px
    },
    "step1_step2_CAVI_mean_field": {
        "family": "q(z,c) = q(z) q(c), q(z)=N(m,s^2), q(c)=Cat(r)",
        "closed_form_updates": {
            "s2": "1 / (1/sigma_obs^2 + sum_k r_k/tau_k^2)",
            "m": "s2 * (x/sigma_obs^2 + sum_k r_k mu_k/tau_k^2)",
            "r": "softmax_k { log pi_k - 0.5 log(2 pi tau_k^2) - [(m-mu_k)^2+s2]/(2 tau_k^2) }"
        },
        "runs": {
            key: {k_: v_ for k_, v_ in cavi_runs[key].items() if k_ != "history"}
            for key in cavi_runs
        },
        "elbo_monotone_all_runs": all(v["monotone"] for v in cavi_runs.values()),
        "interpretation": [
            "对称解 m=0,s=0.7071,r=(0.5,0.5) 是不稳定驻点：它到两个等权后验分量 "
            "N(±1,0.7071²) 的平均KL恰为 1²/(2·0.5)=1（同方差均值漂移），故 gap=1.000；"
            "任意微扰即自发对称破缺",
            "破缺后两个镜像局部最优等优（ELBO=-2.93899，gap=0.6735），"
            "锁左峰或右峰由初始化决定",
            "gap=0.6735 来自 q(z)⊥q(c) 的独立性约束（mode-seeking / zero-avoiding）"
        ]
    },
    "step3_family_capacity": {
        "single_gaussian_free_family": {
            "note": "只有标量z，无因子分解，无独立性约束；间隙纯为高斯族容量不足",
            "best_mu": g_mu,
            "best_sigma": g_sigma,
            "ELBO": g_elbo,
            "gap": g_gap,
            "KL_quadrature_independent_check": g_kl,
            "all_multistart_converge_same_point": all_centered,
            "behavior": "居中摊平，mode-covering，抬高两峰间凹陷"
        },
        "two_component_mixture_family": {
            "note": "真实后验本身是2分量高斯混合，q*=p(z|x) 属于该族，gap恒为0",
            "ELBO_quadrature_check": mix_elbo,
            "gap": max(0.0, log_px - mix_elbo)
        }
    },
    "figures": [
        os.path.basename(fig2_path),
        os.path.basename(fig3_path)
    ],
    "note": (
        "变分间隙有两个不同来源，本实验分开验证：(1) 平均场独立性间隙 "
        "q(z)⊥q(c)，CAVI闭式坐标上升，ELBO严格非降；严格对称初始化停在不稳定驻点"
        "(gap=1.000，等于 q(z) 到两个等权后验分量的平均高斯KL)，微扰后对称破缺、"
        "锁单峰，局部最优 gap=0.6735；(2) 族容量间隙：对单标量z无独立性可言，"
        "最优单高斯居中摊平 gap=0.0577，换2分量混合族(仍无独立性结构)即gap=0。"
        "所有 gap 均用独立 scipy quad 的 KL 积分交叉核对到 1e-6 量级一致。"
        "ELBO 求积采用 128 节点 Gauss-Hermite（求积误差<1e-9、无MC噪声）。"
    )
}

json_path = os.path.join(SAVE_DIR, 'results_summary.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"结果已保存: {os.path.basename(json_path)}")


# ============================================================
# 核心结论
# ============================================================
print(f"\n{'=' * 64}")
print("[核心结论]")
print(f"{'=' * 64}")
print(f"""
  1. CAVI：平均场 q(z)·q(c) 下闭式交替更新，ELBO 逐轮非降；
     对称模型上存在不稳定鞍点，微扰即对称破缺、陷入二选一的局部最优。

  2. 变分间隙 log p(x)−ELBO = KL(q‖后验)，但间隙有两种必须区分的来源：
     (i)  独立性结构（平均场）：q(z)⊥q(c) 丢失 z-c 相关，
          CAVI 解锁定单峰，gap = {mf['gap']:.4f}；
     (ii) 参数族容量：单高斯装不下双峰，最优解居中摊平，
          gap = {g_gap:.4f}；扩大为混合族（无独立性可放松）即 gap = 0。

  3. 缩小间隙的对症方法：
     - 对 (i)：结构化变分/辅助变量、放松因子分解、更好的初始化或多起点；
     - 对 (ii)：扩大变分族（混合分布、归一化流等增强表达能力）。
     张冠李戴（如在单变量问题上把族容量间隙归因于"平均场"）是常见概念错误。

  第八章配套实验8.3-1完成！
""")
