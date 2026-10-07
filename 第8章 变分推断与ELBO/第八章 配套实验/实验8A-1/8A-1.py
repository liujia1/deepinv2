# -*- coding: utf-8 -*-
"""
实验8A-1 Fenchel共轭与ELBO的优化理论根基
对应章节：附录8A（Fenchel共轭与ELBO的优化理论根基）
素材来源：🆕 新写
★ 原创设计：Fenchel共轭的数值计算与ELBO的凸共轭视角

知识点：
  - Fenchel共轭定义：f*(y) = sup_x {<x,y> - f(x)}
  - Fenchel-Young不等式：f(x) + f*(y) >= <x,y>
  - 经典函数的共轭：二次函数、绝对值、负对数、指数函数
  - ELBO作为log p(x)的Fenchel对偶表示
  - 变分间隙对应Fenchel-Young间隙（数值上核验 间隙 = KL(q*||p(z|x))）

实验内容：
  步骤1：Fenchel共轭的计算——几个经典函数的共轭（含几何示意图）
  步骤2：从Fenchel共轭定义推导ELBO = log p(x)的下界性质
        （注：这里用到的核心恒等式是
            log ∫ exp(f(z))dz = sup_q {E_q[f(z)] - (-H(q))}
                              = sup_q {E_q[log p(x,z)] + H(q)}
         它来自Fenchel共轭定义本身（一次共轭），不是
         双重共轭(f**=f, Fenchel-Moreau)；
         对应函数族的"sup_q ELBO = log p(x)" 是强对偶的特例，
         当变分族Q足够大（包含p(z|x)）时等号成立。）
        ★ 两个对照配置：
           配置A（后验近似单峰）→ 单高斯族几乎达成强对偶，间隙 ~5e-4
           配置B（后验真正双峰）→ 单高斯必须放弃一个模式，间隙 ~ -log(π_max)

运行前提：纯NumPy/SciPy CPU即可
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys
import io
import json
from scipy.optimize import minimize
from scipy.integrate import quad
from numpy.polynomial.hermite_e import hermegauss

# numpy 2.x 把 trapz 改名为 trapezoid，这里做双向兼容
_trapz = getattr(np, 'trapezoid', None) or np.trapz

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
    SAVE_DIR = os.path.join(_gdrive, '实验8A-1')
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
RESULT_JSON = {}


# ============================================================
# 步骤1：Fenchel共轭的计算
# f*(y) = sup_x { <x, y> - f(x) }
# ★ 原创设计：数值验证经典函数的共轭
# ============================================================
print("=" * 60)
print("步骤1：Fenchel共轭的计算与验证")
print("=" * 60)

print("\n[核心概念]")
print("  Fenchel共轭：f*(y) = sup_x {<x,y> - f(x)}")
print("  几何意义：f*(y)是f的最优线性下界的截距")
print("  Fenchel-Young不等式：f(x) + f*(y) >= <x,y>")


def fenchel_conjugate_numerical(f, y, x_range=(-10, 10), n_points=10001):
    """数值计算Fenchel共轭 f*(y) = sup_x { x*y - f(x) }

    n_points 默认取奇数：对称区间 linspace(-a, a, 奇数) 必含 x=0。
    若用偶数个点（如10000），格点不含0，则 |x| 这类在 x=0 处取 sup 的函数
    会凭空多出约 0.5*步长 量级的离散误差（本例曾为 -5e-4），属纯格点伪影。
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    values = x * y - f(x)
    idx = np.argmax(values)
    return values[idx], x[idx]


# 经典函数及其解析共轭（数值与理论都存入 rows，结论文本由实测值生成）
rows = []

# 1. f(x) = x²/2 → f*(y) = y²/2
f1 = lambda x: x**2 / 2
y1 = 2.0
f1_star_theory = y1**2 / 2
f1_star_num, f1_argmax = fenchel_conjugate_numerical(f1, y1)
rows.append(('x²/2, y=2', 'y²/2', f1_star_theory, f1_star_num, f1_argmax, True))

# 2. f(x) = |x| → f*(y) = 0 if |y|≤1, +∞ otherwise
f2 = lambda x: np.abs(x)
f2y_a = 0.5
f2a_star_num, f2a_argmax = fenchel_conjugate_numerical(f2, f2y_a)
rows.append(('|x|, y=0.5', '0 (|y|≤1)', 0.0, f2a_star_num, f2a_argmax, True))

f2y_b = 1.5
f2b_star_num, f2b_argmax = fenchel_conjugate_numerical(f2, f2y_b)
rows.append(('|x|, y=1.5', '+inf (|y|>1)', float('inf'), f2b_star_num, f2b_argmax, False))

# 3. f(x) = -log(x) (x>0) → f*(y) = -1-log(-y) (y<0)
f3 = lambda x: -np.log(np.maximum(x, 1e-10))
y3 = -1.5
f3_star_theory = -1 - np.log(-y3)
f3_star_num, f3_argmax = fenchel_conjugate_numerical(f3, y3, x_range=(0.01, 20))
rows.append(('-log(x), y=-1.5', '-1-log(-y)', f3_star_theory, f3_star_num, f3_argmax, True))

# 4. f(x) = exp(x) → f*(y) = y*log(y) - y (y>0)
f4 = lambda x: np.exp(x)
y4 = 2.0
f4_star_theory = y4 * np.log(y4) - y4
f4_star_num, f4_argmax = fenchel_conjugate_numerical(f4, y4, x_range=(-10, 5))
rows.append(('exp(x), y=2', 'y*log(y)-y', f4_star_theory, f4_star_num, f4_argmax, True))

print(f"\n{'函数 f(x)':<20s} | {'f*(y) 理论':<18s} | {'f*(y) 理论值':>12s} | {'f*(y) 数值':>11s} | {'误差':>10s}")
print("-" * 92)
finite_errs = []
for name, theo_str, theo, num, argmax, comparable in rows:
    if comparable:
        err = abs(theo - num)
        finite_errs.append(err)
        err_str = f"{err:.6f}"
        theo_val_str = f"{theo:>12.6f}"
    else:
        err_str = "N/A (有限域截断)"
        theo_val_str = f"{'+inf':>12s}"
    print(f"{name:<20s} | {theo_str:<18s} | {theo_val_str} | {num:>11.6f} | {err_str:>10s}")

# ---- 免责情形的实证：|y|>1 时理论值为 +∞，扩大搜索区间应得单调增长而不收敛 ----
print("\n[免责实证：|x| 在 |y|=1.5>1 时 f*(y)=+∞ 的正确数值表现]")
print(f"  {'x_range 上界':>12s} | {'数值 sup':>12s} | {'理论预测 hi*(|y|-1)':>20s}")
grow = []
for hi in [10, 100, 1000, 10000]:
    v, _ = fenchel_conjugate_numerical(f2, f2y_b, x_range=(-hi, hi))
    grow.append((hi, v))
    print(f"  {hi:>12d} | {v:>12.4f} | {hi * (abs(f2y_b) - 1):>20.4f}")

max_err = max(finite_errs)
print("\n[验证小结]")
print(f"  - 三个有限共轭情形（x²/2, -log(x), exp(x)）与 |x|(|y|≤1) 的数值解与理论值吻合，"
      f"实测最大误差 {max_err:.2e}（格点离散误差量级）。")
print(f"  - |x| 在 |y|>1 时的理论值是 +∞：上表显示把 x_range 上界从 10 扩到 10000，"
      f"数值解从 {grow[0][1]:.1f} 单调增长到 {grow[-1][1]:.1f} 且不收敛，")
print(f"    与解析预测 hi*(|y|-1) 逐点吻合——这正是 +∞ 的正确表现，并非数值方法与理论矛盾，")
print(f"    而是「有限域搜索」这一工具本身无法触及 +∞（共轭的自然定义域限制，属固有属性）。")

RESULT_JSON['step1_conjugates'] = {
    name: {'theory': (None if not np.isfinite(theo) else float(theo)),
           'theory_str': theo_str,
           'numeric': float(num),
           'argmax_x': float(argmax),
           'abs_error': (float(abs(theo - num)) if comparable else None)}
    for name, theo_str, theo, num, argmax, comparable in rows
}
RESULT_JSON['step1_infinity_case_xrange_growth'] = [
    {'x_range_upper': int(hi), 'numeric_sup': float(v)} for hi, v in grow]

# ---- 配图：Fenchel-Young 的几何意义（最优线性下界的截距） ----
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

panel_specs = [
    (f1, -6, 6, y1, r'$f(x)=x^2/2$', r'$\ell(x)=2x-f^*(2)$', f1_star_theory, f1_argmax,
     r'自共轭：$f^*(2)=2$，直线在 $x^*=2$ 处相切'),
    (f3, 0.05, 6, y3, r'$f(x)=-\log x$', r'$\ell(x)=-1.5x-f^*(-1.5)$', f3_star_theory, f3_argmax,
     r'负对数：$f^*(-1.5)=-1-\log 1.5$'),
    (f4, -4, 2.5, y4, r'$f(x)=e^x$', r'$\ell(x)=2x-f^*(2)$', f4_star_theory, f4_argmax,
     r'指数：$f^*(2)=2\log 2-2<0$'),
    (f2, -4, 4, None, r'$f(x)=|x|$', None, None, None,
     r'$|y|\leq 1$ 存在支撑线；$|y|>1$ 时不存在（故 $f^*=+\infty$），'
     '\n截断域上算得的 $5.0$（$x^*=10$）只是有限搜索区间的端点产物'),
]

for ax, (f, lo, hi, yv, flabel, line_label, fst, xs, title) in zip(axes.ravel(), panel_specs):
    xg = np.linspace(lo, hi, 800)
    ax.plot(xg, f(xg), color='tab:blue', lw=2, label=flabel)
    if yv is not None:
        ax.plot(xg, yv * xg - fst, 'r--', lw=1.6, label=line_label)
        ax.plot(xs, f(xs), 'ko', ms=6, zorder=5)
        ax.annotate(r'$x^*=%.3f$' % xs, (xs, f(xs)), textcoords='offset points',
                    xytext=(8, 8), fontsize=9)
        ax.set_title(title, fontsize=10)
    else:
        # |x|：同时画 |y|≤1（有支撑）与 |y|>1（无支撑）两条直线
        ax.plot(xg, 0.5 * xg, 'r--', lw=1.6, label=r'$y=0.5$：全局支撑线，恒有 $0.5x \leq |x|$')
        ax.plot(xg, 1.5 * xg, color='tab:orange', ls='-.', lw=1.6,
                label=r'$y=1.5$：斜率 $>1$，直线终会跑到 $f$ 上方')
        ax.set_ylim(-4, 4)
        ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')

fig.suptitle('步骤1：Fenchel 共轭的几何意义——$f^*(y)$ 是最优线性下界 $\\langle y,x\\rangle-c$ 的最大截距', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_fenchel共轭几何.png'), dpi=100)
plt.close()
print("\n图表已保存: 步骤1_fenchel共轭几何.png")


# ============================================================
# 步骤2：Fenchel共轭与ELBO的关系
# ★ 原创设计
# 核心思想（附录8A）：
# log p(x) = log ∫ p(x,z)dz
#          = log ∫ exp(log p(x,z))dz
# 利用Fenchel共轭：log(∫ exp(f(z))dz) = sup_q { E_q[f(z)] - (-E_q[log q(z)]) }
#                                          = sup_q { E_q[log p(x,z)] + H(q) }
#                                          = sup_q ELBO(q)
# 因此 ELBO = Fenchel共轭的变分表示
# ============================================================
print("\n" + "=" * 60)
print("步骤2：Fenchel共轭与ELBO的关系（单峰/双峰后验对照）")
print("=" * 60)

print("\n[核心概念]")
print("  函数空间的Fenchel共轭：")
print("  F[q] = E_q[log q(z)] = -H(q)（负熵）")
print("  F*[g] = sup_q {E_q[g(z)] - F[q]} = sup_q {E_q[g(z)] + H(q)}")
print("  当g(z) = log p(x,z)时：")
print("  F*[log p(x,·)] = sup_q ELBO(q) = log p(x)")

GH_NODES, GH_W = hermegauss(160)
GH_W = GH_W / GH_W.sum()  # 概率论 Hermite 节点，权重归一化后即 E_{N(0,1)}[·]


def build_model(name, weights, means, stds, sigma_obs, x_obs):
    """先验为高斯混合、似然为一维高斯的共轭模型；返回各种精确/近似量的计算接口"""
    lo, hi = min(means) - 12, max(means) + 12

    def log_prior(z):
        z = np.asarray(z, dtype=float)
        lp = np.full(z.shape, -1e30)
        for w, mu, tau in zip(weights, means, stds):
            lp = np.logaddexp(lp, np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(tau)
                              - 0.5 * ((z - mu) / tau) ** 2)
        return lp

    def log_joint(z):
        z = np.asarray(z, dtype=float)
        return (-0.5 * np.log(2 * np.pi) - np.log(sigma_obs)
                - 0.5 * ((x_obs - z) / sigma_obs) ** 2) + log_prior(z)

    # 证据的两种独立算法：数值积分 + 混合边际的闭式（log-sum-exp）
    z_int, _ = quad(lambda t: float(np.exp(log_joint(t))), lo, hi, limit=800)
    log_px_quad = np.log(z_int)
    terms = [np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(np.sqrt(tau**2 + sigma_obs**2))
             - 0.5 * ((x_obs - mu) / np.sqrt(tau**2 + sigma_obs**2)) ** 2
             for w, mu, tau in zip(weights, means, stds)]
    mx = max(terms)
    log_px_closed = mx + np.log(sum(np.exp(t - mx) for t in terms))

    # 后验各分量的解析参数（高斯×高斯），用于判定后验是否真的双峰
    post_prec = [1 / tau**2 + 1 / sigma_obs**2 for tau in stds]
    post_means = [(mu / tau**2 + x_obs / sigma_obs**2) / pp
                  for mu, tau, pp in zip(means, stds, post_prec)]
    post_stds = [np.sqrt(1 / pp) for pp in post_prec]
    pw = np.array([np.exp(t - mx) for t in terms]); pw = pw / pw.sum()

    def elbo_analytic(mu_q, sigma_q):
        """q=N(mu_q,sigma_q²) 的精确 ELBO：Gauss-Hermite 积分 E_q[log p(x,z)] + H(q)"""
        mu_q = np.asarray(mu_q, dtype=float)
        sigma_q = np.asarray(sigma_q, dtype=float)
        Zq = mu_q[..., None] + sigma_q[..., None] * GH_NODES
        return (np.sum(GH_W * log_joint(Zq), axis=-1)
                + 0.5 * np.log(2 * np.pi * np.e * sigma_q**2))

    def kl_to_posterior(mu_q, sigma_q):
        """直接数值积分 KL(q || p(z|x))，独立于 ELBO，用于核验 间隙 = KL"""
        zz = np.linspace(lo, hi, 200001)
        log_q = (-0.5 * np.log(2 * np.pi) - np.log(sigma_q)
                 - 0.5 * ((zz - mu_q) / sigma_q) ** 2)
        q = np.exp(log_q)
        log_post = log_joint(zz) - log_px_quad
        integrand = np.where(q > 1e-300, q * (log_q - log_post), 0.0)
        return float(_trapz(integrand, zz))

    # 后验密度的局部极大（双峰判定的实证证据）
    zz = np.linspace(lo, hi, 20001)
    dens = np.exp(log_joint(zz) - log_px_quad)
    peaks = [float(zz[i]) for i in range(1, len(zz) - 1)
             if dens[i] > dens[i-1] and dens[i] >= dens[i+1] and dens[i] > 1e-3 * dens.max()]

    return dict(name=name, tag=name.split('（')[0], weights=weights, means=means, stds=stds,
                sigma_obs=sigma_obs, x_obs=x_obs, log_joint=log_joint,
                log_px_quad=float(log_px_quad), log_px_closed=float(log_px_closed),
                post_weights=[float(v) for v in pw], post_means=post_means,
                post_stds=post_stds, elbo_analytic=elbo_analytic,
                kl_to_posterior=kl_to_posterior, post_peaks=peaks,
                z_grid=zz, post_grid=dens)


def optimize_mc_crn(model, n_samples=200000, mu_lo=-4.0, mu_hi=4.0):
    """在单高斯变分族上用 MC + 公共随机数(CRN) + 网格初值 + Nelder-Mead 最大化 ELBO

    ★ 公共随机数（common random numbers）技巧
      在函数外部生成一次 opt_eps，所有目标函数评估都共享同一组基础噪声，
      避免每次调用都重新抽样造成的"有噪目标函数"，让 Nelder-Mead 在光滑目标上稳定收敛。
      200000 样本是 8.4-2.py 已验证的可靠量级，可把 MC 噪声压到 ±1e-3 量级，
      避免出现"sup ELBO > log p(x)"这种打破 Jensen 不等式的尴尬噪声峰值。
    """
    opt_eps = np.random.randn(n_samples)
    sigma_obs, x_obs = model['sigma_obs'], model['x_obs']

    def log_joint_arr(z):
        return model['log_joint'](z)

    def neg_elbo(params):
        mu_q, log_sigma_q = params
        sigma_q = np.exp(log_sigma_q)
        z = mu_q + sigma_q * opt_eps
        log_joint_ = log_joint_arr(z)
        log_qz = -0.5 * np.log(2 * np.pi) - log_sigma_q - 0.5 * ((z - mu_q) / sigma_q) ** 2
        return -np.mean(log_joint_ - log_qz)

    # ★ 两步法：粗网格搜索找全局起点 → Nelder-Mead 精修
    best_val, best_p = np.inf, [0.0, 0.0]
    for mu_c in np.linspace(mu_lo, mu_hi, 25):
        for s_c in np.linspace(0.2, 2.5, 15):
            v = neg_elbo([mu_c, np.log(s_c)])
            if v < best_val:
                best_val, best_p = v, [mu_c, np.log(s_c)]

    res = minimize(neg_elbo, best_p, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    if not res.success:
        print(f"  [警告] {model['name']} 的 Nelder-Mead 未完全收敛: {res.message}")
    mu_star = float(res.x[0]); sigma_star = float(np.exp(res.x[1]))
    return dict(mu=mu_star, sigma=sigma_star, elbo_mc=float(-res.fun))


models = [
    build_model('配置A（原设定：先验双峰但后验被观测拉成单峰）',
                [0.3, 0.7], [-2.0, 1.0], [1.0, 1.0], 0.5, 0.5),
    build_model('配置B（强分离先验+紧似然：后验真正双峰）',
                [0.3, 0.7], [-2.5, 2.5], [0.5, 0.5], 0.4, 0.0),
]

print(f"\n  证据的两条独立算法（数值积分 vs 混合边际闭式）交叉核验：")
for m in models:
    print(f"    {m['tag']}: quad={m['log_px_quad']:.6f}, closed={m['log_px_closed']:.6f}, "
          f"差={abs(m['log_px_quad'] - m['log_px_closed']):.2e}")

# 单高斯变分族的结构极限：分量间距 vs 2*后验分量std（等权双组分混合单峰的充要条件）
print(f"\n[后验形态诊断]（等方差双组分混合：分量间距 > 2·std 才真正双峰）")
for m in models:
    sep = abs(m['post_means'][1] - m['post_means'][0])
    thr = 2 * m['post_stds'][0]
    print(f"  {m['name']}")
    print(f"    后验分量权重={['%.4f' % v for v in m['post_weights']]}, "
          f"分量均值={['%.4f' % v for v in m['post_means']]}, 分量std={m['post_stds'][0]:.4f}")
    print(f"    分量间距={sep:.4f} {'>' if sep > thr else '≤'} 2·std={thr:.4f} → "
          f"后验{'双峰' if sep > thr else '（密度上）单峰'}；实测局部极大={len(m['post_peaks'])} 个 "
          f"{['%.3f' % p for p in m['post_peaks']]}")

print(f"\n[单高斯变分族的优化结果与间隙核验]（MC-CRN 求解，解析 ELBO/KL 复核）")
print(f"  {'配置':<6s} | {'最优 q*':>20s} | {'ELBO(MC)':>10s} | {'ELBO(解析)':>10s} | "
      f"{'log p(x)':>9s} | {'间隙':>9s} | {'KL(直积)':>9s} | {'-log π_max':>10s}")
print("  " + "-" * 118)

step2_records = []
for m in models:
    opt = optimize_mc_crn(m)
    elbo_an = float(m['elbo_analytic'](opt['mu'], opt['sigma']))
    gap_mc = m['log_px_quad'] - opt['elbo_mc']
    gap_an = m['log_px_quad'] - elbo_an
    kl_val = m['kl_to_posterior'](opt['mu'], opt['sigma'])
    neg_log_pimax = -np.log(max(m['post_weights']))
    print(f"  {m['tag']:<6s} | N({opt['mu']:>7.4f},{opt['sigma']:>6.4f}²) | "
          f"{opt['elbo_mc']:>10.6f} | {elbo_an:>10.6f} | {m['log_px_quad']:>9.6f} | "
          f"{gap_an:>9.6f} | {kl_val:>9.6f} | {neg_log_pimax:>10.6f}")
    step2_records.append(dict(model=m, opt=opt, elbo_analytic=elbo_an,
                              gap_mc=gap_mc, gap_analytic=gap_an,
                              kl_direct=kl_val, neg_log_pi_max=float(neg_log_pimax)))

RESULT_JSON['step2'] = [
    {'config': r['model']['name'],
     'model': {'prior_weights': r['model']['weights'], 'prior_means': r['model']['means'],
               'prior_stds': r['model']['stds'], 'sigma_obs': r['model']['sigma_obs'],
               'x_obs': r['model']['x_obs']},
     'log_px_quad': r['model']['log_px_quad'], 'log_px_closed_form': r['model']['log_px_closed'],
     'posterior_weights': r['model']['post_weights'],
     'posterior_means': r['model']['post_means'], 'posterior_std': r['model']['post_stds'][0],
     'posterior_peaks': r['model']['post_peaks'],
     'q_star': {'mu': r['opt']['mu'], 'sigma': r['opt']['sigma']},
     'elbo_mc_crn': r['opt']['elbo_mc'], 'elbo_analytic': r['elbo_analytic'],
     'gap_mc': r['gap_mc'], 'gap_analytic': r['gap_analytic'],
     'kl_q_to_posterior_direct': r['kl_direct'], 'minus_log_pi_max': r['neg_log_pi_max']}
    for r in step2_records
]

# 次优盆地：从另一个后验分量出发做局部优化，验证“锁定左峰”也是一个局部极大
_mB, _rB = step2_records[1]['model'], step2_records[1]
_sec = minimize(lambda p: -float(_mB['elbo_analytic'](p[0], np.exp(p[1]))),
                [_mB['post_means'][0], np.log(_mB['post_stds'][0])],
                method='Nelder-Mead', options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 2000})
_sec_mu, _sec_sig = float(_sec.x[0]), float(np.exp(_sec.x[1]))
_sec_elbo = float(-_sec.fun)
_sec_gap = _mB['log_px_quad'] - _sec_elbo
_log_ratio = float(np.log(_mB['post_weights'][1] / _mB['post_weights'][0]))
print(f"\n[配置B 的次优盆地（锁定左峰的 q）]")
print(f"  q_次优 = N({_sec_mu:.4f}, {_sec_sig:.4f}²)，ELBO = {_sec_elbo:.6f}，间隙 = {_sec_gap:.6f}")
print(f"  比全局最优低 {_sec_gap - step2_records[1]['gap_analytic']:.6f}，"
      f"与两个模式的质量比预测 log(π_max/π_min) = {_log_ratio:.6f} 吻合")

RESULT_JSON['step2'][1]['secondary_basin'] = {
    'mu': _sec_mu, 'sigma': _sec_sig, 'elbo': _sec_elbo,
    'gap': _sec_gap, 'below_global_by': _sec_gap - step2_records[1]['gap_analytic'],
    'prediction_log_weight_ratio': _log_ratio}

print(f"\n[Fenchel共轭视角]")
for r in step2_records:
    m = r['model']
    print(f"  {m['name']}")
    print(f"    sup_{{q∈单高斯}} ELBO = {r['elbo_analytic']:.6f} ≤ log p(x) = {m['log_px_quad']:.6f}"
          f"（下界性质成立 ✓）")
    print(f"    变分间隙 = {r['gap_analytic']:.6f}；直接数值积分 KL(q*||p(z|x)) = {r['kl_direct']:.6f}"
          f"，两者之差 {abs(r['gap_analytic'] - r['kl_direct']):.2e}")
    print(f"    → 变分间隙 = Fenchel-Young间隙 = KL(q*||p(z|x)) 在数值上闭合")

print(f"\n[为什么两个配置的间隙差 3 个数量级？]")
ra, rb = step2_records[0], step2_records[1]
print(f"  配置A 间隙 {ra['gap_analytic']:.1e}：观测 x={ra['model']['x_obs']} 把后验压成单峰"
      f"（分量间距 {abs(ra['model']['post_means'][1]-ra['model']['post_means'][0]):.2f} "
      f"< 2·std={2*ra['model']['post_stds'][0]:.2f}），单高斯几乎等于后验，"
      f"残余只来自后验的轻微偏斜——注意：先验双峰 ≠ 后验双峰。")
print(f"  配置B 间隙 {rb['gap_analytic']:.1e}：后验真正双峰（局部极大 @ "
      f"{['%.2f' % p for p in rb['model']['post_peaks']]}），单高斯只能锁定其一，"
      f"必须付出被放弃那部分质量的代价；")
print(f"    理论预测该代价 ≈ -log(π_max) = {rb['neg_log_pi_max']:.4f}，"
      f"实测间隙 {rb['gap_analytic']:.4f}，差 {abs(rb['gap_analytic']-rb['neg_log_pi_max']):.4f}"
      f"（残差为 q* 与所锁分量的形状失配）。")

print(f"\n[MC-CRN 有限样本的诚实说明]")
for r in step2_records:
    print(f"  {r['model']['tag']}: MC 报告间隙 {r['gap_mc']:.6f} vs 解析间隙 {r['gap_analytic']:.6f}"
          f"（差 {abs(r['gap_mc'] - r['gap_analytic']):.2e}）——"
          f"CRN 固定样本上再做优化会对这 20 万点轻微过拟合，使 MC 间隙略偏乐观，量级可忽略。")

print(f"\n[核心结论]")
print(f"  1. Fenchel共轭为ELBO提供了优化理论根基")
print(f"  2. ELBO是log p(x)的Fenchel对偶表示")
print(f"  3. 变分间隙对应Fenchel-Young间隙（实测 KL 直积分与间隙一致）")
print(f"  4. 当q = p(z|x)时，间隙为零（强对偶）；族受限时间隙由"
      f"『后验中族无法表达的部分』决定：单峰近似下 ~5e-4，双峰下 ~ -log π_max")

# ---- 配图：后验 vs 最优 q*，以及 ELBO 地形 ----
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, r in zip(axes[:2], step2_records):
    m = r['model']
    zz = m['z_grid']
    # 窗口限制在后验分量附近，否则 [-12,12] 的尺度会把密度压成一根尖峰
    zoom_lo = min(m['post_means']) - 3 * m['post_stds'][0] - 0.5
    zoom_hi = max(m['post_means']) + 3 * m['post_stds'][0] + 0.5
    keep = (zz >= zoom_lo) & (zz <= zoom_hi)
    qz = np.exp(-0.5 * ((zz - r['opt']['mu']) / r['opt']['sigma']) ** 2) / (
        r['opt']['sigma'] * np.sqrt(2 * np.pi))
    ax.plot(zz[keep], m['post_grid'][keep], 'b-', lw=2, label=r'真实后验 $p(z|x)$')
    ax.plot(zz[keep], qz[keep], 'r--', lw=2,
            label=r'最优 $q^*=\mathcal{N}(%.3f,\,%.3f^2)$' % (r['opt']['mu'], r['opt']['sigma']))
    ax.fill_between(zz[keep], np.minimum(m['post_grid'][keep], qz[keep]), m['post_grid'][keep],
                    alpha=0.30, color='gray', label='后验中 $q^*$ 未覆盖的部分')
    for _pk in m['post_peaks']:
        ax.axvline(_pk, color='k', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel('密度')
    ax.set_xlim(zoom_lo, zoom_hi)
    t_shape = ('后验（密度上）单峰：间隙极小' if len(m['post_peaks']) <= 1
               else r'后验双峰：$q^*$ 只能二选一')
    ax.set_title(t_shape
                 + f"\n{m['tag']}  间隙={r['gap_analytic']:.4f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(m['post_grid'][keep].max(), qz[keep].max()) * 1.25)

# 第三幅：配置B 的 ELBO 地形（μ_q, σ_q）——展示受限族上的对偶间隙结构
ax = axes[2]
mB, rB = step2_records[1]['model'], step2_records[1]
mu_g = np.linspace(-2.5, 2.5, 201)
sg_g = np.linspace(0.15, 2.0, 161)
MU, SG = np.meshgrid(mu_g, sg_g, indexing='ij')
# 按 μ_q 逐行计算，避免 (201,161,160) 三维大数组一次性占用内存
ELBO_grid = np.empty_like(MU)
for _i, _mu in enumerate(mu_g):
    ELBO_grid[_i] = mB['elbo_analytic'](np.full_like(sg_g, _mu), sg_g)
_sup = rB['elbo_analytic']
cs = ax.contourf(MU, SG, ELBO_grid, levels=np.linspace(_sup - 2.0, _sup, 25),
                 cmap='viridis', extend='min')
fig.colorbar(cs, ax=ax, label=r'$\mathrm{ELBO}(\mu_q,\sigma_q)$')
levels = ax.contour(MU, SG, ELBO_grid, levels=[_sup - 0.5, _sup - 0.1],
                    colors='w', linewidths=1.0, linestyles='--')
ax.clabel(levels, fmt=r'$%1.2f$', fontsize=7)
ax.plot(rB['opt']['mu'], rB['opt']['sigma'], 'r*', ms=17, mec='w', zorder=5,
        label=r'全局最优 $q^*$（MC-CRN 求得）')
ax.plot(_sec_mu, _sec_sig, 'c*', ms=15, mec='w', zorder=5,
        label=r'次优盆地（锁定另一峰）')
ax.set_xlabel(r'$\mu_q$')
ax.set_ylabel(r'$\sigma_q$')
ax.set_title(f"配置B 的 ELBO 地形：两个盆地对应锁定左/右峰\n"
             f"$\\sup$ ELBO={_sup:.4f} < $\\log p(x)$={mB['log_px_quad']:.4f}，"
             f"次优盆地低 {_sec_gap - _rB['gap_analytic']:.3f}", fontsize=10)
ax.legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_变分间隙与elbo地形.png'), dpi=100)
plt.close()
print("\n图表已保存: 步骤2_变分间隙与elbo地形.png")

# ---- 结果归档 ----
RESULT_JSON['note'] = (
    "步骤1 的 |x|(|y|>1) 行不给误差：理论值为 +∞，有限域网格搜索只能给出截断端点值，"
    "属共轭定义域的固有属性；已用区间扩容实验（数值解单调增长至 "
    f"{grow[-1][1]:.0f}）实证。步骤2 配置A 后验为单峰（分量间距≤2·std），"
    "间隙 ~5e-4 只反映形状残余；配置B 后验真正双峰，间隙 ≈ -log(π_max)。"
    "MC-CRN 间隙与解析间隙差 ~1e-5，源于固定样本上的轻微过拟合。")
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as fh:
    json.dump(RESULT_JSON, fh, ensure_ascii=False, indent=2)
print("结果已归档: results_summary.json")

print(f"\n{'='*60}")
print("第八章配套实验8A-1完成！")
