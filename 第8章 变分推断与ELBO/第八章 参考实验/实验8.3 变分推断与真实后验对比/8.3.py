"""
实验8.3 变分推断与真实后验对比
对应章节：8.3（变分推断作为优化问题）
素材来源：🆕 新写
★ 原创设计：CAVI算法实现 + 平均场近似效果对比 + 前向KL vs 逆向KL

实验内容：
  步骤1：CAVI算法在1D高斯混合上的实现
  步骤2：平均场近似 vs 真实后验——变分间隙
  步骤3：前向KL vs 逆向KL——零强迫 vs 零避免

运行前提：纯NumPy/SciPy CPU即可
"""

import numpy as np
import os
import sys

# ====== Windows控制台UTF-8输出 ======
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib.pyplot as plt
import warnings
import logging

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

import platform
from matplotlib.font_manager import FontManager
def _find_chinese_font():
    candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong'] if platform.system() == 'Windows' else ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei']
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available:
            return font
    import os as _os, re
    for f in fm.ttflist:
        for pat in ['cjk', 'wqy', 'noto.*cjk', 'simhei']:
            if re.search(pat, f.name.lower()):
                return f.name
    return None

_cn_font = _find_chinese_font()
if _cn_font:
    plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['font.family'] = 'sans-serif'
    print(f"[Font] 已检测到中文字体: {_cn_font}")

np.random.seed(42)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()


# ============================================================
# 模型设定：1D高斯混合 + 高斯观测（同8.1）
# ============================================================
prior_weights = [0.3, 0.7]
prior_means = [-2.0, 1.0]
prior_stds = [1.0, 1.0]
sigma_obs = 0.5
x_obs = 0.5

print("=" * 60)
print("实验8.3：变分推断与真实后验对比")
print("=" * 60)

# 计算真实后验
def compute_posterior(x, weights, means, stds, sigma_obs):
    post_weights, post_means, post_stds = [], [], []
    for w, mu, tau in zip(weights, means, stds):
        s2 = 1.0 / (1.0 / sigma_obs**2 + 1.0 / tau**2)
        s = np.sqrt(s2)
        m = s2 * (x / sigma_obs**2 + mu / tau**2)
        ms = np.sqrt(sigma_obs**2 + tau**2)
        ml = -0.5 * ((x - mu) / ms)**2 - np.log(ms)
        post_weights.append(w * np.exp(ml))
        post_means.append(m)
        post_stds.append(s)
    total = sum(post_weights)
    post_weights = [pw / total for pw in post_weights]
    return post_weights, post_means, post_stds

post_w, post_m, post_s = compute_posterior(x_obs, prior_weights, prior_means, prior_stds, sigma_obs)

# log p(x)
def log_marginal(x, weights, means, stds, sigma_obs):
    terms = []
    for w, mu, tau in zip(weights, means, stds):
        ms = np.sqrt(sigma_obs**2 + tau**2)
        terms.append(np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(ms) - 0.5 * ((x - mu) / ms)**2)
    mx = max(terms)
    return mx + np.log(sum(np.exp(t - mx) for t in terms))

log_px = log_marginal(x_obs, prior_weights, prior_means, prior_stds, sigma_obs)

# PDF计算
def gm_pdf(x, weights, means, stds):
    pdf = np.zeros_like(x)
    for w, m, s in zip(weights, means, stds):
        pdf += w * np.exp(-0.5 * ((x - m) / s)**2) / (s * np.sqrt(2 * np.pi))
    return pdf


# ============================================================
# 步骤1：CAVI算法实现
# ★ 原创设计
# CAVI（坐标上升变分推断）在平均场假设下交替更新每个因子
# 对于高斯混合后验，CAVI的最优更新可以闭式计算
# ============================================================
print("\n" + "=" * 60)
print("步骤1：CAVI算法实现")
print("=" * 60)

# 在这个模型中，后验是双峰高斯混合
# 平均场近似 q(z) = N(z; μ_q, σ_q²)（单高斯）
# CAVI更新：q*(z) ∝ exp(E[log p(x,z)]) 但这里只有一个潜变量z
# 对于单变量，平均场就是单高斯，CAVI退化为优化μ_q和σ_q

# 用梯度下降优化ELBO（等价于CAVI的连续版本）
from scipy.optimize import minimize

def neg_elbo_gaussian(params):
    mu_q, log_sigma_q = params
    sigma_q = np.exp(log_sigma_q)
    z = np.random.randn(30000) * sigma_q + mu_q
    log_pxz = -0.5 * np.log(2 * np.pi) - np.log(sigma_obs) - 0.5 * ((x_obs - z) / sigma_obs)**2
    log_pz = np.full_like(z, -1e30)
    for w, mu, tau in zip(prior_weights, prior_means, prior_stds):
        log_comp = np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(tau) - 0.5 * ((z - mu) / tau)**2
        log_pz = np.logaddexp(log_pz, log_comp)
    log_joint = log_pxz + log_pz
    log_qz = -0.5 * np.log(2 * np.pi) - np.log(sigma_q) - 0.5 * ((z - mu_q) / sigma_q)**2
    return -np.mean(log_joint - log_qz)

# CAVI迭代（交替优化μ和σ）
print("CAVI迭代过程（交替优化μ_q和σ_q）：")

def compute_elbo_samples(mu_q, sigma_q, n_samples=30000):
    """给定q=N(mu_q, sigma_q²)，数值计算ELBO"""
    np.random.seed(42)
    z = np.random.randn(n_samples) * sigma_q + mu_q
    log_pxz = -0.5 * np.log(2 * np.pi) - np.log(sigma_obs) - 0.5 * ((x_obs - z) / sigma_obs)**2
    log_pz = np.full_like(z, -1e30)
    for w, mu, tau in zip(prior_weights, prior_means, prior_stds):
        log_comp = np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(tau) - 0.5 * ((z - mu) / tau)**2
        log_pz = np.logaddexp(log_pz, log_comp)
    log_joint = log_pxz + log_pz
    log_qz = -0.5 * np.log(2 * np.pi) - np.log(sigma_q) - 0.5 * ((z - mu_q) / sigma_q)**2
    return np.mean(log_joint - log_qz)

mu_q, sigma_q = 0.0, 1.0  # 初始化为先验均值
elbo_history = []

for iteration in range(20):
    # 固定σ，在μ方向搜索
    best_mu_curr, best_elbo_curr = mu_q, -1e10
    for mu_candidate in np.linspace(-3, 3, 100):
        elbo_val = compute_elbo_samples(mu_candidate, sigma_q)
        if elbo_val > best_elbo_curr:
            best_elbo_curr = elbo_val
            best_mu_curr = mu_candidate
    mu_q = best_mu_curr
    
    # 固定μ，在σ方向搜索
    best_sigma_curr, best_elbo_curr = sigma_q, -1e10
    for log_sigma_candidate in np.linspace(-1, 1, 50):
        sigma_candidate = np.exp(log_sigma_candidate)
        elbo_val = compute_elbo_samples(mu_q, sigma_candidate)
        if elbo_val > best_elbo_curr:
            best_elbo_curr = elbo_val
            best_sigma_curr = sigma_candidate
    sigma_q = best_sigma_curr
    
    elbo = compute_elbo_samples(mu_q, sigma_q)
    elbo_history.append(elbo)
    
    if iteration % 5 == 0:
        print(f"  iter {iteration:2d}: mu_q={mu_q:.4f}, sigma_q={sigma_q:.4f}, ELBO={elbo:.4f}")

# 为了避免上面的CAVI实现过于复杂，直接用联合优化
np.random.seed(42)
result = minimize(neg_elbo_gaussian, [0.0, 0.0], method='Nelder-Mead',
                  options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
best_mu, best_log_sigma = result.x
best_sigma = np.exp(best_log_sigma)
best_elbo = -result.fun

print(f"\n变分最优 q*(z) = N({best_mu:.4f}, {best_sigma:.4f}²)")
print(f"最优 ELBO = {best_elbo:.4f}")
print(f"log p(x) = {log_px:.4f}")
print(f"变分间隙 = {log_px - best_elbo:.4f}")


# ============================================================
# 步骤2：平均场近似 vs 真实后验
# ★ 原创设计
# ============================================================
print("\n" + "=" * 60)
print("步骤2：平均场近似 vs 真实后验")
print("=" * 60)

z_grid = np.linspace(-5, 5, 500)

# 真实后验
posterior_pdf = gm_pdf(z_grid, post_w, post_m, post_s)

# 变分近似（单高斯）
variational_pdf = np.exp(-0.5 * ((z_grid - best_mu) / best_sigma)**2) / (best_sigma * np.sqrt(2 * np.pi))

# 先验
prior_pdf = gm_pdf(z_grid, prior_weights, prior_means, prior_stds)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 对比图
axes[0].fill_between(z_grid, posterior_pdf, alpha=0.3, color='blue', label='True posterior p(z|x)')
axes[0].plot(z_grid, posterior_pdf, 'b-', lw=2)
axes[0].fill_between(z_grid, variational_pdf, alpha=0.3, color='red', label=f'Variational q*(z)=N({best_mu:.2f},{best_sigma:.2f}²)')
axes[0].plot(z_grid, variational_pdf, 'r--', lw=2)
axes[0].plot(z_grid, prior_pdf, 'k:', lw=1, alpha=0.5, label='Prior p(z)')
axes[0].set_xlabel('z')
axes[0].set_ylabel('Density')
axes[0].set_title('Average-field approximation vs True posterior')
axes[0].legend()
axes[0].grid(alpha=0.3)

# ELBO迭代收敛
axes[1].plot(range(len(elbo_history)), elbo_history, 'b-o', markersize=3)
axes[1].axhline(y=log_px, color='r', linestyle='--', lw=2, label=f'log p(x) = {log_px:.4f}')
axes[1].axhline(y=best_elbo, color='g', linestyle=':', lw=2, label=f'Best ELBO = {best_elbo:.4f}')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('ELBO')
axes[1].set_title('CAVI convergence process')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(_SCRIPT_DIR, '步骤2_平均场vs真实后验.png'), dpi=150)
plt.show()

print(f"\n平均场近似的局限：")
print(f"  真实后验是双峰（w=[{post_w[0]:.3f}, {post_w[1]:.3f}]）")
print(f"  单高斯q*只能覆盖一个峰，无法捕获多模态")
print(f"  变分间隙 = {log_px - best_elbo:.4f}（来自平均场的表达能力限制）")


# ============================================================
# 步骤3：前向KL vs 逆向KL
# ★ 原创设计
# 逆向KL: KL(q||p) = E_q[log q/p]  → 零避免（mode-seeking）
# 前向KL: KL(p||q) = E_p[log p/q]  → 零强迫（mean-seeking/coverage）
# 变分推断使用逆向KL——这解释了为什么q倾向于覆盖一个峰
# ============================================================
print("\n" + "=" * 60)
print("步骤3：前向KL vs 逆向KL")
print("=" * 60)

def kl_reverse(q_pdf, p_pdf, z_grid):
    """逆向KL(q||p) = ∫ q(z) log(q(z)/p(z)) dz"""
    dz = z_grid[1] - z_grid[0]
    mask = q_pdf > 1e-10
    return np.sum(q_pdf[mask] * np.log(q_pdf[mask] / (p_pdf[mask] + 1e-30))) * dz

def kl_forward(p_pdf, q_pdf, z_grid):
    """前向KL(p||q) = ∫ p(z) log(p(z)/q(z)) dz"""
    dz = z_grid[1] - z_grid[0]
    mask = p_pdf > 1e-10
    return np.sum(p_pdf[mask] * np.log(p_pdf[mask] / (q_pdf[mask] + 1e-30))) * dz

# 搜索使逆向KL最小的单高斯q（≈变分推断的结果）
# 搜索使前向KL最小的单高斯q
from scipy.optimize import minimize as sp_minimize

def neg_elbo_for_opt(params):
    """-ELBO = KL(q||p) + const，优化这个等价于最小化逆向KL"""
    mu_q, log_sigma_q = params
    sigma_q = np.exp(log_sigma_q)
    q_pdf = np.exp(-0.5 * ((z_grid - mu_q) / sigma_q)**2) / (sigma_q * np.sqrt(2 * np.pi))
    return kl_reverse(q_pdf, posterior_pdf, z_grid)

def forward_kl_for_opt(params):
    mu_q, log_sigma_q = params
    sigma_q = np.exp(log_sigma_q)
    q_pdf = np.exp(-0.5 * ((z_grid - mu_q) / sigma_q)**2) / (sigma_q * np.sqrt(2 * np.pi))
    return kl_forward(posterior_pdf, q_pdf, z_grid)

# 逆向KL最优
res_rev = sp_minimize(neg_elbo_for_opt, [0.0, 0.0], method='Nelder-Mead', options={'maxiter': 5000})
rev_mu, rev_sigma = res_rev.x[0], np.exp(res_rev.x[1])

# 前向KL最优
res_fwd = sp_minimize(forward_kl_for_opt, [0.0, 0.0], method='Nelder-Mead', options={'maxiter': 5000})
fwd_mu, fwd_sigma = res_fwd.x[0], np.exp(res_fwd.x[1])

print(f"逆向KL(q||p)最优: q = N({rev_mu:.4f}, {rev_sigma:.4f}²), KL = {res_rev.fun:.4f}")
print(f"  → 零避免：q集中在一个峰，避免在p≈0的区域有概率质量")
print(f"前向KL(p||q)最优: q = N({fwd_mu:.4f}, {fwd_sigma:.4f}²), KL = {res_fwd.fun:.4f}")
print(f"  → 零强迫：q覆盖所有峰，方差更大以确保p>0的地方q>0")

# 可视化
q_rev = np.exp(-0.5 * ((z_grid - rev_mu) / rev_sigma)**2) / (rev_sigma * np.sqrt(2 * np.pi))
q_fwd = np.exp(-0.5 * ((z_grid - fwd_mu) / fwd_sigma)**2) / (fwd_sigma * np.sqrt(2 * np.pi))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 逆向KL
axes[0].fill_between(z_grid, posterior_pdf, alpha=0.3, color='blue', label='True p(z|x)')
axes[0].plot(z_grid, posterior_pdf, 'b-', lw=2)
axes[0].fill_between(z_grid, q_rev, alpha=0.3, color='red')
axes[0].plot(z_grid, q_rev, 'r--', lw=2, label=f'Reverse KL: N({rev_mu:.2f},{rev_sigma:.2f}²)')
axes[0].set_xlabel('z')
axes[0].set_title('Reverse KL(q||p) — Zero-avoiding (mode-seeking)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 前向KL
axes[1].fill_between(z_grid, posterior_pdf, alpha=0.3, color='blue', label='True p(z|x)')
axes[1].plot(z_grid, posterior_pdf, 'b-', lw=2)
axes[1].fill_between(z_grid, q_fwd, alpha=0.3, color='green')
axes[1].plot(z_grid, q_fwd, 'g--', lw=2, label=f'Forward KL: N({fwd_mu:.2f},{fwd_sigma:.2f}²)')
axes[1].set_xlabel('z')
axes[1].set_title('Forward KL(p||q) — Zero-forcing (mean-seeking)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(_SCRIPT_DIR, '步骤3_前向KL_vs_逆向KL.png'), dpi=150)
plt.show()

print(f"\n前向KL vs 逆向KL的行为差异：")
print(f"  逆向KL(q||p)：q在p≈0处必须≈0（否则q/p→∞），所以q覆盖单个峰")
print(f"  前向KL(p||q)：p>0处q必须>0（否则p/q→∞），所以q覆盖所有峰")
print(f"  变分推断使用逆向KL → 近似倾向于零避免/覆盖单个峰")
print(f"  MCMC采样（第4章）渐近地从前验采样 → 自然覆盖所有峰")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验8.3 总结")
print("=" * 60)
print("1. CAVI算法：在平均场假设下交替优化变分因子")
print("   对1D问题，CAVI退化为优化单高斯的μ和σ")
print("2. 平均场近似vs真实后验：")
print("   单高斯q*无法拟合双峰后验→变分间隙>0")
print("   间隙来源：平均场的表达能力限制")
print("3. 前向KL vs 逆向KL：")
print("   逆向KL(q||p)：零避免/覆盖单峰——变分推断的选择")
print("   前向KL(p||q)：零强迫/覆盖所有峰——矩匹配/E-M的选择")
