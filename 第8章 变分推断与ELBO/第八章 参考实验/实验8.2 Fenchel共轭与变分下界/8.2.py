"""
实验8.2 Fenchel共轭与变分下界
对应章节：8.4（变分推断与正则化的统一视角）、附录8A
素材来源：MIVAcourse_opt1 理论 + 编程，🔄 需设计
★ 原创设计：Fenchel共轭的数值计算与ELBO的凸共轭视角

实验内容：
  步骤1：Fenchel共轭的计算——几个经典函数的共轭
  步骤2：Fenchel共轭验证ELBO = (log p)⁎⁎ 的下界性质
  步骤3：ELBO作为重建+正则——从变分下界到正则化的统一

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
# 步骤1：Fenchel共轭的计算
# f*(y) = sup_x { <x, y> - f(x) }
# ★ 原创设计：数值验证经典函数的共轭
# ============================================================
print("=" * 60)
print("步骤1：Fenchel共轭的计算与验证")
print("=" * 60)

def fenchel_conjugate_numerical(f, y, x_range=(-10, 10), n_points=10000):
    """数值计算Fenchel共轭 f*(y) = sup_x { x*y - f(x) }"""
    x = np.linspace(x_range[0], x_range[1], n_points)
    values = x * y - f(x)
    idx = np.argmax(values)
    return values[idx], x[idx]

# 经典函数及其解析共轭
print(f"\n{'函数 f(x)':<25s} | {'f*(y) 理论':<25s} | {'f*(y) 数值':>10s} | {'误差':>10s}")
print("-" * 80)

# 1. f(x) = x²/2 → f*(y) = y²/2
f1 = lambda x: x**2 / 2
y_test = 2.0
f1_star_theory = y_test**2 / 2
f1_star_num, _ = fenchel_conjugate_numerical(f1, y_test)
print(f"{'x²/2':<25s} | {'y²/2':<25s} | {f1_star_num:>10.6f} | {abs(f1_star_theory - f1_star_num):>10.6f}")

# 2. f(x) = |x| → f*(y) = 0 if |y|≤1, +∞ otherwise
f2 = lambda x: np.abs(x)
for y_val in [0.5, 1.5]:
    f2_star_theory = 0.0 if abs(y_val) <= 1 else float('inf')
    f2_star_num, _ = fenchel_conjugate_numerical(f2, y_val)
    theory_str = "0" if abs(y_val) <= 1 else "+inf"
    print(f"{'|x|, y='+str(y_val):<25s} | {theory_str:<25s} | {f2_star_num:>10.6f} | {'N/A' if f2_star_theory == float('inf') else f'{abs(f2_star_theory - f2_star_num):.6f}':>10s}")

# 3. f(x) = -log(x) (x>0) → f*(y) = -1-log(-y) (y<0)
f3 = lambda x: -np.log(np.maximum(x, 1e-10))
y_neg = -1.5
f3_star_theory = -1 - np.log(-y_neg)
f3_star_num, _ = fenchel_conjugate_numerical(f3, y_neg, x_range=(0.01, 20))
print(f"{'-log(x), y=-1.5':<25s} | {'-1-log(-y)':<25s} | {f3_star_num:>10.6f} | {abs(f3_star_theory - f3_star_num):>10.6f}")

# 4. f(x) = exp(x) → f*(y) = y*log(y) - y (y>0)
f4 = lambda x: np.exp(x)
y_pos = 2.0
f4_star_theory = y_pos * np.log(y_pos) - y_pos
f4_star_num, _ = fenchel_conjugate_numerical(f4, y_pos, x_range=(-10, 5))
print(f"{'exp(x), y=2':<25s} | {'y*log(y)-y':<25s} | {f4_star_num:>10.6f} | {abs(f4_star_theory - f4_star_num):>10.6f}")

print("\n数值计算与理论值高度一致——Fenchel共轭公式验证成功")


# ============================================================
# 步骤2：Fenchel共轭与ELBO的关系
# ★ 原创设计
# 核心思想（8.4节，附录8A）：
# log p(x) = log ∫ p(x,z)dz
#          = log ∫ exp(log p(x,z))dz
# 利用Fenchel共轭：log(∫ exp(f(z))dz) = sup_q { E_q[f(z)] - (-E_q[log q(z)]) }
#                                          = sup_q { E_q[log p(x,z)] + H(q) }
#                                          = sup_q ELBO(q)
# 因此 ELBO = Fenchel共轭的变分表示
# ============================================================
print("\n" + "=" * 60)
print("步骤2：Fenchel共轭与ELBO的关系")
print("=" * 60)

# 1D高斯混合模型（同8.1）
prior_weights = [0.3, 0.7]
prior_means = [-2.0, 1.0]
prior_stds = [1.0, 1.0]
sigma_obs = 0.5
x_obs = 0.5

# log p(x)的精确值
def log_marginal(x, weights, means, stds, sigma_obs):
    terms = []
    for w, mu, tau in zip(weights, means, stds):
        ms = np.sqrt(sigma_obs**2 + tau**2)
        terms.append(np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(ms) - 0.5 * ((x - mu) / ms)**2)
    mx = max(terms)
    return mx + np.log(sum(np.exp(t - mx) for t in terms))

log_px = log_marginal(x_obs, prior_weights, prior_means, prior_stds, sigma_obs)

# 数值验证：log ∫ exp(f(z))dz = sup_q { E_q[f(z)] + H(q) }
# f(z) = log p(x,z)
# 对q = N(μ, σ²)，参数化搜索ELBO最大值
from scipy.optimize import minimize_scalar, minimize

def neg_elbo_gaussian(params):
    """对q=N(μ,σ²)计算-ELBO"""
    mu_q, log_sigma_q = params
    sigma_q = np.exp(log_sigma_q)
    n_samples = 30000
    z = np.random.randn(n_samples) * sigma_q + mu_q

    # log p(x,z)
    log_pxz = -0.5 * np.log(2 * np.pi) - np.log(sigma_obs) - 0.5 * ((x_obs - z) / sigma_obs)**2
    log_pz = np.full_like(z, -1e30)
    for w, mu, tau in zip(prior_weights, prior_means, prior_stds):
        log_comp = np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(tau) - 0.5 * ((z - mu) / tau)**2
        log_pz = np.logaddexp(log_pz, log_comp)
    log_joint = log_pxz + log_pz

    # log q(z)
    log_qz = -0.5 * np.log(2 * np.pi) - np.log(sigma_q) - 0.5 * ((z - mu_q) / sigma_q)**2

    return -np.mean(log_joint - log_qz)

np.random.seed(42)
result = minimize(neg_elbo_gaussian, [0.0, 0.0], method='Nelder-Mead',
                  options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
best_mu, best_log_sigma = result.x
best_sigma = np.exp(best_log_sigma)
best_elbo = -result.fun

print(f"\n单高斯变分族 q=N(μ,σ²) 的最优解：")
print(f"  最优 q* = N({best_mu:.4f}, {best_sigma:.4f}²)")
print(f"  最优 ELBO = {best_elbo:.4f}")
print(f"  log p(x) = {log_px:.4f}")
print(f"  变分间隙 = {log_px - best_elbo:.4f}")
print(f"\nFenchel共轭视角：")
print(f"  log p(x) = sup_q ELBO(q) （Fenchel共轭的变分表示）")
print(f"  当变分族Q仅含单高斯时，sup ELBO = {best_elbo:.4f} < log p(x) = {log_px:.4f}")
print(f"  间隙来源：单高斯无法拟合双峰后验")


# ============================================================
# 步骤3：ELBO = 重建 + 正则——与第3章MAP的统一
# ★ 原创设计
# ELBO = E_q[log p(x|z)] - KL(q||p(z))
#       [重建项]          [正则项]
# MAP: -log p(y|x) + λ·R(x)  = 数据项 + 正则项
# ============================================================
print("\n" + "=" * 60)
print("步骤3：ELBO = 重建 + 正则——变分推断与正则化的统一")
print("=" * 60)

# 在q=N(μ,σ²)的参数化下，扫描μ和σ，观察ELBO分解的变化
mu_range = np.linspace(-3, 3, 100)
sigma_range = [0.3, 0.5, 1.0, 2.0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for sigma_q in sigma_range:
    reconstruction_terms = []
    kl_terms = []
    elbos_scan = []

    for mu_q in mu_range:
        np.random.seed(42)
        z = np.random.randn(20000) * sigma_q + mu_q

        # E_q[log p(x|z)] = 重建项
        recon = np.mean(-0.5 * np.log(2 * np.pi) - np.log(sigma_obs) - 0.5 * ((x_obs - z) / sigma_obs)**2)

        # KL(q||p(z)) = 正则项
        log_pz = np.full_like(z, -1e30)
        for w, mu, tau in zip(prior_weights, prior_means, prior_stds):
            log_comp = np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(tau) - 0.5 * ((z - mu) / tau)**2
            log_pz = np.logaddexp(log_pz, log_comp)
        log_qz = -0.5 * np.log(2 * np.pi) - np.log(sigma_q) - 0.5 * ((z - mu_q) / sigma_q)**2
        kl = np.mean(log_qz - log_pz)

        reconstruction_terms.append(recon)
        kl_terms.append(kl)
        elbos_scan.append(recon - kl)

    axes[0].plot(mu_range, reconstruction_terms, label=f'σ_q={sigma_q}')

    if sigma_q == 1.0:
        axes[1].plot(mu_range, elbos_scan, 'b-', lw=2, label='ELBO = Recon - KL')
        axes[1].plot(mu_range, reconstruction_terms, 'r--', label='Reconstruction')
        axes[1].plot(mu_range, [-k for k in kl_terms], 'g--', label='-KL(q||p)')
        axes[1].axhline(y=log_px, color='k', linestyle=':', lw=1, label=f'log p(x)={log_px:.2f}')

axes[0].set_xlabel('q的均值 μ')
axes[0].set_ylabel('重建项 E[log p(x|z)]')
axes[0].set_title('重建项随q均值的变化')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xlabel('q的均值 μ')
axes[1].set_ylabel('值')
axes[1].set_title('ELBO = 重建 - 正则 (σ_q=1.0)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(_SCRIPT_DIR, '步骤3_重建与正则.png'), dpi=150)
plt.show()

print("\nELBO = 重建项 - KL正则项 的统一视角（8.4节）：")
print("  重建项 E[log p(x|z)]：q的均值接近x时重建项最大")
print("  KL正则项 KL(q||p(z))：q接近先验时正则项最小")
print("  ELBO最大值 = 两者权衡的最优点")
print("  这与第3章MAP估计的数据项+正则项结构完全对应")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验8.2 总结")
print("=" * 60)
print("1. Fenchel共轭 f*(y) = sup_x {<x,y> - f(x)} 的数值验证")
print("   x²/2 → y²/2, |x| → I(|y|≤1), -log(x) → -1-log(-y)")
print("2. ELBO = Fenchel共轭的变分表示：")
print("   log p(x) = sup_q ELBO(q) = sup_q {E_q[log p(x,z)] + H(q)}")
print("3. ELBO = 重建 + 正则 的统一视角：")
print("   最大化重建质量 + 最小化与先验的偏离")
print("   这与第3章MAP的数据项+正则项结构对应")
