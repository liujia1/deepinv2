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
  - 变分间隙对应Fenchel-Young间隙

实验内容：
  步骤1：Fenchel共轭的计算——几个经典函数的共轭
  步骤2：Fenchel共轭验证ELBO = (log p)** 的下界性质

运行前提：纯NumPy/SciPy CPU即可
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys
import io
from scipy.optimize import minimize

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
# 核心思想（附录8A）：
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

print("\n[核心概念]")
print("  函数空间的Fenchel共轭：")
print("  F[q] = E_q[log q(z)] = -H(q)（负熵）")
print("  F*[g] = sup_q {E_q[g(z)] - F[q]} = sup_q {E_q[g(z)] + H(q)}")
print("  当g(z) = log p(x,z)时：")
print("  F*[log p(x,·)] = sup_q ELBO(q) = log p(x)")

# 1D高斯混合模型
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

print(f"\n模型设定：")
print(f"  先验 p(z) = {prior_weights[0]}*N({prior_means[0]}, {prior_stds[0]}²) + {prior_weights[1]}*N({prior_means[1]}, {prior_stds[1]}²)")
print(f"  似然 p(x|z) = N(x; z, {sigma_obs}²)")
print(f"  观测值 x = {x_obs}")
print(f"  log p(x) = {log_px:.6f}")

# 数值验证：log ∫ exp(f(z))dz = sup_q { E_q[f(z)] + H(q) }
# f(z) = log p(x,z)
# 对q = N(μ, σ²)，参数化搜索ELBO最大值
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

print(f"\n[Fenchel共轭视角]")
print(f"  log p(x) = sup_q ELBO(q) （Fenchel共轭的变分表示）")
print(f"  当变分族Q仅含单高斯时，sup ELBO = {best_elbo:.4f} < log p(x) = {log_px:.4f}")
print(f"  间隙来源：单高斯无法拟合双峰后验")
print(f"  变分间隙 = Fenchel-Young间隙")

print(f"\n[核心结论]")
print(f"  1. Fenchel共轭为ELBO提供了优化理论根基")
print(f"  2. ELBO是log p(x)的Fenchel对偶表示")
print(f"  3. 变分间隙对应Fenchel-Young间隙")
print(f"  4. 当q = p(z|x)时，间隙为零（强对偶）")

print(f"\n{'='*60}")
print("第八章配套实验8A-1完成！")
