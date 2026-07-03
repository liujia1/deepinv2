# -*- coding: utf-8 -*-
"""
实验13.3-1：DPS的Tweedie闭环验证
对应章节：13.3.2节 DPS深度剖析

★ 原创设计：DPS的Tweedie闭环验证
  DPS核心近似：p(x_0|x_t) ~ delta(x_0 - x_hat_{0|t})
  -> p(y|x_t) ~ p(y|x_hat_{0|t})（积分坍缩为单点求值）

实验内容：
  - 验证DPS近似似然得分 vs 精确似然得分
  - 含Jacobian与忽略Jacobian两种DPS变体
  - 误差随噪声水平t的变化规律

本实验不需要GPU，通过1D解析情形验证DPS近似质量。
"""

import sys
import io
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import logging
import warnings

# 设置控制台输出为 UTF-8 (Windows下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# 静默matplotlib相关警告
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验13.3-1')
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
    print("警告: chinese_font模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)

print("\n" + "=" * 60)
print("实验13.3-1: DPS的Tweedie闭环验证")
print("=" * 60)
print("对应章节: 13.3.2节 DPS深度剖析")
print("知识点: Tweedie等式, delta函数近似, 似然得分的Laplace近似")


# ============================================================
# 1D高斯混合先验 + VP-SDE框架
# ============================================================
GM_WEIGHTS = [0.3, 0.7]
GM_MEANS = [-2.0, 1.0]
GM_STDS = [1.0, 1.0]

BETA_MIN, BETA_MAX = 0.1, 20.0

def gm1d_pdf(x):
    pdf = np.zeros_like(x)
    for w, m, s in zip(GM_WEIGHTS, GM_MEANS, GM_STDS):
        pdf += w * np.exp(-0.5 * ((x - m) / s)**2) / (s * np.sqrt(2 * np.pi))
    return pdf

def vp_marginal(t):
    log_mean = -0.25 * t**2 * (BETA_MAX - BETA_MIN) - 0.5 * t * BETA_MIN
    mean_t = np.exp(log_mean)
    std_t = np.sqrt(1 - np.exp(2 * log_mean))
    return mean_t, std_t

def vp_score_analytic(x, t):
    mean_t, std_t = vp_marginal(t)
    pdf = np.zeros_like(x)
    dpdf = np.zeros_like(x)
    for w, m, s in zip(GM_WEIGHTS, GM_MEANS, GM_STDS):
        new_mean = mean_t * m
        new_std = np.sqrt(mean_t**2 * s**2 + std_t**2)
        pdf += w * np.exp(-0.5 * ((x - new_mean) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
        dpdf += w * (-(x - new_mean) / new_std**2) * np.exp(-0.5 * ((x - new_mean) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
    return dpdf / (pdf + 1e-30)

def tweedie_estimate(x_t, t):
    """Tweedie估计: E[x_0 | x_t] = (x_t + std_t^2 * score) / mean_t"""
    mean_t, std_t = vp_marginal(t)
    score = vp_score_analytic(x_t, t)
    return (x_t + std_t**2 * score) / (mean_t + 1e-10)

def _compute_likelihood(x_t, t, y_obs, A, sigma_y):
    mean_t, std_t = vp_marginal(t)
    x0_grid = np.linspace(-8, 8, 2000)
    dx0 = x0_grid[1] - x0_grid[0]
    p_xt_given_x0 = np.exp(-0.5 * ((x_t - mean_t * x0_grid) / std_t)**2) / (std_t * np.sqrt(2 * np.pi))
    p_x0 = gm1d_pdf(x0_grid)
    p_x0_given_xt = p_xt_given_x0 * p_x0
    p_y_given_x0 = np.exp(-0.5 * ((y_obs - A * x0_grid) / sigma_y)**2) / (sigma_y * np.sqrt(2 * np.pi))
    return np.sum(p_y_given_x0 * p_x0_given_xt) * dx0

def likelihood_grad_analytic(x_t, t, y_obs, A, sigma_y):
    eps = 1e-4
    p_plus = _compute_likelihood(x_t + eps, t, y_obs, A, sigma_y)
    p_minus = _compute_likelihood(x_t - eps, t, y_obs, A, sigma_y)
    return (p_plus - p_minus) / (2 * eps * (p_plus + p_minus) / 2 + 1e-30)


# ============================================================
# 步骤1：DPS的Tweedie闭环验证（13.3.2节）
# ============================================================
print("\n" + "=" * 60)
print("步骤1：DPS的Tweedie闭环验证（13.3.2节）")
print("=" * 60)

print("""
DPS核心近似（13.3.2节）：
  p(x_0|x_t) ~ delta(x_0 - x_hat_{0|t})  （delta函数近似）
  -> p(y|x_t) ~ p(y|x_hat_{0|t})       （积分坍缩为单点求值）

Tweedie闭环（13.2.3节）：
  得分函数 -> Tweedie估计x_hat_0 -> 一致性梯度 -> 似然得分近似 -> 修正得分

验证：比较DPS近似的似然得分与精确似然得分
""")

# 逆问题设置
A_val = 1.0
sigma_y = 0.5
y_obs = 0.5

t_val = 0.3
x_test = np.linspace(-4, 4, 200)

# 精确似然得分
exact_ll_scores = np.zeros_like(x_test)
for i, xi in enumerate(x_test):
    eps_d = 1e-4
    p_plus = _compute_likelihood(xi + eps_d, t_val, y_obs, A_val, sigma_y)
    p_minus = _compute_likelihood(xi - eps_d, t_val, y_obs, A_val, sigma_y)
    exact_ll_scores[i] = (p_plus - p_minus) / (2 * eps_d * (p_plus + p_minus) / 2 + 1e-30)

# DPS近似似然得分
x0_hat = tweedie_estimate(x_test, t_val)
dps_residual = (y_obs - A_val * x0_hat) / sigma_y**2

eps_j = 1e-4
x0_hat_plus = tweedie_estimate(x_test + eps_j, t_val)
x0_hat_minus = tweedie_estimate(x_test - eps_j, t_val)
grad_x0_hat = (x0_hat_plus - x0_hat_minus) / (2 * eps_j)

dps_approx_full = grad_x0_hat * dps_residual
dps_approx_simple = dps_residual

print("DPS近似 vs 精确似然得分 (t=0.3, y=0.5, A=1, sigma_y=0.5):")
print(f"{'x_t':>6s}  {'精确':>10s}  {'DPS(含Jacobian)':>16s}  {'DPS(忽略Jacobian)':>18s}  {'相对误差(含)':>12s}  {'相对误差(略)':>12s}")
print("-" * 80)
for idx in [40, 60, 80, 100, 120, 140, 160]:
    xi = x_test[idx]
    exact = exact_ll_scores[idx]
    dps_f = dps_approx_full[idx]
    dps_s = dps_approx_simple[idx]
    err_f = abs(dps_f - exact) / (abs(exact) + 1e-10)
    err_s = abs(dps_s - exact) / (abs(exact) + 1e-10)
    print(f"{xi:6.2f}  {exact:10.4f}  {dps_f:16.4f}  {dps_s:18.4f}  {err_f:12.4f}  {err_s:12.4f}")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Tweedie估计
axes[0].plot(x_test, x_test, 'k--', lw=1, alpha=0.5, label=r'$x_t$ (恒等)')
axes[0].plot(x_test, x0_hat, 'b-', lw=2, label=r'Tweedie $\hat{x}_{0|t}$')
axes[0].axhline(y_obs, color='r', linestyle=':', lw=1.5, label=r'观测 $y={}$'.format(y_obs))
axes[0].set_xlabel(r'$x_t$', fontsize=12)
axes[0].set_ylabel('去噪估计', fontsize=12)
axes[0].set_title(f'(a) Tweedie估计 (t={t_val})', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# (b) 似然得分对比
mask = np.abs(x_test) < 3.5
axes[1].plot(x_test[mask], exact_ll_scores[mask], 'b-', lw=2, label=r'精确 $\nabla\log p(y|x_t)$')
axes[1].plot(x_test[mask], dps_approx_full[mask], 'r--', lw=2, label='DPS近似 (含Jacobian)')
axes[1].plot(x_test[mask], dps_approx_simple[mask], 'g:', lw=2, label='DPS近似 (忽略Jacobian)')
axes[1].set_xlabel(r'$x_t$', fontsize=12)
axes[1].set_ylabel('似然得分', fontsize=12)
axes[1].set_title(f'(b) DPS近似 vs 精确似然得分 (t={t_val})', fontsize=13)
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

# (c) DPS误差随t的变化
t_range = np.linspace(0.05, 0.95, 20)
x_test_point = 0.5
dps_errors_full = []
dps_errors_simple = []
for t_r in t_range:
    exact_val = float(likelihood_grad_analytic(np.array([x_test_point]), t_r, y_obs, A_val, sigma_y))
    x0h = float(tweedie_estimate(np.array([x_test_point]), t_r))
    x0h_p = float(tweedie_estimate(np.array([x_test_point + eps_j]), t_r))
    x0h_m = float(tweedie_estimate(np.array([x_test_point - eps_j]), t_r))
    gx = (x0h_p - x0h_m) / (2 * eps_j)
    residual = (y_obs - A_val * x0h) / sigma_y**2
    dps_f = gx * residual
    dps_s = residual
    dps_errors_full.append(abs(dps_f - exact_val) / (abs(exact_val) + 1e-10))
    dps_errors_simple.append(abs(dps_s - exact_val) / (abs(exact_val) + 1e-10))

axes[2].semilogy(t_range, dps_errors_full, 'r-o', markersize=4, label='DPS近似 (含Jacobian)')
axes[2].semilogy(t_range, dps_errors_simple, 'g-s', markersize=4, label='DPS近似 (忽略Jacobian)')
axes[2].set_xlabel('t (噪声水平)', fontsize=12)
axes[2].set_ylabel('相对误差', fontsize=12)
axes[2].set_title('(c) DPS近似误差 vs 噪声水平', fontsize=13)
axes[2].legend(fontsize=10)
axes[2].grid(alpha=0.3)
axes[2].annotate('t大->噪声高->x_hat_0不可靠\n->DPS近似误差增大',
                xy=(0.65, 0.75), xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeaa7', alpha=0.8))

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, 'DPS的Tweedie闭环验证.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图已保存: {fig_path}")

print("""
关键发现：
  1. DPS近似的似然得分方向正确，但幅度在高噪声时偏差增大
  2. 忽略Jacobian项的DPS简化版在高噪声时误差更大
  3. 这解释了13.3.2节指出的DPS局限：高噪声时delta函数近似质量下降
""")

print("\n" + "=" * 60)
print("实验13.3-1 完成!")
print("=" * 60)
print("""
关键结论:
1. DPS的Tweedie闭环（13.3.2节）
   - DPS用delta函数近似将不可解积分转化为单点求值
   - 近似误差在高噪声时增大（Tweedie估计不可靠）
   - 忽略Jacobian项是进一步的简化，引入额外误差

2. 实际启示
   - DPS在高噪声时（t大）需要更小的引导权重zeta
   - 这是13.4.3节介绍的"时变引导权重"方案的动机
""")
