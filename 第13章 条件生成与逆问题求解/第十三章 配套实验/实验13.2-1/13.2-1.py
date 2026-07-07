# -*- coding: utf-8 -*-
"""
实验13.2-1：1D逆问题原理演示

★ 原创设计：1D逆问题原理演示（先理解原理再看图像）
  问题：y = Ax + n，已知y和A，求x的后验p(x|y)
  扩散后验采样：在逆向SDE中加入似然梯度

实验内容：
  - 1D高斯混合先验 + VP-SDE框架
  - 无条件采样 vs 条件采样对比
  - 后验采样的均值更接近观测值（似然项的约束效果）

本实验纯NumPy/PyTorch CPU即可运行。
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
    SAVE_DIR = os.path.join(_gdrive, '实验13.2-1')
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
print("实验13.2-1: 1D逆问题原理演示")
print("=" * 60)
print("知识点: 条件采样 vs 无条件采样，后验p(x|y) vs 先验p(x)")


# ============================================================
# 1D高斯混合先验 + VP-SDE框架
# ============================================================
def gm1d_pdf(x, weights=[0.3, 0.7], means=[-2, 1], stds=[1, 1]):
    pdf = np.zeros_like(x)
    for w, m, s in zip(weights, means, stds):
        pdf += w * np.exp(-0.5 * ((x - m) / s)**2) / (s * np.sqrt(2 * np.pi))
    return pdf

def vp_marginal(t, beta_min=0.1, beta_max=20.0):
    log_mean = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    mean_t = np.exp(log_mean)
    std_t = np.sqrt(1 - np.exp(2 * log_mean))
    return mean_t, std_t

def vp_beta(t, beta_min=0.1, beta_max=20.0):
    return beta_min + t * (beta_max - beta_min)

def vp_score_analytic(x, t, beta_min=0.1, beta_max=20.0):
    mean_t, std_t = vp_marginal(t, beta_min, beta_max)
    pdf = np.zeros_like(x)
    dpdf = np.zeros_like(x)
    weights = [0.3, 0.7]
    means = [-2, 1]
    stds = [1, 1]
    for w, m, s in zip(weights, means, stds):
        new_mean = mean_t * m
        new_std = np.sqrt(mean_t**2 * s**2 + std_t**2)
        pdf += w * np.exp(-0.5 * ((x - new_mean) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
        dpdf += w * (-(x - new_mean) / new_std**2) * np.exp(-0.5 * ((x - new_mean) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
    return dpdf / (pdf + 1e-30)


# ============================================================
# 步骤1：1D逆问题原理——扩散后验采样
# ============================================================
print("\n" + "=" * 60)
print("步骤1：1D逆问题原理——扩散后验采样")
print("=" * 60)

# 逆问题设置：去噪 y = x + n
print("\n1D逆问题：去噪 y = x + n, sigma_obs=0.5")
sigma_obs = 0.5

# 从先验p(x)抽一个真实信号（逆问题的核心叙事）
if np.random.rand() < 0.3:
    x0_star = np.random.randn() - 2  # 从第一个高斯分量抽样
else:
    x0_star = np.random.randn() + 1  # 从第二个高斯分量抽样

# 观测：y = x0_star + n
y_obs = x0_star + sigma_obs * np.random.randn()
print(f"  真实信号 x0_star: {x0_star:.4f}")
print(f"  观测值 y: {y_obs:.4f} (真实信号 + 观测噪声)")

# 采样用的粒子数
N_particles = 5000

# 似然梯度: nabla log p(y|x) = (y - x) / sigma_obs^2
# 后验得分 = 先验得分 + 似然梯度
# nabla log p(x|y) = nabla log p(x) + nabla log p(y|x)

def posterior_score_vp(x, t, y_obs, sigma_obs, zeta=1.0):
    """后验得分 = 先验得分 + 似然梯度（DPS近似）

    DPS近似（Chung et al. 2022）：
    在逆向SDE的每一步，用Tweedie估计的x_hat_0计算似然梯度

    参数:
        zeta: 似然项的权重系数（简化起见取1.0，完整DPS会根据梯度范数自适应调节）
    """
    mean_t, std_t = vp_marginal(t)
    prior_score = vp_score_analytic(x, t)
    x0_hat = (x + std_t**2 * prior_score) / (mean_t + 1e-10)
    likelihood_grad = mean_t / (sigma_obs**2) * (y_obs - x0_hat)
    return prior_score + zeta * likelihood_grad

# ---- 无条件采样（仅先验）----
def unconditional_sample(N_particles, N_steps=500, T=1.0):
    """VP-SDE逆向采样（无条件，仅使用先验得分，逆时参数化）"""
    h = T / N_steps
    x = np.random.randn(N_particles)
    trajectory = [x.copy()]
    for i in range(N_steps):
        t = T - i * h
        beta_t = vp_beta(t)
        score = vp_score_analytic(x, t)
        x = x + beta_t * h * (0.5 * x + score) + np.sqrt(beta_t * h) * np.random.randn(N_particles)
        trajectory.append(x.copy())
    return np.array(trajectory)

# ---- 条件采样（后验）----
def conditional_sample(y_obs, sigma_obs, N_particles, N_steps=500, T=1.0):
    """VP-SDE逆向后验采样（条件于观测y，DPS方法，逆时参数化）"""
    h = T / N_steps
    x = np.random.randn(N_particles)
    trajectory = [x.copy()]
    for i in range(N_steps):
        t = T - i * h
        beta_t = vp_beta(t)
        score = posterior_score_vp(x, t, y_obs, sigma_obs)
        x = x + beta_t * h * (0.5 * x + score) + np.sqrt(beta_t * h) * np.random.randn(N_particles)
        trajectory.append(x.copy())
    return np.array(trajectory)

# 运行采样
np.random.seed(42)
traj_uncond = unconditional_sample(N_particles, 300)
np.random.seed(42)
traj_cond = conditional_sample(y_obs, sigma_obs, N_particles, 300)

uncond_final = traj_uncond[-1]
cond_final = traj_cond[-1]

print(f"\n无条件采样: 均值={np.mean(uncond_final):.4f}, 方差={np.var(uncond_final):.4f}")
print(f"后验采样:   均值={np.mean(cond_final):.4f}, 方差={np.var(cond_final):.4f}")
print(f"真实信号:   x0_star={x0_star:.4f} (从先验中抽取的单一信号)")
print(f"观测值:     y={y_obs:.4f}")
sys.stdout.flush()  # 确保输出立即显示
print(f"\n关键：后验采样的均值更接近观测值y（似然项的约束效果）")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

x_hist = np.linspace(-6, 6, 500)

# (a) 采样分布对比
axes[0].hist(uncond_final, bins=60, density=True, alpha=0.5, color='blue',
             range=(-6, 6), label='无条件采样 p(x)')
axes[0].hist(cond_final, bins=60, density=True, alpha=0.5, color='red',
             range=(-6, 6), label='条件采样 p(x|y)')
axes[0].plot(x_hist, gm1d_pdf(x_hist), 'k--', lw=2, alpha=0.7, label='先验 p(x)')
axes[0].axvline(x0_star, color='orange', linestyle='-', lw=2, label=r'真实信号 $x_0^\star={:.2f}$'.format(x0_star))
axes[0].axvline(y_obs, color='green', linestyle=':', lw=2, label=r'观测 $y={:.2f}$'.format(y_obs))
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('概率密度', fontsize=12)
axes[0].set_title('(a) 无条件 vs 条件采样分布', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# (b) 采样轨迹
time_steps = np.linspace(0, 1, traj_uncond.shape[0])
for i, t_idx in enumerate([0, traj_uncond.shape[0]//4, traj_uncond.shape[0]//2, traj_uncond.shape[0]-1]):
    axes[1].hist(traj_uncond[t_idx], bins=40, density=True, alpha=0.3, color='blue', range=(-6, 6))
    axes[1].hist(traj_cond[t_idx], bins=40, density=True, alpha=0.3, color='red', range=(-6, 6))
axes[1].axvline(x0_star, color='orange', linestyle='-', lw=2, label=r'真实信号 $x_0^\star$')
axes[1].axvline(y_obs, color='green', linestyle=':', lw=2, label=r'观测 $y={:.2f}$'.format(y_obs))
axes[1].set_xlabel('x', fontsize=12)
axes[1].set_ylabel('概率密度', fontsize=12)
axes[1].set_title('(b) 采样轨迹演化（蓝=无条件, 红=条件）', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

# (c) 均值与方差对比
metrics = [r'均值 $\mu$', r'方差 $\sigma^2$', r'$|x-y|$均值', r'$|x-x_0^\star|$均值']
uncond_vals = [np.mean(uncond_final), np.var(uncond_final),
               np.mean(np.abs(uncond_final - y_obs)), np.mean(np.abs(uncond_final - x0_star))]
cond_vals = [np.mean(cond_final), np.var(cond_final),
             np.mean(np.abs(cond_final - y_obs)), np.mean(np.abs(cond_final - x0_star))]

x_pos = np.arange(len(metrics))
width = 0.35
axes[2].bar(x_pos - width/2, uncond_vals, width, label='无条件采样', color='blue', alpha=0.7)
axes[2].bar(x_pos + width/2, cond_vals, width, label='条件采样', color='red', alpha=0.7)
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(metrics, fontsize=11)
axes[2].set_title('(c) 采样统计量对比', fontsize=13)
axes[2].legend(fontsize=10)
axes[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '1D逆问题原理演示.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图已保存: {fig_path}")

print("\n" + "=" * 60)
print("实验13.2-1 完成!")
print("=" * 60)
print("""
关键结论:
1. 1D逆问题原理（13.2节）
   - 扩散后验采样 = 逆向SDE + 似然梯度
   - nabla log p(x_t|y) = nabla log p(x_t) + zeta * nabla log p(y|x_hat_{0|t})
   - 本实验取 zeta=1.0（简化起见），完整DPS会根据梯度范数自适应调节
   - 似然梯度将无条件采样推向观测一致区域

2. 条件 vs 无条件采样
   - 无条件采样：分布接近先验 p(x)
   - 条件采样：分布更接近观测 y，方差更小（后验更集中）
   - 后验采样的 |x-y|均值和 |x-x0_star|均值均显著小于无条件采样

3. 物理意义
   - 似然项起"软约束"作用：把粒子从整个先验空间拉向 y 附近的区域
   - 同时先验项保持"多样性"：不会坍缩到单点（与MAP的本质区别）
   - 后验采样能反映出真实信号 x0_star 来自哪个高斯分量
""")
