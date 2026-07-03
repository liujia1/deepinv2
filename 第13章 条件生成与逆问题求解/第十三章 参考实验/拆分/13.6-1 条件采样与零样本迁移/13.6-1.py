# -*- coding: utf-8 -*-
"""
实验13.6-1：条件采样与零样本迁移
对应章节：13.6节 闭环：回到逆问题（条件扩散的三大优势）

素材来源：实验13.1-步骤4

★ 原创设计：固定随机种子，对比无条件 vs 条件采样
  13.1节/13.6节叙事：
  - 无条件采样：从p(x)生成"看起来像数据"的样本
  - 条件采样：从p(x|y)生成"看起来像数据且与观测一致"的样本
  - 逆问题求解 = 条件扩散采样

扩散模型作为先验的三大优势（13.6节）：
  1. 任意复杂先验（从数据学习，非手工设计）
  2. 不确定性量化（多次采样→后验方差）
  3. 零样本迁移（同一模型，不同A和y）

本实验不需要GPU，通过1D解析情形展示条件扩散的三大优势。
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import warnings

# ====== 解决中文乱码的核心代码 ======
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
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
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
# ========================================================

np.random.seed(42)

SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()


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

def vp_beta(t):
    return BETA_MIN + t * (BETA_MAX - BETA_MIN)

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


def dps_posterior_sample(y_obs, A, sigma_y, zeta, N_particles=5000, N_steps=300, T=1.0, seed=42):
    """VP-SDE后验采样（DPS近似）"""
    np.random.seed(seed)
    h = T / N_steps
    x = np.random.randn(N_particles)
    for i in range(N_steps):
        t = T - i * h
        beta_t = vp_beta(t)
        mean_t, std_t = vp_marginal(t)
        prior_score = vp_score_analytic(x, t)
        x0_hat = (x + std_t**2 * prior_score) / (mean_t + 1e-10)
        likelihood_grad = mean_t * (y_obs - A * x0_hat) / sigma_y**2
        posterior_score = prior_score + zeta * likelihood_grad
        x = x + beta_t * h * (0.5 * x + posterior_score) + np.sqrt(beta_t * h) * np.random.randn(N_particles)
    return x


# ============================================================
# 步骤1：无条件采样 vs 条件采样
# ============================================================
print("=" * 60)
print("步骤1：无条件采样 vs 条件采样（13.1节/13.6节）")
print("=" * 60)

print("""
13.1节/13.6节叙事：
  - 无条件采样：从p(x)生成"看起来像数据"的样本
  - 条件采样：从p(x|y)生成"看起来像数据且与观测一致"的样本
  - 逆问题求解 = 条件扩散采样

扩散模型作为先验的三大优势（13.6节）：
  1. 任意复杂先验（从数据学习，非手工设计）
  2. 不确定性量化（多次采样→后验方差）
  3. 零样本迁移（同一模型，不同A和y）
""")

A_val = 1.0
sigma_y = 0.5
y_obs = 0.5
N_particles = 10000

def unconditional_sample(N, N_steps=300, T=1.0):
    np.random.seed(42)
    h = T / N_steps
    x = np.random.randn(N)
    for i in range(N_steps):
        t = T - i * h
        beta_t = vp_beta(t)
        score = vp_score_analytic(x, t)
        x = x + beta_t * h * (0.5 * x + score) + np.sqrt(beta_t * h) * np.random.randn(N)
    return x

# 无条件采样
uncond_samples = unconditional_sample(N_particles)
cond_samples = dps_posterior_sample(y_obs, A_val, sigma_y, zeta=1.0, N_particles=N_particles)

# 多次条件采样（不同随机种子）→ 不确定性量化
multi_cond = []
for seed in [42, 123, 456, 789, 1024]:
    s = dps_posterior_sample(y_obs, A_val, sigma_y, zeta=1.0, N_particles=2000, seed=seed)
    multi_cond.append(s)

# 不同A值的零样本迁移
A_different = 0.5
y_obs_diff = A_different * 0.5
cond_different = dps_posterior_sample(y_obs_diff, A_different, sigma_y, zeta=1.0, N_particles=N_particles, seed=42)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
x_hist = np.linspace(-6, 6, 500)

# (a) 无条件 vs 条件
axes[0, 0].hist(uncond_samples, bins=60, density=True, alpha=0.5, color='blue',
                range=(-6, 6), label='无条件采样 p(x)')
axes[0, 0].hist(cond_samples, bins=60, density=True, alpha=0.5, color='red',
                range=(-6, 6), label='条件采样 p(x|y)')
axes[0, 0].plot(x_hist, gm1d_pdf(x_hist), 'k--', lw=2, alpha=0.7, label='先验 p(x)')
axes[0, 0].axvline(y_obs, color='green', linestyle=':', lw=2, label=f'观测 y={y_obs}')
axes[0, 0].set_title('(a) 无条件采样 vs 条件采样', fontsize=13)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(alpha=0.3)

# (b) 不确定性量化
for i, s in enumerate(multi_cond):
    axes[0, 1].hist(s, bins=40, density=True, alpha=0.3, range=(-6, 6))
axes[0, 1].axvline(y_obs, color='green', linestyle=':', lw=2, label=f'观测 y={y_obs}')
all_cond = np.concatenate(multi_cond)
axes[0, 1].hist(all_cond, bins=60, density=True, alpha=0.7, color='purple',
                range=(-6, 6), label='聚合后验')
axes[0, 1].set_title('(b) 不确定性量化：5次独立后验采样', fontsize=13)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(alpha=0.3)

# (c) 零样本迁移
axes[1, 0].hist(cond_samples, bins=60, density=True, alpha=0.5, color='red',
                range=(-6, 6), label=f'A=1.0, y={y_obs}')
axes[1, 0].hist(cond_different, bins=60, density=True, alpha=0.5, color='orange',
                range=(-6, 6), label=f'A=0.5, y={y_obs_diff}')
axes[1, 0].set_title('(c) 零样本迁移：同一模型，不同正向算子A', fontsize=13)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(alpha=0.3)

# (d) 全书叙事闭环
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

steps = [
    (1, 8, 'Ch1\n逆问题', '#e74c3c'),
    (2.5, 8, 'Ch3\nMAP', '#e67e22'),
    (4, 8, 'Ch5\nPnP-ULA', '#3498db'),
    (5.5, 8, 'Ch7\nScore SDE', '#2ecc71'),
    (7, 8, 'Ch12\nDSM≡VLB', '#9b59b6'),
    (8.5, 8, 'Ch13\n条件扩散', '#e74c3c'),
]

for x_pos, y_pos, label, color in steps:
    ax.add_patch(plt.Circle((x_pos, y_pos), 0.4, color=color, alpha=0.3))
    ax.text(x_pos, y_pos, label, fontsize=8, ha='center', va='center', fontweight='bold')

for i in range(len(steps) - 1):
    ax.annotate('', xy=(steps[i+1][0] - 0.5, steps[i+1][1]),
                xytext=(steps[i][0] + 0.5, steps[i][1]),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

ax.annotate('闭环', xy=(8.5, 7.5), xytext=(1, 7.0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2, connectionstyle='arc3,rad=-0.3'),
            fontsize=12, color='red', fontweight='bold')

ax.set_title('(d) 全书叙事闭环（13.1/13.6节）', fontsize=13)
ax.axis('off')

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '条件采样与零样本迁移.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图已保存: {fig_path}")

print(f"\n关键观察:")
print(f"  无条件采样均值: {np.mean(uncond_samples):.3f} (接近先验均值)")
print(f"  条件采样均值:   {np.mean(cond_samples):.3f} (更接近观测y={y_obs})")
print(f"  条件采样标准差:  {np.std(cond_samples):.3f} (< 先验标准差，后验更集中)")
print(f"  多次采样均值范围: [{min(np.mean(s) for s in multi_cond):.3f}, {max(np.mean(s) for s in multi_cond):.3f}]")
print(f"  → 不确定性量化：不同采样给出不同解，体现后验分布的完整性")

print(f"\n{'='*60}")
print("实验13.6-1 完成!")
print("=" * 60)
print("""
关键结论:
1. 条件扩散的三大优势（13.1/13.6节）
   - 任意复杂先验（从数据学习）
   - 不确定性量化（多次采样→后验方差）
   - 零样本迁移（同一模型，不同A和y）

2. 实践意义
   - 无条件采样：用于生成、补全、超分
   - 条件采样：用于逆问题求解、不确定性量化
   - 零样本迁移：训练一次模型，应对多种A（去噪/去模糊/inpainting）
""")
