# -*- coding: utf-8 -*-
"""
实验13.4-1：引导权重ζ权衡曲线
对应章节：13.4.3节 引导权重与质量-多样性权衡

★ 原创设计：固定随机种子，用不同zeta执行后验采样，
  对比采样分布的均值（->数据一致性）和方差（->多样性）

实验内容：
  - 不同zeta值下的后验采样分布
  - zeta-数据一致性 / zeta-多样性权衡曲线
  - 与第2-3章正则化参数lambda的类比

本实验不需要GPU，通过1D解析情形研究zeta效应。
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
    SAVE_DIR = os.path.join(_gdrive, '实验13.4-1')
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
print("实验13.4-1: 引导权重zeta权衡曲线")
print("=" * 60)
print("对应章节: 13.4.3节 引导权重与质量-多样性权衡")
print("知识点: 引导权重zeta, 质量-多样性权衡, Tweedie一致性")


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
    """VP-SDE后验采样（DPS近似），可调引导权重zeta"""
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
# 步骤1：引导权重zeta与质量-多样性权衡
# ============================================================
print("\n" + "=" * 60)
print("步骤1：引导权重zeta与质量-多样性权衡（13.4.3节）")
print("=" * 60)

print("""
13.4.3节：引导权重zeta控制先验与似然的相对强度
  nabla log p(x_t|y) ~ nabla log p(x_t) + zeta * nabla log p(y|x_hat_{0|t})

  zeta大 -> 强数据一致性（低多样性）-> 类似MAP
  zeta小 -> 强先验（高多样性）-> 类似无条件采样
""")

# 逆问题设置
A_val = 1.0
sigma_y = 0.5
y_obs = 0.5

zeta_values = [0.0, 0.3, 0.7, 1.0, 2.0, 5.0]
sampling_results = {}

for zeta in zeta_values:
    samples = dps_posterior_sample(y_obs, A_val, sigma_y, zeta)
    sampling_results[zeta] = samples
    mean_s = np.mean(samples)
    std_s = np.std(samples)
    consistency = np.mean(np.abs(samples - y_obs))
    print(f"  zeta={zeta:4.1f}: 均值={mean_s:6.3f}, 标准差={std_s:5.3f}, |x-y|均值={consistency:5.3f}")

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
x_hist = np.linspace(-6, 6, 500)

for idx, zeta in enumerate(zeta_values):
    ax = axes[idx // 3, idx % 3]
    samples = sampling_results[zeta]
    ax.hist(samples, bins=60, density=True, alpha=0.5, color='steelblue',
            range=(-6, 6), label=r'后验采样 ($\zeta={}$)'.format(zeta))
    ax.plot(x_hist, gm1d_pdf(x_hist), 'k--', lw=1.5, alpha=0.7, label='先验 p(x)')
    ax.axvline(y_obs, color='red', linestyle=':', lw=2, label=r'观测 $y={}$'.format(y_obs))
    mean_s = np.mean(samples)
    ax.axvline(mean_s, color='blue', linestyle='-', lw=1.5, alpha=0.7, label=r'采样均值={:.2f}'.format(mean_s))
    ax.set_title(r'$\zeta$ = {} ({})'.format(zeta, "无条件" if zeta == 0 else "弱引导" if zeta < 0.5 else "标准" if zeta < 1.5 else "强引导"), fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 0.8)

fig.suptitle('引导权重zeta与质量-多样性权衡（13.4.3节）', fontsize=14, y=1.01)
plt.tight_layout()
fig_path1 = os.path.join(SAVE_DIR, '引导权重权衡.png')
plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图1已保存: {fig_path1}")

# 权衡曲线
consistency_list = []
diversity_list = []
for zeta in zeta_values:
    samples = sampling_results[zeta]
    consistency_list.append(np.mean(np.abs(samples - y_obs)))
    diversity_list.append(np.std(samples))

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(diversity_list, consistency_list, 'ro-', markersize=8, lw=2)
for i, zeta in enumerate(zeta_values):
    ax.annotate(r'$\zeta={}$'.format(zeta), (diversity_list[i], consistency_list[i]),
                textcoords="offset points", xytext=(10, 5), fontsize=10)
ax.set_xlabel('多样性（采样标准差）', fontsize=12)
ax.set_ylabel(r'数据一致性（$|x-y|$均值）', fontsize=12)
ax.set_title('质量-多样性权衡曲线（13.4.3节）', fontsize=13)
ax.grid(alpha=0.3)
ax.annotate('右上: 弱引导(高多样性, 低一致性)\n左下: 强引导(低多样性, 高一致性)\n对应第2-3章的正则化参数lambda',
            xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#dfe6e9', alpha=0.8))

plt.tight_layout()
fig_path2 = os.path.join(SAVE_DIR, '权衡曲线.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig_path2}")

print("\n" + "=" * 60)
print("实验13.4-1 完成!")
print("=" * 60)
print("""
关键结论:
1. 引导权重zeta（13.4.3节）原创设计
   - zeta控制先验与似然的相对强度
   - zeta大->数据一致性强、多样性低
   - zeta小->先验贡献大、多样性高
   - 对应第2-3章正则化参数lambda的角色

2. 实际启示
   - zeta=0: 无条件采样（忽略观测）
   - zeta=1: 标准DPS（平衡先验与似然）
   - zeta过大: 类似MAP，多样性坍缩
   - 最优zeta通常在0.5-1.5之间
""")
