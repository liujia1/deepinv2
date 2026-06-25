# -*- coding: utf-8 -*-
"""
实验7.1-1 从Langevin到扩散——连续时间推广
对应章节: 7.1（从Langevin到扩散：连续时间推广）
素材来源: ★ 原创设计，基于7.1节"Langevin→扩散SDE连续时间推广"

实验内容:
  步骤1: 1D高斯混合上，单步Langevin vs 多步扩散采样对比
  步骤2: 从离散噪声调度到连续SDE——β(t)调度可视化
  步骤3: "离散→连续→再离散"认知螺旋的数值验证

运行前提: 纯NumPy/PyTorch CPU即可
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys
import io
import warnings
import logging

# 设置控制台输出为 UTF-8 (Windows 下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# 静默 matplotlib 相关警告
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
    SAVE_DIR = os.path.join(_gdrive, '实验7.1-1')
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
import torch
torch.manual_seed(42)


# ============================================================
# 1D高斯混合分布
# ============================================================
def gm1d_pdf(x):
    """真实分布 p(x) = 0.3·N(-2,1) + 0.7·N(1,1) 的密度"""
    return 0.3*np.exp(-0.5*(x+2)**2)/np.sqrt(2*np.pi) + \
           0.7*np.exp(-0.5*(x-1)**2)/np.sqrt(2*np.pi)

def gm1d_score_sigma(x, sigma):
    """被高斯核 N(0,σ²) 平滑后的得分函数 ∇log p_σ(x)。
    sigma=0 时退化为真实分布的得分 ∇log p(x)。"""
    v = 1 + sigma**2
    p1 = np.exp(-0.5*(x+2)**2/v)/np.sqrt(2*np.pi*v)
    p2 = np.exp(-0.5*(x-1)**2/v)/np.sqrt(2*np.pi*v)
    p = 0.3*p1 + 0.7*p2
    return (0.3*-(x+2)/v*p1 + 0.7*-(x-1)/v*p2) / p


# ============================================================
# 步骤1：单步Langevin vs 多步扩散采样对比
# ★ 原创设计：在1D高斯混合上直观展示NCSN的三个根本局限
# ============================================================
print("=" * 60)
print("步骤1：单步Langevin vs 多步扩散采样对比")
print("=" * 60)

# 方法1：单步Langevin（固定σ，使用"较粗"的得分函数做所有步骤）
def langevin_1d(n_samples, n_iter, delta, sigma_score):
    """单步Langevin：用固定σ_score的得分函数采样整个过程。
    ★ 当σ_score较大时（远大于真实分布的"细粒度"尺度），
       得分函数是p被高斯核平滑后的版本∇log p_{σ_score}，
       模态间信号变弱，链难以跨越模态——这正是NCSN要解决的失败模式。"""
    x = np.random.randn(n_samples) * 3
    for _ in range(n_iter):
        score = gm1d_score_sigma(x, sigma_score)
        x = x + delta * score + np.sqrt(2*delta) * np.random.randn(n_samples)
    return x

# 方法2：退火Langevin（多步扩散，第6章NCSN）
def annealed_langevin_1d(n_samples, sigmas, T_per_level, epsilon):
    """退火Langevin：从大σ到小σ逐步采样。
    ★ 注意：NCSN原始公式 alpha_i = ε·(σ_i/σ_L)² 在粗粒度级别会产生较大步长
       （本例 σ_0=5 时 alpha≈20），但大σ下 score 量级按 1/σ² 衰减，
       使得 |alpha·score| 保持有限，1D 演示中数值稳定。
       教学用途下可放心使用；移植到高维或其他分布时应给 alpha 施加上界。"""
    x = np.random.randn(n_samples) * sigmas[0]
    for i, sigma in enumerate(sigmas):
        alpha = epsilon * (sigma / sigmas[-1])**2
        for _ in range(T_per_level):
            score = gm1d_score_sigma(x, sigma)
            x = x + alpha/2 * score + np.sqrt(alpha) * np.random.randn(n_samples)
    return x

# 运行对比
# ★ 关键：单步Langevin的sigma_score=1.0（远大于真实分布的尺度），
#   模拟"用一个粗粒度的score函数做所有步骤"的失败模式。
#   此时p_σ = 0.3*N(-2,2) + 0.7*N(1,2)，模态被平滑，
#   即使链收敛也只能采样到p_σ≠p，因此模态比例会偏离真实值。
N = 20000
x_langevin = langevin_1d(N, n_iter=2000, delta=0.05, sigma_score=1.0)
sigmas = np.exp(np.linspace(np.log(5.0), np.log(0.05), 10))
x_annealed = annealed_langevin_1d(N, sigmas, T_per_level=200, epsilon=2e-3)

# 真实样本
x_true = np.concatenate([np.random.randn(int(N*0.3))-2, np.random.randn(int(N*0.7))+1])

# 评估：各模态的样本比例
# ★ 阈值选在两高斯密度相等处：0.3·N(x;-2,1) = 0.7·N(x;1,1)
#   解得 x ≈ -0.78，这是贝叶斯最优决策边界，比任意的 -0.5 更严谨。
def mode_proportions(samples, threshold=-0.78):
    p_left = np.mean(samples < threshold)
    p_right = np.mean(samples >= threshold)
    return p_left, p_right

true_props = mode_proportions(x_true)
langevin_props = mode_proportions(x_langevin)
annealed_props = mode_proportions(x_annealed)

print(f"模态比例对比 (左模态N(-2,1):右模态N(1,1)):")
print(f"  真实分布:    {true_props[0]:.3f} : {true_props[1]:.3f}")
print(f"  单步Langevin: {langevin_props[0]:.3f} : {langevin_props[1]:.3f}")
print(f"  退火Langevin: {annealed_props[0]:.3f} : {annealed_props[1]:.3f}")
print(f"\nNCSN的三个根本局限（7.1节）：")
print(f"  1. 跳跃式切换噪声水平——离散σ_i不连续")
print(f"  2. 每个水平的步数T无系统选择原则")
print(f"  3. NCSN和DDPM框架不统一——看起来是两种方法")
print(f"\n→ 以上局限正是连续化的动机：把离散的σ_i/β_i视为连续函数σ(t)/β(t)上的采样，")
print(f"  就能把NCSN和DDPM统一在同一个SDE框架下，并获得逆向方程等「免费」工具。")


# ============================================================
# 步骤2：从离散噪声调度到连续SDE
# ★ 原创设计：展示β(t)调度的连续化
# ============================================================
print("\n" + "=" * 60)
print("步骤2：从离散噪声调度到连续SDE——β(t)调度可视化")
print("=" * 60)

# VP-SDE的离散噪声调度（DDPM）
T_ddpm = 1000
beta_min, beta_max = 0.0001, 0.02
betas_ddpm = np.linspace(beta_min, beta_max, T_ddpm)
alphas_ddpm = 1 - betas_ddpm
alpha_bars_ddpm = np.cumprod(alphas_ddpm)

# VP-SDE的连续调度
# ★ 关键：连续时间下的β(t)需要乘以T_ddpm（按"单位时间"标定），
#   因为DDPM的β_i是"每步"的小量，而SDE的β(t)是"单位时间"的量。
#   当SDE离散化成T步时，每步方差增量为β(t_i)·dt = β(t_i)/T，
#   应等于DDPM的β_i，因此连续β(t) = T_ddpm × 每步β。
#   ★ 从 1/T_ddpm 开始（而非0）以对齐DDPM第1步，避免 SNR 尖峰
t_continuous = np.linspace(1/T_ddpm, 1, 1000)
beta_t_continuous = T_ddpm * (beta_min + t_continuous * (beta_max - beta_min))

# ᾱ(t) = exp(-∫₀ᵗ β(s) ds) = exp(-T·[β_min·t + 0.5·(β_max-β_min)·t²])
# 注：均值系数为 √ᾱ(t)，即 x(t)的均值 = √ᾱ(t)·x(0)，方差 = 1 - ᾱ(t)
#    /2 是均值系数 √ᾱ(t) 推导中出现的，不属于 ᾱ(t) 的定义
alpha_bar_continuous = np.exp(
    -T_ddpm * (beta_min * t_continuous + 0.5 * (beta_max - beta_min) * t_continuous**2)
)

# SNR对比：SNR = ᾱ / (1 - ᾱ)
# 用 eps 钳制避免 t=1 处的除零警告（t=0 处的尖峰已通过 t_continuous 起点处理）
snr_ddpm = alpha_bars_ddpm / np.maximum(1 - alpha_bars_ddpm, 1e-12)
snr_continuous = np.maximum(alpha_bar_continuous, 1e-12) / np.maximum(1 - alpha_bar_continuous, 1e-12)

# VE-SDE的离散噪声调度（SMLD/NCSN）
L_smld = 10
sigma_min, sigma_max = 0.01, 50.0
sigmas_smld = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), L_smld))

# VE-SDE的连续调度
t_ve = np.linspace(0, 1, 1000)
sigma_t_ve = sigma_max * (sigma_min/sigma_max)**t_ve  # 几何插值

print(f"VP-SDE (DDPM离散 vs 连续):")
print(f"  T={T_ddpm}, β_min={beta_min}, β_max={beta_max}（每步的β）")
print(f"  连续β(t) = T_ddpm × (β_min + t·(β_max-β_min)) = {T_ddpm*beta_min:.2f} → {T_ddpm*beta_max:.2f}（单位时间）")
print(f"  ᾱ(T)={alpha_bars_ddpm[-1]:.6f} (几乎为0，信号消失)")

print(f"\nVE-SDE (SMLD离散 vs 连续):")
print(f"  L={L_smld}, σ_min={sigma_min}, σ_max={sigma_max}")
print(f"  连续σ(t) = {sigma_max}×({sigma_min}/{sigma_max})^t")


# ============================================================
# 步骤3："离散→连续→再离散"认知螺旋
# ★ 原创设计：展示三种离散化（DDPM/SMLD/DDIM）的统一
# ============================================================
print("\n" + "=" * 60)
print("步骤3：'离散→连续→再离散'认知螺旋")
print("=" * 60)

print("""
7.1节核心洞见：三种认知层次

第一步：离散→连续
  - DDPM的离散β_i → VP-SDE: dx = -β(t)/2·x dt + √β(t) dw
  - SMLD的离散σ_i → VE-SDE: dx = √(d[σ²]/dt) dw
  - 统一框架：NCSN和DDPM不再是两种方法，而是同一SDE的不同参数化

第二步：连续→新发现
  - Anderson (1982) 定理：正向SDE已知→逆向SDE自动获得
  - 得分函数是逆向SDE中唯一的未知量
  - PF-ODE：逆向SDE的确定性等价（7.4节）

第三步：连续→再离散
  - 逆向VP-SDE + Euler-Maruyama → DDPM采样
  - 逆向VE-SDE + Euler-Maruyama → SMLD采样
  - PF-ODE + Euler → DDIM采样
  - PF-ODE + RK4/DPM-Solver → 高效采样（7.5节）

关键收益：连续框架提供"免费"的逆向方程和ODE等价
  - 在离散框架中，DDPM和DDIM看起来完全不同
  - 在连续框架中，它们只是同一方程的不同离散化
""")

# 数值验证：DDPM的α_bar ≈ VP-SDE的连续解
# 使用与步骤2完全一致的公式（含T_ddpm标度因子）
t_norm = np.arange(T_ddpm) / T_ddpm  # 归一化到[0,1]
alpha_bar_continuous_verify = np.exp(
    -T_ddpm * (beta_min * t_norm + 0.5 * (beta_max - beta_min) * t_norm**2)
)

print(f"VP-SDE连续解 vs DDPM离散α_bar:")
for t_label, idx in [("0.1", 100), ("0.5", 500), ("0.9", 900)]:
    print(f"  t={t_label}: 连续={alpha_bar_continuous_verify[idx]:.6f}, 离散={alpha_bars_ddpm[idx]:.6f}")
print(f"  两者数量级一致——离散是连续的Euler-Maruyama近似（O(Δt)误差来自累乘→积分）")


# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 子图1：采样对比（方案B：保留 x_true，line style 区分两条几乎重叠的 p 曲线）
#   黑色实线 = 真密度 p(x)        x_true = 蓝色实线（"经验 p"，与绿实线应重合）
#   红色 step = 单步Langevin（远离黑线 = 失败）  绿色 step = 退火Langevin（贴合黑线 = 成功）
#   绘制顺序：黑 → 红 → 绿 → 蓝，蓝实线最后画确保不被绿实线覆盖。
#   ★ alpha=0.5 让蓝实线覆盖处仍能透出黑色密度曲线，避免视觉遮挡。
x_grid = np.linspace(-6, 6, 200)
axes[0].plot(x_grid, gm1d_pdf(x_grid), 'k-', lw=2, alpha=0.5, label='真实密度 p(x)')
axes[0].hist(x_langevin, bins=60, density=True, histtype='step', lw=1.5, color='red', alpha=0.5, label='单步Langevin')
axes[0].hist(x_annealed, bins=60, density=True, histtype='step', lw=1.5, color='green', alpha=0.5, label='退火Langevin')
axes[0].hist(x_true, bins=60, density=True, histtype='step', lw=1.5, color='blue', alpha=0.5, label='真实分布(经验)')
axes[0].set_xlabel('$x$')
axes[0].set_ylabel('密度')
axes[0].set_title('单步Langevin vs 退火Langevin')
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# 子图2：噪声调度与SNR
ax2 = axes[1]
ax2.plot(t_norm, 10*np.log10(alpha_bars_ddpm/(1-alpha_bars_ddpm)), 'b-', lw=2, label='DDPM SNR')
ax2.plot(t_continuous, 10*np.log10(snr_continuous), 'r--', lw=2, label='VP-SDE连续SNR')
ax2.set_xlabel('归一化时间 t')
ax2.set_ylabel('SNR (dB)')
ax2.set_title('VP-SDE: 离散与连续SNR对比')
ax2.legend()
ax2.grid(alpha=0.3)

# 子图3：不同σ下的得分函数∇log p_σ
# ★ 直接呼应步骤1的"粗粒度score"失败模式：
#   σ越大，p_σ越平滑，模态间信号越弱（"跨不过去"）；
#   σ越小，p_σ越接近p，但模态间沟壑越深（"mixing慢"）。
#   这就是退火Langevin"从粗到细"的几何直觉。
# ★ ylabel 与 σ 标签用 mathtext 渲染，避免 SimHei 缺字符变方框
# ★ σ=0.01 曲线在模态分界处趋近 δ 函数（峰值极高、宽度极窄），
#   用虚线 + 标注说明，避免学生误以为"蓝色线不可见"
ax3 = axes[2]
x_score = np.linspace(-5, 5, 500)
sigmas_plot = [0.01, 0.1, 1.0, 5.0]
colors_plot = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
linestyles_plot = ['--', '-', '-', '-']  # σ=0.01 用虚线，表示"接近真实得分的极限"
# ★ σ=0.01 和 σ=0.1 的得分函数在模态分界外几乎完全重合（都≈∇log p），
#   把 σ=0.1 的 alpha 降到 0.4，让蓝色虚线透出来
alphas_plot = [1.0, 0.4, 1.0, 1.0]
for sigma_p, color_p, ls_p, alpha_p in zip(sigmas_plot, colors_plot, linestyles_plot, alphas_plot):
    s = gm1d_score_sigma(x_score, sigma_p)
    label = r'$\sigma={}$'.format(sigma_p) + (' (步骤1所用)' if sigma_p == 1.0 else '')
    ax3.plot(x_score, s, lw=2, color=color_p, linestyle=ls_p, alpha=alpha_p, label=label)
ax3.axhline(0, color='k', lw=0.5, alpha=0.3)
ax3.axvline(-0.78, color='gray', ls=':', lw=1, alpha=0.5, label=r'模态分界 $x=-0.78$')
ax3.set_xlabel('$x$')
ax3.set_ylabel(r'$\nabla \log p_\sigma(x)$')
ax3.set_title('不同噪声水平下的得分函数')
ax3.legend(fontsize=8, loc='lower right')
ax3.grid(alpha=0.3)
# ★ y 轴放宽到 ±5，让 σ=0.01 在分界处的尖峰露出一点
ax3.set_ylim(-5, 5)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_Langevin到扩散.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"图表已保存: 步骤1_Langevin到扩散.png")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验7.1-1 总结")
print("=" * 60)
print("1. 单步Langevin vs 退火Langevin：")
print(f"   单步Langevin难以准确采样多模态分布（左模态比例{langevin_props[0]:.3f} vs 真实{true_props[0]:.3f}）")
print(f"   退火Langevin更接近真实分布（左模态比例{annealed_props[0]:.3f}）")
print("2. NCSN的三个根本局限→推动连续化：")
print("   跳跃式切换、步数无原则、框架不统一")
print("3. 连续SDE框架的收益：")
print("   统一NCSN/DDPM为同一框架，Anderson定理免费给出逆向方程")
print("   VP-SDE连续解与DDPM离散α_bar高度一致")
print("4. '离散→连续→再离散'认知螺旋：")
print("   DDPM/SMLD是SDE的离散特例，DDIM是PF-ODE的离散特例")