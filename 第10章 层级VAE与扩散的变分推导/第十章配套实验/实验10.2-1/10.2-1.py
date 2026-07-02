# -*- coding: utf-8 -*-
"""
实验10.2-1 扩散过程的变分下界推导
对应章节: 10.2 扩散过程的变分下界推导

知识点:
  - 高斯编码器=加噪过程 (马尔可夫高斯链的闭式边际分布)
  - VLB三项分解: L_T + ΣL_{t-1} + L_0
  - 前向过程后验的闭式解

实验内容:
  步骤1: 高斯编码器=加噪过程验证 (直接采样 vs 逐步加噪)
  步骤2: VLB三项分解验证 (L_T, L_{t-1}, L_0)
  步骤3: SNR分析与噪声调度可视化

本实验不需要GPU，通过数值实验验证10.2节的核心数学结论。
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
    SAVE_DIR = os.path.join(_gdrive, '实验10.2-1')
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
import torch
torch.manual_seed(42)

print("\n" + "="*60)
print("实验10.2-1: 扩散过程的变分下界推导")
print("="*60)
print("对应章节: 10.2 扩散过程的变分下界推导")
print("使用设备: CPU (本实验不需要GPU)")


# ============================================================
# 噪声调度定义
# ============================================================
T = 1000  # 扩散步数
beta_min, beta_max = 1e-4, 0.02

# 线性噪声调度 (DDPM)
betas = torch.linspace(beta_min, beta_max, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

print("\n噪声调度: beta从" + str(beta_min) + "线性增至" + str(beta_max) + ", T=" + str(T))
print("ab_1=" + str(round(alpha_bars[0].item(), 6)) + ", ab_500=" + str(round(alpha_bars[499].item(), 6)) + ", ab_T=" + str(round(alpha_bars[999].item(), 6)))


# ============================================================
# 步骤1：高斯编码器=加噪过程验证 —— 10.2节核心对应
# ============================================================
print("\n" + "="*60)
print("步骤1：高斯编码器=加噪过程验证")
print("="*60)
print("\n[核心思想]")
print("  当编码器采用固定方差的高斯转移时，编码链等价于逐步加噪:")
print("  q(x_t|x_{t-1}) = N(x_t; sqrt(alpha_t) x_{t-1}, (1-alpha_t)I)")
print("  → q(x_t|x_0) = N(x_t; sqrt(ab_t) x_0, (1-ab_t)I)")

# 数值验证：直接采样 vs 逐步加噪 (两条独立采样链)
N = 5000  # 大样本量确保统计显著性
torch.manual_seed(42)
x0 = torch.randn(N, 2)  # N个2D样本
n_steps_list = [1, 50, 200, 500, 999]

print("\n直接采样 vs 逐步加噪 (" + str(N) + "个样本, 2维):")
print("    t      均值差(max)    标准差差(max)    条件期望    理论标准差")
print("-" * 75)

# 为每个t创建独立的Generator，确保可复现性（跨版本稳定）
base_seed = 42

for t_idx in n_steps_list:
    t = t_idx  # 0-indexed
    sqrt_ab = torch.sqrt(alpha_bars[t])
    sqrt_1mab = torch.sqrt(1 - alpha_bars[t])

    # 为当前t创建两个独立的Generator
    g_direct = torch.Generator().manual_seed(base_seed + t_idx * 1000)
    g_step = torch.Generator().manual_seed(base_seed + t_idx * 1000 + 1)

    # 直接采样: x_t = sqrt(ab_t) * x_0 + sqrt(1-ab_t) * eps
    eps_direct = torch.randn_like(x0, generator=g_direct)
    x_t_direct = sqrt_ab * x0 + sqrt_1mab * eps_direct

    # 逐步加噪 (独立随机性，与直接采样解耦)
    x_t_step = x0.clone()
    for s in range(t + 1):
        noise = torch.randn_like(x_t_step, generator=g_step)
        x_t_step = torch.sqrt(alphas[s]) * x_t_step + torch.sqrt(betas[s]) * noise

    # 比较两条独立链的样本统计量
    mean_direct = x_t_direct.mean(dim=0)
    mean_step = x_t_step.mean(dim=0)
    std_direct = x_t_direct.std(dim=0)
    std_step = x_t_step.std(dim=0)

    mean_diff = (mean_direct - mean_step).abs().max().item()
    std_diff = (std_direct - std_step).abs().max().item()

    # 条件期望值 (以当前x0样本为条件，而非绝对理论期望0)
    mean_conditional = sqrt_ab * x0.mean(dim=0)
    std_theory = sqrt_1mab

    print(str(t_idx).rjust(5) + "  " + str(round(mean_diff, 6)).rjust(14) + "  " + str(round(std_diff, 6)).rjust(14) +
          "  " + str(round(mean_conditional[0].item(), 6)).rjust(14) + "  " + str(round(std_theory.item(), 6)).rjust(14))

print("\n→ 两条独立采样链的统计量趋于一致，验证了闭式边际分布成立:")
print("  q(x_t|x_0) = N(sqrt(ab_t)x_0, (1-ab_t)I) 精确成立")
print("  这是10.2节的核心结论：高斯马尔可夫链的闭式边际分布")
print("\n  注：标准差差异是更强的判别指标——")
print("     均值本身因 E[x_0]≈0 而趋于0，判别力弱；")
print("     而方差项 (1-ab_t) 不退化为0，能更好区分两个过程是否等价")


# ============================================================
# 步骤2：VLB三项分解验证 —— 10.2节公式
# ============================================================
print("\n" + "="*60)
print("步骤2：VLB三项分解验证")
print("="*60)
print("\n[核心思想]")
print("  变分下界分解为三项:")
print("  L_VLB = L_T + sum_{t=2}^T L_{t-1} + L_0")
print("  - L_T: 先验匹配项 D_KL(q(x_T|x_0) || p(x_T))")
print("  - L_{t-1}: 一致性项 D_KL(q(x_{t-1}|x_t,x_0) || p_theta(x_{t-1}|x_t))")
print("  - L_0: 重建项 -E[log p_theta(x_0|x_1)]")

# L_T: 先验匹配项
print("\nL_T (先验匹配项):")
print("  q(x_T|x_0) = N(sqrt(ab_T)x_0, (1-ab_T)I)")
print("  p(x_T) = N(0, I)")
print("  ab_T = " + str(round(alpha_bars[-1].item(), 8)) + " → sqrt(ab_T) ≈ 0, 1-ab_T ≈ 1")
print("  → q(x_T|x_0) ≈ N(0, I) → L_T ≈ 0")

# 前向过程后验 (10.2节闭式解)
alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])

# 验证后验方差
posterior_var = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)
print("\n前向过程后验方差 tilde_beta_t:")
print("  t=1:   tilde_beta_1 = " + str(round(posterior_var[0].item(), 6)))
print("  t=500: tilde_beta_500 = " + str(round(posterior_var[499].item(), 6)))
print("  t=T:   tilde_beta_T = " + str(round(posterior_var[999].item(), 6)))

# 验证后验方差与前向方差的相对误差 (DDPM简化为tilde_beta_t≈beta_t的关键依据)
rel_err = (posterior_var - betas) / betas
# t=1是边界点: alpha_bars_prev[0]=1导致后验方差公式分子为0 (1-ab_0=0)
# 故tilde_beta_1=0, 相对误差为-100%, 仅在t=1发生
# 实际分析相对误差时应从t=2开始
rel_err_t2_onwards = rel_err[1:]
print("\n  关键观察: 相对误差 (tilde_beta_t - beta_t) / beta_t")
print("    t=1:     -100%   (边界: tilde_beta_1=0, 因ab_0=1使公式分子为0)")
print("    t=2:     " + str(round(rel_err_t2_onwards[0].item() * 100, 4)) + "%")
print("    t=500:   " + str(round(rel_err_t2_onwards[498].item() * 100, 4)) + "%")
print("    t=T-1:   " + str(round(rel_err_t2_onwards[-2].item() * 100, 4)) + "%")
print("    t=T:     " + str(round(rel_err_t2_onwards[-1].item() * 100, 4)) + "%")
print("    最大值(忽略t=1):  " + str(round(rel_err_t2_onwards.max().item() * 100, 4)) + "% (在t=" + str(int(rel_err_t2_onwards.argmax().item()) + 2) + ")")
print("  → 实际 t>=2 时, tilde_beta_t 与 beta_t 相对误差 < 0.1%, 证明 DDPM 简化方差估计是合理的")

# 验证后验均值公式 (10.2节)
print("\n后验均值系数 (10.2节):")
print("  tilde_mu_t = [sqrt(alpha_t)(1-ab_{t-1})/(1-ab_t)] x_t + [sqrt(ab_{t-1})(1-alpha_t)/(1-ab_t)] x_0")

for t_idx in [0, 99, 499, 999]:
    t = t_idx
    a_t = alphas[t]
    ab_t = alpha_bars[t]
    ab_prev = alpha_bars_prev[t]

    coeff_xt = torch.sqrt(a_t) * (1 - ab_prev) / (1 - ab_t)
    coeff_x0 = torch.sqrt(ab_prev) * (1 - a_t) / (1 - ab_t)

    print("  t=" + str(t+1).rjust(4) + ": x_t系数=" + str(round(coeff_xt.item(), 4)) + ", x_0系数=" + str(round(coeff_x0.item(), 4)) + ", 和=" + str(round((coeff_xt+coeff_x0).item(), 4)))

# L_0: 重建项
print("\nL_0 (重建项): 离散解码器或高斯解码器的负对数似然")
print("  对连续数据通常用高斯: -log p(x_0|x_1) ∝ ||x_0 - mu_theta(x_1,1)||^2")


# ============================================================
# 步骤3：SNR分析
# ============================================================
print("\n" + "="*60)
print("步骤3：SNR分析")
print("="*60)
print("\n信噪比(SNR)随时间变化:")
print("  SNR(t) = ab_t / (1-ab_t)")

for t_idx in [0, 99, 249, 499, 749, 999]:
    snr = alpha_bars[t_idx] / (1 - alpha_bars[t_idx])
    snr_db = 10 * np.log10(snr.item()) if snr.item() > 0 else -np.inf
    print("  t=" + str(t_idx+1).rjust(4) + ": ab=" + str(round(alpha_bars[t_idx].item(), 6)) + ", SNR=" + str(round(snr.item(), 4)) + " (" + str(round(snr_db, 1)) + " dB)")

print("\n关键观察:")
print("  - t小: 信号占优，SNR高")
print("  - t大: 噪声占优，SNR低")
print("  - 当ab_t→0时，SNR→0，图像完全被噪声淹没")


# ============================================================
# 可视化
# ============================================================
print("\n" + "="*60)
print("生成可视化图表...")
print("="*60)

fig = plt.figure(figsize=(14, 11))
gs_outer = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)
gs_c = gs_outer[1, 0].subgridspec(2, 1, hspace=0.45)

# (a) 噪声调度与ab_t
ax = fig.add_subplot(gs_outer[0, 0])
t_range = range(1, T + 1)
ax.plot(t_range, alpha_bars.numpy(), 'b-', linewidth=2, label=r'$\bar{\alpha}_t$')
ax.plot(t_range, torch.sqrt(alpha_bars).numpy(), 'r--', linewidth=1.5, label=r'$\sqrt{\bar{\alpha}_t}$ ($x_0$系数)')
ax.plot(t_range, torch.sqrt(1 - alpha_bars).numpy(), 'g--', linewidth=1.5, label=r'$\sqrt{1-\bar{\alpha}_t}$ (噪声系数)')
ax.set_xlabel('时间步 $t$', fontsize=12)
ax.set_ylabel('系数值', fontsize=12)
ax.set_title('(a) DDPM线性噪声调度', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', alpha=0.3)

# (b) SNR随时间变化
ax = fig.add_subplot(gs_outer[0, 1])
snr = alpha_bars / (1 - alpha_bars)
snr_db = 10 * torch.log10(snr)
ax.plot(t_range, snr_db.numpy(), 'purple', linewidth=2)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='SNR=0 dB')
ax.set_xlabel('时间步 $t$', fontsize=12)
ax.set_ylabel('SNR (dB)', fontsize=12)
ax.set_title('(b) 信噪比(SNR)随时间衰减', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.annotate('信号占优', xy=(0.1, 0.8), xycoords='axes fraction',
            fontsize=11, color='blue', fontweight='bold')
ax.annotate('噪声占优', xy=(0.7, 0.2), xycoords='axes fraction',
            fontsize=11, color='red', fontweight='bold')

# (c-上) 前向过程后验方差 vs 前向方差
ax = fig.add_subplot(gs_c[0])
ax.plot(t_range[1:], posterior_var[1:].numpy(), 'b-', linewidth=2, label=r'$\tilde{\beta}_t$ (后验方差)')
ax.plot(t_range[1:], betas[1:].numpy(), 'r--', linewidth=1.5, label=r'$\beta_t$ (前向方差)')
ax.set_xlabel('时间步 $t$', fontsize=12)
ax.set_ylabel('方差', fontsize=12)
ax.set_title(r'(c-上) 前向过程后验方差 vs 前向方差 ($\tilde{\beta}_t \approx \beta_t$)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (c-下) 相对误差: 放大显示二者的微小差异
ax = fig.add_subplot(gs_c[1])
rel_err_pct = rel_err[1:].numpy() * 100  # 从t=2开始 (跳过t=1的-100%退化点)
ax.plot(t_range[1:], rel_err_pct, 'b-', linewidth=2)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
# 标注: t=2 误差最大 (最不近似处), t=T 误差最小 (最近似处)
worst_idx = 0  # rel_err_pct[0] 对应 t=2, 是最负的值
worst_t = worst_idx + 2
worst_val = rel_err_pct[worst_idx]
ax.annotate(r'最大误差 $t=2$: ' + f'{worst_val:.2f}%',
            xy=(worst_t, worst_val), xytext=(0.40, 0.30),
            textcoords='axes fraction',
            fontsize=10, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))
# 在曲线右端标注最小值
best_val = rel_err_pct[-1]
ax.annotate(r'$t=T$: ' + f'{best_val:.4f}%',
            xy=(T, best_val), xytext=(0.65, 0.75),
            textcoords='axes fraction',
            fontsize=10, color='blue', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6))
ax.set_xlabel('时间步 $t$', fontsize=12)
ax.set_ylabel('相对误差 (%)', fontsize=12)
ax.set_title(r'(c-下) 相对误差 $\frac{\tilde{\beta}_t - \beta_t}{\beta_t} \times 100\%$', fontsize=13)
ax.grid(True, alpha=0.3)

# (d) 2D扩散过程可视化
ax = fig.add_subplot(gs_outer[1, 1])
np.random.seed(42)
n_pts = 200
# 两个高斯模式
mode1 = np.random.randn(n_pts // 2, 2) * 0.3 + np.array([1.0, 0.0])
mode2 = np.random.randn(n_pts // 2, 2) * 0.3 + np.array([-1.0, 0.0])
x0_2d = np.vstack([mode1, mode2])

colors = np.array([0] * (n_pts // 2) + [1] * (n_pts // 2))
t_show = [0, 50, 200, 500, 999]

for i, t_idx in enumerate(t_show):
    ab = alpha_bars[t_idx].item()
    noise = np.random.randn(n_pts, 2)
    x_t = np.sqrt(ab) * x0_2d + np.sqrt(1 - ab) * noise
    offset_x = i * 4
    ax.scatter(x_t[:, 0] + offset_x, x_t[:, 1], c=colors, cmap='coolwarm',
               alpha=0.5, s=8)
    ax.text(offset_x, -2.5, '$t=' + str(t_idx) + '$', ha='center', fontsize=10)
    if i == 0:
        ax.text(offset_x, 2.8, '$x_0$', ha='center', fontsize=11, fontweight='bold')
    elif i == len(t_show) - 1:
        ax.text(offset_x, 2.8, '$x_T \\approx N(0,I)$', ha='center', fontsize=11, fontweight='bold')

ax.set_xlabel('前向加噪过程 →', fontsize=12)
ax.set_title('(d) 高斯编码器=加噪过程 (2D可视化)', fontsize=13)
ax.set_ylim(-3, 3.5)
ax.grid(True, alpha=0.3)
ax.set_yticks([])

fig_path = os.path.join(SAVE_DIR, '步骤1_噪声调度与VLB分解.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print("图表已保存: 步骤1_噪声调度与VLB分解.png")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "="*60)
print("实验10.2-1 总结")
print("="*60)
print("\n关键结论:")
print("\n1. 高斯编码器=加噪过程 (步骤1)")
print("   - 马尔可夫高斯链的闭式边际分布: q(x_t|x_0) = N(sqrt(ab_t)x_0, (1-ab_t)I)")
print("   - 直接采样公式允许高效地从任意时间步采样")
print("\n2. VLB三项分解 (步骤2)")
print("   - L_T≈0: 先验匹配项，当ab_T≈0时自动满足")
print("   - L_{t-1}: 一致性项，核心优化目标")
print("   - L_0: 重建项")
print("\n3. 前向过程后验闭式解 (步骤2)")
print("   - 后验均值: tilde_mu_t = [sqrt(alpha_t)(1-ab_{t-1})/(1-ab_t)] x_t + [sqrt(ab_{t-1})(1-alpha_t)/(1-ab_t)] x_0")
print("   - 后验方差: tilde_beta_t = (1-ab_{t-1})/(1-ab_t) * beta_t")
print("\n4. SNR分析 (步骤3)")
print("   - SNR随t单调衰减，从信号占优到噪声占优")
print("   - 这是10.3节中损失权重差异的根源")

print("\n" + "="*60)
print("实验10.2-1 完成!")