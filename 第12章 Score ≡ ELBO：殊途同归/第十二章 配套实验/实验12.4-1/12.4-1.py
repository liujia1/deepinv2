# -*- coding: utf-8 -*-
"""
实验12.4-1 DSM≡VLB等价性的数值验证
对应章节: 12.4节 DSM≡VLB：等价性的形式化证明

知识点:
  - λ(t)的选择使DSM权重≡VLB权重
  - 简化目标(λ(σ)=σ²)下DSM权重退化为1/2
  - ε/x₀/score三种参数化下DSM≡VLB同时成立

约定说明:
  本实验采用标准DSM损失定义: J_DSM = Σ λ(t)/[2σ_t²] · E[||ε - ε̂||²]
  (含1/2系数，与score matching文献惯例一致)
  实验12.3-1中为了直观展示权重匹配，省略了1/2系数，定义DSM权重为λ(σ)/σ²
  两份实验的结论本质相同：当λ(t)满足匹配条件时，DSM≡VLB仅差常数因子

实验内容:
  步骤1: DSM权重=VLB权重的数值验证（12.4节核心定理）
  步骤2: 简化目标下的等价性（12.4节推论）
  步骤3: 三种参数化下的DSM≡VLB统一验证（12.4节）

本实验不需要GPU，纯数值验证。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys
import io
import logging
import warnings

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
    SAVE_DIR = os.path.join(_gdrive, '实验12.4-1')
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

# 设置随机种子
np.random.seed(42)
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print(f"\n{'='*60}")
print(f"实验12.4-1: DSM≡VLB等价性的数值验证")
print(f"对应章节: 12.4节 DSM≡VLB：等价性的形式化证明")
print(f"{'='*60}")


# ============================================================
# 噪声调度（DDPM线性调度，T=1000）
# ============================================================
T = 1000
beta_min, beta_max = 1e-4, 0.02
betas = torch.linspace(beta_min, beta_max, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])  # ᾱ_{t-1}
posterior_var = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)  # β̃_t

# VLB权重 (ε-prediction): w_t = β_t² / [2β̃_t·α_t·(1-ᾱ_t)]
wt_vlb = betas**2 / (2 * posterior_var * alphas * (1 - alpha_bars))

# 有效时间步掩码（t=0时ᾱ_0=1导致分母为零，排除之）
valid_mask = torch.arange(T) >= 1

# 噪声水平 σ_t = √(1-ᾱ_t)
sigma_t = torch.sqrt(1 - alpha_bars)

print(f"\n噪声调度: DDPM线性, T={T}, β_min={beta_min}, β_max={beta_max}")
print(f"ᾱ_1 = {alpha_bars[0].item():.6f}, ᾱ_T = {alpha_bars[-1].item():.8f}")


# ============================================================
# 步骤1：DSM权重=VLB权重的数值验证（12.4节核心定理）
# ============================================================
print(f"\n{'='*60}")
print("步骤1：DSM权重=VLB权重的数值验证（12.4节核心定理）")
print("=" * 60)

print("""
12.4节定理: 在VP-SDE下，选择 λ(t) 使得:
  λ(t) / [2(1-ᾱ_t)] = w_t = β_t² / [2β̃_t·α_t·(1-ᾱ_t)]

解出: λ(t) = β_t·(1-ᾱ_t) / [(1-ᾱ_{t-1})·α_t]

当此条件满足时: J_DSM(θ) = L_VLB(θ) + 常数
""")

# 计算使DSM≡VLB所需的λ(t)
lambda_equiv = betas * (1 - alpha_bars) / ((1 - alpha_bars_prev) * alphas)

# 验证: λ(t) / [2(1-ᾱ_t)] 应等于 w_t
wt_from_lambda = lambda_equiv / (2 * (1 - alpha_bars))

print("验证 λ(t)/[2(1-ᾱ_t)] = w_t（逐时间步打印比值）:")
print(f"{'t':>5s}  {'λ(t)':>12s}  {'λ/[2(1-ᾱ)]':>12s}  {'w_t(VLB)':>12s}  {'比值':>8s}")
print("-" * 55)
for t_idx in [1, 10, 50, 100, 250, 500, 750, 999]:
    lam = lambda_equiv[t_idx].item()
    w_from_lam = wt_from_lambda[t_idx].item()
    w_vlb = wt_vlb[t_idx].item()
    ratio = w_from_lam / max(w_vlb, 1e-30)
    print(f"{t_idx:5d}  {lam:12.6f}  {w_from_lam:12.6f}  {w_vlb:12.6f}  {ratio:8.4f}")

# 全局验证（跳过t=0，因ᾱ_0=1导致分母为零）
wt_from_lambda_valid = wt_from_lambda[valid_mask]
wt_vlb_valid = wt_vlb[valid_mask]
max_ratio_err = torch.max(torch.abs(wt_from_lambda_valid - wt_vlb_valid) / torch.max(wt_vlb_valid, torch.tensor(1e-30)))
print(f"\n全时间步(t≥1)最大相对误差: {max_ratio_err.item():.2e}")
print("→ 所有时间步比值≈1.0000，DSM≡VLB权重匹配精确成立！")
print("  这就是12.4节定理的数值验证。")


# ============================================================
# 步骤2：简化目标下的等价性（12.4节推论）
# ============================================================
print(f"\n{'='*60}")
print("步骤2：简化目标下的等价性（12.4节推论）")
print("=" * 60)

print("""
12.4节推论: 当λ(σ) = σ²时（NCSN默认选择），DSM权重退化为1/2:
  λ(σ_t)/[2σ_t²] = σ_t²/[2σ_t²] = 1/2

此时DSM目标变为均匀加权，即每个时间步贡献相同。
与完整VLB权重的差异恰好是12.6节讨论的核心。
""")

# NCSN默认 λ(σ) = σ² 时的DSM权重
lambda_ncsn = sigma_t**2  # λ(σ_t) = σ_t² = (1-ᾱ_t)
wt_ncsn = lambda_ncsn / (2 * sigma_t**2)  # = 1/2

print("NCSN默认权重 λ(σ)=σ² 时:")
print(f"  DSM权重 = λ(σ)/[2σ²] = σ²/[2σ²] = 1/2 (均匀权重)")
print(f"  简化VLB权重 = 1 (均匀权重)")
print(f"  两者仅差常数因子2，训练行为完全一致")

# 与完整VLB权重的对比
print(f"\n完整VLB权重 vs 简化目标权重:")
print(f"{'t':>5s}  {'w_t(VLB)':>12s}  {'w_t(简化)':>10s}  {'w_t(VLB)/w_t(简化)':>18s}")
print("-" * 50)
for t_idx in [1, 10, 50, 100, 250, 500, 750, 999]:
    w_vlb = wt_vlb[t_idx].item()
    w_simple = 1.0
    print(f"{t_idx:5d}  {w_vlb:12.6f}  {w_simple:10.4f}  {w_vlb/w_simple:18.4f}")

wt_vlb_valid = wt_vlb[valid_mask]  # 排除t=0边界
wt_vlb_min = wt_vlb_valid.min().item()
wt_vlb_max = wt_vlb_valid.max().item()
n_decades = np.log10(wt_vlb_max / wt_vlb_min) if wt_vlb_min > 0 else float('inf')
print(f"\n→ VLB权重跨越{n_decades:.1f}个数量级({wt_vlb_min:.4f}~{wt_vlb_max:.4f})，简化目标为等权重1")
print("  这解释了11.4节Ho et al. (2020)的发现：简化目标训练更稳定")


# ============================================================
# 步骤3：三种参数化下的DSM≡VLB统一验证（12.4节）
# ============================================================
print(f"\n{'='*60}")
print("步骤3：三种参数化下的DSM≡VLB统一验证（12.4节）")
print("=" * 60)

print("""
12.4节指出三种参数化携带完全相同的信息，DSM≡VLB在三种参数化下同时成立:
  ε-prediction:   ||ε - ε̂_θ||²
  x₀-prediction:  ᾱ_t/(1-ᾱ_t) · ||x₀ - x̂_θ||²
  score-prediction: (1-ᾱ_t) · ||s - s_θ||²

三者之间的转换关系（11.3节）保证等价性在所有参数化下同时成立。
""")

# 数值验证：给定一个非完美预测，三种参数化损失的一致性
n_samples = 5000
x0_test = torch.randn(n_samples, 2)
eps_test = torch.randn_like(x0_test)

print(f"{'t':>5s}  {'||ε-ε̂||²':>12s}  {'ᾱ/(1-ᾱ)·||x₀-x̂||²':>20s}  {'(1-ᾱ)·||s-ŝ||²':>16s}  {'三种一致':>8s}")
print("-" * 70)

for t_idx in [10, 100, 500, 900]:
    ab = alpha_bars[t_idx]
    sqrt_ab = torch.sqrt(ab)
    sqrt_1mab = torch.sqrt(1 - ab)

    # 构造含噪输入
    x_t = sqrt_ab * x0_test + sqrt_1mab * eps_test

    # 模拟非完美预测（加微小扰动）
    eps_hat = eps_test + 0.01 * torch.randn_like(eps_test)

    # ε-prediction损失: ||ε - ε̂||²
    loss_eps = ((eps_test - eps_hat)**2).mean().item()

    # x₀-prediction损失: ᾱ_t/(1-ᾱ_t) · ||x₀ - x̂₀||²
    x0_hat = (x_t - sqrt_1mab * eps_hat) / sqrt_ab
    loss_x0 = (ab / (1 - ab)).item() * ((x0_test - x0_hat)**2).mean().item()

    # score-prediction损失: (1-ᾱ_t) · ||s + s_θ||²
    # s = -ε/σ_t, s_θ = -ε̂/σ_t  →  ||s - s_θ||² = ||ε̂-ε||²/σ_t²
    # (1-ᾱ_t) · ||s - s_θ||² = (1-ᾱ_t) · ||ε̂-ε||²/(1-ᾱ_t) = ||ε̂-ε||²
    s_hat = -eps_hat / sqrt_1mab
    s_target = -eps_test / sqrt_1mab
    loss_score = (1 - ab).item() * ((s_hat - s_target)**2).mean().item()

    # 检查一致性
    consistent = abs(loss_eps - loss_x0) / max(loss_eps, 1e-10) < 0.01 and \
                 abs(loss_eps - loss_score) / max(loss_eps, 1e-10) < 0.01

    print(f"{t_idx:5d}  {loss_eps:12.8f}  {loss_x0:20.8f}  {loss_score:16.8f}  {'✓' if consistent else '✗':>8s}")

print("\n→ 三种参数化损失数值一致，DSM≡VLB在所有参数化下同时成立")


# ============================================================
# 可视化：DSM与VLB等价性验证
# ============================================================
print(f"\n{'='*60}")
print("生成可视化图表...")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
t_range = range(1, T + 1)

# (a) 两条路径的权重对比：VLB权重和使DSM≡VLB的DSM权重完全重合
ax = axes[0]
wt_dsm = lambda_equiv / (2 * (1 - alpha_bars))  # 使DSM≡VLB的DSM权重
ax.semilogy(t_range, wt_vlb.numpy(), 'b-', linewidth=2.5,
            label=r'VLB权重 $w_t = \frac{\beta_t^2}{2\tilde{\beta}_t \alpha_t (1-\bar{\alpha}_t)}$')
ax.semilogy(t_range, wt_dsm.numpy(), 'r--', linewidth=2.5,
            label=r'DSM权重 $\frac{\lambda(t)}{2(1-\bar{\alpha}_t)}$ (使DSM≡VLB)')
ax.axhline(1, color='gray', linestyle=':', linewidth=1.5, label=r'简化目标权重 $=1$')
ax.set_xlabel(r'时间步 $t$', fontsize=13)
ax.set_ylabel(r'权重 (对数尺度)', fontsize=13)
ax.set_title(r'(a) VLB权重 $\equiv$ DSM权重：两条路径完全重合', fontsize=13)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

# (b) 完整VLB权重 vs 简化目标权重：fill_between展示丢弃的权重区域
ax = axes[1]
wt_simple = np.ones(T)
ax.semilogy(t_range, wt_vlb.numpy(), 'b-', linewidth=2.5,
            label=r'完整VLB权重 $w_t$')
ax.semilogy(t_range, wt_simple, 'r--', linewidth=2.5,
            label=r'简化目标权重 $=1$')
# fill_between展示权重差异
ax.fill_between(t_range, wt_vlb.numpy(), wt_simple,
                where=wt_vlb.numpy() > wt_simple, alpha=0.15, color='blue',
                label='丢弃的低t权重(小噪声步)')
ax.fill_between(t_range, wt_vlb.numpy(), wt_simple,
                where=wt_vlb.numpy() < wt_simple, alpha=0.15, color='red',
                label='丢弃的高t权重(大噪声步)')
ax.set_xlabel(r'时间步 $t$', fontsize=13)
ax.set_ylabel(r'权重 (对数尺度)', fontsize=13)
ax.set_title(r'(b) 完整VLB权重 vs 简化目标权重', fontsize=13)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
ax.annotate('VLB权重非均匀\n简化目标 = 均匀权重1\n→ 训练更稳定',
            xy=(0.4, 0.6), xycoords='axes fraction', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeaa7', alpha=0.8))

plt.suptitle('DSM与VLB等价性验证', fontsize=15, fontweight='bold')
plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, 'DSM与VLB等价性验证.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图表已保存: {fig_path}")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print("实验12.4-1 总结")
print("=" * 60)
print("""
关键结论:

1. DSM≡VLB的形式化验证（12.4节核心定理）
   - 选择 λ(t) = β_t(1-ᾱ_t)/[(1-ᾱ_{t-1})·α_t] 时，DSM权重 = VLB权重
   - 数值验证: 所有时间步比值≈1.0000，匹配精确成立
   - 含义: 两条路径推导出的训练目标在数学上完全等价

2. 简化目标的等价性（12.4节推论）
   - λ(σ)=σ²时DSM权重 = 1/2（均匀权重）
   - 这正是DDPM/NCSN简化目标的理论基础
   - VLB权重跨越约2个数量级，简化目标为等权重1
   - 均匀权重 → 训练更稳定，生成质量更好

3. 三种参数化的统一（12.4节）
   - ε/x₀/score三种参数化下损失数值一致
   - 三种参数化是同一目标的不同坐标表示
   - DSM≡VLB在所有参数化下同时成立
   - 网络只需学一种参数化，其余可解析转换

4. 实践意义（12.6节）
   - 简化目标(均匀权重) → 生成质量好(FID/IS)
   - 完整VLB(自然权重) → 似然好(BPD)但训练不稳定
   - 两者差异仅在权重选择，网络结构完全相同
""")

print(f"{'='*60}")
print("第十二章配套实验12.4-1 完成!")
