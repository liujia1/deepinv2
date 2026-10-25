# -*- coding: utf-8 -*-
"""
实验12.3-1 结构对比：殊途为何同归（数值验证）
对应章节: 12.3节 结构对比：殊途为何同归

知识点:
  - 采样路径(DSM)与变分路径(VLB)的损失结构并排对比
  - 含噪输入的形式差异与Karras统一参数化
  - 权重匹配：简化VLB ≡ 简化DSM，加权DSM ≡ 完整VLB
  - VLB权重与简化目标权重的量级差异
  - 从噪声调度到权重的因果链：β_i → ᾱ_i → SNR_i → w_i

约定说明:
  本实验为了直观展示权重匹配关系，定义DSM权重为λ(σ)/σ²(不含1/2系数)
  实验12.4-1采用标准DSM损失定义λ(t)/[2σ_t²](含1/2系数，与文献惯例一致)
  两份实验的结论本质相同：当λ(t)满足匹配条件时，DSM≡VLB仅差常数因子
  本实验中"DSM权重 = 2×VLB权重"（精确常数比），12.4-1中"DSM权重=VLB权重"（精确相等）
  为让常数比的数值验证达到浮点舍入级别，本实验调度全部以float64计算

本实验不需要GPU，纯数值验证。

素材来源:
  - 实验12.1步骤1（结构对比部分）
  - 12.3节正文
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
    SAVE_DIR = os.path.join(_gdrive, '实验12.3-1')
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

print(f"\n{'='*60}")
print("实验12.3-1: 结构对比——殊途为何同归")
print(f"{'='*60}")
print(f"运行环境: {'Google Colab' if _IN_COLAB else '本地'}")
print(f"输出目录: {SAVE_DIR}")


# ============================================================
# 噪声调度（DDPM线性调度，T=1000）
# ============================================================
T = 1000
beta_min, beta_max = 1e-4, 0.02
betas = torch.linspace(beta_min, beta_max, T, dtype=torch.float64)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bars[:-1]])
posterior_var = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)

# VP边际噪声标准差 s(t)σ(t)=√(1-ᾱ_t)（本实验记法约定下记作σ_t，见12.3正文"记法说明"）
# 注意与Karras噪声水平 σ(t)=√((1-ᾱ_t)/ᾱ_t) 区分；权重匹配分母用的是 1-ᾱ_t
sigma_t = torch.sqrt(1 - alpha_bars)

# VLB权重 (ε-prediction)
wt_vlb = betas**2 / (2 * posterior_var * alphas * (1 - alpha_bars))


# ============================================================
# 步骤1：两条路径的损失结构对比（12.3节核心内容）
# ============================================================
print(f"\n{'='*60}")
print("步骤1：两条路径的损失结构对比（12.3节核心内容）")
print(f"{'='*60}")

print("""
采样路径（第6章→第7章）：
  目标：学习得分函数 s(x,t) = ∇log p_t(x) 以驱动采样
  DSM损失: J_DSM(θ) = Σ_i λ(σ_i)/σ_i² · E[||ε - ε̂_θ(x_0+σ_iε, σ_i)||²]

变分路径（第8章→第11章）：
  目标：最大化数据似然 log p(x_0) 的下界
  VLB损失: L_VLB(θ) = Σ_t w_t · E[||ε - ε̂_θ(x_t, t)||²]

核心相似性: 都是 ||ε - ε̂_θ(含噪输入, 条件)||²
唯一差异: (1) 含噪输入的形式; (2) 时间权重
""")

# --- 1a. VP-SDE噪声调度表 ---
print("VP-SDE噪声调度: σ_t = √(1-ᾱ_t) 与时间步t的对应")
print(f"{'t':>5s}  {'ᾱ_t':>10s}  {'σ_t':>10s}  {'SNR_t':>10s}")
print("-" * 40)
for t_idx in [1, 10, 50, 100, 250, 500, 750, 999]:
    ab = alpha_bars[t_idx].item()
    sig = sigma_t[t_idx].item()
    snr = ab / (1 - ab)
    print(f"{t_idx:5d}  {ab:10.6f}  {sig:10.6f}  {snr:10.4f}")

# --- 1b. Karras统一参数化消除含噪输入的形式差异 ---
print("""
--- Karras统一参数化 ---

DSM路径的含噪输入 (VE-SDE):  x_t = x_0 + σ·ε              (s=1, σ为加性噪声标准差)
VLB路径的含噪输入 (VP-SDE):  x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε  (s=√ᾱ_t, σ(t)=√((1-ᾱ_t)/ᾱ_t), s·σ=√(1-ᾱ_t))

Karras et al. (2022) 统一参数化:
  x_t = s(t)·x_0 + s(t)·σ(t)·ε

  VE-SDE: s(t)=1,              σ(t)=σ_t
  VP-SDE: s(t)=√ᾱ_t,          σ(t)=√((1-ᾱ_t)/ᾱ_t)

统一后，含噪输入的差异仅仅是 s(t) 和 σ(t) 的不同取值，
不影响损失函数的核心结构 ||ε - ε̂_θ||²。
""")

# Karras参数化数值展示
# σ(t) = √((1-ᾱ_t)/ᾱ_t)：Karras框架下VP的噪声水平，与VE-SDE原生σ同口径（见附录12B）
# s(t)σ(t) = √(1-ᾱ_t)：VP前向过程实际注入x_0的噪声标准差（边际扰动，恒≤1）
# VE-SDE下s(t)=1，故σ(t)与s(t)σ(t)退化为同一列；VP下高噪声步两者差异显著
print("Karras参数化数值对照:")
print(f"{'t':>5s}  {'s(t)=√ᾱ_t':>10s}  {'σ(t)=√((1-ᾱ_t)/ᾱ_t)':>22s}  {'s(t)σ(t)=√(1-ᾱ_t)':>19s}")
print("-" * 64)
for t_idx in [1, 50, 100, 250, 500, 750, 999]:
    ab_t = alpha_bars[t_idx].item()
    s_t = np.sqrt(ab_t)
    sigma_karras = np.sqrt((1 - ab_t) / ab_t)   # Karras噪声水平（与VE的σ同口径）
    sigma_marg = np.sqrt(1 - ab_t)              # VP边际扰动 s(t)σ(t)
    print(f"{t_idx:5d}  {s_t:10.6f}  {sigma_karras:22.6f}  {sigma_marg:19.6f}")

print("\n→ VE-SDE和VP-SDE只是 (s,σ) 参数化的两种选择，")
print("  核心结构 ||ε - ε̂_θ(x_t, 条件)||² 完全相同。")
print("  注：与VE-SDE的σ同口径的是Karras噪声水平 σ(t)=√((1-ᾱ_t)/ᾱ_t)；")
print("      √(1-ᾱ_t)=s(t)σ(t) 是VP前向过程实际注入的噪声标准差，两者勿混")


# ============================================================
# 步骤2：差异仅在于时间权重（12.3节核心观察3）
# ============================================================
print(f"\n{'='*60}")
print("步骤2：差异仅在于时间权重（12.3节核心观察3）")
print(f"{'='*60}")

# 两条路径的权重对比
print("""
两条路径的唯一实质差异是时间权重:

  DSM: λ(σ)/σ²   (由权重函数λ决定)
  VLB: w_t = β_t²/[2β̃_t·α_t·(1-ᾱ_t)]   (由噪声调度决定)

12.3节的三个观察引出等价性条件:
  观察1: 预测目标完全相同 → 都是 ||ε - ε̂_θ||²
  观察2: 含噪输入可统一 → Karras参数化消除形式差异
  观察3: 差异仅是权重 → 当权重匹配时两者等价
""")

# 权重匹配条件（结论性陈述，详细数值验证见实验12.4-1）
print("权重匹配的两种情况（详细数值验证见实验12.4-1）:")
print()
print("情况1: 简化VLB ≡ 简化DSM")
print("  当 λ(σ) = σ² 时，DSM权重 = σ²/σ² = 1 (均匀)")
print("  简化VLB权重 = 1 (均匀)")
print("  → 两者训练行为完全一致")
print()
print("情况2: 加权DSM ≡ 完整VLB")
print("  当 λ(t) = β_t(1-ᾱ_t)/[(1-ᾱ_{t-1})·α_t] 时")
print("  本实验约定下 DSM权重 = 2 × VLB权重（仅差常数2）")
print("  标准约定 λ/(2σ²) 下则 DSM权重 = VLB权重（精确相等）")
print("  → 12.4节将严格证明并数值验证此等价性")

# VLB权重的量级特征（简要展示）
wt_vlb_min = wt_vlb[1:].min().item()
wt_vlb_max = wt_vlb[1:].max().item()
print(f"\nVLB权重范围: {wt_vlb_min:.6f} ~ {wt_vlb_max:.2f} "
      f"(跨越 {np.log10(wt_vlb_max/wt_vlb_min):.1f} 个数量级)")
print("简化目标权重恒为1（均匀）")


# ============================================================
# 可视化：DSM与VLB结构对比
# ============================================================
print(f"\n{'='*60}")
print("生成可视化图表...")
print(f"{'='*60}")

# 计算权重匹配的DSM权重
# 匹配条件: λ(t) = β_t(1-ᾱ_t) / ((1-ᾱ_{t-1})·α_t)
# DSM权重 = λ(t) / σ_t^2 = [β_t(1-ᾱ_t) / ((1-ᾱ_{t-1})·α_t)] / (1-ᾱ_t)
#                   = β_t / ((1-ᾱ_{t-1})·α_t)
lambda_match = betas[1:] * (1 - alpha_bars[1:]) / ((1 - alpha_bars_prev[1:]) * alphas[1:])
wt_dsm_matched = lambda_match / (1 - alpha_bars[1:])  # DSM权重 = λ(t) / σ_t^2

# VLB权重（已有）
wt_vlb_t = wt_vlb[1:].numpy()

# 恒等性自检：λ(t)由w_t反解得到，故 DSM权重 = 2 * VLB权重 是代数恒等式（与调度无关）
ratio = wt_dsm_matched.numpy() / wt_vlb_t
ratio_mean = np.mean(ratio)
ratio_std = np.std(ratio)
ratio_max_dev = np.max(np.abs(ratio - 2.0))

print("\n=== 权重匹配的恒等性自检 ===")
print(f"DSM权重 / VLB权重 的统计特性:")
print(f"  理论值: 2.000 (本实验省略DSM损失1/2系数所致的常数倍)")
print(f"  实际均值: {ratio_mean:.6f}")
print(f"  标准差: {ratio_std:.2e}")
print(f"  与理论值最大偏差: {ratio_max_dev:.2e} (float64浮点舍入级别)")
print(f"→ 说明: 匹配λ(t)本就是从w_t反解得到的，比值=2是与噪声调度无关的代数恒等式")
print(f"  (换成任意调度仍精确为2)；此处自检的是代码实现与12.4节公式严格一致，")
print(f"  而非对一条经验规律的验证。严格证明见12.4节(其标准约定下比值为1)。")

# 复核：换成一个随机非均匀调度（排序后单调递增），比值应仍然精确为 2
# 用局部 generator，避免扰动全局 RNG（后续绘图的可复现性不受影响）
_gen_rand = torch.Generator().manual_seed(2026)
betas_rand, _ = torch.sort(torch.rand(T, generator=_gen_rand, dtype=torch.float64) * 0.02 + 1e-4)
alphas_rand = 1 - betas_rand
ab_rand = torch.cumprod(alphas_rand, dim=0)
abp_rand = torch.cat([torch.ones(1, dtype=torch.float64), ab_rand[:-1]])
pv_rand = betas_rand * (1 - abp_rand) / (1 - ab_rand)
w_rand = betas_rand**2 / (2 * pv_rand * alphas_rand * (1 - ab_rand))
lam_rand = betas_rand[1:] * (1 - ab_rand[1:]) / ((1 - abp_rand[1:]) * alphas_rand[1:])
dsm_rand = lam_rand / (1 - ab_rand[1:])
ratio_rand = (dsm_rand / w_rand[1:]).numpy()
ratio_rand_max_dev = float(np.max(np.abs(ratio_rand - 2.0)))
print(f"随机非均匀调度复核: 比值均值 {ratio_rand.mean():.6f}，"
      f"与 2 的最大偏差 {ratio_rand_max_dev:.2e}")

# 可视化：同一图上对比
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
t_range = np.arange(1, T)

# DSM匹配权重（缩放后）
ax.semilogy(t_range, wt_dsm_matched.numpy() / 2, 'b-', linewidth=2.5,
            label=r'加权DSM权重 $\lambda(t)/\sigma_t^2$ (缩放1/2)')
# VLB权重
ax.semilogy(t_range, wt_vlb_t, 'r--', linewidth=2.5,
            label=r'完整VLB权重 $w_t$')
# 简化目标权重
ax.axhline(1, color='green', linestyle=':', linewidth=2,
           label=r'简化目标权重 (=1)')

ax.set_xlabel('时间步 $t$', fontsize=13)
ax.set_ylabel('权重', fontsize=13)
ax.set_title('DSM权重与VLB权重的匹配（恒等性自检）', fontsize=14)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, 'DSM与VLB结构对比.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图表已保存: {fig_path}")
print("\n→ 蓝色实线（DSM权重/2）与红色虚线（VLB权重）完全重合")
print("  自检匹配公式的代码实现: λ(t) = β_t(1-ᾱ_t) / [(1-ᾱ_{t-1})·α_t]")
print("  （该重合是代数恒等性的体现，对任意噪声调度均成立；严格证明见12.4节）")


# ============================================================
# 步骤3：从噪声调度到权重——w_t 的形状从哪儿来
# ============================================================
# 上一张图把 DSM 与 VLB 曲线叠在一起，验证了"两者重合"，但曲线完全遮住，
# 反而看不清 w_t 自己的形状。本步把 β_i → ᾱ_i → SNR_i → w_i 的因果链拆开画。
print(f"\n{'='*60}")
print("步骤3：从噪声调度到权重——w_t 的形状从哪儿来")
print(f"{'='*60}")

snr_all = alpha_bars / (1 - alpha_bars)
i_snr1 = int(torch.argmin((snr_all - 1.0).abs()).item())
i_wmax = int(torch.argmax(wt_vlb[1:]).item()) + 1   # 换回 0 起下标
# wt_vlb[0] 为 inf（ᾱ_0 处的构造），故最小值只在 i>=1 上取
i_wmin = int(torch.argmin(wt_vlb[1:]).item()) + 1
w_tail = wt_vlb[1:]
snr_decades = np.log10(snr_all[0].item() / snr_all[-1].item())
w_decades = np.log10(w_tail.max().item() / w_tail.min().item())

print(f"β:  {betas[0].item():.2e} → {betas[-1].item():.2e}（线性增长 {betas[-1].item()/betas[0].item():.0f} 倍）")
print(f"ᾱ:  {alpha_bars[0].item():.6f} → {alpha_bars[-1].item():.2e}（终端 √ᾱ_T = {np.sqrt(alpha_bars[-1].item()):.4f}）")
print(f"SNR: {snr_all[0].item():.1f} → {snr_all[-1].item():.2e}（跨越 {snr_decades:.1f} 个数量级），"
      f"SNR=1 出现在下标 i={i_snr1}（论文第 {i_snr1 + 1} 步）")
print(f"w:   最大 {w_tail.max().item():.4f} @ i={i_wmax}，最小 {w_tail.min().item():.4f} @ i={i_wmin}，"
      f"末端 {w_tail[-1].item():.4f}（跨 {w_decades:.2f} 个数量级）")
print("→ w_t 呈 'L 形 + 末端回升'：低噪声端 β 小但 1-ᾱ 更小，比值反而最大；")
print("  中段 1-ᾱ 追上 β，比值触底；末端 1-ᾱ→1 饱和，w_t 只剩 β_t/[2α_t] 的缓慢增长")

fig, axes = plt.subplots(2, 2, figsize=(14, 8.6))
fig.suptitle('线性调度 $T=1000$：从调度 $\\beta_i$ 到权重 $w_i$ 的因果链', fontsize=15)
tt = np.arange(T)

# (a) β_i 本身
ax = axes[0, 0]
ax.plot(tt, betas.numpy(), color='C0', lw=1.8)
ax.set_xlabel('时间步下标 $i$', fontsize=12)
ax.set_ylabel(r'$\beta_i$', fontsize=12)
ax.set_title(r'(a) 调度本身：$\beta_i$ 从 $10^{-4}$ 线性涨到 $0.02$', fontsize=12)
ax.text(0.03, 0.92,
        f'$\\beta_1$ = {betas[0].item():.1e}\n$\\beta_T$ = {betas[-1].item():.3f}\n'
        f'两端相差 {betas[-1].item()/betas[0].item():.0f} 倍',
        transform=ax.transAxes, va='top', fontsize=9.5)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.001, 0.023)

# (b) 累计效应
ax = axes[0, 1]
ax.semilogy(tt, alpha_bars.numpy(), color='C0', lw=1.8,
            label=r'$\bar\alpha_i$（信号保留比例）')
ax.semilogy(tt, (1 - alpha_bars).numpy(), color='C3', lw=1.6, ls='--',
            label=r'$1-\bar\alpha_i$（噪声方差比例）')
ax.set_xlabel('时间步下标 $i$', fontsize=12)
ax.set_ylabel('比例（log）', fontsize=12)
ax.set_title(r'(b) 累计效应：$\bar\alpha_i$ 从 0.9999 衰减到 $4\times10^{-5}$', fontsize=12)
ax.legend(fontsize=9, loc='lower left')
ax.grid(True, alpha=0.3, which='both')

# (c) 信噪比
ax = axes[1, 0]
ax.semilogy(tt, snr_all.numpy(), color='C2', lw=1.8,
            label=r'$\mathrm{SNR}_i=\bar\alpha_i/(1-\bar\alpha_i)$')
ax.axhline(1.0, color='k', ls=':', lw=1.2, label='SNR = 1')
ax.axvline(i_snr1, color='gray', ls=':', lw=1.0)
ax.plot([i_snr1], [1.0], 'ko', ms=5)
ax.text(0.40, 0.80, f'SNR=1 @ i={i_snr1}\n（论文第 {i_snr1 + 1} 步）',
        transform=ax.transAxes, fontsize=9.5,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.85))
ax.set_xlabel('时间步下标 $i$', fontsize=12)
ax.set_ylabel(r'$\mathrm{SNR}_i$（log）', fontsize=12)
ax.set_title(f'(c) 信噪比单调跨越 {snr_decades:.1f} 个数量级', fontsize=12)
ax.legend(fontsize=9, loc='lower left')
ax.grid(True, alpha=0.3, which='both')
ax.set_ylim(2e-5, 3e4)

# (d) 权重本身
ax = axes[1, 1]
idx = np.arange(1, T)
ax.semilogy(idx, w_tail.numpy(), color='C1', lw=2.0,
            label=r'完整 VLB 权重 $w_i=\beta_i/[2(1-\bar\alpha_{i-1})\alpha_i]$')
ax.axhline(1.0, color='C0', ls='--', lw=1.2, label=r'简化目标权重 $c=1$')
ax.plot([i_wmax], [w_tail.max().item()], 'o', color='C1', ms=5)
ax.plot([i_wmin], [w_tail.min().item()], 'o', color='C1', ms=5)
ax.annotate(f'max {w_tail.max().item():.4f} @ i={i_wmax}',
            xy=(i_wmax, w_tail.max().item()), xytext=(i_wmax + 60, 0.30),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
ax.annotate(f'min {w_tail.min().item():.4f} @ i={i_wmin}',
            xy=(i_wmin, w_tail.min().item()), xytext=(i_wmin + 120, 0.030),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
ax.set_xlabel('时间步下标 $i$', fontsize=12)
ax.set_ylabel(r'$w_i$（log）', fontsize=12)
ax.set_title(f'(d) 权重：低噪声端高、中段触底、末端小幅回升（跨 {w_decades:.1f} 个数量级）',
             fontsize=12)
ax.legend(fontsize=8.5, loc='upper right')
ax.grid(True, alpha=0.3, which='both')
ax.set_ylim(0.0035, 3.0)

fig.text(0.5, -0.055,
         '注：$(a)\\to(b)\\to(c)\\to(d)$ 是一条因果链——$(a)$ 的 $\\beta_i$ 累乘得 $(b)$ 的 $\\bar\\alpha_i$，'
         '二者之比得 $(c)$ 的 SNR，而 $(d)$ 的 $w_i$ 由 $\\beta_i$ 与 $1-\\bar\\alpha_i$ 的相对大小决定。\n'
         '线性 $T=1000$ 下 $w_i$ 是“L 形 + 末端回升”（最大/最小约 120 倍）；换成 cosine $T=200$ 时形状会变成 U 形（见 12.6）。',
         ha='center', fontsize=9, color='dimgray')
plt.tight_layout()
fig_path2 = os.path.join(SAVE_DIR, '调度与权重演化.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图表已保存: {fig_path2}")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print("实验12.3-1 总结")
print(f"{'='*60}")
print("""
关键结论:

1. 损失结构的三个关键观察（12.3节）
   (1) 预测目标完全相同: 都是 ||ε - ε̂_θ||²
   (2) 含噪输入可通过Karras统一参数化消除形式差异
   (3) 唯一实质差异是时间权重: DSM用λ(σ)/σ²，VLB用w_t

2. 权重匹配的恒等性自检（本实验，严格证明见12.4节）
   (1) 简化VLB ≡ 简化DSM: λ(σ)=σ²时，两者权重均退化为常数1
   (2) 加权DSM ≡ 完整VLB:
       - 匹配条件: λ(t) = β_t(1-ᾱ_t) / [(1-ᾱ_{t-1})·α_t]
       - DSM权重 = 2 × VLB权重（仅差常数因子2，源于本实验省略1/2系数的约定）
       - 自检结果: 两条曲线在缩放后完全重合（偏差~1e-15，浮点舍入级别）
       - 注意: 该比值对任意噪声调度恒为2，是匹配公式的代数恒等式，
         此处验证的是代码实现与公式一致，而非经验规律

3. VLB权重量级跨越约2个数量级，简化目标等权重1
   → 简化目标训练更稳定（Ho et al. 2020的关键实践发现）
   → 完整VLB用于似然优化（BPD指标）

4. VE-SDE和VP-SDE的参数化差异
   - VE-SDE: s(t)=1, σ(t)为原生加性噪声标准差（sσ=σ，两列退化）
   - VP-SDE: s(t)=√ᾱ_t, σ(t)=√((1-ᾱ_t)/ᾱ_t)，实际边际噪声s(t)σ(t)=√(1-ᾱ_t)
   - Karras噪声水平σ(t)与VP边际噪声s(t)σ(t)勿混（见附录12B）

5. 从调度到权重的因果链（调度与权重演化.png）
   - β_i 线性增长 → ᾱ_i 累乘衰减 → SNR_i=ᾱ_i/(1-ᾱ_i) 单调跨 8.4 个数量级
   - w_i 的形状由 β_i 与 1-ᾱ_i 的相对大小决定：低噪声端 β 小但 1-ᾱ 更小，比值最大；
     中段 1-ᾱ 追上 β，比值触底（i=349）；末端 1-ᾱ→1 饱和，w_i 只剩 β/(2α) 的缓慢增长
   - 形状为“L 形 + 末端回升”（最大/最小约 120 倍），与 cosine T=200 的 U 形形成对照（见12.6）
""")

print(f"{'='*60}")
print("第十二章配套实验12.3-1完成!")

# ===== 保存数值结果 =====
import json

def _to_native(obj):
    """递归转换numpy/torch类型为Python原生类型"""
    import numpy as np
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return _to_native(obj.tolist())
    try:
        import torch
        if isinstance(obj, torch.Tensor): return _to_native(obj.detach().cpu().tolist())
    except: pass
    return obj

results_summary = {
    'VLB_weight_range': {
        'min': wt_vlb[1:].min().item(),
        'max': wt_vlb[1:].max().item(),
        'n_decades': np.log10(wt_vlb[1:].max().item() / wt_vlb[1:].min().item()),
    },
    'DSM_VLB_weight_matching': {
        'ratio_mean': ratio_mean,
        'ratio_std': ratio_std,
        'ratio_max_dev_from_2': ratio_max_dev,
        'random_schedule_ratio_mean': float(ratio_rand.mean()),
        'random_schedule_max_dev_from_2': ratio_rand_max_dev,
        'note': 'DSM权重 = 2 × VLB权重；匹配公式的代数恒等性自检（与噪声调度无关，任意调度下恒为2）',
    },
    'VLB_weights_sampled': {
        f't_{t}': wt_vlb[t].item()
        for t in [1, 10, 50, 100, 250, 500, 750, 999]
    },
    'schedule_profile': {
        'beta_first': betas[0].item(),
        'beta_last': betas[-1].item(),
        'beta_ratio': betas[-1].item() / betas[0].item(),
        'alpha_bar_first': alpha_bars[0].item(),
        'alpha_bar_last': alpha_bars[-1].item(),
        'sqrt_alpha_bar_T': float(np.sqrt(alpha_bars[-1].item())),
        'snr_first': snr_all[0].item(),
        'snr_last': snr_all[-1].item(),
        'snr_decades': float(snr_decades),
        'snr1_index': i_snr1,
        'snr1_step': i_snr1 + 1,
        'w_max': w_tail.max().item(),
        'w_max_index': i_wmax,
        'w_min': w_tail.min().item(),
        'w_min_index': i_wmin,
        'w_last': w_tail[-1].item(),
        'w_decades': float(w_decades),
        'note': '下标为 0 起数组下标（与 VLB_weights_sampled 的 t_* 键同口径，即论文第 i+1 步）；w 的 max/min 只在 i>=1 上取',
    },
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
