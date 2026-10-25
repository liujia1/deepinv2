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
  步骤4: 一维高斯混合双网络训练对照（12.4节"一维数值验证示例"）
         - 同步组：同初始化/同数据序列的两个网络分别用完整VLB与匹配DSM目标训练，
           损失曲线应逐点重合、参数轨迹同步（等价性在优化层面的直接检验）
         - 独立组：各3个独立种子训练，比较最终去噪MSE分布（经验层面统计不可区分）

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
# 逐点相对误差: 每个时间步 |DSM权重-VLB权重|/w_t 后取最大值
# （注: 分母是逐点w_t而非全局max(w_t)，在权重最小的t处同样要求匹配）
max_rel_err = torch.max(torch.abs(wt_from_lambda_valid - wt_vlb_valid)
                        / torch.clamp(wt_vlb_valid, min=1e-30))
print(f"\n全时间步(t≥1)逐点最大相对误差: {max_rel_err.item():.2e}")
print("→ 所有时间步比值≈1.0000，DSM≡VLB权重匹配精确成立！")
print("  这就是12.4节定理的数值验证。")

# ★ 自检：验证12.4节λ(t)解出公式的边界行为与极值
# 公式: λ(t) = β_t·(1-ᾱ_t) / [(1-ᾱ_{t-1})·α_t]
# 理论性质:
#   P1: t=1时ᾱ_0=1, 故分母(1-ᾱ_0)=0, λ(1)→∞ (注意: VLB中对应的L_1项在DDPM
#       惯例下由解码器项L_0处理，闭式权重公式仅对t≥2成立)
#   P2: t=T时ᾱ_T→0, 1-ᾱ_T→1, λ(T)≈β_T/α_T (因为ᾱ_{T-1}≈0, 1-ᾱ_{T-1}→1)
#   P3: λ(t)在数组下标≥3（即数学t≥4）后严格递增；数组下标1→2（数学t=2→3）
#       有一次解析性的小幅下降（非数值噪声）：由恒等变形
#       λ(t)=β_t+β_t²/[α_t·(1-ᾱ_{t-1})] 可见，t很小时分母(1-ᾱ_{t-1})
#       近似线性增长，第二项β_t²/(1-ᾱ_{t-1})∝1/t递减并占主导；
#       之后β_t线性增长占主导，λ(t)转为严格递增
# 检验规则: 边界值在量级上正确, 整体单调性成立
print(f"[λ(t)自检] 边界行为与单调性验证(实际运行, 非手算):")
print(f"  P1: λ(1)={lambda_equiv[0].item():.2e} (应→∞, 因分母1-ᾱ_0=0)")
print(f"  P2: λ(T)={lambda_equiv[-1].item():.6f} (应≈β_T/α_T={betas[-1].item()/alphas[-1].item():.6f})")
print(f"  P3: 单调性(数组下标t≥3, 即数学t≥4, 严格递增): {'OK' if (lambda_equiv[3:] > lambda_equiv[2:-1]).all().item() else 'FAIL'}")
# 典型t处的λ值
for _t in [10, 100, 500, 999]:
    print(f"  λ(t={_t+1})={lambda_equiv[_t].item():.6f}, 对应β·(1-ᾱ)/[(1-ᾱ_prev)·α]={betas[_t].item() * (1-alpha_bars[_t].item()) / ((1-alpha_bars_prev[_t].item()) * alphas[_t].item()):.6f}")


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
# 口径说明: 本实验在VP内部取 σ_t² = 1-ᾱ_t（见前文定义），故 λ=σ² 给出均匀权重1/2；
#           严格VE→VP翻译为 σ²=(1-ᾱ_t)/ᾱ_t，对应权重1/(2ᾱ_t)（见本节末推论），
#           两种口径勿混，此处按VP内部口径验证
lambda_ncsn = sigma_t**2  # VP口径: λ(σ_t) = σ_t² = (1-ᾱ_t)
wt_ncsn = lambda_ncsn / (2 * sigma_t**2)  # = 1/2

print("NCSN默认权重 λ(σ)=σ² 时:")
print(f"  DSM权重 = λ(σ)/[2σ²] = σ²/[2σ²] = 1/2 (VP口径σ²=1-ᾱ_t, 均匀权重)")
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

    # score-prediction损失: (1-ᾱ_t) · ||s - s_θ||²
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
# 数学步 t 对应数组下标 t-1：t=1 处 β̃_1=0 使 w_1=inf（该项即离散解码器 L_0），
# matplotlib 会自动跳过该点，因此曲线实际从 t=2 开始。
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
# fill_between展示权重差异（本调度下 w_t 处处<1，故蓝色 w_t>1 区域为空）
ax.fill_between(t_range, wt_vlb.numpy(), wt_simple,
                where=wt_vlb.numpy() > wt_simple, alpha=0.15, color='blue',
                label=r'$w_t>1$（本调度未出现）')
ax.fill_between(t_range, wt_vlb.numpy(), wt_simple,
                where=wt_vlb.numpy() < wt_simple, alpha=0.15, color='red',
                label=r'$w_t<1$：VLB 相对均匀权重下调')
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
# 步骤4：训练目标等价性的一维双网络对照（12.4节"一维数值验证示例"）
# ============================================================
# 设计（口径固定，便于审计）:
#   数据: p(x0)=0.5 N(-3,1)+0.5 N(3,1)，一维
#   调度: VP-SDE, T=100, 常数β=0.01（12.4节正文指定的一维示例设置）
#   网络: 同一结构的小MLP（sinusoidal时间嵌入16维+3×64隐层），ε-prediction
#   VLB目标: mean_t [ w_t · ||ε-εθ||² ]，t~U{1,99}（t=0处权重为inf，按惯例排除）
#   DSM目标: mean_t [ λ(t)/(2(1-ᾱ_t)) · ||ε-εθ||² ]，λ按12.4定理匹配
#   同步组: 两网络同初始化、每步喂同一批(x0,ε,t)，仅损失定义不同 →
#           VLB与匹配DSM的每步损失逐点相等、梯度相同、参数轨迹同步
#           注意：匹配λ是由w_t反解出来的，两个目标是同一标量函数的两种写法，
#           因此这组重合是代数必然（它检验的是代码两条分支确实等价），
#           不能单独作为"两目标等价"的经验证据
#   负对照: 第三个网络，同初始化、同数据流，改用简化目标（λ(σ)=σ²，权重恒1/2，
#           即不匹配的原始DSM）→ 若权重不匹配时轨迹必须分开，则说明上面的
#           "重合"不是因为所有网络都会重合，同步组具备判别力
#   独立组: VLB/匹配DSM/简化目标各3个独立种子（初始化与数据流均独立）
#           → 最终去噪MSE：前两者应一致（同一目标），简化目标给出可比的参照
#   全流程float64（小网络+一维数据，CPU可承受；与步骤1的浮点级验证口径一致）
print(f"\n{'='*60}")
print("步骤4：训练目标等价性的一维双网络对照（12.4节一维示例）")
print("=" * 60)

import math
import copy
import time as _time

T4 = 100
BETA4 = 0.01
betas4 = torch.full((T4,), BETA4, dtype=torch.float64)
alphas4 = 1.0 - betas4
ab4 = torch.cumprod(alphas4, dim=0)
ab4_prev = torch.cat([torch.ones(1, dtype=torch.float64), ab4[:-1]])
pv4 = betas4 * (1 - ab4_prev) / (1 - ab4)
w4 = betas4**2 / (2 * pv4 * alphas4 * (1 - ab4))             # VLB系数
lam4 = betas4 * (1 - ab4) / ((1 - ab4_prev) * alphas4)       # 匹配λ(t)
dsm_c4 = lam4 / (2 * (1 - ab4))                              # DSM系数（=w4）

# 训练前先确认两套系数一致（排除t=0）
_coef_err = torch.max(torch.abs(w4[1:] - dsm_c4[1:])).item()
print(f"[训练前] VLB系数 vs DSM匹配系数 最大偏差(t≥1): {_coef_err:.2e}")


class TinyEpsNet(torch.nn.Module):
    """一维 ε-prediction 小网络: (x, sinusoidal(t)) -> ε̂"""
    def __init__(self, emb_dim=16, h=64):
        super().__init__()
        self.emb_dim = emb_dim
        freqs = torch.exp(torch.arange(0, emb_dim, 2, dtype=torch.float64)
                          * (-math.log(10000.0) / emb_dim))
        self.register_buffer('freqs', freqs)
        layers = []
        d_in = 1 + emb_dim
        for _ in range(3):
            lin = torch.nn.Linear(d_in if not layers else h, h)
            layers += [lin.double(), torch.nn.SiLU()]
        last = torch.nn.Linear(h, 1)
        layers.append(last.double())
        self.net = torch.nn.Sequential(*layers)

    def t_embed(self, t):
        ang = t.to(torch.float64)[:, None] * self.freqs[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

    def forward(self, x, t):
        return self.net(torch.cat([x, self.t_embed(t)], dim=1))


def sample_batch4(batch_size, gen):
    comp = (torch.randint(0, 2, (batch_size, 1), generator=gen).double() * 2 - 1)
    x0 = 3.0 * comp + torch.randn(batch_size, 1, generator=gen).double()
    eps = torch.randn(batch_size, 1, generator=gen).double()
    t = torch.randint(1, T4, (batch_size,), generator=gen)  # 1..99
    return x0, eps, t


def loss4(model, x0, eps, t, kind):
    ab_t = ab4[t][:, None]
    xt = ab_t.sqrt() * x0 + (1 - ab_t).sqrt() * eps
    eps_hat = model(xt, t)
    per = ((eps - eps_hat) ** 2).sum(dim=1)
    if kind == 'vlb':
        coef = w4[t]                    # 完整VLB权重
    elif kind == 'dsm':
        coef = dsm_c4[t]                # 匹配λ的DSM权重（与w4数值相同）
    elif kind == 'simple':
        coef = torch.full_like(t, 0.5, dtype=torch.float64)  # λ(σ)=σ² ⇒ 常数权重1/2
    else:
        raise ValueError(kind)
    return (coef * per).mean()


@torch.no_grad()
def denoise_mse4(model, n=40000, seed=999):
    """固定测试集上的去噪MSE：x̂0=(x_t-√(1-ᾱ)ε̂)/√ᾱ，对4个噪声级取平均"""
    ge = torch.Generator().manual_seed(seed)
    comp = (torch.randint(0, 2, (n, 1), generator=ge).double() * 2 - 1)
    x0 = 3.0 * comp + torch.randn(n, 1, generator=ge).double()
    eps = torch.randn(n, 1, generator=ge).double()
    mses = []
    for tv in [10, 30, 50, 80]:
        a = ab4[tv]
        t = torch.full((n,), tv, dtype=torch.long)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        x_hat = (xt - (1 - a).sqrt() * model(xt, t)) / a.sqrt()
        mses.append(((x0 - x_hat) ** 2).mean().item())
    return float(np.mean(mses)), mses


def build_model4(seed):
    torch.manual_seed(seed)
    return TinyEpsNet()


# ---- 4A. 同步组：同初始化 + 同数据序列，仅损失定义不同 ----
BATCH4 = 512
N_SYNC = 2000
LR4 = 1e-3
m_vlb = build_model4(42)
m_dsm = TinyEpsNet()
m_simple = TinyEpsNet()
m_dsm.load_state_dict(copy.deepcopy(m_vlb.state_dict()))     # 强制同一初始化
m_simple.load_state_dict(copy.deepcopy(m_vlb.state_dict()))  # 负对照：同初始化
opt_v = torch.optim.Adam(m_vlb.parameters(), lr=LR4)
opt_d = torch.optim.Adam(m_dsm.parameters(), lr=LR4)
opt_s = torch.optim.Adam(m_simple.parameters(), lr=LR4)
gen_sync = torch.Generator().manual_seed(2026)

# ---- 固定验证集：用于“目标是否在下降”的判断与绘图 ----
# 训练 batch 损失每步都在换数据，且 t~U{1,99} 而 w_t 跨越两个数量级，单次 batch
# 损失方差很大（实测在 0.017~0.038 间震荡，看不出趋势）。因此曲线改为在固定
# 验证集（固定样本 + 固定 33 个时间步）上重算完整目标值。
N_VAL4 = 4096
_gen_val4 = torch.Generator().manual_seed(7777)
_comp_v4 = (torch.randint(0, 2, (N_VAL4, 1), generator=_gen_val4).double() * 2 - 1)
X0_VAL4 = 3.0 * _comp_v4 + torch.randn(N_VAL4, 1, generator=_gen_val4).double()
EPS_VAL4 = torch.randn(N_VAL4, 1, generator=_gen_val4).double()
T_GRID4 = sorted(set(np.linspace(1, T4 - 1, 33).round().astype(int).tolist()))


@torch.no_grad()
def val_loss4(model, kind):
    """固定验证集 + 固定 33 个时间步上的目标值 E_t[coef_t·||ε-ε̂||²]。"""
    vals = []
    for tv in T_GRID4:
        t = torch.full((N_VAL4,), int(tv), dtype=torch.long)
        vals.append(loss4(model, X0_VAL4, EPS_VAL4, t, kind).item())
    return float(np.mean(vals))


# 训练前基线（同一初始化、尚未更新参数时的去噪MSE），用于证明训练确实有效
mse_baseline, mse_baseline_each = denoise_mse4(m_vlb)
val_v_init = val_loss4(m_vlb, 'vlb')

rec_steps, rec_lv, rec_ld, rec_ls, rec_param_diff = [], [], [], [], []
rec_val_steps, rec_valv, rec_vald, rec_vals = [], [], [], []
max_loss_diff = 0.0
_t0 = _time.time()
for step in range(N_SYNC):
    x0, eps, t = sample_batch4(BATCH4, gen_sync)  # 同一批数据喂三个网络
    lv = loss4(m_vlb, x0, eps, t, 'vlb')
    ld = loss4(m_dsm, x0, eps, t, 'dsm')
    ls = loss4(m_simple, x0, eps, t, 'simple')
    opt_v.zero_grad(); lv.backward(); opt_v.step()
    opt_d.zero_grad(); ld.backward(); opt_d.step()
    opt_s.zero_grad(); ls.backward(); opt_s.step()
    d_l = abs(lv.item() - ld.item())
    max_loss_diff = max(max_loss_diff, d_l)
    if step % 25 == 0:
        rec_steps.append(step)
        rec_lv.append(lv.item()); rec_ld.append(ld.item()); rec_ls.append(ls.item())
    if step % 50 == 0 or step == N_SYNC - 1:
        rec_val_steps.append(step)
        rec_valv.append(val_loss4(m_vlb, 'vlb'))
        rec_vald.append(val_loss4(m_dsm, 'dsm'))
        rec_vals.append(val_loss4(m_simple, 'simple'))
    if step % 500 == 0 or step == N_SYNC - 1:
        pd_ = max((p1 - p2).abs().max().item()
                  for p1, p2 in zip(m_vlb.parameters(), m_dsm.parameters()))
        pd_s = max((p1 - p3).abs().max().item()
                   for p1, p3 in zip(m_vlb.parameters(), m_simple.parameters()))
        rec_param_diff.append((step, pd_, pd_s))
_sync_secs = _time.time() - _t0

mse_v_sync, mse_v_sync_each = denoise_mse4(m_vlb)
mse_d_sync, mse_d_sync_each = denoise_mse4(m_dsm)
mse_s_sync, mse_s_sync_each = denoise_mse4(m_simple)
final_param_diff = rec_param_diff[-1][1]
final_param_diff_simple = rec_param_diff[-1][2]
max_rel_val = max(abs(a - b) / max(abs(a), 1e-12) for a, b in zip(rec_valv, rec_vald))

print(f"[4A 同步组] {N_SYNC}步, batch={BATCH4}, 耗时{_sync_secs:.1f}s")
print(f"  训练前去噪MSE基线: {mse_baseline:.6f}（证明训练有效，非两条线空转）")
print(f"  全程 |L_VLB - L_DSM| 最大值: {max_loss_diff:.2e}")
print(f"  固定验证集目标值: {val_v_init:.6f}(未训练) → {rec_valv[-1]:.6f}（{N_SYNC}步后）")
print(f"  验证集上 |VLB-DSM| 最大相对差: {max_rel_val:.2e}")
print(f"  训练后参数最大绝对偏差: |VLB-DSM|={final_param_diff:.2e}（同一目标，代数必然）")
print(f"                           |VLB-简化|={final_param_diff_simple:.2e}（负对照，应显著非零）")
print(f"  固定测试集去噪MSE: VLB网络={mse_v_sync:.8f}  DSM网络={mse_d_sync:.8f}  "
      f"差={abs(mse_v_sync-mse_d_sync):.2e}")
print(f"                     简化目标负对照={mse_s_sync:.8f}")

# ---- 4B. 独立组：各3个独立种子（初始化、数据流均独立） ----
N_INDEP = 1200


def train_indep4(kind, seed):
    m = build_model4(seed)
    opt = torch.optim.Adam(m.parameters(), lr=LR4)
    ge = torch.Generator().manual_seed(seed * 7 + 13)
    for _ in range(N_INDEP):
        x0, eps, t = sample_batch4(BATCH4, ge)
        loss = loss4(m, x0, eps, t, kind)
        opt.zero_grad(); loss.backward(); opt.step()
    mse_mean, _ = denoise_mse4(m)
    return mse_mean


_t0 = _time.time()
vlb_mse_indep = [train_indep4('vlb', s) for s in (101, 102, 103)]
dsm_mse_indep = [train_indep4('dsm', s) for s in (201, 202, 203)]
simple_mse_indep = [train_indep4('simple', s) for s in (301, 302, 303)]
_indep_secs = _time.time() - _t0
v_mean, d_mean = float(np.mean(vlb_mse_indep)), float(np.mean(dsm_mse_indep))
s_mean = float(np.mean(simple_mse_indep))
v_std, d_std = float(np.std(vlb_mse_indep, ddof=1)), float(np.std(dsm_mse_indep, ddof=1))
s_std = float(np.std(simple_mse_indep, ddof=1))
rel_mse_diff = abs(v_mean - d_mean) / max(v_mean, d_mean)
rel_simple_diff = abs(s_mean - v_mean) / max(v_mean, s_mean)

print(f"[4B 独立组] 每组3种子×{N_INDEP}步, 耗时{_indep_secs:.1f}s")
print(f"  VLB目标 去噪MSE: {['%.6f' % x for x in vlb_mse_indep]} 均值={v_mean:.6f}±{v_std:.6f}")
print(f"  匹配DSM目标 去噪MSE: {['%.6f' % x for x in dsm_mse_indep]} 均值={d_mean:.6f}±{d_std:.6f}")
print(f"  简化目标负对照 去噪MSE: {['%.6f' % x for x in simple_mse_indep]} "
      f"均值={s_mean:.6f}±{s_std:.6f}")
print(f"  VLB vs 匹配DSM 均值相对差异: {rel_mse_diff:.2%}（同一目标，差异仅来自种子）")
print(f"  VLB vs 简化目标 均值相对差异: {rel_simple_diff:.2%}（不同权重，负对照）")

# ---- 步骤4可视化 ----
# (a) 不平滑训练 batch 损失（batch 内 t 随机，方差极大），直接画固定验证集上的目标值
fig4, (ax4a, ax4b, ax4c) = plt.subplots(1, 3, figsize=(19, 5.2))
ax4a.plot(rec_val_steps, rec_valv, 'b-', linewidth=3.0, label='VLB 验证目标')
ax4a.plot(rec_val_steps, rec_vald, 'r--', linewidth=1.6, label='匹配DSM 验证目标（与VLB重合）')
ax4a.axhline(val_v_init, color='b', linestyle=':', linewidth=1.4, alpha=0.7,
             label=f'训练前={val_v_init:.4f}')
ax4a.set_xlabel('训练步', fontsize=12)
ax4a.set_ylabel('固定验证集目标值（左轴）', fontsize=12)
ax4a.set_title(f'(a) 固定验证集：目标快速下降后走平，两目标重合\n'
               f'{val_v_init:.4f}→{rec_valv[-1]:.4f}，最大相对差={max_rel_val:.1e}',
               fontsize=11)
ax4R = ax4a.twinx()
ax4R.plot(rec_val_steps, rec_vals, color='0.35', linestyle=':', linewidth=2.2,
          label='简化目标（右轴，负对照）')
ax4R.set_ylabel('简化目标值（右轴）', fontsize=12)
ax4R.tick_params(axis='y', colors='0.35')
_h1, _l1 = ax4a.get_legend_handles_labels()
_h2, _l2 = ax4R.get_legend_handles_labels()
ax4a.legend(_h1 + _h2, _l1 + _l2, fontsize=8, loc='center right')
ax4a.grid(True, alpha=0.3)

# (b) 参数轨迹偏差：匹配λ → 逐位同步；非匹配权重（简化目标）→ 显著分离
_pd_steps = [r[0] for r in rec_param_diff]
_pd_vd = [max(r[1], 1e-18) for r in rec_param_diff]
_pd_vs = [max(r[2], 1e-18) for r in rec_param_diff]
ax4b.semilogy(_pd_steps, _pd_vd, 'o-', color='r', linewidth=2,
              label='|VLB − 匹配DSM|（λ匹配）')
ax4b.semilogy(_pd_steps, _pd_vs, 's--', color='0.35', linewidth=2,
              label='|VLB − 简化目标|（λ不匹配）')
ax4b.set_xlabel('训练步', fontsize=12)
ax4b.set_ylabel('两网络参数最大绝对偏差', fontsize=12)
ax4b.set_title(f'(b) 负对照：权重不匹配时轨迹必须分开\n'
               f'匹配λ: {final_param_diff:.1e}；简化目标: {final_param_diff_simple:.1e}',
               fontsize=11)
ax4b.legend(fontsize=9, loc='center right')
ax4b.grid(True, alpha=0.3, which='both')

ax4c.scatter([0]*len(vlb_mse_indep), vlb_mse_indep, s=90, c='b', zorder=3, label='VLB（独立种子）')
ax4c.scatter([1]*len(dsm_mse_indep), dsm_mse_indep, s=90, c='r', marker='^', zorder=3, label='匹配DSM（独立种子）')
ax4c.scatter([2]*len(simple_mse_indep), simple_mse_indep, s=90, c='0.35', marker='s', zorder=3, label='简化目标（负对照）')
ax4c.hlines(v_mean, -0.18, 0.18, colors='b', linewidths=2)
ax4c.hlines(d_mean, 0.82, 1.18, colors='r', linewidths=2)
ax4c.hlines(s_mean, 1.82, 2.18, colors='0.35', linewidths=2)
ax4c.errorbar([0, 1, 2], [v_mean, d_mean, s_mean], yerr=[v_std, d_std, s_std],
              fmt='none', ecolor='k', elinewidth=1.2, capsize=5, alpha=0.7, zorder=2)
ax4c.set_xticks([0, 1, 2])
ax4c.set_xticklabels(['完整VLB', '匹配DSM', '简化目标'], fontsize=12)
ax4c.set_ylabel('固定测试集去噪MSE（4个噪声级平均）', fontsize=12)
ax4c.set_title(f'(c) 独立种子组（各3种子）\n'
               f'VLB {v_mean:.4f}±{v_std:.4f} vs DSM {d_mean:.4f}±{d_std:.4f}，'
               f'简化 {s_mean:.4f}±{s_std:.4f}', fontsize=11)
ax4c.legend(fontsize=8, loc='upper right')
ax4c.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig4_path = os.path.join(SAVE_DIR, '步骤4_训练目标等价性.png')
plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图表已保存: {fig4_path}")


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

5. 训练目标等价性（12.4节一维数值验证示例，本实验步骤4）
   - 同步组：匹配λ由w_t反解，两目标实为同一标量函数，故每步损失逐点相等、
     参数轨迹同步是代数必然；它检验的是代码两条分支确实等价
   - 负对照：同初始化、同数据下改用简化目标（权重恒1/2）→ 参数轨迹显著分开、
     去噪MSE不同，证明同步组的重合并非“所有网络都会重合”
   - 独立组：VLB与匹配DSM各3个独立种子，去噪MSE均值差异只反映种子噪声；
     简化目标同时作为该指标判别力的参照
""")

print(f"{'='*60}")
print("第十二章配套实验12.4-1 完成!")

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
    'DSM_VLB_equivalence': {
        'max_pointwise_relative_error': max_rel_err.item(),
        'note': 'lambda(t)/[2(1-ab_t)] = w_t 精确成立(数值为逐点|差|/w_t的最大值, float32精度)',
    },
    'lambda_equivalent': {
        # 与步骤1控制台表格一致：直接取数组下标 t 处（下标0因1-ᾱ_0=0发散，不列入）
        f't_{t}': float(lambda_equiv[t].item())
        for t in [1, 10, 50, 100, 250, 500, 750, 999]
    },
    'wt_vlb': {
        f't_{t}': float(wt_vlb[t].item())
        for t in [1, 10, 50, 100, 250, 500, 750, 999]
    },
    'VLB_weight_range': {
        'min': float(wt_vlb_min),
        'max': float(wt_vlb_max),
        'n_decades': float(n_decades),
    },
    'simplified_DSM': {
        'weight': 0.5,
        'note': 'lambda(sigma)=sigma^2时DSM权重=1/2(均匀)',
    },
    'step4_training_equivalence': {
        'setup': {
            'data_distribution': '0.5*N(-3,1) + 0.5*N(3,1), 1D',
            'T': T4, 'beta_const': BETA4,
            'network': 'MLP(sinusoidal_t_emb=16, 3x64 hidden, SiLU), eps-prediction',
            'dtype': 'float64', 'batch_size': BATCH4, 'lr': LR4,
            't_sampling': 'uniform over {1,...,99} (t=0 excluded: weight inf)',
            'denoise_eval': 'Tweedie x0_hat on fixed testset n=40000, seed=999, t in {10,30,50,80}',
        },
        'synchronized_pair': {
            'steps': N_SYNC,
            'init_seed': 42, 'data_seed': 2026,
            'max_abs_step_loss_diff': max_loss_diff,
            'final_param_max_abs_diff': final_param_diff,
            'final_param_max_abs_diff_vs_simple': final_param_diff_simple,
            'denoise_mse_untrained_baseline': mse_baseline,
            'denoise_mse_vlb': mse_v_sync,
            'denoise_mse_dsm': mse_d_sync,
            'denoise_mse_simple_negative_control': mse_s_sync,
            'denoise_mse_per_t': {
                'vlb': mse_v_sync_each,
                'dsm': mse_d_sync_each,
                'simple': mse_s_sync_each,
            },
            'note': ('匹配λ由w_t反解，VLB与DSM是同一标量函数：损失逐点重合、参数逐位同步是代数必然；'
                     '简化目标（λ(σ)=σ²，权重恒1/2）为负对照，用于证明该对照具有判别力'),
        },
        'validation_curve': {
            'n_samples': N_VAL4,
            'sample_seed': 7777,
            't_grid': T_GRID4,
            'eval_every': 50,
            'steps': rec_val_steps,
            'vlb': rec_valv,
            'dsm': rec_vald,
            'simple_negative_control': rec_vals,
            'vlb_untrained': val_v_init,
            'max_rel_diff_vlb_vs_dsm': max_rel_val,
            'note': ('训练 batch 损失（单步）在 0.017~0.038 间震荡，看不出趋势；'
                     '图 (a) 改为固定验证集（固定样本 + 固定 33 个时间步）上重算的完整目标值'),
        },
        'independent_seeds': {
            'steps_per_run': N_INDEP,
            'vlb_seeds': [101, 102, 103],
            'dsm_seeds': [201, 202, 203],
            'simple_seeds': [301, 302, 303],
            'vlb_denoise_mse': vlb_mse_indep,
            'dsm_denoise_mse': dsm_mse_indep,
            'simple_denoise_mse': simple_mse_indep,
            'vlb_mean': v_mean, 'vlb_std_ddof1': v_std,
            'dsm_mean': d_mean, 'dsm_std_ddof1': d_std,
            'simple_mean': s_mean, 'simple_std_ddof1': s_std,
            'relative_mean_diff': rel_mse_diff,
            'relative_mean_diff_simple_vs_vlb': rel_simple_diff,
            'note': ('VLB与匹配DSM为同一目标，均值差异只能反映种子噪声；'
                     '简化目标为不同的加权方式，其均值差异用于衡量该指标的判别力'),
        },
    },
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
