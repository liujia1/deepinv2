# -*- coding: utf-8 -*-
"""
实验11.3-1 VLB数值计算与三种参数化
对应章节: 11.1-11.3节

知识点:
  - 11.1节 VLB分解与正向过程后验
  - 11.2节 一致性项化简：从KL散度到均值匹配
  - 11.3节 三种参数化：ε预测、得分预测与x₀预测

实验内容:
  步骤1: VLB三项分解数值计算
  步骤2: KL散度→均值匹配等价性验证
  步骤3: 三种参数化互推与权重分析
  步骤4: VLB与DSM的结构对比

本实验不需要GPU，通过数值验证VLB三项分解、均值匹配等价性、
三种参数化的权重与互推关系。
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
    SAVE_DIR = os.path.join(_gdrive, '实验11.3-1')
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
print("实验11.3-1: VLB数值计算与三种参数化")
print("="*60)
print("对应章节: 11.1-11.3节")
print(f"保存目录: {SAVE_DIR}")

# ============================================================
# 噪声调度（DDPM线性）
# ============================================================
T = 1000
beta_min, beta_max = 1e-4, 0.02
betas = torch.linspace(beta_min, beta_max, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])

# 后验方差
posterior_var = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)
# 用于L_simple的固定方差: σ_t² = β_t
posterior_log_var = torch.log(posterior_var)


# ============================================================
# 步骤1：VLB三项分解数值计算（11.1节）
# ============================================================
print("\n" + "="*60)
print("步骤1：VLB三项分解数值计算（11.1节）")
print("="*60)

print("\n[核心思想]")
print("  VLB（变分下界）可分解为三项:")
print("  - L_T: 先验匹配项，衡量q(x_T|x_0)与p(x_T)的KL散度")
print("  - L_{t-1}: 一致性项，衡量后验q(x_{t-1}|x_t,x_0)与模型p_θ(x_{t-1}|x_t)的匹配程度")
print("  - L_0: 重建项，衡量从x_1重建x_0的能力")
print("  对于DDPM的噪声调度，L_T≈0，训练重点是L_{t-1}和L_0")

# L_T = D_KL(q(x_T|x_0) || p(x_T))
# q(x_T|x_0) = N(√ᾱ_T x_0, (1-ᾱ_T)I)
# p(x_T) = N(0, I)
# L_T = 0.5 * (ᾱ_T * (μ₀² + tr(Σ₀)) + (1-ᾱ_T) + d * log(ᾱ_T*σ₀² + (1-ᾱ_T)) - d)
# 对单位高斯数据 x_0 ~ N(0,I): L_T = 0.5 * (ᾱ_T + (1-ᾱ_T) + d*log(1) - d) ≈ 0

print("L_T (先验匹配项):")
ab_T = alpha_bars[-1]
print(f"  ᾱ_T = {ab_T:.8f}")
print(f"  √ᾱ_T = {torch.sqrt(ab_T):.6f} ≈ 0")
print(f"  1-ᾱ_T = {1-ab_T:.6f} ≈ 1")
print(f"  → q(x_T|x_0) ≈ N(0, I) = p(x_T)")
print(f"  → L_T ≈ 0 (无需优化)")

# L_{t-1}: 一致性项 (11.2节均值匹配)
# L_{t-1} = E_q[1/(2σ_t²) * ||μ̃_t - μ_θ||²] + C
# 权重 w_t = β_t² / (2σ_t² α_t (1-ᾱ_t))
print(f"\nL_{{t-1}} 各时间步的权重 w_t (11.3节):")

# 三种参数化的权重
wt_eps = betas**2 / (2 * posterior_var * alphas * (1 - alpha_bars))
wt_x0 = alpha_bars_prev * betas**2 / (2 * posterior_var * (1 - alpha_bars)**2)
wt_score = betas**2 / (2 * posterior_var * alphas)

print(f"{'t':>5s}  {'w_t(ε)':>12s}  {'w_t(x₀)':>12s}  {'w_t(s)':>12s}  {'w_t(ε)/w_t(s)':>14s}")
print("-" * 60)
# 注意：t=1时后验方差退化为0（alpha_bars_prev[0]=1导致分母为0），权重为inf，故从t=2开始采样
for t_idx in [2, 50, 100, 250, 500, 750, 999]:
    idx = t_idx - 1  # 数组下标idx对应物理时间步t=idx+1
    print(f"{t_idx:5d}  {wt_eps[idx].item():12.6f}  {wt_x0[idx].item():12.6f}  "
          f"{wt_score[idx].item():12.6f}  {wt_eps[idx].item()/max(wt_score[idx].item(),1e-30):14.6f}")

print(f"\n[观察]")
print(f"  w_t(ε) = β_t²/(2σ_t²α_t(1-ᾱ_t)): U形曲线，小t时大(~0.6)，中间t最小(~0.005)，大t时回升(~0.01)")
print(f"  w_t(x₀) = ᾱ_{{t-1}}β_t²/(2σ_t²(1-ᾱ_t)²): 单调递减，t小时极大(~10³)，t大时趋于0")
print(f"  w_t(ε)/w_t(s) = 1/(1-ᾱ_t): ε-prediction的权重比score-prediction多1/(1-ᾱ_t)因子")

# ★ 自检：验证三种权重的理论恒等式(避免注释里的"~0.6, ~0.005"等手算值与代码脱节)
# 理论关系(由权重定义直接推导):
#   R1: w_t(ε)/w_t(s) = 1/(1-ᾱ_t)            (wt_eps=β²/(2σ²·α·(1-ᾱ)), wt_score=β²/(2σ²·α))
#   R2: w_t(x₀)/w_t(ε) = ᾱ_{t-1}·α_t/(1-ᾱ_t) (wt_x0=ᾱ_{t-1}·β²/(2σ²(1-ᾱ)²), wt_eps=β²/(2σ²·α·(1-ᾱ)))
# 验证规则: 相对误差 < 1e-3 即认为公式实现正确
print(f"[权重自检] 三种VLB权重的理论恒等式(实际运行, 非手算):")
_t_check = [2, 50, 100, 500, 999]
for _t in _t_check:
    _i = _t - 1
    _ratio_eps_score = wt_eps[_i].item() / max(wt_score[_i].item(), 1e-30)
    _theory_eps_score = 1.0 / (1.0 - alpha_bars[_i].item())
    _err1 = abs(_ratio_eps_score - _theory_eps_score) / max(abs(_theory_eps_score), 1e-30)
    _ratio_x0_eps = wt_x0[_i].item() / max(wt_eps[_i].item(), 1e-30)
    _theory_x0_eps = alpha_bars_prev[_i].item() * alphas[_i].item() / (1.0 - alpha_bars[_i].item())
    _err2 = abs(_ratio_x0_eps - _theory_x0_eps) / max(abs(_theory_x0_eps), 1e-30)
    _flag1 = "OK" if _err1 < 1e-3 else "FAIL"
    _flag2 = "OK" if _err2 < 1e-3 else "FAIL"
    print(f"  t={_t:4d}: R1(w_ε/w_s)={_ratio_eps_score:.4f} vs 1/(1-ᾱ_t)={_theory_eps_score:.4f} [{_flag1}]; "
          f"R2(w_x0/w_ε)={_ratio_x0_eps:.4f} vs ᾱ_{{t-1}}/[(1-ᾱ_t)·α_t]={_theory_x0_eps:.4f} [{_flag2}]")

# L_0: 重建项
print(f"\nL_0 (重建项): 对高斯解码器, -E[log p(x_0|x_1)] ∝ ||x_0 - μ_θ(x_1,1)||²")


# ============================================================
# 步骤2：均值匹配验证（11.2节）
# KL散度 → 均值匹配的等价性
# ============================================================
print(f"\n{'='*60}")
print("步骤2：KL散度→均值匹配等价性验证（11.2节）")
print("=" * 60)

print("\n[核心思想]")
print("  11.2节的关键结论：当两个高斯分布有相同方差时，")
print("  D_KL(N(μ₁,σ²I) || N(μ₂,σ²I)) = 1/(2σ²) ||μ₁-μ₂||²")
print("  这将复杂的分布匹配问题转化为简单的回归问题，")
print("  训练扩散模型 = 回归后验均值 μ̃_t")

# 验证：对L_{t-1}项，用MC采样计算KL和均值匹配的比值
np.random.seed(42)
d = 2  # 维度
n_mc = 50000

# 随机生成μ₁, μ₂, σ²
mu1 = torch.randn(d)
mu2 = torch.randn(d)
sigma2 = 0.5  # 方差

# MC估计 KL
z = mu1.unsqueeze(0) + torch.sqrt(torch.tensor(sigma2)) * torch.randn(n_mc, d)
log_q = -0.5 * ((z - mu1.unsqueeze(0))**2).sum(dim=1) / sigma2
log_p = -0.5 * ((z - mu2.unsqueeze(0))**2).sum(dim=1) / sigma2
kl_mc = (log_q - log_p).mean().item()

# 均值匹配公式
kl_formula = 0.5 * torch.sum((mu1 - mu2)**2).item() / sigma2

print(f"\n验证结果:")
print(f"  MC估计 KL = {kl_mc:.6f}")
print(f"  均值匹配公式 1/(2σ²)||μ₁-μ₂||² = {kl_formula:.6f}")
print(f"  误差 = {abs(kl_mc - kl_formula):.6f}")

print(f"\n[关键结论]")
print(f"  11.2节等价性精确成立: KL散度 = 均值匹配（相同方差时）")
print(f"  这意味着扩散模型的训练可以简化为均值回归问题")


# ============================================================
# 步骤3：三种参数化互推与权重分析（11.3节）
# ============================================================
print(f"\n{'='*60}")
print("步骤3：三种参数化互推与权重分析（11.3节）")
print("=" * 60)

print("\n[核心思想]")
print("  11.3节介绍的三种参数化方式在数学上完全等价:")
print("  - ε-prediction: 预测噪声 ε")
print("  - x₀-prediction: 预测原始数据 x₀")
print("  - score-prediction: 预测得分函数 ∇log p(x_t)")
print("  它们之间可以通过公式精确互推，但VLB权重不同")

# 11.3节互推关系:
# x̂₀ = (x_t - √(1-ᾱ_t)·ε̂) / √ᾱ_t
# ŝ = -ε̂ / √(1-ᾱ_t)
# ε̂ = -√(1-ᾱ_t)·ŝ = (x_t - √ᾱ_t·x̂₀) / √(1-ᾱ_t)

# 验证：给定ε̂，计算x̂₀和ŝ，再反推回ε̂
x0_test = torch.randn(5, 2)
eps_test = torch.randn_like(x0_test)

print(f"\n互推验证: ε̂ → x̂₀ → ε̂ (闭环) 和 ε̂ → ŝ → ε̂ (闭环)")
print(f"{'t':>5s}  {'ε̂→x̂₀→ε̂ 误差':>15s}  {'ε̂→ŝ→ε̂ 误差':>15s}")
print("-" * 45)

for t_idx in [10, 100, 500, 900]:
    idx = t_idx - 1  # 数组下标idx对应物理时间步t=idx+1
    ab = alpha_bars[idx]
    sqrt_ab = torch.sqrt(ab)
    sqrt_1mab = torch.sqrt(1 - ab)

    x_t = sqrt_ab * x0_test + sqrt_1mab * eps_test

    # ε̂ → x̂₀
    eps_hat = eps_test
    x0_hat = (x_t - sqrt_1mab * eps_hat) / sqrt_ab
    # x̂₀ → ε̂
    eps_recovered = (x_t - sqrt_ab * x0_hat) / sqrt_1mab

    # ε̂ → ŝ
    s_hat = -eps_hat / sqrt_1mab
    # ŝ → ε̂
    eps_from_score = -sqrt_1mab * s_hat

    err_x0 = (eps_hat - eps_recovered).abs().max().item()
    err_score = (eps_hat - eps_from_score).abs().max().item()

    print(f"{t_idx:5d}  {err_x0:15.2e}  {err_score:15.2e}")

print(f"\n[关键结论]")
print(f"  三种参数化互推精确成立（误差<浮点精度）")

# 权重对比: 三种参数化的VLB损失权重
print(f"\n三种参数化的VLB损失权重 w_t:")
print(f"{'t':>5s}  {'ε-pred':>12s}  {'x₀-pred':>12s}  {'score-pred':>12s}  {'ε/x₀比':>8s}")
print("-" * 55)

# 注意：t=1时后验方差退化为0（alpha_bars_prev[0]=1导致分母为0），权重为inf，故从t=2开始采样
for t_idx in [2, 10, 50, 100, 250, 500, 750, 999]:
    idx = t_idx - 1  # 数组下标idx对应物理时间步t=idx+1
    we = wt_eps[idx].item()
    wx = wt_x0[idx].item()
    ws = wt_score[idx].item()
    print(f"{t_idx:5d}  {we:12.6f}  {wx:12.6f}  {ws:12.6f}  {we/max(wx,1e-30):8.4f}")

print(f"\n[观察]")
print(f"  - ε-pred权重: U形曲线，小t时大(~0.6)，中间t最小(~0.005，约t=350)，大t时回升(~0.01)")
print(f"  - x₀-pred权重: 单调递减，t小时极大(~10³)，t大时趋于0")
print(f"  - score-pred权重: 单调递增，小t时极小(~10⁻⁴)，大t时较大(~10⁻²)")


# ============================================================
# 可视化
# ============================================================
print(f"\n{'='*60}")
print("生成可视化图表...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
t_range = range(1, T + 1)

# (a) VLB三项的量级分析
ax = axes[0, 0]
# L_T≈0
# L_{t-1}的权重随t变化
ax.semilogy(t_range, wt_eps.numpy(), 'b-', linewidth=1.5, label=r'$w_t^{(\varepsilon)}$ ($\varepsilon$-pred权重)')
ax.axhline(1, color='r', linestyle='--', linewidth=1.5, label=r'$L_{simple}$权重 (=1)')
ax.set_xlabel('时间步 $t$', fontsize=12)
ax.set_ylabel('权重 (对数尺度)', fontsize=12)
ax.set_title('(a) $\\varepsilon$-prediction的VLB权重 vs $L_{simple}$', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (b) 三种参数化权重对比
ax = axes[0, 1]
ax.semilogy(t_range, wt_eps.numpy(), 'b-', linewidth=1.5, label=r'$w_t^{(\varepsilon)}$')
ax.semilogy(t_range, wt_x0.numpy(), 'r-', linewidth=1.5, label=r'$w_t^{(x_0)}$')
ax.semilogy(t_range, wt_score.numpy(), 'g-', linewidth=1.5, label=r'$w_t^{(s)}$')
ax.set_xlabel('时间步 $t$', fontsize=12)
ax.set_ylabel('权重 (对数尺度)', fontsize=12)
ax.set_title('(b) 三种参数化的VLB权重对比', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.annotate('$x_0$-pred权重单调递减\n$\\varepsilon$-pred权重U形，score-pred权重单调递增',
            xy=(0.35, 0.75), xycoords='axes fraction', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeaa7', alpha=0.8))

# (c) 前向过程后验均值系数（11.1节）
coeff_x0 = torch.sqrt(alpha_bars_prev) * betas / (1 - alpha_bars)
coeff_xt = torch.sqrt(alphas) * (1 - alpha_bars_prev) / (1 - alpha_bars)

ax = axes[1, 0]
ax.plot(t_range, coeff_x0.numpy(), 'r-', linewidth=2, label=r'$x_0$系数')
ax.plot(t_range, coeff_xt.numpy(), 'b-', linewidth=2, label=r'$x_t$系数')
ax.plot(t_range, (coeff_x0 + coeff_xt).numpy(), 'k--', linewidth=1, label='和')
ax.set_xlabel('时间步 $t$', fontsize=12)
ax.set_ylabel('系数值', fontsize=12)
ax.set_title('(c) 后验均值 $\\tilde{\\mu}_t$ 的系数（11.1节）', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (d) ★ 原创设计: VLB vs L_simple的有效梯度分析
# 模拟: 当ε预测有固定误差时，VLB和L_simple对各时间步的梯度大小
ax = axes[1, 1]
eps_error = 0.1  # 固定预测误差
# VLB梯度 ∝ w_t * ε_error
# L_simple梯度 ∝ 1 * ε_error (等权重)
vlb_grad = wt_eps.numpy() * eps_error
simple_grad = np.ones(T) * eps_error

ax.semilogy(t_range, vlb_grad + 1e-30, 'b-', linewidth=2, label=r'VLB梯度 $\propto w_t$')
ax.semilogy(t_range, simple_grad, 'r--', linewidth=2, label=r'$L_{simple}$梯度 $\propto 1$')
ax.set_xlabel('时间步 $t$', fontsize=12)
ax.set_ylabel('有效梯度 (对数尺度)', fontsize=12)
ax.set_title('(d) VLB vs $L_{simple}$ 有效梯度（★ 原创设计）', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.annotate('VLB: 小$t$梯度极大，大$t$梯度极小\n→ 训练不稳定',
            xy=(0.3, 0.7), xycoords='axes fraction', fontsize=9, color='blue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#dfe6e9', alpha=0.8))
ax.annotate('$L_{simple}$: 均匀梯度\n→ 训练稳定',
            xy=(0.6, 0.3), xycoords='axes fraction', fontsize=9, color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdecea', alpha=0.8))

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤1_VLB权重与参数化.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图1已保存: {fig_path}")


# ============================================================
# 步骤4：★ 原创设计 - VLB与DSM的结构对比（11.5节预览）
# ============================================================
print(f"\n{'='*60}")
print("步骤4：VLB与DSM的结构对比（11.5节预览）")
print("=" * 60)

print("\n[核心思想]")
print("  11.5节指出: ε-prediction的VLB与第6章DSM在结构上完全一致,")
print("  唯一区别是时间权重 w_t:")
print("")
print("  VLB (ε-prediction):  L = Σ_t w_t · E[||ε - ε̂_θ(x_t,t)||²]")
print("  DSM (第6章):          L = Σ_t λ(σ_t) · E[||s_θ(x_t,σ_t) + ε/σ_t||²]")
print("")
print("  利用 ε̂_θ = -√(1-ᾱ_t) · s_θ 的转换:")
print("    ||ε - ε̂_θ||² = (1-ᾱ_t) · ||-ε/√(1-ᾱ_t) - s_θ||²")
print("                 = (1-ᾱ_t) · ||ε/√(1-ᾱ_t) + s_θ||²")
print("")
print("  因此:  VLB权重 w_t = (1-ᾱ_t) · λ(σ_t)/(σ_t²) (经适当变换)")
print("         → 当 λ(σ_t) = σ_t² 时, VLB权重 = (1-ᾱ_t) → DSM简化目标")
print("         → 当 λ(σ_t) = σ_t²/(1-ᾱ_t) · w_t 时, DSM = VLB (精确等价)")

# 数值验证: ε-prediction VLB 与 score-prediction 的损失关系
np.random.seed(42)
x0_v = torch.randn(100, 2)
eps_v = torch.randn_like(x0_v)

print(f"\n数值验证: ||ε-ε̂||² ≈ (1-ᾱ_t)·||ŝ-s*||²")
print(f"{'t':>5s}  {'||ε-ε̂||²':>12s}  {'(1-ᾱ_t)·||s+s*||²':>18s}  {'比值':>8s}")
print("-" * 50)

for t_idx in [10, 100, 500, 900]:
    idx = t_idx - 1  # 数组下标idx对应物理时间步t=idx+1
    ab = alpha_bars[idx]
    x_t = torch.sqrt(ab) * x0_v + torch.sqrt(1 - ab) * eps_v

    # ε-prediction损失: ||ε - ε̂||² (假设完美预测: ε̂=ε, 则误差=0)
    # 用微小扰动模拟非完美预测
    eps_hat = eps_v + 0.01 * torch.randn_like(eps_v)
    loss_eps = ((eps_v - eps_hat)**2).mean().item()

    # score-prediction损失: ||s_θ + ε/√(1-ᾱ_t)||²
    s_hat = -eps_hat / torch.sqrt(1 - ab)
    s_target = -eps_v / torch.sqrt(1 - ab)
    loss_score = ((s_hat - s_target)**2).mean().item()

    ratio = loss_eps / max(loss_score * (1 - ab.item()), 1e-30)
    print(f"{t_idx:5d}  {loss_eps:12.8f}  {loss_score*(1-ab.item()):18.8f}  {ratio:8.4f}")

print(f"\n[关键结论]")
print(f"  ||ε-ε̂||² ≈ (1-ᾱ_t)·||ŝ-s*||² 精确成立")
print(f"  这是11.5节Score≡VLB等价性的数值验证")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print("实验11.3-1 完成!")
print("=" * 60)
print("""
关键结论:
1. VLB三项分解（11.1节）
   - L_T≈0 (先验自动匹配), L_{t-1} (一致性项), L_0 (重建项)
   - 后验均值μ̃_t是x_t和x_0的线性组合，系数和≈1

2. 均值匹配等价性（11.2节）
   - KL散度 = 1/(2σ²)||μ̃_t-μ_θ||²（同方差高斯时）
   - 训练扩散模型=回归后验均值，从分布匹配→回归问题

3. 三种参数化（11.3节）
   - ε/x₀/score三种参数化互推精确成立
   - VLB权重: ε-pred呈U形曲线, x₀-pred单调递减, score-pred单调递增

4. VLB vs L_simple（11.4节）
   - VLB权重跨越多个数量级→训练不稳定
   - L_simple等权重1→训练稳定（11.4节Ho et al. 2020的关键简化）

5. VLB与DSM的结构等价性（11.5节预览）
   - ||ε-ε̂||² = (1-ᾱ_t)·||ŝ-s*||²
   - 这是第12章Score≡ELBO等价性的数值基础
""")

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
    'L_T_prior_matching': {
        'alpha_bar_T': alpha_bars[-1].item(),
    },
    'KL_mean_matching_verification': {
        'KL_MC_estimate': kl_mc,
        'KL_formula': kl_formula,
        'error': abs(kl_mc - kl_formula),
    },
    'VLB_weights_sampled': {
        f't_{t}': {
            'wt_eps': wt_eps[t-1].item(),
            'wt_x0': wt_x0[t-1].item(),
            'wt_score': wt_score[t-1].item(),
        }
        for t in [2, 50, 100, 250, 500, 750, 999]
    },
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")