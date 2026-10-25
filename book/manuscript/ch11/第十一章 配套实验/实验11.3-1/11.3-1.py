# -*- coding: utf-8 -*-
"""
实验11.3-1 VLB数值计算与三种参数化
对应章节: 11.1-11.3节

知识点:
  - 11.1节 VLB分解与正向过程后验
  - 11.2节 一致性项化简：从KL散度到均值匹配
  - 11.3节 三种参数化：ε预测、得分预测与x₀预测

实验内容:
  步骤1: VLB三项结构与一致性项权重数值分析
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

# 后验方差 \tilde{\beta}_t；t=1时退化为0，对应确定性后验/L0项，需单独处理
posterior_var = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)


# ============================================================
# 步骤1：VLB三项结构与一致性项权重数值分析（11.1节）
# ============================================================
print("\n" + "="*60)
print("步骤1：VLB三项结构与一致性项权重数值分析（11.1节）")
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
# 固定x0时: L_T/d = 0.5 * (ᾱ_T·||x0||²/d + (1-ᾱ_T) - 1 - log(1-ᾱ_T))
# 对单位高斯数据 x_0 ~ N(0,I)，有E||x0||²/d=1，因此 E[L_T]/d = -0.5*log(1-ᾱ_T) ≈ 0

print("L_T (先验匹配项):")
ab_T = alpha_bars[-1]
# 若 x0~N(0,I)，按维平均的期望KL：-0.5*log(1-\bar\alpha_T)
lt_per_dim_standard_gaussian = -0.5 * torch.log1p(-ab_T)
print(f"  ᾱ_T = {ab_T:.8f}")
print(f"  √ᾱ_T = {torch.sqrt(ab_T):.6f} ≈ 0")
print(f"  1-ᾱ_T = {1-ab_T:.8f} ≈ 1")
print(f"  x0~N(0,I)时，E[L_T]/d = -0.5·log(1-ᾱ_T) = {lt_per_dim_standard_gaussian.item():.8f}")
print(f"  → q(x_T|x_0) ≈ N(0, I) = p(x_T)")
print(f"  → L_T是模型无关的近似零常数项，因此不作为训练优化重点")

# L_{t-1}: 一致性项 (11.2节均值匹配)
# L_{t-1} = E_q[1/(2σ_t²) * ||μ̃_t - μ_θ||²] + C
# 权重 w_t = β_t² / (2σ_t² α_t (1-ᾱ_t))
print(f"\nL_{{t-1}} 各时间步的权重 w_t (11.3节):")

# 三种参数化的权重
wt_eps = betas**2 / (2 * posterior_var * alphas * (1 - alpha_bars))
wt_x0 = alpha_bars_prev * betas**2 / (2 * posterior_var * (1 - alpha_bars)**2)
wt_score = betas**2 / (2 * posterior_var * alphas)

# 普通高斯一致性项仅适用于t>=2；t=1是确定性后验，对应单独的L0项
valid_idx = torch.arange(1, T, dtype=torch.long)
valid_t = valid_idx + 1
wt_eps_valid = wt_eps[valid_idx]
wt_x0_valid = wt_x0[valid_idx]
wt_score_valid = wt_score[valid_idx]

eps_min_idx = int(torch.argmin(wt_eps_valid).item())
score_increasing_from_t3 = bool(torch.all(wt_score[3:] > wt_score[2:-1]).item())
x0_decreasing_valid = bool(torch.all(wt_x0_valid[1:] < wt_x0_valid[:-1]).item())

print(f"{'t':>5s}  {'w_t(ε)':>12s}  {'w_t(x₀)':>12s}  {'w_t(s)':>12s}  {'w_t(ε)/w_t(s)':>14s}")
print("-" * 60)
# 注意：t=1时后验方差退化为0（alpha_bars_prev[0]=1导致分母为0），权重为inf，故从t=2开始采样
for t_idx in [2, 50, 100, 250, 500, 750, 999]:
    idx = t_idx - 1  # 数组下标idx对应物理时间步t=idx+1
    print(f"{t_idx:5d}  {wt_eps[idx].item():12.6f}  {wt_x0[idx].item():12.6f}  "
          f"{wt_score[idx].item():12.6f}  {wt_eps[idx].item()/max(wt_score[idx].item(),1e-30):14.6f}")

print(f"\n[观察]")
print(f"  w_t(ε) = β_t²/(2σ_t²α_t(1-ᾱ_t)): U形曲线，小t时大(~0.6)，t={int(valid_t[eps_min_idx].item())}附近最小(~{wt_eps_valid[eps_min_idx].item():.5f})，大t时回升(~0.01)")
print(f"  w_t(x₀) = ᾱ_{{t-1}}β_t²/(2σ_t²(1-ᾱ_t)²): t>=2时单调递减，t小时极大(~10³)，t大时趋于0")
print(f"  w_t(ε)/w_t(s) = 1/(1-ᾱ_t): ε-prediction的权重比score-prediction多1/(1-ᾱ_t)因子")
print(f"  score权重在t=2→3处有轻微下降({wt_score[1].item():.8f}→{wt_score[2].item():.8f})，自t=3起严格递增")

# ★ 自检：验证三种权重的理论恒等式(避免注释里的"~0.6, ~0.005"等手算值与代码脱节)
# 理论关系(由权重定义直接推导):
#   R1: w_t(ε)/w_t(s) = 1/(1-ᾱ_t)            (wt_eps=β²/(2σ²·α·(1-ᾱ)), wt_score=β²/(2σ²·α))
#   R2: w_t(x₀)/w_t(ε) = ᾱ_{t-1}·α_t/(1-ᾱ_t) (wt_x0=ᾱ_{t-1}·β²/(2σ²(1-ᾱ)²), wt_eps=β²/(2σ²·α·(1-ᾱ)))
# 验证规则: 相对误差 < 1e-3 即认为公式实现正确
print(f"[权重自检] 三种VLB权重的理论恒等式(实际运行, 非手算):")
_t_check = [2, 50, 100, 500, 999]
weight_identity_checks = []
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
    weight_identity_checks.append({
        't': _t,
        'eps_over_score_ratio': _ratio_eps_score,
        'eps_over_score_theory': _theory_eps_score,
        'eps_over_score_relative_error': _err1,
        'x0_over_eps_ratio': _ratio_x0_eps,
        'x0_over_eps_theory': _theory_x0_eps,
        'x0_over_eps_relative_error': _err2,
        'passed': _flag1 == "OK" and _flag2 == "OK",
    })
    print(f"  t={_t:4d}: R1(w_ε/w_s)={_ratio_eps_score:.4f} vs 1/(1-ᾱ_t)={_theory_eps_score:.4f} [{_flag1}]; "
          f"R2(w_x0/w_ε)={_ratio_x0_eps:.4f} vs ᾱ_t/(1-ᾱ_t)={_theory_x0_eps:.4f} [{_flag2}]")

if not all(item['passed'] for item in weight_identity_checks):
    raise AssertionError("VLB权重理论恒等式自检失败，请检查三种参数化权重公式")

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

# 刻意把两均值拉开到固定距离 ||μ₁-μ₂||=0.5，
# 避免随机种子恰好给出 μ₁≈μ₂ 的退化用例（那样 KL 本身≈0，任何估计都能“通过”）
_delta = mu2 - mu1
mu2 = mu1 + 0.5 * _delta / _delta.norm().clamp(min=1e-12)

# MC估计 KL
z = mu1.unsqueeze(0) + torch.sqrt(torch.tensor(sigma2)) * torch.randn(n_mc, d)
log_q = -0.5 * ((z - mu1.unsqueeze(0))**2).sum(dim=1) / sigma2
log_p = -0.5 * ((z - mu2.unsqueeze(0))**2).sum(dim=1) / sigma2
kl_samples = log_q - log_p
kl_mc = kl_samples.mean().item()
kl_mc_standard_error = (kl_samples.std(unbiased=True) / np.sqrt(n_mc)).item()

# 均值匹配公式
kl_formula = 0.5 * torch.sum((mu1 - mu2)**2).item() / sigma2
kl_abs_error = abs(kl_mc - kl_formula)
kl_relative_error = kl_abs_error / max(abs(kl_formula), 1e-30)
kl_z_score = kl_abs_error / max(kl_mc_standard_error, 1e-30)

print(f"\n验证结果:")
print(f"  MC估计 KL = {kl_mc:.6f}")
print(f"  MC标准误 = {kl_mc_standard_error:.6f}")
print(f"  均值匹配公式 1/(2σ²)||μ₁-μ₂||² = {kl_formula:.6f}")
print(f"  绝对误差 = {kl_abs_error:.6f}，相对误差 = {kl_relative_error:.4%}")
print(f"  误差/标准误 = {kl_z_score:.3f}（应 < 3，否则视为公式与估计不相容）")

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
parametrization_conversion_checks = []

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
    parametrization_conversion_checks.append({
        't': t_idx,
        'eps_x0_eps_max_abs_error': err_x0,
        'eps_score_eps_max_abs_error': err_score,
        'passed': err_x0 < 1e-5 and err_score < 1e-5,
    })

    print(f"{t_idx:5d}  {err_x0:15.2e}  {err_score:15.2e}")

if not all(item['passed'] for item in parametrization_conversion_checks):
    raise AssertionError("三种参数化闭环换算自检失败")

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
print(f"  - ε-pred权重: U形曲线，小t时大(~0.6)，t={int(valid_t[eps_min_idx].item())}附近最小(~{wt_eps_valid[eps_min_idx].item():.5f})，大t时回升(~0.01)")
print(f"  - x₀-pred权重: t>=2时单调递减，t小时极大(~10³)，t大时趋于0")
print(f"  - score-pred权重: 整体随时长增大，但t=2→3轻微下降；自t=3起严格递增")


# ============================================================
# 可视化
# ============================================================
print(f"\n{'='*60}")
print("生成可视化图表...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
t_range = range(1, T + 1)

# (a) VLB一致性项的权重分析（t=1的确定性L0项不纳入普通高斯权重）
ax = axes[0, 0]
ax.semilogy(valid_t.numpy(), wt_eps_valid.numpy(), 'b-', linewidth=1.5,
            label=r'$w_t^{(\varepsilon)}$ ($\varepsilon$-pred权重)')
ax.axhline(1, color='r', linestyle='--', linewidth=1.5, label=r'$L_{simple}$权重 (=1)')
ax.set_xlabel('时间步 $t$（$t\geq2$）', fontsize=12)
ax.set_ylabel('权重 (对数尺度)', fontsize=12)
ax.set_title('(a) $\\varepsilon$-prediction的VLB权重 vs $L_{simple}$', fontsize=13)
ax.set_xlim(1.5, T + 0.5)
ax.set_xticks([2, 200, 400, 600, 800, 1000])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (b) 三种参数化权重对比
ax = axes[0, 1]
ax.semilogy(valid_t.numpy(), wt_eps_valid.numpy(), 'b-', linewidth=1.5, label=r'$w_t^{(\varepsilon)}$')
ax.semilogy(valid_t.numpy(), wt_x0_valid.numpy(), 'r-', linewidth=1.5, label=r'$w_t^{(x_0)}$')
ax.semilogy(valid_t.numpy(), wt_score_valid.numpy(), 'g-', linewidth=1.5, label=r'$w_t^{(s)}$')
ax.set_xlabel('时间步 $t$（$t\geq2$）', fontsize=12)
ax.set_ylabel('权重 (对数尺度)', fontsize=12)
ax.set_title('(b) 三种参数化的VLB权重对比', fontsize=13)
ax.set_xlim(1.5, T + 0.5)
ax.set_xticks([2, 200, 400, 600, 800, 1000])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.annotate('$x_0$-pred权重自t=2递减\n$\\varepsilon$-pred权重U形；score权重自t=3递增',
            xy=(0.35, 0.75), xycoords='axes fraction', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeaa7', alpha=0.8))

# (c) 正向过程后验均值系数（11.1节）
coeff_x0 = torch.sqrt(alpha_bars_prev) * betas / (1 - alpha_bars)
coeff_xt = torch.sqrt(alphas) * (1 - alpha_bars_prev) / (1 - alpha_bars)
coeff_sum = coeff_x0 + coeff_xt
coeff_sum_valid = coeff_sum[valid_idx]

ax = axes[1, 0]
ax.plot(t_range, coeff_x0.numpy(), 'r-', linewidth=2, label=r'$x_0$系数')
ax.plot(t_range, coeff_xt.numpy(), 'b-', linewidth=2, label=r'$x_t$系数')
ax.plot(t_range, (coeff_x0 + coeff_xt).numpy(), 'k--', linewidth=1, label='和')
ax.set_xlabel('时间步 $t$', fontsize=12)
ax.set_ylabel('系数值', fontsize=12)
ax.set_title('(c) 后验均值 $\\tilde{\\mu}_t$ 的系数（11.1节）', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (d) VLB vs L_simple的时间步损失占比（均匀采样t时的归一化尺度，不是网络参数梯度）
ax = axes[1, 1]
vlb_uniform_share = wt_eps_valid / wt_eps_valid.sum()
simple_uniform_share = torch.full_like(wt_eps_valid, 1.0 / len(wt_eps_valid))

ax.semilogy(valid_t.numpy(), vlb_uniform_share.numpy(), 'b-', linewidth=2,
            label=r'VLB归一化时间步占比 $w_t/\sum w$')
ax.semilogy(valid_t.numpy(), simple_uniform_share.numpy(), 'r--', linewidth=2,
            label=r'$L_{simple}$占比 $=1/(T-1)$')
ax.set_xlabel('时间步 $t$（$t\geq2$）', fontsize=12)
ax.set_ylabel('归一化损失占比 (对数尺度)', fontsize=12)
ax.set_title('(d) VLB vs $L_{simple}$ 的时间步贡献尺度', fontsize=13)
ax.set_xlim(1.5, T + 0.5)
ax.set_xticks([2, 200, 400, 600, 800, 1000])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.annotate('VLB: 小$t$占比高，中间$t$占比低\n→ 各时间步损失尺度不均衡',
            xy=(0.32, 0.70), xycoords='axes fraction', fontsize=9, color='blue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#dfe6e9', alpha=0.8))
ax.annotate('$L_{simple}$: 均匀采样下每步占比相同\n→ 避免权重跨数量级',
            xy=(0.56, 0.22), xycoords='axes fraction', fontsize=9, color='red',
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
print("  令 σ_t² = 1-ᾱ_t，利用 ε̂_θ = -σ_t·s_θ 的转换:")
print("    ||s_θ + ε/σ_t||² = 1/σ_t² · ||ε - ε̂_θ||²")
print("    λ(σ_t)·||s_θ + ε/σ_t||² = λ(σ_t)/σ_t² · ||ε - ε̂_θ||²")
print("")
print("  因此:")
print("         → 当 λ(σ_t)=σ_t² 时, DSM目标化为等权重的 ||ε-ε̂_θ||²，即L_simple形式")
print("         → 当 λ(σ_t)=σ_t²·w_t^(ε)=w_t^(s) 时, DSM = ε-prediction VLB (精确等价)")

# 数值验证: ε-prediction VLB 与 score-prediction 的损失关系
np.random.seed(42)
x0_v = torch.randn(100, 2)
eps_v = torch.randn_like(x0_v)

print(f"\n数值验证: ||ε-ε̂||² = (1-ᾱ_t)·||ŝ-s*||²，其中s*=-ε/√(1-ᾱ_t)")
print(f"{'t':>5s}  {'||ε-ε̂||²':>12s}  {'(1-ᾱ_t)·||ŝ-s*||²':>18s}  {'比值':>8s}")
print("-" * 50)
dsm_equivalence_checks = []

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

    scaled_loss_score = loss_score * (1 - ab.item())
    ratio = loss_eps / max(scaled_loss_score, 1e-30)
    ratio_error = abs(ratio - 1.0)
    dsm_equivalence_checks.append({
        't': t_idx,
        'epsilon_loss': loss_eps,
        'scaled_score_loss': scaled_loss_score,
        'epsilon_over_scaled_score': ratio,
        'relative_error': ratio_error,
        'passed': ratio_error < 1e-6,
    })
    print(f"{t_idx:5d}  {loss_eps:12.8f}  {scaled_loss_score:18.8f}  {ratio:8.4f}")

if not all(item['passed'] for item in dsm_equivalence_checks):
    raise AssertionError("epsilon-MSE与score-MSE换算自检失败")

print(f"\n[关键结论]")
print(f"  ||ε-ε̂||² = (1-ᾱ_t)·||ŝ-s*||² 在数值上精确成立")
print(f"  这是第12章Score≡ELBO等价性的数值基础")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print("实验11.3-1 完成!")
print("=" * 60)
print("""
关键结论:
1. VLB三项分解（11.1节）
   - L_T是模型无关的近似零常数项，L_{t-1}是一致性项，L_0是重建项
   - 后验均值μ̃_t是x_t和x_0的线性组合，系数和≈1；t=1的确定性后验需单独处理

2. 均值匹配等价性（11.2节）
   - KL散度 = 1/(2σ²)||μ̃_t-μ_θ||²（同方差高斯时）
   - 训练扩散模型=回归后验均值，从分布匹配→回归问题

3. 三种参数化（11.3节）
   - ε/x₀/score三种参数化互推精确成立
   - VLB权重: ε-pred呈U形曲线，x₀-pred自t=2递减，score-pred自t=3递增

4. VLB vs L_simple（11.4节）
   - 本实验验证的是时间步损失尺度：VLB权重跨越多个数量级，各时间步贡献不均衡
   - L_simple在均匀采样t时给予等权重，可避免该尺度差异；实际训练稳定性需另行训练验证

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
    'metadata': {
        'experiment': '11.3-1 VLB数值计算与三种参数化',
        'T': T,
        'beta_min': beta_min,
        'beta_max': beta_max,
        'schedule': 'DDPM linear beta schedule',
        'numpy_seed': 42,
        'torch_seed': 42,
        'torch_version': torch.__version__,
        'gaussian_consistency_terms': 't=2,...,T; t=1 is deterministic and is treated as L0',
    },
    'L_T_prior_matching': {
        'alpha_bar_T': alpha_bars[-1].item(),
        'sqrt_alpha_bar_T': torch.sqrt(alpha_bars[-1]).item(),
        'one_minus_alpha_bar_T': (1.0 - alpha_bars[-1]).item(),
        'expected_LT_per_dim_for_standard_gaussian_x0': lt_per_dim_standard_gaussian.item(),
        'interpretation': 'model-independent near-zero constant for the DDPM schedule',
    },
    'KL_mean_matching_verification': {
        'dimension': d,
        'n_monte_carlo': n_mc,
        'sigma_squared': sigma2,
        'KL_MC_estimate': kl_mc,
        'KL_MC_standard_error': kl_mc_standard_error,
        'KL_formula': kl_formula,
        'absolute_error': kl_abs_error,
        'relative_error': kl_relative_error,
        'z_score': kl_z_score,
        'error': kl_abs_error,
        'passed': kl_abs_error < 3.0 * kl_mc_standard_error,
    },
    'VLB_weights_sampled': {
        f't_{t}': {
            'wt_eps': wt_eps[t-1].item(),
            'wt_x0': wt_x0[t-1].item(),
            'wt_score': wt_score[t-1].item(),
        }
        for t in [2, 50, 100, 250, 500, 750, 999]
    },
    'VLB_weight_statistics': {
        'eps_prediction': {
            'minimum_weight': wt_eps_valid[eps_min_idx].item(),
            'minimum_t': int(valid_t[eps_min_idx].item()),
            't_2_weight': wt_eps[1].item(),
            't_1000_weight': wt_eps[-1].item(),
            'shape': 'U-shaped for t>=2',
        },
        'x0_prediction': {
            't_2_weight': wt_x0[1].item(),
            't_1000_weight': wt_x0[-1].item(),
            'strictly_decreasing_for_t_ge_2': x0_decreasing_valid,
        },
        'score_prediction': {
            't_2_weight': wt_score[1].item(),
            't_3_weight': wt_score[2].item(),
            't_1000_weight': wt_score[-1].item(),
            'strictly_increasing_from_t_3': score_increasing_from_t3,
            'monotonicity_note': 'linear DDPM schedule has a small decrease from t=2 to t=3',
        },
    },
    'weight_identity_checks': weight_identity_checks,
    'posterior_mean_coefficients': {
        'coefficient_sum_min_t_ge_2': float(coeff_sum_valid.min().item()),
        'coefficient_sum_max_t_ge_2': float(coeff_sum_valid.max().item()),
        'coefficient_sum_t_1': float(coeff_sum[0].item()),
    },
    'parametrization_conversion_checks': parametrization_conversion_checks,
    'DSM_equivalence_checks': dsm_equivalence_checks,
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")