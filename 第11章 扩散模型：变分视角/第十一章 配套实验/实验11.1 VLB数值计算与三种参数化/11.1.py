# -*- coding: utf-8 -*-
"""
实验11.1 VLB数值计算与三种参数化
对应知识点：
  - 11.1节 VLB分解与正向过程后验
  - 11.2节 一致性项化简：从KL散度到均值匹配
  - 11.3节 三种参数化：ε预测、得分预测与x₀预测

本实验不需要GPU，通过数值验证VLB三项分解、均值匹配等价性、
三种参数化的权重与互推关系。
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

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
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
# ========================================================

np.random.seed(42)
import torch
torch.manual_seed(42)

SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()


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
print("=" * 60)
print("步骤1：VLB三项分解数值计算")
print("=" * 60)

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
for t_idx in [1, 50, 100, 250, 500, 750, 999]:
    t = t_idx
    print(f"{t_idx:5d}  {wt_eps[t].item():12.6f}  {wt_x0[t].item():12.6f}  "
          f"{wt_score[t].item():12.6f}  {wt_eps[t].item()/max(wt_score[t].item(),1e-30):14.6f}")

print(f"\n关键观察:")
print(f"  w_t(ε) = β_t²/(2σ_t²α_t(1-ᾱ_t)): 小t时极大，大t时递减")
print(f"  w_t(x₀) = ᾱ_{{t-1}}β_t²/(2σ_t²(1-ᾱ_t)²): 峰值在中间t")
print(f"  w_t(ε)/w_t(s) = 1/(1-ᾱ_t): ε-prediction的权重比score-prediction多1/(1-ᾱ_t)因子")

# L_0: 重建项
print(f"\nL_0 (重建项): 对高斯解码器, -E[log p(x_0|x_1)] ∝ ||x_0 - μ_θ(x_1,1)||²")


# ============================================================
# 步骤2：均值匹配验证（11.2节）
# KL散度 → 均值匹配的等价性
# ============================================================
print(f"\n{'='*60}")
print("步骤2：KL散度→均值匹配等价性验证（11.2节）")
print("=" * 60)

# 11.2节: 当两个高斯有相同方差时,
# D_KL(N(μ₁,σ²I) || N(μ₂,σ²I)) = 1/(2σ²) ||μ₁-μ₂||²

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

print(f"MC估计 KL = {kl_mc:.6f}")
print(f"均值匹配公式 1/(2σ²)||μ₁-μ₂||² = {kl_formula:.6f}")
print(f"误差 = {abs(kl_mc - kl_formula):.6f}")
print(f"→ 11.2节等价性精确成立: KL散度 = 均值匹配（相同方差时）")


# ============================================================
# 步骤3：三种参数化互推与权重分析（11.3节）
# ============================================================
print(f"\n{'='*60}")
print("步骤3：三种参数化互推与权重分析（11.3节）")
print("=" * 60)

# 11.3节互推关系:
# x̂₀ = (x_t - √(1-ᾱ_t)·ε̂) / √ᾱ_t
# ŝ = -ε̂ / √(1-ᾱ_t)
# ε̂ = -√(1-ᾱ_t)·ŝ = (x_t - √ᾱ_t·x̂₀) / √(1-ᾱ_t)

# 验证：给定ε̂，计算x̂₀和ŝ，再反推回ε̂
x0_test = torch.randn(5, 2)
eps_test = torch.randn_like(x0_test)

print(f"互推验证: ε̂ → x̂₀ → ε̂ (闭环) 和 ε̂ → ŝ → ε̂ (闭环)")
print(f"{'t':>5s}  {'ε̂→x̂₀→ε̂ 误差':>15s}  {'ε̂→ŝ→ε̂ 误差':>15s}")
print("-" * 45)

for t_idx in [10, 100, 500, 900]:
    t = t_idx
    ab = alpha_bars[t]
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

print(f"\n→ 三种参数化互推精确成立（误差<浮点精度）")

# 权重对比: 三种参数化的VLB损失权重
print(f"\n三种参数化的VLB损失权重 w_t:")
print(f"{'t':>5s}  {'ε-pred':>12s}  {'x₀-pred':>12s}  {'score-pred':>12s}  {'ε/x₀比':>8s}")
print("-" * 55)

for t_idx in [1, 10, 50, 100, 250, 500, 750, 999]:
    t = t_idx
    we = wt_eps[t].item()
    wx = wt_x0[t].item()
    ws = wt_score[t].item()
    print(f"{t_idx:5d}  {we:12.6f}  {wx:12.6f}  {ws:12.6f}  {we/max(wx,1e-30):8.4f}")


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
ax.semilogy(t_range, wt_eps.numpy(), 'b-', linewidth=1.5, label=r'$w_t^{(\varepsilon)}$ (ε-pred权重)')
ax.axhline(1, color='r', linestyle='--', linewidth=1.5, label=r'$L_{simple}$权重 (=1)')
ax.set_xlabel('时间步 t', fontsize=12)
ax.set_ylabel('权重 (对数尺度)', fontsize=12)
ax.set_title('(a) ε-prediction的VLB权重 vs L_simple', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (b) 三种参数化权重对比
ax = axes[0, 1]
ax.semilogy(t_range, wt_eps.numpy(), 'b-', linewidth=1.5, label=r'$w_t^{(\varepsilon)}$')
ax.semilogy(t_range, wt_x0.numpy(), 'r-', linewidth=1.5, label=r'$w_t^{(x_0)}$')
ax.semilogy(t_range, wt_score.numpy(), 'g-', linewidth=1.5, label=r'$w_t^{(s)}$')
ax.set_xlabel('时间步 t', fontsize=12)
ax.set_ylabel('权重 (对数尺度)', fontsize=12)
ax.set_title('(b) 三种参数化的VLB权重对比', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.annotate('x₀-pred权重峰值在中间t\nε/s-pred权重单调递减',
            xy=(0.35, 0.75), xycoords='axes fraction', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeaa7', alpha=0.8))

# (c) 前向过程后验均值系数（11.1节）
coeff_x0 = torch.sqrt(alpha_bars_prev) * betas / (1 - alpha_bars)
coeff_xt = torch.sqrt(alphas) * (1 - alpha_bars_prev) / (1 - alpha_bars)

ax = axes[1, 0]
ax.plot(t_range, coeff_x0.numpy(), 'r-', linewidth=2, label=r'$x_0$系数')
ax.plot(t_range, coeff_xt.numpy(), 'b-', linewidth=2, label=r'$x_t$系数')
ax.plot(t_range, (coeff_x0 + coeff_xt).numpy(), 'k--', linewidth=1, label='和')
ax.set_xlabel('时间步 t', fontsize=12)
ax.set_ylabel('系数值', fontsize=12)
ax.set_title('(c) 后验均值 μ̃_t 的系数（11.1节）', fontsize=13)
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

ax.semilogy(t_range, vlb_grad + 1e-30, 'b-', linewidth=2, label='VLB梯度 ∝ w_t')
ax.semilogy(t_range, simple_grad, 'r--', linewidth=2, label=r'$L_{simple}$梯度 ∝ 1')
ax.set_xlabel('时间步 t', fontsize=12)
ax.set_ylabel('有效梯度 (对数尺度)', fontsize=12)
ax.set_title('(d) VLB vs L_simple 有效梯度（★ 原创设计）', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.annotate('VLB: 小t梯度极大，大t梯度极小\n→ 训练不稳定',
            xy=(0.3, 0.7), xycoords='axes fraction', fontsize=9, color='blue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#dfe6e9', alpha=0.8))
ax.annotate('L_simple: 均匀梯度\n→ 训练稳定',
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

print("""
11.5节指出: ε-prediction的VLB与第6章DSM在结构上完全一致,
唯一区别是时间权重 w_t:

VLB (ε-prediction):  L = Σ_t w_t · E[||ε - ε̂_θ(x_t,t)||²]
DSM (第6章):          L = Σ_t λ(σ_t) · E[||s_θ(x_t,σ_t) + ε/σ_t||²]

利用 ε̂_θ = -√(1-ᾱ_t) · s_θ 的转换:
  ||ε - ε̂_θ||² = (1-ᾱ_t) · ||-ε/√(1-ᾱ_t) - s_θ||²
               = (1-ᾱ_t) · ||ε/√(1-ᾱ_t) + s_θ||²

因此:  VLB权重 w_t = (1-ᾱ_t) · λ(σ_t)/(σ_t²) (经适当变换)
       → 当 λ(σ_t) = σ_t² 时, VLB权重 = (1-ᾱ_t) → DSM简化目标
       → 当 λ(σ_t) = σ_t²/(1-ᾱ_t) · w_t 时, DSM = VLB (精确等价)
""")

# 数值验证: ε-prediction VLB 与 score-prediction 的损失关系
np.random.seed(42)
x0_v = torch.randn(100, 2)
eps_v = torch.randn_like(x0_v)

print(f"{'t':>5s}  {'||ε-ε̂||²':>12s}  {'(1-ᾱ_t)·||s+s*||²':>18s}  {'比值':>8s}")
print("-" * 50)

for t_idx in [10, 100, 500, 900]:
    t = t_idx
    ab = alpha_bars[t]
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

print(f"\n→ ||ε-ε̂||² ≈ (1-ᾱ_t)·||ŝ-s*||² 精确成立")
print(f"  这是11.5节Score≡VLB等价性的数值验证")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print("实验11.1 完成!")
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
   - VLB权重: ε-pred单调递减, x₀-pred峰值在中间, score-pred与ε-pred差1/(1-ᾱ_t)因子

4. VLB vs L_simple（11.4节）
   - VLB权重跨越多个数量级→训练不稳定
   - L_simple等权重1→训练稳定（11.4节Ho et al. 2020的关键简化）

5. VLB与DSM的结构等价性（11.5节预览）
   - ||ε-ε̂||² = (1-ᾱ_t)·||ŝ-s*||²
   - 这是第12章Score≡ELBO等价性的数值基础
""")
