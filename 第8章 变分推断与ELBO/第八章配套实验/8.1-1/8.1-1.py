# -*- coding: utf-8 -*-
"""
实验8.1-1 为什么需要变分推断？——从采样到优化的范式转换
对应章节：8.1（为什么需要变分推断？）
素材来源：🆕 新写（教学用简化案例）
★ 原创设计：对比MCMC、Laplace、MAP与变分推断的计算代价与近似质量

知识点：
  - 采样路径的局限：MCMC/Langevin迭代计算代价高
  - 真实后验不可解的普遍性：高维积分障碍
  - 变分推断核心思想：从采样到优化，用近似换效率
  - 已有策略对比：MCMC（精确但慢）、Laplace近似（快但仅单峰）、MAP（最快但无分布信息）

实验内容：
  步骤1：后验不可解演示——归一化常数Z的计算困难
  步骤2：四种策略对比——MCMC、Laplace、MAP、变分推断
  步骤3：计算代价 vs 近似质量可视化

运行前提：纯NumPy/SciPy CPU即可
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys
import io
import time

# 设置控制台输出为 UTF-8 (Windows 下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验8.1-1')
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


# ============================================================
# 模型设定：1D高斯混合后验
# 模拟一个双峰后验分布 p(z|x) ∝ p(x|z)·p(z)
# p(z) = 0.4*N(-2, 0.5²) + 0.6*N(2, 0.5²)  （先验）
# p(x|z) = N(x; z, 0.3²)                    （似然）
# ============================================================

# 先验参数
prior_weights = [0.4, 0.6]
prior_means = [-2.0, 2.0]
prior_stds = [0.5, 0.5]

# 似然参数
sigma_obs = 0.3
x_obs = 0.0  # 观测值

print("=" * 60)
print("实验8.1-1：为什么需要变分推断？")
print("=" * 60)
print(f"\n模型设定：")
print(f"  先验 p(z) = {prior_weights[0]}*N({prior_means[0]}, {prior_stds[0]}²) + {prior_weights[1]}*N({prior_means[1]}, {prior_stds[1]}²)")
print(f"  似然 p(x|z) = N(x; z, {sigma_obs}²)")
print(f"  观测值 x = {x_obs}")


# ============================================================
# 解析计算真实后验 p(z|x)（用于评估近似质量）
# ============================================================

def compute_posterior(x, weights, means, stds, sigma_obs):
    """解析计算高斯混合先验+高斯似然的后验"""
    post_weights = []
    post_means = []
    post_stds = []

    for w, mu, tau in zip(weights, means, stds):
        # 后验分量参数
        s2 = 1.0 / (1.0 / sigma_obs**2 + 1.0 / tau**2)
        s = np.sqrt(s2)
        m = s2 * (x / sigma_obs**2 + mu / tau**2)

        # 后验分量权重 ∝ w * N(x; mu, sigma_obs²+tau²)
        marginal_std = np.sqrt(sigma_obs**2 + tau**2)
        marginal_log = -0.5 * ((x - mu) / marginal_std)**2 - np.log(marginal_std)

        post_weights.append(w * np.exp(marginal_log))
        post_means.append(m)
        post_stds.append(s)

    # 归一化权重
    total = sum(post_weights)
    post_weights = [pw / total for pw in post_weights]

    return post_weights, post_means, post_stds

post_w, post_m, post_s = compute_posterior(x_obs, prior_weights, prior_means, prior_stds, sigma_obs)

print(f"\n真实后验 p(z|x={x_obs})（解析解）:")
for i in range(len(post_w)):
    print(f"  分量{i+1}: w={post_w[i]:.4f}, μ={post_m[i]:.4f}, σ={post_s[i]:.4f}")


# ============================================================
# 步骤1：后验不可解演示——归一化常数Z的计算困难
# ★ 原创设计：展示高维积分的不可解性
# ============================================================
print("\n" + "=" * 60)
print("步骤1：后验不可解演示——归一化常数Z的计算困难")
print("=" * 60)

print("\n[核心困难]")
print("  后验分布 p(x|y) = p(y|x)·p(x) / Z")
print("  其中 Z = ∫p(y|x)·p(x)dx 是高维积分")
print("  对于复杂先验（如深度学习先验），Z 不可解析计算")

# 1D情况：Z可计算
print("\n[1D情况：Z可计算]")
# Z = p(x) = Σ_k w_k · N(x; μ_k, σ_obs² + τ_k²)
z_1d = 0.0
for w, mu, tau in zip(prior_weights, prior_means, prior_stds):
    marginal_std = np.sqrt(sigma_obs**2 + tau**2)
    z_1d += w * np.exp(-0.5 * ((x_obs - mu) / marginal_std)**2) / (np.sqrt(2 * np.pi) * marginal_std)
print(f"  1D高斯混合: Z = {z_1d:.6f}（解析可算）")

# 高维情况：维数诅咒
print("\n[高维情况：维数诅咒]")
print("  假设 d 维独立高斯混合，Z 需要 d 维积分")
print("  数值积分需要 O(N^d) 个网格点，N=10, d=100 时:")
print(f"    网格点数: 10^100 ≈ 10^100（宇宙原子数约 10^80）")
print("  → 高维积分在计算上不可行")

# 蒙特卡罗估计Z（重要性采样）
print("\n[蒙特卡罗估计Z]")
n_mc = 1000000
proposal_std = 3.0
z_samples = np.random.randn(n_mc) * proposal_std  # 从提议分布 q(z)=N(0, proposal_std²) 采样
# 计算 log p(x,z)
log_pz = np.full(n_mc, -1e30)
for w, mu, tau in zip(prior_weights, prior_means, prior_stds):
    log_comp = np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(tau) - 0.5 * ((z_samples - mu) / tau)**2
    log_pz = np.logaddexp(log_pz, log_comp)
log_pxz = -0.5 * np.log(2 * np.pi) - np.log(sigma_obs) - 0.5 * ((x_obs - z_samples) / sigma_obs)**2
log_joint = log_pxz + log_pz
# 重要性采样：Z = E_q[p(x,z)/q(z)]，需要除以提议分布密度 q(z)
log_q = -0.5 * np.log(2 * np.pi) - np.log(proposal_std) - 0.5 * (z_samples / proposal_std)**2
log_weights = log_joint - log_q  # log[p(x,z)/q(z)]
z_mc = np.mean(np.exp(log_weights - np.max(log_weights))) * np.exp(np.max(log_weights))
print(f"  蒙特卡罗估计 (N={n_mc}, 重要性采样): Z ≈ {z_mc:.6f}")
print(f"  与解析值比较: 误差 = {abs(z_mc - z_1d) / z_1d * 100:.2f}%")
print("  → 重要性采样可以准确估计Z，但高维时提议分布难以匹配目标分布")


# ============================================================
# 步骤2：四种策略对比——MCMC、Laplace、MAP、变分推断
# ★ 原创设计：直观展示各策略的计算代价与近似质量
# ============================================================
print("\n" + "=" * 60)
print("步骤2：四种策略对比")
print("=" * 60)

# 定义未归一化后验（用于MCMC和评估）
def unnormalized_posterior(z):
    """未归一化的后验 p̃(z|x) ∝ p(x|z)·p(z)"""
    # 似然
    log_pxz = -0.5 * np.log(2 * np.pi) - np.log(sigma_obs) - 0.5 * ((x_obs - z) / sigma_obs)**2
    # 先验（混合高斯）
    log_pz = np.full_like(z, -1e30)
    for w, mu, tau in zip(prior_weights, prior_means, prior_stds):
        log_comp = np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(tau) - 0.5 * ((z - mu) / tau)**2
        log_pz = np.logaddexp(log_pz, log_comp)
    return np.exp(log_pxz + log_pz)

def log_unnormalized_posterior(z):
    """未归一化后验的对数"""
    log_pxz = -0.5 * np.log(2 * np.pi) - np.log(sigma_obs) - 0.5 * ((x_obs - z) / sigma_obs)**2
    log_pz = np.full_like(z, -1e30)
    for w, mu, tau in zip(prior_weights, prior_means, prior_stds):
        log_comp = np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(tau) - 0.5 * ((z - mu) / tau)**2
        log_pz = np.logaddexp(log_pz, log_comp)
    return log_pxz + log_pz

def grad_log_unnormalized_posterior(z):
    """未归一化后验的梯度（用于Langevin动力学）"""
    # 数值梯度
    eps = 1e-5
    return (log_unnormalized_posterior(z + eps) - log_unnormalized_posterior(z - eps)) / (2 * eps)

# 真实后验PDF（用于评估）
def true_posterior_pdf(z):
    """真实后验的PDF"""
    unnorm = unnormalized_posterior(z)
    return unnorm / z_1d

# 计算KL散度（近似后验 vs 真实后验）
def compute_kl(q_samples, log_q, true_pdf_func, n_bins=100):
    """近似计算KL(q||p)"""
    z_min, z_max = -5, 5
    z_grid = np.linspace(z_min, z_max, n_bins)
    dz = z_grid[1] - z_grid[0]
    
    # 真实后验
    p_vals = true_pdf_func(z_grid)
    p_vals = np.maximum(p_vals, 1e-10)  # 避免log(0)
    
    # 近似后验（直方图估计）
    q_hist, _ = np.histogram(q_samples, bins=n_bins, range=(z_min, z_max), density=True)
    q_vals = np.maximum(q_hist, 1e-10)
    
    # KL散度
    kl = np.sum(q_vals * np.log(q_vals / p_vals) * dz)
    return kl

# ========================================
# 策略1：MCMC采样（Metropolis-Hastings）
# ========================================
print("\n[策略1] MCMC采样（Metropolis-Hastings）")
t_start = time.time()

n_mcmc = 10000  # 采样数
burn_in = 2000  # 预热期
proposal_std = 0.3

z_current = 0.0
samples = []
n_accept = 0

for i in range(n_mcmc + burn_in):
    # 提议分布
    z_proposal = z_current + np.random.randn() * proposal_std
    
    # 接受率
    log_alpha = log_unnormalized_posterior(z_proposal) - log_unnormalized_posterior(z_current)
    
    if np.log(np.random.rand()) < log_alpha:
        z_current = z_proposal
        n_accept += 1
    
    if i >= burn_in:
        samples.append(z_current)

t_mcmc = time.time() - t_start
mcmc_samples = np.array(samples)

print(f"  采样数: {n_mcmc}, 预热期: {burn_in}")
print(f"  接受率: {n_accept / (n_mcmc + burn_in):.3f}")
print(f"  计算时间: {t_mcmc:.4f}秒")
print(f"  样本均值: {np.mean(mcmc_samples):.4f}, 样本标准差: {np.std(mcmc_samples):.4f}")

# ========================================
# 策略2：Laplace近似（单高斯近似）
# ========================================
print("\n[策略2] Laplace近似（单高斯近似后验）")
t_start = time.time()

# 找到后验众数（MAP）
from scipy.optimize import minimize
neg_log_post = lambda z: -log_unnormalized_posterior(z)
result = minimize(neg_log_post, x0=0.0, method='Nelder-Mead')
z_map = result.x[0]

# 在众数处计算Hessian（二阶导数）
eps = 1e-4
hessian = (log_unnormalized_posterior(z_map + eps) - 2 * log_unnormalized_posterior(z_map) + 
           log_unnormalized_posterior(z_map - eps)) / (eps**2)
sigma_laplace = np.sqrt(-1.0 / hessian)

t_laplace = time.time() - t_start

print(f"  后验众数 (MAP): z = {z_map:.4f}")
print(f"  Laplace近似: q(z) = N({z_map:.4f}, {sigma_laplace:.4f}²)")
print(f"  计算时间: {t_laplace:.6f}秒")
print(f"  注意: Laplace只能捕获单峰，无法表示双峰后验")

# ========================================
# 策略3：MAP估计（点估计）
# ========================================
print("\n[策略3] MAP估计（点估计）")
t_start = time.time()

# MAP独立优化：最大化后验（与Laplace共享众数，但独立计时）
neg_log_post_map = lambda z: -log_unnormalized_posterior(z)
result_map = minimize(neg_log_post_map, x0=0.0, method='Nelder-Mead')
z_map_estimate = result_map.x[0]

t_map = time.time() - t_start

print(f"  MAP估计: z = {z_map_estimate:.4f}")
print(f"  计算时间: {t_map:.6f}秒")
print(f"  注意: MAP只给出点估计，完全丢弃分布信息")

# ========================================
# 策略4：变分推断（优化方法）
# ========================================
print("\n[策略4] 变分推断（优化方法）")
t_start = time.time()

# 使用单高斯变分族 q(z) = N(μ, σ²)
# 目标：最小化 KL(q||p(z|x))
# 等价于：最大化 ELBO = E_q[log p(x,z)] - E_q[log q(z)]

# 使用公共随机数（common random numbers）消除蒙特卡罗噪声
# 固定一组ε ~ N(0,1)，通过重参数化 z = μ + σ·ε 评估不同(μ,σ)
n_vi_samples = 10000
vi_eps = np.random.randn(n_vi_samples)  # 固定的基础噪声

def compute_elbo_single_gaussian(mu, sigma):
    """计算单高斯变分族的ELBO（使用公共随机数）"""
    # 重参数化：z = μ + σ·ε，ε固定
    z_samples = mu + sigma * vi_eps
    
    # log p(x,z)
    log_pxz = -0.5 * np.log(2 * np.pi) - np.log(sigma_obs) - 0.5 * ((x_obs - z_samples) / sigma_obs)**2
    log_pz = np.full(n_vi_samples, -1e30)
    for w, mu_prior, tau in zip(prior_weights, prior_means, prior_stds):
        log_comp = np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(tau) - 0.5 * ((z_samples - mu_prior) / tau)**2
        log_pz = np.logaddexp(log_pz, log_comp)
    log_joint = log_pxz + log_pz
    
    # log q(z)
    log_qz = -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((z_samples - mu) / sigma)**2
    
    # ELBO
    elbo = np.mean(log_joint - log_qz)
    return elbo

# 优化ELBO
from scipy.optimize import minimize
neg_elbo = lambda params: -compute_elbo_single_gaussian(params[0], np.exp(params[1]))
result = minimize(neg_elbo, x0=[0.0, 0.0], method='Nelder-Mead')
mu_vi = result.x[0]
sigma_vi = np.exp(result.x[1])

t_vi = time.time() - t_start

print(f"  变分族: q(z) = N(μ, σ²)")
print(f"  最优参数: μ = {mu_vi:.4f}, σ = {sigma_vi:.4f}")
print(f"  最优ELBO: {-result.fun:.4f}")
print(f"  计算时间: {t_vi:.6f}秒")
print(f"  注意: 变分推断用优化替代采样，速度快但受限于变分族表达能力")

# ========================================
# 对比总结
# ========================================
print("\n" + "=" * 60)
print("策略对比总结")
print("=" * 60)

# 计算各策略的KL散度
kl_mcmc = compute_kl(mcmc_samples, None, true_posterior_pdf)
kl_laplace = compute_kl(np.random.randn(10000) * sigma_laplace + z_map, None, true_posterior_pdf)
kl_vi = compute_kl(np.random.randn(10000) * sigma_vi + mu_vi, None, true_posterior_pdf)

print(f"\n{'策略':<20s} | {'计算时间(秒)':<15s} | {'KL散度':<12s} | {'分布信息':<10s}")
print("-" * 70)
print(f"{'MCMC采样':<20s} | {t_mcmc:<15.4f} | {kl_mcmc:<12.4f} | {'✓ 完整':<10s}")
print(f"{'Laplace近似':<20s} | {t_laplace:<15.6f} | {kl_laplace:<12.4f} | {'✓ 单峰':<10s}")
print(f"{'MAP估计':<20s} | {t_map:<15.6f} | {'N/A':<12s} | {'✗ 无':<10s}")
print(f"{'变分推断':<20s} | {t_vi:<15.6f} | {kl_vi:<12.4f} | {'✓ 近似':<10s}")

print(f"\n核心结论：")
print(f"  1. MCMC最精确（KL最小），但计算代价最高（迭代采样）")
print(f"  2. Laplace近似快速，但只能捕获单峰（无法表示双峰后验）")
print(f"  3. MAP最快，但完全丢弃分布信息（只有点估计）")
print(f"  4. 变分推断在速度和精度间取得平衡（优化方法，快速且有分布信息）")
print(f"  5. 这就是'用近似换效率'的核心思想——变分推断的动机")


# ============================================================
# 步骤3：计算代价 vs 近似质量可视化
# ★ 原创设计：直观展示各策略的权衡
# ============================================================
print("\n" + "=" * 60)
print("步骤3：计算代价 vs 近似质量可视化")
print("=" * 60)

# 创建可视化
fig = plt.figure(figsize=(16, 10))

# 子图1：后验分布与各策略的近似
ax1 = plt.subplot(2, 2, 1)
z_grid = np.linspace(-5, 5, 1000)
true_pdf = true_posterior_pdf(z_grid)

ax1.plot(z_grid, true_pdf, 'k-', lw=2, label='True posterior $p(z|x)$')
ax1.hist(mcmc_samples, bins=50, density=True, alpha=0.5, color='blue', label='MCMC samples')
ax1.plot(z_grid, 1/(np.sqrt(2*np.pi)*sigma_laplace) * np.exp(-0.5*((z_grid-z_map)/sigma_laplace)**2), 
        'r--', lw=2, label='Laplace approx')
ax1.plot(z_grid, 1/(np.sqrt(2*np.pi)*sigma_vi) * np.exp(-0.5*((z_grid-mu_vi)/sigma_vi)**2), 
        'g--', lw=2, label='VI approx')
ax1.axvline(z_map_estimate, color='purple', linestyle=':', lw=2, label='MAP estimate')
ax1.set_xlabel('$z$')
ax1.set_ylabel('$p(z|x)$')
ax1.set_title('Posterior and Approximations')
ax1.legend()
ax1.grid(alpha=0.3)

# 子图2：计算代价对比（对数尺度）
ax2 = plt.subplot(2, 2, 2)
strategies = ['MCMC', 'Laplace', 'MAP', 'VI']
times = [t_mcmc, t_laplace, t_map, t_vi]
colors = ['blue', 'red', 'purple', 'green']

ax2.bar(strategies, times, color=colors, alpha=0.7)
ax2.set_ylabel('Computation Time (seconds)')
ax2.set_title('Computation Cost Comparison')
ax2.set_yscale('log')
ax2.grid(alpha=0.3, axis='y')

# 添加数值标签
for i, (strategy, t) in enumerate(zip(strategies, times)):
    ax2.text(i, t * 1.2, f'{t:.4f}s', ha='center', va='bottom', fontsize=9)

# 子图3：近似质量对比（KL散度）
ax3 = plt.subplot(2, 2, 3)
kl_values = [kl_mcmc, kl_laplace, kl_vi]
kl_strategies = ['MCMC', 'Laplace', 'VI']
kl_colors = ['blue', 'red', 'green']

bars = ax3.bar(kl_strategies, kl_values, color=kl_colors, alpha=0.7)
ax3.set_ylabel('KL Divergence $KL(q \| p(z|x))$')
ax3.set_title('Approximation Quality (lower is better)')
ax3.grid(alpha=0.3, axis='y')

# 添加数值标签
for i, (strategy, kl) in enumerate(zip(kl_strategies, kl_values)):
    ax3.text(i, kl + 0.01, f'{kl:.4f}', ha='center', va='bottom', fontsize=9)

# 子图4：计算代价 vs 近似质量（权衡曲线）
ax4 = plt.subplot(2, 2, 4)
kl_times = [t_mcmc, t_laplace, t_vi]
ax4.scatter(kl_times, [kl_mcmc, kl_laplace, kl_vi], 
           c=['blue', 'red', 'green'], s=100, alpha=0.7)
for i, strategy in enumerate(['MCMC', 'Laplace', 'VI']):
    ax4.annotate(strategy, (kl_times[i], [kl_mcmc, kl_laplace, kl_vi][i]), 
                textcoords="offset points", xytext=(10, 5), fontsize=10)

ax4.set_xlabel('Computation Time (seconds)')
ax4.set_ylabel('KL Divergence (lower is better)')
ax4.set_title('Trade-off: Speed vs Accuracy')
ax4.set_xscale('log')
ax4.grid(alpha=0.3)

# 添加说明文字
ax4.text(0.02, 0.02,
        'Lower-left: Fast + Accurate (Ideal)\n'
        'Upper-right: Slow + Inaccurate (Worst)',
        transform=ax4.transAxes, fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_计算代价vs近似质量.png'), dpi=100)
plt.close()
print(f"\n图表已保存: 步骤3_计算代价vs近似质量.png")

print(f"\n可视化说明：")
print(f"  子图1：真实后验（双峰）与各策略的近似")
print(f"    - MCMC：渐近精确，但需要大量迭代")
print(f"    - Laplace：只能捕获单峰（红色虚线），丢失了另一个峰")
print(f"    - VI：单高斯近似（绿色虚线），快速但有偏")
print(f"    - MAP：点估计（紫色点线），无分布信息")
print(f"  子图2：计算代价对比（对数尺度）")
print(f"    - MCMC最慢（迭代采样）")
print(f"    - Laplace/MAP/VI都很快（优化方法）")
print(f"  子图3：近似质量对比（KL散度）")
print(f"    - MCMC最精确（KL最小）")
print(f"    - Laplace和VI有近似误差")
print(f"  子图4：计算代价 vs 近似质量的权衡")
print(f"    - 理想：左下角（快且准）")
print(f"    - MCMC：右下角（慢但准）")
print(f"    - VI：左中（快且较准）——最佳权衡")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验8.1-1 总结")
print("=" * 60)
print("1. 后验不可解：高维积分Z在计算上不可行")
print("2. 四种策略对比：")
print("   - MCMC：精确但慢（迭代采样）")
print("   - Laplace：快但仅单峰（高斯近似）")
print("   - MAP：最快但无分布信息（点估计）")
print("   - 变分推断：快且有分布信息（优化方法）")
print("3. 核心洞见：变分推断 = 从采样到优化的范式转换")
print("   用近似换效率，在速度和精度间取得平衡")
print("4. 两条路径的分野：")
print("   - 采样路径（Part II）：追求精确，计算慢")
print("   - 变分路径（Part III）：追求高效，计算快")
print("5. 下一章预告：ELBO是变分推断的优化目标")

print(f"\n{'='*60}")
print("第八章配套实验8.1-1完成！")
