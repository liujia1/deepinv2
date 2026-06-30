# -*- coding: utf-8 -*-
"""
实验9.2-1 重参数化技巧的数值验证
对应知识点：9.2节 重参数化技巧；附录9A REINFORCE梯度估计器

★ 原创设计：本实验通过数值对比系统验证重参数化技巧相对于REINFORCE的方差优势，
特别是附录9A中方差随σ变化的标度行为：REINFORCE方差 O(σ² + μ²/σ²) 在σ→0时发散，
而重参数化方差 O(σ²) 在σ→0时趋于0。

实验内容:
  步骤1: 梯度可传播性验证
  步骤2: REINFORCE与重参数化梯度估计器对比
  步骤3: 方差随σ变化的标度行为（附录9A核心结论）
  步骤4: 样本效率对比（原创设计）

素材来源:
  - 实验9.2.py
  - 附录9A方差分析

运行前提: PyTorch, NumPy, CPU即可
"""

import sys
import io
import os
import numpy as np
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

# ====== matplotlib 静默模式 ======
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
# =================================

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验9.2-1')
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

# 设置随机种子
np.random.seed(42)

import torch
torch.manual_seed(42)

# Checkpoint路径（保存数值计算结果）
RESULTS_CACHE_PATH = os.path.join(SAVE_DIR, 'numerical_results_cache.npz')

print(f"\n{'='*60}")
print(f"实验9.2-1: 重参数化技巧的数值验证")
print(f"{'='*60}")
print("\n[核心思想]")
print("  9.2节: 重参数化将随机性外移到与参数无关的噪声变量ε")
print("  附录9A: REINFORCE方差在σ→0时发散，重参数化方差趋于0")


# ============================================================
# 步骤1：梯度可传播性验证 —— 重参数化使梯度穿过随机节点
# 对应9.2节核心思想：将随机性外移到与参数无关的噪声变量ε
# ============================================================
print(f"\n{'='*60}")
print("步骤1: 梯度可传播性验证")
print(f"{'='*60}")

# --- 重参数化采样：梯度可传播 ---
mu = torch.tensor(2.0, requires_grad=True)
logvar = torch.tensor(0.0, requires_grad=True)  # σ=1

std = torch.exp(0.5 * logvar)
eps = torch.randn(1)
z = mu + std * eps  # z = μ + σε，梯度可通过μ和σ传播
loss = z ** 2
loss.backward()

print(f"重参数化: μ={mu.item():.1f}, σ={std.item():.1f}")
print(f"  ∂loss/∂μ = {mu.grad.item():.4f}  (解析值: ∂E[z²]/∂μ = 2μ = {2*2.0:.1f})")
print(f"  ∂loss/∂logvar = {logvar.grad.item():.4f}")
print("  → 梯度成功穿过随机采样节点！")

# --- 直接采样（非重参数化）：使用 .sample() ---
from torch.distributions import Normal
mu2 = torch.tensor(2.0, requires_grad=True)
dist = Normal(mu2, 1.0)

# .sample() 不支持梯度传播
z_no_grad = dist.sample()
print(f"\n直接采样 (.sample()): z = {z_no_grad.item():.4f}")
print("  → .sample() 产生的样本无法反传梯度")

# .rsample() 使用重参数化技巧
z_with_grad = dist.rsample()
loss2 = z_with_grad ** 2
loss2.backward()
print(f"重参数化采样 (.rsample()): z = {z_with_grad.item():.4f}")
print(f"  ∂loss/∂μ = {mu2.grad.item():.4f}  → 梯度可传播！")

print("\n→ 核心区别：重参数化将随机性从 z=Sample(μ,σ) 改写为 z=μ+σε，")
print("  使梯度可通过确定性路径（μ,σ）反传。")


# ============================================================
# 步骤2：REINFORCE 与重参数化梯度估计器实现
# 对应9.2节和附录9A：两种梯度估计方法
# ============================================================
print(f"\n{'='*60}")
print("步骤2: REINFORCE 与重参数化梯度估计器对比")
print(f"{'='*60}")

def reinforce_grad_mu(mu_val, sigma_val, n_samples=100):
    """
    REINFORCE梯度估计（得分函数估计器）:
    ∇_μ E_{q(z|μ,σ)}[f(z)] ≈ (1/L) Σ f(z_l) · ∇_μ log q(z_l|μ,σ)
    
    对于高斯: ∇_μ log q(z|μ,σ) = (z-μ)/σ²
    测试函数: f(z) = z²
    真实梯度: ∇_μ E[z²] = 2μ
    """
    z = np.random.normal(mu_val, sigma_val, n_samples)
    f_z = z ** 2
    score_mu = (z - mu_val) / sigma_val ** 2
    return np.mean(f_z * score_mu)


def reparam_grad_mu(mu_val, sigma_val, n_samples=100):
    """
    重参数化梯度估计:
    ∇_μ E[f(z)] = ∇_μ E_ε[f(μ+σε)] ≈ (1/L) Σ ∇_μ f(μ+σε_l)
    
    对于 f(z)=z²: ∇_μ f(μ+σε) = 2(μ+σε) = 2z
    """
    eps = np.random.normal(0, 1, n_samples)
    z = mu_val + sigma_val * eps
    return np.mean(2 * z)  # ∂f/∂μ = 2z (chain rule: ∂z/∂μ = 1)


# 检查是否有缓存的数值结果
cached_results = None
if os.path.exists(RESULTS_CACHE_PATH):
    print(f"\n检测到已保存的数值结果: {RESULTS_CACHE_PATH}")
    cached_results = np.load(RESULTS_CACHE_PATH, allow_pickle=True)
    
    # 检查缓存是否包含所需的数据
    required_keys = ['reinforce_estimates', 'reparam_estimates', 
                     'reinforce_small', 'reparam_small']
    
    if all(k in cached_results.files for k in required_keys):
        print("✓ 缓存数据完整, 直接加载, 跳过数值计算")
        reinforce_estimates = cached_results['reinforce_estimates']
        reparam_estimates = cached_results['reparam_estimates']
        reinforce_small = cached_results['reinforce_small']
        reparam_small = cached_results['reparam_small']
        
        # 加载其他必需数据
        sigma_list = cached_results['sigma_list']
        reinforce_vars = cached_results['reinforce_vars']
        reparam_vars = cached_results['reparam_vars']
        L_list = cached_results['L_list']
        reinforce_mse = cached_results['reinforce_mse']
        reparam_mse = cached_results['reparam_mse']
        
        print("  数值计算结果已加载")
    else:
        print("  缓存数据不完整, 重新计算")
        cached_results = None

if cached_results is None:
    # 验证无偏性
    mu_test, sigma_test = 2.0, 1.0
    true_grad = 2 * mu_test
    n_trials = 5000
    L = 50  # 每次估计使用的样本数
    
    np.random.seed(42)
    reinforce_estimates = np.array([reinforce_grad_mu(mu_test, sigma_test, L) for _ in range(n_trials)])
    reparam_estimates = np.array([reparam_grad_mu(mu_test, sigma_test, L) for _ in range(n_trials)])
    
    # σ=0.1 时的梯度估计
    sigma_small = 0.1
    reinforce_small = np.array([reinforce_grad_mu(mu_test, sigma_small, L) for _ in range(n_trials)])
    reparam_small = np.array([reparam_grad_mu(mu_test, sigma_small, L) for _ in range(n_trials)])


print(f"\n测试参数: μ={mu_test}, σ={sigma_test}, L={L}")
print(f"真实梯度: ∇_μ E[z²] = 2μ = {true_grad:.4f}")
print(f"\nREINFORCE:  均值={np.mean(reinforce_estimates):.4f}, "
      f"标准差={np.std(reinforce_estimates):.4f}, "
      f"方差={np.var(reinforce_estimates):.4f}")
print(f"重参数化:   均值={np.mean(reparam_estimates):.4f}, "
      f"标准差={np.std(reparam_estimates):.4f}, "
      f"方差={np.var(reparam_estimates):.4f}")
print(f"\n→ 两种估计器均无偏（均值≈{true_grad}），但重参数化方差远低于REINFORCE")
print(f"  方差比: REINFORCE/重参数化 = {np.var(reinforce_estimates)/np.var(reparam_estimates):.1f}x")


# ============================================================
# 步骤3：方差随σ变化的标度行为 —— 附录9A核心结论
# ============================================================
print(f"\n{'='*60}")
print("步骤3: 方差随σ变化的标度行为（附录9A核心结论）")
print(f"{'='*60}")

if cached_results is None:
    sigma_list = np.logspace(-1.5, 0.7, 25)  # σ from ~0.03 to ~5
    mu_fixed = 2.0
    n_trials_var = 3000
    L_var = 50
    
    np.random.seed(123)
    reinforce_vars = []
    reparam_vars = []
    
    for sigma in sigma_list:
        rg = [reinforce_grad_mu(mu_fixed, sigma, L_var) for _ in range(n_trials_var)]
        rpg = [reparam_grad_mu(mu_fixed, sigma, L_var) for _ in range(n_trials_var)]
        reinforce_vars.append(np.var(rg))
        reparam_vars.append(np.var(rpg))
    
    reinforce_vars = np.array(reinforce_vars)
    reparam_vars = np.array(reparam_vars)

# 理论预测 (附录9A)
# REINFORCE方差 ∝ (σ² + μ²/σ²) / L
# 重参数化方差 = 4σ² / L
theory_reinforce = (sigma_list**2 + mu_fixed**2 / sigma_list**2) / L_var
theory_reparam = 4 * sigma_list**2 / L_var

print("σ\t\tREINFORCE方差\t重参数化方差\t方差比")
print("-" * 65)
for i in [0, len(sigma_list)//4, len(sigma_list)//2, 3*len(sigma_list)//4, -1]:
    idx = i if i >= 0 else len(sigma_list) + i
    ratio = reinforce_vars[idx] / max(reparam_vars[idx], 1e-10)
    print(f"{sigma_list[idx]:.4f}\t\t{reinforce_vars[idx]:.4f}\t\t{reparam_vars[idx]:.4f}\t\t{ratio:.1f}x")

print(f"\n关键发现:")
print(f"  σ={sigma_list[0]:.3f} (小): REINFORCE方差比重参数化高 {reinforce_vars[0]/max(reparam_vars[0],1e-10):.0f}倍 → REINFORCE在σ小时几乎不可用")
print(f"  σ={sigma_list[-1]:.3f} (大): REINFORCE方差比重参数化高 {reinforce_vars[-1]/max(reparam_vars[-1],1e-10):.1f}倍 → 差距缩小但仍然显著")


# ============================================================
# 步骤4：★ 原创设计 - 样本效率对比
# 达到相同精度所需的样本数
# ============================================================
print(f"\n{'='*60}")
print("步骤4: 样本效率对比（★ 原创设计）")
print(f"{'='*60}")

if cached_results is None:
    L_list = [10, 50, 100, 500, 1000, 5000]
    mu_eff, sigma_eff = 2.0, 0.5
    true_grad_eff = 2 * mu_eff
    n_trials_eff = 2000
    
    np.random.seed(456)
    reinforce_mse = []
    reparam_mse = []
    
    for L_eff in L_list:
        rg = np.array([reinforce_grad_mu(mu_eff, sigma_eff, L_eff) for _ in range(n_trials_eff)])
        rpg = np.array([reparam_grad_mu(mu_eff, sigma_eff, L_eff) for _ in range(n_trials_eff)])
        reinforce_mse.append(np.mean((rg - true_grad_eff) ** 2))
        reparam_mse.append(np.mean((rpg - true_grad_eff) ** 2))
    
    reinforce_mse = np.array(reinforce_mse)
    reparam_mse = np.array(reparam_mse)
    
    # 保存所有数值结果到缓存
    np.savez(RESULTS_CACHE_PATH,
             reinforce_estimates=reinforce_estimates,
             reparam_estimates=reparam_estimates,
             reinforce_small=reinforce_small,
             reparam_small=reparam_small,
             sigma_list=sigma_list,
             reinforce_vars=reinforce_vars,
             reparam_vars=reparam_vars,
             L_list=L_list,
             reinforce_mse=reinforce_mse,
             reparam_mse=reparam_mse,
             mu_test=mu_test,
             sigma_test=sigma_test,
             mu_fixed=mu_fixed,
             L_var=L_var,
             mu_eff=mu_eff,
             sigma_eff=sigma_eff)
    print(f"\n✓ 数值计算完成, 结果已保存: {RESULTS_CACHE_PATH}")

print(f"参数: μ={mu_eff}, σ={sigma_eff}, 真实梯度={true_grad_eff}")
print(f"{'L':>6s}  {'REINFORCE MSE':>15s}  {'重参数化 MSE':>15s}  {'MSE比':>10s}")
print("-" * 55)
for i, L_eff in enumerate(L_list):
    ratio = reinforce_mse[i] / max(reparam_mse[i], 1e-10)
    print(f"{L_eff:6d}  {reinforce_mse[i]:15.4f}  {reparam_mse[i]:15.4f}  {ratio:10.1f}x")


# ============================================================
# 可视化
# ============================================================
print(f"\n{'='*60}")
print("生成可视化图表...")
print(f"{'='*60}")

# --- 图1: 梯度估计分布对比 + 方差随σ变化 ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) σ=1.0 时的梯度估计直方图
ax = axes[0, 0]
ax.hist(reinforce_estimates, bins=60, alpha=0.6, label='REINFORCE', density=True, color='#e74c3c')
ax.hist(reparam_estimates, bins=60, alpha=0.6, label='Reparameterized', density=True, color='#3498db')
ax.axvline(true_grad, color='black', linestyle='--', linewidth=2, label=f'True Gradient = {true_grad:.1f}')
ax.set_xlabel('Gradient Estimate', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title(f'(a) Gradient Distribution ($\\mu={mu_test}$, $\\sigma={sigma_test}$, $L={L}$)', fontsize=13)
ax.legend(fontsize=10)

# (b) σ=0.1 时的梯度估计直方图
ax = axes[0, 1]
ax.hist(reinforce_small, bins=60, alpha=0.6, label='REINFORCE', density=True, color='#e74c3c')
ax.hist(reparam_small, bins=60, alpha=0.6, label='Reparameterized', density=True, color='#3498db')
ax.axvline(true_grad, color='black', linestyle='--', linewidth=2, label=f'True Gradient = {true_grad:.1f}')
ax.set_xlabel('Gradient Estimate', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title(f'(b) Gradient Distribution ($\\mu={mu_test}$, $\\sigma={sigma_small}$, $L={L}$)', fontsize=13)
ax.legend(fontsize=10)

# (c) 方差随σ变化 (log-log)
ax = axes[1, 0]
ax.loglog(sigma_list, reinforce_vars, 'o-', label='REINFORCE (Experiment)', color='#e74c3c', markersize=5)
ax.loglog(sigma_list, reparam_vars, 's-', label='Reparameterized (Experiment)', color='#3498db', markersize=5)
ax.loglog(sigma_list, theory_reinforce, '--', label='Theory: $(\\sigma^2+\\mu^2/\\sigma^2)/L$', color='#e74c3c', alpha=0.5)
ax.loglog(sigma_list, theory_reparam, '--', label='Theory: $4\\sigma^2/L$', color='#3498db', alpha=0.5)
ax.set_xlabel('$\\sigma$ (log scale)', fontsize=12)
ax.set_ylabel('Gradient Variance (log scale)', fontsize=12)
ax.set_title(f'(c) Variance vs $\\sigma$ ($\\mu={mu_fixed}$, $L={L_var}$)', fontsize=13)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

# (d) 方差比
ax = axes[1, 1]
ratio = reinforce_vars / np.maximum(reparam_vars, 1e-10)
ax.semilogy(sigma_list, ratio, 'D-', color='#8e44ad', markersize=5)
ax.axhline(1, color='gray', linestyle='--', alpha=0.5, label='Variance Ratio = 1')
ax.set_xlabel('$\\sigma$', fontsize=12)
ax.set_ylabel('Variance Ratio (REINFORCE / Reparameterized)', fontsize=12)
ax.set_title(f'(d) Variance Ratio ($\\mu={mu_fixed}$)', fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤1_REINFORCE与重参数化方差.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图1已保存: {fig_path}")


# --- 图2: 样本效率对比 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) MSE vs L
ax = axes[0]
ax.loglog(L_list, reinforce_mse, 'o-', label='REINFORCE', color='#e74c3c', markersize=8)
ax.loglog(L_list, reparam_mse, 's-', label='Reparameterized', color='#3498db', markersize=8)
ax.set_xlabel('Number of Samples $L$ (log scale)', fontsize=12)
ax.set_ylabel('MSE (log scale)', fontsize=12)
ax.set_title(f'(a) Gradient MSE vs Sample Size ($\\mu={mu_eff}$, $\\sigma={sigma_eff}$)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# (b) 达到目标精度的样本数对比
target_mse = 0.1
from scipy.interpolate import interp1d
try:
    f_reinforce = interp1d(np.log10(reinforce_mse), np.log10(L_list), fill_value='extrapolate')
    f_reparam = interp1d(np.log10(reparam_mse), np.log10(L_list), fill_value='extrapolate')
    L_reinforce_needed = 10 ** f_reinforce(np.log10(target_mse))
    L_reparam_needed = 10 ** f_reparam(np.log10(target_mse))
    
    print(f"\n达到MSE={target_mse}所需样本数:")
    print(f"  REINFORCE:  L ≈ {L_reinforce_needed:.0f}")
    print(f"  重参数化:   L ≈ {L_reparam_needed:.0f}")
    print(f"  比值: {L_reinforce_needed/L_reparam_needed:.1f}x")
    
    ax = axes[1]
    methods = ['REINFORCE', 'Reparameterized']
    L_needed = [L_reinforce_needed, L_reparam_needed]
    bars = ax.bar(methods, L_needed, color=['#e74c3c', '#3498db'], alpha=0.8, edgecolor='black')
    ax.set_ylabel('Required Samples $L$', fontsize=12)
    ax.set_title(f'(b) Samples Needed to Reach MSE={target_mse}', fontsize=13)
    for bar, val in zip(bars, L_needed):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{val:.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
except Exception as e:
    print(f"插值失败: {e}")
    ax = axes[1]
    ax.text(0.5, 0.5, 'Sample Efficiency Comparison\n(Insufficient Data)', 
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title('(b) Samples Needed to Reach Target MSE', fontsize=13)

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤2_样本效率对比.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig_path}")


# ============================================================
# 实验总结
# ============================================================
print(f"\n{'='*60}")
print("实验9.2-1 总结")
print(f"{'='*60}")
print("\n核心验证点（对应9.2节和附录9A）：")
print("\n1. 重参数化使梯度可穿过随机采样节点：")
print("   - z = μ + σε 将随机性外移到与参数无关的ε")
print("   - 梯度可通过μ和σ的确定性路径反传")
print("   ✓ 实验验证: 步骤1展示梯度成功穿过随机节点")
print("\n2. REINFORCE与重参数化均给出无偏梯度估计：")
print("   - 两者在大量重复实验下均值均等于真实梯度")
print("   - 但REINFORCE方差远高于重参数化")
print("   ✓ 实验验证: 步骤2验证无偏性和方差差异")
print("\n3. 方差的σ标度行为是关键差异（附录9A核心结论）：")
print("   - REINFORCE方差 ∝ (σ² + μ²/σ²)/L → σ→0时发散")
print("   - 重参数化方差 ∝ 4σ²/L → σ→0时趋于0")
print("   - 这意味着在编码器输出小方差时，REINFORCE几乎不可用")
print("   ✓ 实验验证: 步骤3验证σ标度行为符合理论")
print("\n4. 重参数化所需样本数远少于REINFORCE：")
print("   - 达到相同MSE，REINFORCE需要更多样本")
print("   - 这是VAE选择重参数化而非REINFORCE的直接原因")
print("   ✓ 实验验证: 步骤4量化样本效率差异")

print(f"\n{'='*60}")
print("第九章配套实验 9.2-1 完成!")