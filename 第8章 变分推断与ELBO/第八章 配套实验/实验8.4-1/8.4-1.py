# -*- coding: utf-8 -*-
"""
实验8.4-1 前向KL vs 逆向KL：零强迫与零避免
对应章节：8.4（变分推断与正则化的统一视角）
素材来源：🆕 新写
★ 原创设计：前向KL vs 逆向KL的行为对比

知识点：
  - 前向KL(p||q)：零强迫（zero-forcing / mean-seeking），覆盖所有模态
  - 逆向KL(q||p)：零避免（zero-avoiding / mode-seeking），聚焦单个模态
  - 变分推断使用逆向KL的原因
  - KL方向选择对近似质量的影响

实验内容：
  步骤1：前向KL vs 逆向KL的数值对比
  步骤2：最优变分分布的可视化

运行前提：纯NumPy/SciPy CPU即可
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys
import io
from scipy.optimize import minimize

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
    SAVE_DIR = os.path.join(_gdrive, '实验8.4-1')
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

np.random.seed(42)


# ============================================================
# 模型设定：直接构造一个不对称双峰后验
# 注意：使用不对称双峰，使逆向KL能锁定较重的峰，而前向KL覆盖两峰
# ============================================================
# 后验分布：p(z) = 0.35*N(-2.5, 0.6²) + 0.65*N(2.0, 0.6²)
# 左峰权重35%，右峰权重65%，两峰相距4.5（约7.5σ），明显分离
post_w = [0.35, 0.65]
post_m = [-2.5, 2.0]
post_s = [0.6, 0.6]

print("=" * 60)
print("实验8.4-1：前向KL vs 逆向KL")
print("=" * 60)
print(f"\n真实后验 p(z):")
for i in range(len(post_w)):
    print(f"  分量{i+1}: w={post_w[i]:.2f}, μ={post_m[i]:.2f}, σ={post_s[i]:.2f}")


# ============================================================
# 辅助函数
# ============================================================

def gm_pdf(x, weights, means, stds):
    """高斯混合分布的PDF"""
    pdf = np.zeros_like(x)
    for w, m, s in zip(weights, means, stds):
        pdf += w * np.exp(-0.5 * ((x - m) / s)**2) / (s * np.sqrt(2 * np.pi))
    return pdf


# ============================================================
# 步骤1：前向KL vs 逆向KL的数值对比
# ★ 原创设计
# ============================================================
print("\n" + "=" * 60)
print("步骤1：前向KL vs 逆向KL的数值对比")
print("=" * 60)

print("\n[核心概念]")
print("  前向KL(p||q) = E_p[log(p/q)]：零强迫（zero-forcing / mean-seeking）")
print("    → q必须在p>0的地方也有概率质量")
print("    → 覆盖所有模态，但可能过度分散")
print("  逆向KL(q||p) = E_q[log(q/p)]：零避免（zero-avoiding / mode-seeking）")
print("    → q在p≈0的地方必须也≈0，否则KL→∞")
print("    → 聚焦单个模态，但可能欠估计不确定性")

# 定义网格
z_grid = np.linspace(-6, 6, 1000)
dz = z_grid[1] - z_grid[0]

# 真实后验
posterior_pdf = gm_pdf(z_grid, post_w, post_m, post_s)

# 定义KL散度计算函数
def kl_forward(p, q, dz):
    """前向KL(p||q)"""
    mask = p > 1e-10
    return np.sum(p[mask] * np.log(p[mask] / (q[mask] + 1e-10))) * dz

def kl_reverse(q, p, dz):
    """逆向KL(q||p)"""
    mask = q > 1e-10
    return np.sum(q[mask] * np.log(q[mask] / (p[mask] + 1e-10))) * dz

# 定义单高斯变分分布
def gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

# 优化目标函数
def objective_forward(params):
    """最小化前向KL(p||q)"""
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    q = gaussian_pdf(z_grid, mu, sigma)
    return kl_forward(posterior_pdf, q, dz)

def objective_reverse(params):
    """最小化逆向KL(q||p)"""
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    q = gaussian_pdf(z_grid, mu, sigma)
    return kl_reverse(q, posterior_pdf, dz)

# 优化前向KL
print("\n优化前向KL(p||q)...")
result_forward = minimize(objective_forward, [0.0, 0.0], method='Nelder-Mead', 
                         options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
mu_forward = result_forward.x[0]
sigma_forward = np.exp(result_forward.x[1])
kl_forward_val = result_forward.fun

print(f"  最优参数: μ = {mu_forward:.4f}, σ = {sigma_forward:.4f}")
print(f"  前向KL(p||q) = {kl_forward_val:.6f}")

# 优化逆向KL - 从每个峰附近初始化，找到锁定单峰的局部最优解
print("\n优化逆向KL(q||p)...")
print("  从每个峰附近初始化，寻找锁定单峰的局部最优解...")

# 从每个峰附近用小sigma初始化
mode_lock_results = []
for i, (mode_mu, mode_sigma) in enumerate(zip(post_m, post_s)):
    init = [mode_mu, np.log(mode_sigma * 0.8)]
    result = minimize(objective_reverse, init, method='Nelder-Mead', 
                     options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
    mu_temp = result.x[0]
    sigma_temp = np.exp(result.x[1])
    kl_temp = result.fun
    mode_lock_results.append((mu_temp, sigma_temp, kl_temp))
    print(f"  从峰{i+1}初始化 (μ={mode_mu:.1f}): μ={mu_temp:.4f}, σ={sigma_temp:.4f}, KL={kl_temp:.6f}")

# 选择KL最小的锁定单峰解
mode_lock_result = min(mode_lock_results, key=lambda x: x[2])
mu_mode_lock, sigma_mode_lock, kl_mode_lock = mode_lock_result

# 也从中心初始化，找到全局最优（覆盖型）
result_center = minimize(objective_reverse, [0.0, np.log(1.5)], method='Nelder-Mead', 
                        options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
mu_center = result_center.x[0]
sigma_center = np.exp(result_center.x[1])
kl_center = result_center.fun
print(f"  从中心初始化 (μ=0.0): μ={mu_center:.4f}, σ={sigma_center:.4f}, KL={kl_center:.6f}")

# 全局最优
if kl_center < kl_mode_lock:
    mu_reverse, sigma_reverse, kl_reverse_val = mu_center, sigma_center, kl_center
    reverse_type = "覆盖型（居中）"
else:
    mu_reverse, sigma_reverse, kl_reverse_val = mu_mode_lock, sigma_mode_lock, kl_mode_lock
    reverse_type = "锁定单峰"

print(f"\n[对比结果]")
print(f"  前向KL最优: μ = {mu_forward:.4f}, σ = {sigma_forward:.4f}")
print(f"  逆向KL最优 ({reverse_type}): μ = {mu_reverse:.4f}, σ = {sigma_reverse:.4f}")
print(f"  逆向KL锁定单峰: μ = {mu_mode_lock:.4f}, σ = {sigma_mode_lock:.4f}")
print(f"  差异分析:")
print(f"    - 前向KL: σ={sigma_forward:.4f}（较大，尝试覆盖两个模态）")
print(f"    - 逆向KL: σ={sigma_reverse:.4f}（较小，聚焦单个模态）")


# ============================================================
# 步骤2：最优变分分布的可视化
# ★ 原创设计
# ============================================================
print("\n" + "=" * 60)
print("步骤2：最优变分分布的可视化")
print("=" * 60)

# 计算最优变分分布
q_forward = gaussian_pdf(z_grid, mu_forward, sigma_forward)
q_reverse = gaussian_pdf(z_grid, mu_mode_lock, sigma_mode_lock)  # 使用锁定单峰的解

# 创建可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：前向KL最优（覆盖两峰）
axes[0].fill_between(z_grid, posterior_pdf, alpha=0.3, color='blue', label=r'True posterior $p(z)$')
axes[0].plot(z_grid, posterior_pdf, 'b-', lw=2)
axes[0].fill_between(z_grid, q_forward, alpha=0.3, color='orange', label=r'Forward KL optimal $q$')
axes[0].plot(z_grid, q_forward, 'orange', lw=2, linestyle='--')
axes[0].set_xlabel(r'$z$')
axes[0].set_ylabel(r'Density')
axes[0].set_title('Forward KL(p||q): Zero-forcing')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_xlim([-6, 6])

# 右图：逆向KL最优（锁定单峰）
axes[1].fill_between(z_grid, posterior_pdf, alpha=0.3, color='blue', label=r'True posterior $p(z)$')
axes[1].plot(z_grid, posterior_pdf, 'b-', lw=2)
axes[1].fill_between(z_grid, q_reverse, alpha=0.3, color='red', label=r'Reverse KL optimal $q$')
axes[1].plot(z_grid, q_reverse, color='red', lw=2, linestyle='--')
axes[1].set_xlabel(r'$z$')
axes[1].set_ylabel(r'Density')
axes[1].set_title('Reverse KL(q||p): Zero-avoiding')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_xlim([-6, 6])

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_前向KL_vs_逆向KL.png'), dpi=100)
plt.close()
print(f"\n图表已保存: 步骤2_前向KL_vs_逆向KL.png")

print(f"\n[可视化分析]")
print(f"  左图（前向KL）:")
print(f"    - q尝试覆盖p的两个模态")
print(f"    - σ={sigma_forward:.4f}，分布较宽")
print(f"    - 在p≈0的地方q也有概率质量（零强迫）")
print(f"  右图（逆向KL）:")
print(f"    - q锁定在p的较重模态（右峰）")
print(f"    - σ={sigma_mode_lock:.4f}，分布较窄")
print(f"    - 在p≈0的地方q也≈0（零避免）")

print(f"\n[核心结论]")
print(f"  1. 变分推断使用逆向KL(q||p)的原因:")
print(f"     - 根本原因：前向KL(p||q)=E_p[log p/q]需要对真实后验p求期望，")
print(f"       而p本身往往不可解析（这正是8.1节讲过的核心困难），无法采样无法估计；")
print(f"       逆向KL(q||p)=E_q[log q/p]只需要对q求期望，q是可控的变分分布，")
print(f"       可以直接采样、用重参数化技巧高效估计——所以实践中只能用逆向KL")
print(f"     - 附带效果：逆向KL容易产生更锐利的近似（模式锁定），")
print(f"       但可能欠估计不确定性（zero-forcing）")
print(f"  2. 前向KL(p||q)的特点:")
print(f"     - 零强迫/均值寻优：覆盖所有模态，但可能过度分散")
print(f"     - 实践中难以直接优化，但在期望传播(EP)等场景中扮演理论角色")
print(f"  3. KL方向的选择是变分推断的核心权衡之一")

print(f"\n{'='*60}")
print("第八章配套实验8.4-1完成！")
