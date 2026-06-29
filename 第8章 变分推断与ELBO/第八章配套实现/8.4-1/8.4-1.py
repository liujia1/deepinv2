# -*- coding: utf-8 -*-
"""
实验8.4-1 前向KL vs 逆向KL：零强迫与零避免
对应章节：8.4（变分推断与正则化的统一视角）
素材来源：🆕 新写
★ 原创设计：前向KL vs 逆向KL的行为对比

知识点：
  - 前向KL(p||q)：零强迫（mean-seeking），覆盖所有模态
  - 逆向KL(q||p)：零避免（mode-seeking），聚焦单个模态
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
# ========================================================

np.random.seed(42)


# ============================================================
# 模型设定：1D高斯混合 + 高斯观测（与8.3-1相同）
# ============================================================
prior_weights = [0.3, 0.7]
prior_means = [-2.0, 1.0]
prior_stds = [1.0, 1.0]
sigma_obs = 0.5
x_obs = 0.5

print("=" * 60)
print("实验8.4-1：前向KL vs 逆向KL")
print("=" * 60)
print(f"\n模型设定：")
print(f"  先验 p(z) = {prior_weights[0]}*N({prior_means[0]}, {prior_stds[0]}²) + {prior_weights[1]}*N({prior_means[1]}, {prior_stds[1]}²)")
print(f"  似然 p(x|z) = N(x; z, {sigma_obs}²)")
print(f"  观测值 x = {x_obs}")


# ============================================================
# 辅助函数：计算真实后验、边际似然、高斯混合PDF
# ============================================================

def compute_posterior(x, weights, means, stds, sigma_obs):
    """解析计算高斯混合先验+高斯似然的后验"""
    post_weights, post_means, post_stds = [], [], []
    for w, mu, tau in zip(weights, means, stds):
        s2 = 1.0 / (1.0 / sigma_obs**2 + 1.0 / tau**2)
        s = np.sqrt(s2)
        m = s2 * (x / sigma_obs**2 + mu / tau**2)
        ms = np.sqrt(sigma_obs**2 + tau**2)
        ml = -0.5 * ((x - mu) / ms)**2 - np.log(ms)
        post_weights.append(w * np.exp(ml))
        post_means.append(m)
        post_stds.append(s)
    total = sum(post_weights)
    post_weights = [pw / total for pw in post_weights]
    return post_weights, post_means, post_stds

post_w, post_m, post_s = compute_posterior(x_obs, prior_weights, prior_means, prior_stds, sigma_obs)

print(f"\n真实后验 p(z|x={x_obs}):")
for i in range(len(post_w)):
    print(f"  分量{i+1}: w={post_w[i]:.4f}, μ={post_m[i]:.4f}, σ={post_s[i]:.4f}")


def log_marginal(x, weights, means, stds, sigma_obs):
    """计算log p(x) = log Σₖ wₖ·N(x; μₖ, σ_obs²+τₖ²)"""
    terms = []
    for w, mu, tau in zip(weights, means, stds):
        ms = np.sqrt(sigma_obs**2 + tau**2)
        terms.append(np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(ms) - 0.5 * ((x - mu) / ms)**2)
    mx = max(terms)
    return mx + np.log(sum(np.exp(t - mx) for t in terms))

log_px = log_marginal(x_obs, prior_weights, prior_means, prior_stds, sigma_obs)
print(f"\nlog p(x={x_obs}) = {log_px:.6f}")


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
print("  前向KL(p||q) = E_p[log(p/q)]：零避免（mean-seeking）")
print("    → q必须在p>0的地方也有概率质量")
print("    → 覆盖所有模态，但可能过度分散")
print("  逆向KL(q||p) = E_q[log(q/p)]：零强迫（mode-seeking）")
print("    → q必须在p=0的地方也为0")
print("    → 聚焦单个模态，但可能欠估计方差")

# 定义网格
z_grid = np.linspace(-5, 5, 500)
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
                         options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
mu_forward = result_forward.x[0]
sigma_forward = np.exp(result_forward.x[1])
kl_forward_val = result_forward.fun

print(f"  最优参数: μ = {mu_forward:.4f}, σ = {sigma_forward:.4f}")
print(f"  前向KL(p||q) = {kl_forward_val:.6f}")

# 优化逆向KL
print("\n优化逆向KL(q||p)...")
result_reverse = minimize(objective_reverse, [0.0, 0.0], method='Nelder-Mead', 
                         options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
mu_reverse = result_reverse.x[0]
sigma_reverse = np.exp(result_reverse.x[1])
kl_reverse_val = result_reverse.fun

print(f"  最优参数: μ = {mu_reverse:.4f}, σ = {sigma_reverse:.4f}")
print(f"  逆向KL(q||p) = {kl_reverse_val:.6f}")

print(f"\n[对比结果]")
print(f"  前向KL最优: μ = {mu_forward:.4f}, σ = {sigma_forward:.4f}")
print(f"  逆向KL最优: μ = {mu_reverse:.4f}, σ = {sigma_reverse:.4f}")
print(f"  差异分析:")
print(f"    - 前向KL倾向于更大的σ（覆盖两个模态）")
print(f"    - 逆向KL倾向于更小的σ（聚焦单个模态）")


# ============================================================
# 步骤2：最优变分分布的可视化
# ★ 原创设计
# ============================================================
print("\n" + "=" * 60)
print("步骤2：最优变分分布的可视化")
print("=" * 60)

# 计算最优变分分布
q_forward = gaussian_pdf(z_grid, mu_forward, sigma_forward)
q_reverse = gaussian_pdf(z_grid, mu_reverse, sigma_reverse)

# 创建可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：前向KL最优
axes[0].fill_between(z_grid, posterior_pdf, alpha=0.3, color='blue', label=r'True posterior $p(z|x)$')
axes[0].plot(z_grid, posterior_pdf, 'b-', lw=2)
axes[0].fill_between(z_grid, q_forward, alpha=0.3, color='orange', label=r'Forward KL optimal $q$')
axes[0].plot(z_grid, q_forward, 'orange', lw=2, linestyle='--')
axes[0].set_xlabel(r'$z$')
axes[0].set_ylabel(r'Density')
axes[0].set_title('Forward KL(p||q): Zero-forcing')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_xlim([-5, 5])

# 右图：逆向KL最优
axes[1].fill_between(z_grid, posterior_pdf, alpha=0.3, color='blue', label=r'True posterior $p(z|x)$')
axes[1].plot(z_grid, posterior_pdf, 'b-', lw=2)
axes[1].fill_between(z_grid, q_reverse, alpha=0.3, color='red', label=r'Reverse KL optimal $q$')
axes[1].plot(z_grid, q_reverse, 'r-', lw=2, linestyle='--')
axes[1].set_xlabel(r'$z$')
axes[1].set_ylabel(r'Density')
axes[1].set_title('Reverse KL(q||p): Zero-avoiding')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_xlim([-5, 5])

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_前向KL_vs_逆向KL.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图表已保存: 步骤2_前向KL_vs_逆向KL.png")

print(f"\n[可视化分析]")
print(f"  左图（前向KL）:")
print(f"    - q尝试覆盖p的两个模态")
print(f"    - σ较大，分布较宽")
print(f"    - 在p=0的地方q也有概率质量（零强迫）")
print(f"  右图（逆向KL）:")
print(f"    - q聚焦在p的主要模态")
print(f"    - σ较小，分布较窄")
print(f"    - 在p=0的地方q也为0（零避免）")

print(f"\n[核心结论]")
print(f"  1. 变分推断使用逆向KL(q||p)的原因:")
print(f"     - 逆向KL更容易优化（基于q的期望）")
print(f"     - 产生更锐利的近似（聚焦主要模态）")
print(f"     - 但可能欠估计不确定性")
print(f"  2. 前向KL(p||q)的特点:")
print(f"     - 覆盖所有模态，但可能过度分散")
print(f"     - 在某些应用场景（如期望传播）中使用")
print(f"  3. KL方向的选择是变分推断的核心权衡之一")

print(f"\n{'='*60}")
print("第八章配套实验8.4-1完成！")
