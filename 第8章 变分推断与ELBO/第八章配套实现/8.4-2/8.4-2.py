# -*- coding: utf-8 -*-
"""
实验8.4-2 ELBO = 重建 + 正则：变分推断与正则化的统一
对应章节：8.4（变分推断与正则化的统一视角）
素材来源：🆕 新写
★ 原创设计：ELBO分解为重建项与KL正则项，与MAP估计的统一

知识点：
  - ELBO = 重建项 - KL正则项
  - 重建项：E_q[log p(x|z)]，衡量数据拟合质量
  - KL正则项：KL(q||p(z))，惩罚q偏离先验
  - 与MAP估计的数据项+正则项结构对应
  - 确定性正则化是变分推断的退化形式

实验内容：
  步骤1：ELBO分解为重建项与KL正则项
  步骤2：重建-正则权衡的可视化

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
    SAVE_DIR = os.path.join(_gdrive, '实验8.4-2')
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
# 模型设定：1D高斯混合 + 高斯观测
# ============================================================
prior_weights = [0.3, 0.7]
prior_means = [-2.0, 1.0]
prior_stds = [1.0, 1.0]
sigma_obs = 0.5
x_obs = 0.5

print("=" * 60)
print("实验8.4-2：ELBO = 重建 + 正则")
print("=" * 60)
print(f"\n模型设定：")
print(f"  先验 p(z) = {prior_weights[0]}*N({prior_means[0]}, {prior_stds[0]}²) + {prior_weights[1]}*N({prior_means[1]}, {prior_stds[1]}²)")
print(f"  似然 p(x|z) = N(x; z, {sigma_obs}²)")
print(f"  观测值 x = {x_obs}")


# ============================================================
# 辅助函数：计算log p(x)
# ============================================================

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


# ============================================================
# 步骤1：ELBO分解为重建项与KL正则项
# ★ 原创设计
# ============================================================
print("\n" + "=" * 60)
print("步骤1：ELBO分解为重建项与KL正则项")
print("=" * 60)

print("\n[核心概念]")
print("  ELBO = E_q[log p(x|z)] - KL(q||p(z))")
print("       = 重建项 - 正则项")
print("  重建项：衡量q编码的数据信息")
print("  KL正则项：惩罚q偏离先验p(z)")

# 定义计算重建项和KL项的函数
def compute_reconstruction_and_kl(mu_q, sigma_q, n_samples=20000):
    """计算ELBO的两个组成部分"""
    np.random.seed(42)
    z = np.random.randn(n_samples) * sigma_q + mu_q
    
    # 重建项：E_q[log p(x|z)]
    reconstruction = np.mean(-0.5 * np.log(2 * np.pi) - np.log(sigma_obs) - 0.5 * ((x_obs - z) / sigma_obs)**2)
    
    # KL项：KL(q||p(z)) = E_q[log q(z) - log p(z)]
    log_pz = np.full_like(z, -1e30)
    for w, mu, tau in zip(prior_weights, prior_means, prior_stds):
        log_comp = np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(tau) - 0.5 * ((z - mu) / tau)**2
        log_pz = np.logaddexp(log_pz, log_comp)
    log_qz = -0.5 * np.log(2 * np.pi) - np.log(sigma_q) - 0.5 * ((z - mu_q) / sigma_q)**2
    kl = np.mean(log_qz - log_pz)
    
    return reconstruction, kl

# 扫描不同的q参数，观察重建项和KL项的变化
mu_range = np.linspace(-3, 3, 100)
sigma_range = [0.3, 0.5, 1.0, 2.0]

print(f"\n扫描不同的q=N(μ,σ²)，观察重建项与KL项的变化：")
print(f"{'μ_q':>8s} | {'σ_q':>6s} | {'重建项':>10s} | {'KL项':>10s} | {'ELBO':>10s}")
print("-" * 60)

for sigma_q in [0.5, 1.0, 2.0]:
    for mu_q in [-2.0, 0.0, 0.5, 1.0]:
        recon, kl = compute_reconstruction_and_kl(mu_q, sigma_q)
        elbo = recon - kl
        print(f"{mu_q:>8.2f} | {sigma_q:>6.2f} | {recon:>10.4f} | {kl:>10.4f} | {elbo:>10.4f}")

print(f"\n[关键观察]")
print(f"  1. 当μ_q接近x_obs={x_obs}时，重建项最大")
print(f"  2. 当σ_q较小时，KL项较小（q接近先验）")
print(f"  3. ELBO最大值 = 重建项与KL项的权衡")


# ============================================================
# 步骤2：重建-正则权衡的可视化
# ★ 原创设计
# ============================================================
print("\n" + "=" * 60)
print("步骤2：重建-正则权衡的可视化")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：重建项随q均值的变化
for sigma_q in sigma_range:
    reconstruction_terms = []
    for mu_q in mu_range:
        recon, _ = compute_reconstruction_and_kl(mu_q, sigma_q)
        reconstruction_terms.append(recon)
    axes[0].plot(mu_range, reconstruction_terms, label=r'$\sigma_q=' + f'{sigma_q}$')

axes[0].set_xlabel(r'$\mu_q$')
axes[0].set_ylabel(r'$\mathbb{E}_q[\log p(x|z)]$')
axes[0].set_title('Reconstruction term vs $q$ mean')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 右图：ELBO分解（σ_q=1.0）
sigma_q = 1.0
elbos_scan = []
reconstruction_terms = []
kl_terms = []

for mu_q in mu_range:
    recon, kl = compute_reconstruction_and_kl(mu_q, sigma_q)
    reconstruction_terms.append(recon)
    kl_terms.append(kl)
    elbos_scan.append(recon - kl)

axes[1].plot(mu_range, elbos_scan, 'b-', lw=2, label=r'$\mathrm{ELBO} = \mathrm{Recon} - \mathrm{KL}$')
axes[1].plot(mu_range, reconstruction_terms, 'r--', label=r'Reconstruction $\mathbb{E}_q[\log p(x|z)]$')
axes[1].plot(mu_range, [-k for k in kl_terms], 'g--', label=r'$-\mathrm{KL}(q\|p)$')
axes[1].axhline(y=log_px, color='k', linestyle=':', lw=1, label=r'$\log p(x) = ' + f'{log_px:.2f}$')

axes[1].set_xlabel(r'$\mu_q$')
axes[1].set_ylabel(r'Value')
axes[1].set_title(r'ELBO decomposition ($\sigma_q=1.0$)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_重建与正则.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图表已保存: 步骤2_重建与正则.png")

print(f"\n[可视化分析]")
print(f"  左图：重建项随q均值的变化")
print(f"    - 当μ_q接近x_obs时，重建项最大")
print(f"    - σ_q越大，重建项越平滑（对μ_q不敏感）")
print(f"  右图：ELBO分解（σ_q=1.0）")
print(f"    - ELBO = 重建项 - KL项")
print(f"    - 重建项驱动q拟合数据")
print(f"    - KL项惩罚q偏离先验")
print(f"    - ELBO最大值 = 两者权衡的最优点")

print(f"\n[与MAP估计的统一]")
print(f"  MAP估计：min_x [-log p(y|x) + λR(x)]")
print(f"           = min_x [数据项 + 正则项]")
print(f"  变分推断：max_q [E_q[log p(x|z)] - KL(q||p(z))]")
print(f"           = max_q [重建项 - 正则项]")
print(f"  两者具有完全相同的结构！")
print(f"  关键差异：")
print(f"    - MAP是点估计（优化x）")
print(f"    - 变分推断是分布估计（优化q）")
print(f"    - 当q退化为δ函数时，ELBO退化为MAP目标")

print(f"\n[核心结论]")
print(f"  1. ELBO = 重建项 - KL正则项")
print(f"  2. 重建项驱动数据拟合，KL项约束先验偏离")
print(f"  3. 与MAP估计的数据项+正则项结构完全对应")
print(f"  4. 确定性正则化是变分推断的退化形式")

print(f"\n{'='*60}")
print("第八章配套实验8.4-2完成！")
