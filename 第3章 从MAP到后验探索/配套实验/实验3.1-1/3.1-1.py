"""
实验3.1-1 MAP估计：后验能量分解
对应章节：3.1 MAP估计——从后验众数到优化问题
知识点：后验概率与优化目标的等价性；后验能量分解（数据项 + 正则项）；
        λ = sigma^2/sigma_x^2 的概率诠释；MAP解与闭式解

修改说明：
  从原参考实验3.1.py拆分，聚焦MAP估计与后验能量分解概念，
  去除梯度下降细节（移至3.2-1）和λ扫描（移至3.3-1）。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio
from scipy.ndimage import gaussian_filter
import os
import sys

# ====== 中文字体配置（兼容本地和Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验3.1-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(SAVE_DIR, exist_ok=True)
else:
    _chinese_path = '.chinese'
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)

# ══════════════════════════════════════════════════════════
# 1. 问题设定：小尺寸模糊逆问题
# ══════════════════════════════════════════════════════════
n = 32
from skimage.data import shepp_logan_phantom
phantom = resize(shepp_logan_phantom(), (n, n), order=0, preserve_range=True, anti_aliasing=False)
x = phantom / phantom.max()      # 原始图像，值域 [0,1]
N = n * n

# 高斯模糊核
def gaussian_psf(size, sigma):
    ax = np.concatenate((np.arange(0, size // 2), np.arange(-size // 2, 0)))
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return h / h.sum()

blur_sigma = 2.0
h = gaussian_psf(n, blur_sigma)
H_fft = np.fft.fft2(h)

# 构建模糊矩阵 A (N×N)
A = np.zeros((N, N))
for j in range(N):
    e_j = np.zeros(N)
    e_j[j] = 1.0
    A[:, j] = np.real(np.fft.ifft2(H_fft * np.fft.fft2(e_j.reshape(n, n)))).ravel()

sigma_noise = 0.05
y = A @ x.ravel() + sigma_noise * np.random.randn(N)

# ══════════════════════════════════════════════════════════
# 2. MAP估计：后验能量分解
# ══════════════════════════════════════════════════════════
# 高斯似然 + 高斯先验 → 后验为高斯分布
# 后验能量: -ln p(x|y) = 0.5||Ax-y||^2/sigma^2 + 0.5||x||^2/sigma_x^2 + const
# MAP估计 = 最小化 J(x) = 0.5||Ax-y||^2 + 0.5 lambda ||x||^2
# 其中 lambda = sigma^2/sigma_x^2

sigma_prior = 0.5
lam = sigma_noise ** 2 / sigma_prior ** 2
sigma_x = sigma_prior

AtA = A.T @ A
Aty = A.T @ y

# 闭式解（MAP估计）
Sigma_post = np.linalg.inv(AtA / sigma_noise ** 2 + np.eye(N) / sigma_prior ** 2)
mu_post = Sigma_post @ (Aty / sigma_noise ** 2)

# 后验方差（不确定性量化）
# [注意] 这仅在高斯-高斯共轭模型下可用（闭式后验协方差）
# [预告] 非高斯后验的不确定性量化见第4章（MCMC/ULA采样方法）
# [关联] 3.7节将讨论：为什么仅有MAP点估计是不够的
post_var = np.diag(Sigma_post).reshape(n, n)

# ══════════════════════════════════════════════════════════
# 3. 后验能量验证
# ══════════════════════════════════════════════════════════
def posterior_energy(x_vec, y, A, sigma_noise, sigma_prior):
    """计算后验能量 -ln p(x|y)（省略常数项）"""
    data_term = 0.5 / sigma_noise**2 * np.linalg.norm(A @ x_vec - y) ** 2
    reg_term = 0.5 / sigma_prior**2 * np.linalg.norm(x_vec) ** 2
    return data_term + reg_term, data_term, reg_term

# 计算三种情况下的后验能量
x_init = A.T @ y
x_map = mu_post

E_init, d_init, r_init = posterior_energy(x_init, y, A, sigma_noise, sigma_prior)
E_map, d_map, r_map = posterior_energy(x_map, y, A, sigma_noise, sigma_prior)
E_true, d_true, r_true = posterior_energy(x.ravel(), y, A, sigma_noise, sigma_prior)

psnr_map = peak_signal_noise_ratio(x, np.clip(x_map.reshape(n, n), 0, 1))
psnr_init = peak_signal_noise_ratio(x, np.clip(x_init.reshape(n, n), 0, 1))

print("=" * 60)
print("实验3.1-1 MAP估计：后验能量分解")
print("=" * 60)
print(f"\n[参数设定]")
print(f"  噪声标准差 sigma = {sigma_noise:.3f}")
print(f"  先验标准差 sigma_x = {sigma_prior:.3f}")
print(f"  正则化参数 lambda = sigma^2/sigma_x^2 = {lam:.4f}")
print(f"\n[后验能量分解 -ln p(x|y) = 数据项 + 正则项]")
print(f"  {'':>15} {'数据项':>12} {'正则项':>12} {'后验能量':>12} {'PSNR':>10}")
print(f"  {'初始点':>15} {d_init:>12.2f} {r_init:>12.2f} {E_init:>12.2f} {psnr_init:>10.2f}")
print(f"  {'MAP估计':>15} {d_map:>12.2f} {r_map:>12.2f} {E_map:>12.2f} {psnr_map:>10.2f}")
print(f"  {'真解':>15} {d_true:>12.2f} {r_true:>12.2f} {E_true:>12.2f} {'N/A':>10}")
print(f"\n  [验证] MAP解的后验能量 ({E_map:.2f}) < 初始点 ({E_init:.2f})，优化有效")
print(f"  [验证] MAP解PSNR = {psnr_map:.2f} dB，显著优于初始点 {psnr_init:.2f} dB")

# ══════════════════════════════════════════════════════════
# 4. 可视化
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 第一行：原始 → 观测 → MAP重建
axes[0, 0].imshow(x, cmap='gray')
axes[0, 0].set_title(r'原始图像 $x$')
axes[0, 0].axis('off')

axes[0, 1].imshow(y.reshape(n, n), cmap='gray')
axes[0, 1].set_title(r'模糊含噪观测 $y = Ax + \epsilon$')
axes[0, 1].axis('off')

axes[0, 2].imshow(np.clip(x_map.reshape(n, n), 0, 1), cmap='gray')
axes[0, 2].set_title(r'MAP估计 $\hat{x}_{\mathrm{MAP}}$' + f'\nPSNR={psnr_map:.2f}dB')
axes[0, 2].axis('off')

# 第二行：后验不确定性 → 能量分解 → 核心公式
im = axes[1, 0].imshow(post_var, cmap='hot')
axes[1, 0].set_title(r'后验方差 $\mathrm{diag}(\Sigma_{\mathrm{post}})$\n不确定性量化')
axes[1, 0].axis('off')
plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

# 能量分解柱状图
methods = ['初始点', 'MAP估计', '真解']
data_terms = [d_init, d_map, d_true]
reg_terms = [r_init, r_map, r_true]
x_pos = np.arange(len(methods))
width = 0.35
axes[1, 1].bar(x_pos - width/2, data_terms, width, label='数据项', color='steelblue', alpha=0.8)
axes[1, 1].bar(x_pos + width/2, reg_terms, width, label='正则项', color='seagreen', alpha=0.8)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(methods)
axes[1, 1].set_title(r'后验能量分解: $-\ln p(x|y) = $ 数据项 $+$ 正则项')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# 核心公式
formula_text = (
    r'$\hat{x}_{\mathrm{MAP}} = \arg\max_x p(x|y)$'
    '\n'
    r'$= \arg\min_x [ -\ln p(x|y) ]$'
    '\n\n'
    r'$-\ln p(x|y) = \frac{1}{2\sigma^2}\|Ax-y\|^2$'
    '\n'
    r'$\qquad\qquad + \frac{1}{2\sigma_x^2}\|x\|^2 + \mathrm{const}$'
    '\n\n'
    r'$\lambda = \sigma^2 / \sigma_x^2$ (正则化参数)'
)
axes[1, 2].text(0.5, 0.5, formula_text, fontsize=12, ha='center', va='center',
                transform=axes[1, 2].transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
axes[1, 2].set_title('MAP核心公式')
axes[1, 2].axis('off')

plt.suptitle('实验3.1-1: MAP估计——从后验众数到优化问题', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_MAP后验能量分解.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print("1. MAP估计 = 最大化后验 = 最小化 -ln p(x|y)")
print("2. -ln p(x|y) 可分解为: 数据项 + 正则项 + 常数")
print("3. 正则化参数 lambda = sigma^2/sigma_x^2 有明确的概率意义")
print("4. 后验方差反映 MAP 解的不确定性")
print("   [注意] 这仅在高斯-高斯共轭模型下可用（闭式后验协方差）")
print("   [预告] 非高斯后验的不确定性量化见第4章（MCMC/ULA采样方法）")
print("   [关联] 3.7节将讨论：为什么仅有MAP点估计是不够的")
print("5. 在共轭高斯-高斯模型下，MAP解有闭式形式")
print("\n实验完成。结果已保存至:", SAVE_DIR)