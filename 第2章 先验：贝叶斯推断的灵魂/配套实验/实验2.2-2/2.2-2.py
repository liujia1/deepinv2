"""
实验2.2-2 高斯先验的推广：从Tikhonov到Sobolev平滑
对应章节：2.2 经典先验族
知识点：高斯先验→Tikhonov；梯度高斯先验→Sobolev平滑；闭式解；不同协方差结构的影响

素材来源：
  - IP22 (statistical_perspective.md): 高斯随机场采样 + MAP估计
  - 2.2章节: 高斯先验的推广形式
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio
import os
import sys

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.2-2', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.2-2')
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
sys.path.insert(0, _chinese_path)
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=_chinese_path)

np.random.seed(42)

n = 64
x = resize(data.camera(), (n, n))

noise_lev = 0.1
y = x + noise_lev * np.random.randn(n, n)

def tikhonov_denoise(y, lam):
    """
    简单高斯先验: x ~ N(0, σ_x² I)
    正则项: ||x||_2²
    闭式解: x = y / (1 + λ), λ = σ²/σ_x²
    """
    return y / (1 + lam)

def sobolev_denoise(y, lam, n_iter=100):
    """
    梯度高斯先验: ∇x ~ N(0, σ_∇² I)
    正则项: ||∇x||_2²
    使用梯度下降求解
    """
    x = y.copy()
    dt = 0.1
    
    for _ in range(n_iter):
        lap = (np.roll(x, 1, axis=0) + np.roll(x, -1, axis=0) +
               np.roll(x, 1, axis=1) + np.roll(x, -1, axis=1) - 4 * x)
        x = x + dt * ((y - x) + lam * lap)
    
    return x

def gaussian_field_denoise(y, L, alpha, n_iter=50):
    """
    高斯随机场先验: 协方差矩阵由相关长度L控制
    L越大，信号越平滑
    使用迭代方法求解: (αI + Σ)u = Σy
    """
    h, w = y.shape
    u = y.copy()
    
    for _ in range(n_iter):
        lap = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
               np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u)
        u = u + 0.1 * (y - u + alpha * L**2 * lap)
    
    return u

sigma = noise_lev
sigma_x = 1.0
lam_tikh = sigma**2 / sigma_x**2

x_tikh = tikhonov_denoise(y, lam_tikh)

lam_sob = 0.1
x_sob = sobolev_denoise(y, lam_sob, n_iter=200)

L_values = [0.05, 0.1, 0.2]
x_gf = []
for L in L_values:
    x_gf.append(gaussian_field_denoise(y, L, alpha=0.1, n_iter=100))

psnr_noisy = peak_signal_noise_ratio(x, y)
psnr_tikh = peak_signal_noise_ratio(x, x_tikh)
psnr_sob = peak_signal_noise_ratio(x, x_sob)
psnr_gf = [peak_signal_noise_ratio(x, xg) for xg in x_gf]

print("===== 高斯先验的推广形式 =====")
print(f"噪声水平 σ = {sigma:.4f}")
print(f"\n简单高斯先验 (假设: 值小):")
print(f"  正则项: ||x||_2² → Tikhonov")
print(f"  λ = σ²/σ_x² = {lam_tikh:.4f}")
print(f"  PSNR = {psnr_tikh:.2f} dB")
print(f"\n梯度高斯先验 (假设: 相邻像素相似):")
print(f"  正则项: ||∇x||_2² → Sobolev平滑")
print(f"  λ = {lam_sob:.4f}")
print(f"  PSNR = {psnr_sob:.2f} dB")
print(f"\n高斯随机场先验 (相关长度L控制光滑度):")
for i, L in enumerate(L_values):
    print(f"  L = {L}: PSNR = {psnr_gf[i]:.2f} dB")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(x, cmap='gray')
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(y, cmap='gray')
axes[0, 1].set_title(f'含噪图像\nPSNR={psnr_noisy:.2f}dB')
axes[0, 1].axis('off')

axes[0, 2].imshow(x_tikh, cmap='gray')
axes[0, 2].set_title(f'Tikhonov (简单高斯先验)\n假设: 值小\nPSNR={psnr_tikh:.2f}dB')
axes[0, 2].axis('off')

axes[0, 3].imshow(x_sob, cmap='gray')
axes[0, 3].set_title(f'Sobolev (梯度高斯先验)\n假设: 相邻像素相似\nPSNR={psnr_sob:.2f}dB')
axes[0, 3].axis('off')

for i, (L, xg) in enumerate(zip(L_values, x_gf)):
    axes[1, i].imshow(xg, cmap='gray')
    axes[1, i].set_title(f'高斯随机场 (L={L})\nPSNR={psnr_gf[i]:.2f}dB')
    axes[1, i].axis('off')

center = n // 2
axes[1, 3].plot(x[center, :], 'k--', linewidth=1.5, label='真实')
axes[1, 3].plot(y[center, :], 'r-', linewidth=0.5, alpha=0.5, label='含噪')
axes[1, 3].plot(x_tikh[center, :], 'b-', linewidth=1, label='Tikhonov')
axes[1, 3].plot(x_sob[center, :], 'g-', linewidth=1, label='Sobolev')
axes[1, 3].set_title('中心行剖面对比')
axes[1, 3].legend(fontsize=8)
axes[1, 3].set_xlabel('像素索引')

plt.suptitle('高斯先验的推广：不同协方差结构→不同正则化效果', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_高斯先验推广对比.png'), dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

methods = ['Tikhonov\n(简单高斯)', 'Sobolev\n(梯度高斯)'] + [f'高斯随机场\nL={L}' for L in L_values]
psnrs = [psnr_tikh, psnr_sob] + psnr_gf
colors = ['blue', 'green', 'red', 'orange', 'purple']

axes[0].bar(methods, psnrs, color=colors[:len(methods)], alpha=0.7)
axes[0].axhline(y=psnr_noisy, color='black', linestyle='--', label=f'含噪: {psnr_noisy:.2f}dB')
axes[0].set_ylabel('PSNR (dB)')
axes[0].set_title('不同高斯先验的PSNR对比')
axes[0].legend()

t = np.linspace(-2, 2, 400)
axes[1].plot(t, t**2, 'b-', linewidth=2, label='L2: t² (高斯先验正则项)')
axes[1].set_title('高斯先验对应的正则项形态\n(与Laplace先验对比)')
axes[1].legend()
axes[1].set_xlabel('t')
axes[1].set_ylabel('惩罚值')
axes[1].set_ylim(-0.2, 4)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_性能与正则项对比.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n===== 高斯先验的特点总结 =====")
print("\n简单高斯先验 (Tikhonov):")
print("  - 假设: 图像值小")
print("  - 协方差: σ_x² I (对角矩阵)")
print("  - 优点: 有闭式解，计算高效")
print("  - 缺点: 过度平滑，丢失边缘")
print("\n梯度高斯先验 (Sobolev平滑):")
print("  - 假设: 相邻像素相似")
print("  - 协方差: 涉及梯度算子")
print("  - 优点: 编码空间结构，比简单高斯更合理")
print("  - 缺点: 仍会过度平滑边缘")
print("\n高斯随机场先验:")
print("  - 假设: 由相关长度L控制光滑度")
print("  - 协方差: 非对角矩阵，相邻点相关")
print("  - L越大，信号越平滑")
