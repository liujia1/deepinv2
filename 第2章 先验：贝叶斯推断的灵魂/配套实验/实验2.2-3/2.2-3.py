"""
实验2.2-3 经典先验族对比：高斯、Laplace与TV
对应章节：2.2 经典先验族
知识点：高斯先验→Tikhonov；Laplace先验→LASSO；TV先验→ROF模型；三种先验的解形态对比

素材来源：
  - M1 CompImLab25.ipynb: Tikhonov闭式解 + LASSO软阈值
  - 2.2章节: TV先验与ROF模型
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
    _chinese_path = os.path.join(_gdrive, '实验2.2-3', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.2-3')
    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
    print("请确保 .chinese 文件夹已上传到 Google Drive 的正确位置")

np.random.seed(42)

n = 128
x = resize(data.camera(), (n, n))

noise_lev = 0.1
y = x + noise_lev * np.random.randn(n, n)

def soft_thresh(x, l):
    """软阈值函数：Laplace先验的MAP解"""
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - l, np.zeros_like(x)))

def tv_denoise_chambolle(y, lam, n_iter=100):
    """
    Chambolle算法求解TV去噪（ROF模型）
    
    min_x 0.5||x-y||^2 + lam*TV(x)
    
    TV(x) = sum_i sqrt((Dh*x)_i^2 + (Dv*x)_i^2)
    """
    h, w = y.shape
    p_h = np.zeros((h, w))
    p_v = np.zeros((h, w))
    
    tau = 0.25
    
    for _ in range(n_iter):
        div_p = np.roll(p_h, 1, axis=1) + np.roll(p_v, 1, axis=0) - p_h - p_v
        
        u = y - lam * div_p
        
        grad_h = np.roll(u, -1, axis=1) - u
        grad_v = np.roll(u, -1, axis=0) - u
        
        denom = 1 + tau * np.sqrt(grad_h**2 + grad_v**2)
        p_h = (p_h + tau * grad_h) / denom
        p_v = (p_v + tau * grad_v) / denom
    
    div_p = np.roll(p_h, 1, axis=1) + np.roll(p_v, 1, axis=0) - p_h - p_v
    return y - lam * div_p

sigma = noise_lev
sigma_x = 1.0
b_laplace = 0.5
lam_tikh = sigma**2 / sigma_x**2
lam_lasso = sigma**2 / b_laplace
lam_tv = 0.15

x_tikh = y / (1 + lam_tikh)
x_lasso = soft_thresh(y, lam_lasso)
x_tv = tv_denoise_chambolle(y, lam_tv, n_iter=200)

psnr_noisy = peak_signal_noise_ratio(x, y)
psnr_tikh = peak_signal_noise_ratio(x, x_tikh)
psnr_lasso = peak_signal_noise_ratio(x, x_lasso)
psnr_tv = peak_signal_noise_ratio(x, x_tv)

print("===== 经典先验族对比 =====")
print(f"噪声水平 σ = {sigma:.4f}")
print(f"\n高斯先验 (假设: 值小):")
print(f"  正则项: ||x||_2^2 → Tikhonov")
print(f"  λ = σ²/σ_x² = {lam_tikh:.4f}")
print(f"  PSNR = {psnr_tikh:.2f} dB")
print(f"\nLaplace先验 (假设: 值稀疏):")
print(f"  正则项: ||x||_1 → LASSO")
print(f"  λ = σ²/b = {lam_lasso:.4f}")
print(f"  PSNR = {psnr_lasso:.2f} dB")
print(f"\nTV先验 (假设: 梯度稀疏):")
print(f"  正则项: ||∇x||_1 → ROF模型")
print(f"  λ = {lam_tv:.4f}")
print(f"  PSNR = {psnr_tv:.2f} dB")

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

axes[0, 0].imshow(x, cmap='gray')
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(y, cmap='gray')
axes[0, 1].set_title(f'含噪图像\nPSNR={psnr_noisy:.2f}dB')
axes[0, 1].axis('off')

axes[0, 2].imshow(x_tikh, cmap='gray')
axes[0, 2].set_title(f'Tikhonov (高斯先验)\n假设: 值小\nPSNR={psnr_tikh:.2f}dB')
axes[0, 2].axis('off')

axes[1, 0].imshow(x_lasso, cmap='gray')
axes[1, 0].set_title(f'LASSO (Laplace先验)\n假设: 值稀疏\nPSNR={psnr_lasso:.2f}dB')
axes[1, 0].axis('off')

axes[1, 1].imshow(x_tv, cmap='gray')
axes[1, 1].set_title(f'TV去噪 (TV先验)\n假设: 梯度稀疏\nPSNR={psnr_tv:.2f}dB')
axes[1, 1].axis('off')

center = n // 2
axes[1, 2].plot(x[center, :], 'k--', linewidth=1.5, label='真实')
axes[1, 2].plot(x_tikh[center, :], 'b-', linewidth=1, label='Tikhonov')
axes[1, 2].plot(x_lasso[center, :], 'g-', linewidth=1, label='LASSO')
axes[1, 2].plot(x_tv[center, :], 'r-', linewidth=1, label='TV')
axes[1, 2].set_title('中心行剖面对比')
axes[1, 2].legend()
axes[1, 2].set_xlabel('像素索引')

plt.suptitle('经典先验族对比：不同假设→不同正则项→不同解形态', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_三种先验对比.png'), dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

t = np.linspace(-2, 2, 400)
axes[0].plot(t, t**2, 'b-', linewidth=2, label='L2: t² (高斯先验)')
axes[0].plot(t, np.abs(t), 'g-', linewidth=2, label='L1: |t| (Laplace先验)')
axes[0].set_title('正则项形态对比')
axes[0].legend()
axes[0].set_xlabel('t')
axes[0].set_ylabel('惩罚值')
axes[0].set_ylim(-0.2, 4)
axes[0].grid(True, alpha=0.3)

lambdas = np.logspace(-4, 1, 30)
psnr_tikh_list = []
psnr_lasso_list = []
psnr_tv_list = []

for lam in lambdas:
    psnr_tikh_list.append(peak_signal_noise_ratio(x, y / (1 + lam)))
    psnr_lasso_list.append(peak_signal_noise_ratio(x, soft_thresh(y, lam)))
    psnr_tv_list.append(peak_signal_noise_ratio(x, tv_denoise_chambolle(y, lam, n_iter=100)))

axes[1].semilogx(lambdas, psnr_tikh_list, 'b-o', markersize=3, label='Tikhonov')
axes[1].semilogx(lambdas, psnr_lasso_list, 'g-s', markersize=3, label='LASSO')
axes[1].semilogx(lambdas, psnr_tv_list, 'r-^', markersize=3, label='TV')
axes[1].axhline(y=psnr_noisy, color='k', linestyle='--', alpha=0.5, label='含噪')
axes[1].set_xlabel('λ')
axes[1].set_ylabel('PSNR (dB)')
axes[1].set_title('不同先验的PSNR-λ曲线')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

res_tikh = np.abs(x_tikh - x)
res_lasso = np.abs(x_lasso - x)
res_tv = np.abs(x_tv - x)

axes[2].bar(['Tikhonov\n(高斯)', 'LASSO\n(Laplace)', 'TV\n(梯度稀疏)'], 
            [np.mean(res_tikh**2), np.mean(res_lasso**2), np.mean(res_tv**2)],
            color=['blue', 'green', 'red'], alpha=0.7)
axes[2].set_ylabel('MSE')
axes[2].set_title('重建误差对比')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_正则项形态与性能对比.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n===== 三种先验的特点总结 =====")
print("\n高斯先验 (Tikhonov):")
print("  - 假设: 图像值小")
print("  - 优点: 有闭式解，计算高效")
print("  - 缺点: 过度平滑，丢失边缘")
print("\nLaplace先验 (LASSO):")
print("  - 假设: 图像值稀疏")
print("  - 优点: 促稀疏，保留显著特征")
print("  - 缺点: 需要稀疏表示前提，直接用于图像效果有限")
print("\nTV先验 (ROF模型):")
print("  - 假设: 图像梯度稀疏")
print("  - 优点: 同时平滑与保边")
print("  - 缺点: 可能产生阶梯效应")
