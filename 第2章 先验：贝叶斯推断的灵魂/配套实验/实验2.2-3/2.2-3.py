"""
实验2.2-3 TV先验的局限：阶梯效应
对应章节：2.2 经典先验族 - TV先验
知识点：TV保边但产生阶梯效应；TV偏好分段常数；与高斯先验的对比

素材来源：
  - 2.5.py: TV去噪与阶梯效应展示
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio
import os
import sys

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.2-3', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.2-3')
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
sys.path.insert(0, _chinese_path)
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=_chinese_path)

np.random.seed(42)

n = 128
camera = resize(data.camera(), (n, n))

sigma = 0.1
camera_noisy = random_noise(camera, mode='gaussian', var=sigma**2)

alpha = 0.2
camera_tv = denoise_tv_chambolle(camera_noisy, weight=alpha)

lam_tikh = 0.1
camera_tikh = camera_noisy / (1 + lam_tikh)

psnr_noisy = peak_signal_noise_ratio(camera, camera_noisy)
psnr_tikh = peak_signal_noise_ratio(camera, camera_tikh)
psnr_tv = peak_signal_noise_ratio(camera, camera_tv)

print("===== TV先验的局限：阶梯效应 =====")
print(f"\n噪声水平 σ = {sigma}")
print(f"\n去噪结果对比:")
print(f"  含噪图像: PSNR = {psnr_noisy:.2f} dB")
print(f"  Tikhonov (高斯先验): PSNR = {psnr_tikh:.2f} dB")
print(f"  TV先验: PSNR = {psnr_tv:.2f} dB")
print(f"\nTV先验特点:")
print(f"  优势: 保边 - 边缘锐利")
print(f"  局限: 阶梯效应 - 渐变区域出现阶梯状伪影")
print(f"  原因: TV偏好分段常数函数")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(camera_noisy, cmap='gray')
axes[0].set_title(f'含噪图像\nPSNR={psnr_noisy:.2f}dB')
axes[0].axis('off')

axes[1].imshow(camera_tikh, cmap='gray')
axes[1].set_title(f'Tikhonov去噪 (高斯先验)\n边缘模糊\nPSNR={psnr_tikh:.2f}dB')
axes[1].axis('off')

axes[2].imshow(camera_tv, cmap='gray')
axes[2].set_title(f'TV去噪\n边缘锐利，渐变区阶梯化\nPSNR={psnr_tv:.2f}dB')
axes[2].axis('off')

plt.suptitle('高斯先验 vs TV先验：边缘保持的代价', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_TV去噪对比.png'), dpi=150, bbox_inches='tight')
plt.show()

row = n // 2
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(camera[row, :], 'k-', linewidth=2, label='真实信号')
axes[0].plot(camera_tikh[row, :], 'b-', linewidth=1.5, label='Tikhonov (边缘模糊)')
axes[0].plot(camera_tv[row, :], 'r-', linewidth=1.5, label='TV (边缘锐利)')
axes[0].set_title('中心行剖面对比')
axes[0].legend()
axes[0].set_xlabel('像素索引')
axes[0].set_ylabel('灰度值')
axes[0].grid(True, alpha=0.3)

grad_region = slice(n//4, n//2)
axes[1].plot(camera[row, grad_region], 'k-', linewidth=2, label='真实（渐变区域）')
axes[1].plot(camera_tv[row, grad_region], 'r-', linewidth=1.5, label='TV（阶梯效应）')
axes[1].set_title('渐变区域放大：TV阶梯效应')
axes[1].legend()
axes[1].set_xlabel('像素索引')
axes[1].set_ylabel('灰度值')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_阶梯效应分析.png'), dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

t = np.linspace(-3, 3, 100)
axes[0].plot(t, np.abs(t), 'r-', linewidth=2, label='|t| (TV正则项)')
axes[0].set_title('TV正则项形态\n偏好稀疏解')
axes[0].legend()
axes[0].set_xlabel('t (梯度值)')
axes[0].set_ylabel('惩罚值')
axes[0].grid(True, alpha=0.3)

methods = ['含噪', 'Tikhonov\n(高斯先验)', 'TV先验']
psnrs = [psnr_noisy, psnr_tikh, psnr_tv]
colors = ['gray', 'blue', 'red']

axes[1].bar(methods, psnrs, color=colors, alpha=0.7)
axes[1].set_ylabel('PSNR (dB)')
axes[1].set_title('去噪性能对比')
axes[1].set_ylim([0, max(psnrs) + 5])

for i, (m, p) in enumerate(zip(methods, psnrs)):
    axes[1].text(i, p + 0.5, f'{p:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_正则项与性能对比.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n===== TV先验总结 =====")
print("\nTV正则项: ||∇x||₁")
print("\n优势:")
print("  1. 保边性: 边缘不会被过度平滑")
print("  2. 稀疏性: 梯度稀疏假设对自然图像合理")
print("  3. 凸性: 优化问题可高效求解")
print("\n局限:")
print("  1. 阶梯效应: 渐变区域出现阶梯状伪影")
print("  2. 分段常数假设: 无法自然表示渐变")
print("  3. 参数敏感: α需要仔细调节")
print("\n改进方向:")
print("  - TGV (附录2B): 引入二阶信息，允许分段仿射")
print("  - 隐式先验 (2.4节): 从数据中学习先验")
