import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize
import sys
import os

# ====== 中文字体配置（兼容本地和 Google Colab）======
_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验1.2-2', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验1.2-2')
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
sys.path.insert(0, _chinese_path)
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=_chinese_path)
# ========================================================

np.random.seed(42)

# ---- 1. 准备测试图像（Shepp-Logan 幻影）----
n = 128
phantom = resize(shepp_logan_phantom(), (n, n), order=0, preserve_range=True, anti_aliasing=False)
phantom = phantom / phantom.max()

# ---- 2. Radon 变换：生成正弦图（Sinogram）----
theta = np.linspace(0, 180, 90, endpoint=False)
sinogram = radon(phantom, theta=theta, circle=True)

print("="*60)
print("Radon 变换正向模型 y = Ax")
print("="*60)
print(f"原始图像尺寸: {phantom.shape}")
print(f"投影角度数: {len(theta)}")
print(f"正弦图尺寸: {sinogram.shape}")
print(f"  - 行数（探测器位置 t）: {sinogram.shape[0]}")
print(f"  - 列数（投影角度 θ）: {sinogram.shape[1]}")
print("="*60)
print("\n正弦图的含义（1.2 节）：")
print("- 横轴：投影角度 θ（CT 扫描仪旋转的角度）")
print("- 纵轴：探测器位置 t（射线偏移量）")
print("- 像素值：对应射线穿过物体后的线积分")
print("\n为什么叫'正弦图'？")
print("图像中的一个点在 Radon 变换后形成一条正弦曲线")
print("t = a·cos(θ) + b·sin(θ)，所有点的正弦曲线叠加形成正弦图")
print("="*60)

# ---- 3. FBP 重建 ----
reconstruction_fbp = iradon(sinogram, theta=theta, filter_name='ramp', output_size=n)
reconstruction_fbp = np.clip(reconstruction_fbp, 0, 1)

error_map = np.abs(phantom - reconstruction_fbp)
rmse_fbp = np.sqrt(np.mean((phantom - reconstruction_fbp)**2))
psnr_fbp = 20 * np.log10(1.0 / rmse_fbp)

print("\nFBP（滤波反投影）重建结果")
print("="*60)
print(f"滤波器类型: Ramp (Ram-Lak)")
print(f"RMSE: {rmse_fbp:.4f}")
print(f"PSNR: {psnr_fbp:.2f} dB")
print("="*60)

# ---- 4. 可视化 ----
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].imshow(phantom, cmap='gray')
axes[0, 0].set_title('原始截面图像 x\n（衰减系数分布）')
axes[0, 0].axis('off')

im = axes[0, 1].imshow(sinogram, cmap='gray', aspect='auto',
                    extent=[theta.min(), theta.max(), sinogram.shape[0], 0])
axes[0, 1].set_title('正弦图（Sinogram）y\n$y = Ax$，A 是 Radon 变换算子')
axes[0, 1].set_xlabel('投影角度 θ (°)')
axes[0, 1].set_ylabel('探测器位置 t')
plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

example_angle_idx = len(theta) // 2
example_angle = theta[example_angle_idx]
single_projection = sinogram[:, example_angle_idx]

axes[0, 2].plot(single_projection, 'b-', linewidth=2)
axes[0, 2].set_title(f'单角度投影示例\nθ = {example_angle:.1f}°')
axes[0, 2].set_xlabel('探测器位置 t')
axes[0, 2].set_ylabel('线积分值')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

axes[1, 0].imshow(reconstruction_fbp, cmap='gray')
axes[1, 0].set_title(f'FBP 重建结果\n(RMSE={rmse_fbp:.4f}, PSNR={psnr_fbp:.2f} dB)')
axes[1, 0].axis('off')

im2 = axes[1, 1].imshow(error_map, cmap='hot')
axes[1, 1].set_title('误差热力图\n|x - x_FBP|')
axes[1, 1].axis('off')
plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

axes[1, 2].text(0.1, 0.7, f'FBP 重建指标:', fontsize=12, fontweight='bold')
axes[1, 2].text(0.1, 0.5, f'投影角度数: {len(theta)}', fontsize=11)
axes[1, 2].text(0.1, 0.3, f'RMSE: {rmse_fbp:.4f}', fontsize=11)
axes[1, 2].text(0.1, 0.1, f'PSNR: {psnr_fbp:.2f} dB', fontsize=11)
axes[1, 2].axis('off')

plt.suptitle('Radon 变换与 FBP 重建\n（CT 成像的正向模型 $y = Ax$ 与逆问题求解）', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_2_2_Radon变换.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n可视化结果已保存。")
print("说明：FBP 是 CT 重建的经典算法，通过滤波+反投影操作近似求解 $x = A^{-1}y$。")
print("当投影角度充足时，FBP 能较好地恢复原始图像。")
