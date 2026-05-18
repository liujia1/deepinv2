import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize
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
# 根据 1.2 节：CT 扫描仪从 K 个角度采集数据，每个角度有 D 个探测器单元
# 这里使用 90 个均匀分布的投影角度
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

# ---- 3. 可视化 ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 左图：原始图像
axes[0].imshow(phantom, cmap='gray')
axes[0].set_title('原始截面图像 x\n（衰减系数分布）')
axes[0].axis('off')

# 中图：正弦图
im = axes[1].imshow(sinogram, cmap='gray', aspect='auto',
                    extent=[theta.min(), theta.max(), sinogram.shape[0], 0])
axes[1].set_title('正弦图（Sinogram）y\n$y = Ax$，A 是 Radon 变换算子')
axes[1].set_xlabel('投影角度 θ (°)')
axes[1].set_ylabel('探测器位置 t')
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

# 右图：单角度投影示例
example_angle_idx = len(theta) // 2  # 选择中间角度
example_angle = theta[example_angle_idx]
single_projection = sinogram[:, example_angle_idx]

axes[2].plot(single_projection, 'b-', linewidth=2)
axes[2].set_title(f'单角度投影示例\nθ = {example_angle:.1f}°')
axes[2].set_xlabel('探测器位置 t')
axes[2].set_ylabel('线积分值')
axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

plt.suptitle('Radon 变换：从截面图像到正弦图\n（CT 成像的正向模型 $y = Ax$）', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_2_2_Radon变换.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n可视化结果已保存。")
print("说明：正弦图本身无法直观解读，需要通过逆问题求解（如滤波反投影或迭代重建）")
print("才能恢复出可理解的截面图像。这就是 CT 重建的核心挑战。")
