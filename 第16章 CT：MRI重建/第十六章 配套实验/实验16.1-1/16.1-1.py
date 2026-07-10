# -*- coding: utf-8 -*-
"""
实验16.1-1 CT成像基础——Radon变换与FBP重建

实验目的：理解CT正向模型的完整链条——从X射线衰减到线积分，
          从Radon变换到sinogram，再到FBP滤波反投影重建

素材来源：基于1.4.py扩展
运行前提：CPU可运行
"""

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import sys
import io
import warnings
import logging

# 设置控制台输出为 UTF-8 (Windows 下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

plt.rcParams['axes.unicode_minus'] = False

_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验16.1-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(_chinese_path, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")

np.random.seed(42)

print(f"\n{'='*60}")
print(f"实验16.1-1: CT成像基础——Radon变换与FBP重建")
print(f"{'='*60}")

# ---- 准备Shepp-Logan幻影 ----
n = 128
phantom = resize(shepp_logan_phantom(), (n, n), order=0, preserve_range=True, anti_aliasing=False)
phantom = phantom / phantom.max()


# ========================================================================
# 步骤1：Shepp-Logan幻影与Sinogram（16.1.2节 Radon变换与sinogram）
# ========================================================================
print("\n" + "=" * 60)
print("步骤1：Shepp-Logan幻影与Sinogram（16.1.2节）")
print("=" * 60)

theta_full = np.linspace(0, 180, 180, endpoint=False)
theta_sparse = np.linspace(0, 180, 30, endpoint=False)

sinogram_full = radon(phantom, theta=theta_full, circle=True)
sinogram_sparse = radon(phantom, theta=theta_sparse, circle=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].imshow(phantom, cmap='gray')
axes[0].set_title('Shepp-Logan幻影 $u(x)$')
axes[0].axis('off')

axes[1].imshow(sinogram_full, cmap='gray', aspect='auto',
               extent=[0, 180, sinogram_full.shape[0], 0])
axes[1].set_title('Sinogram $\\mathcal{R}u(\\theta, s)$ (180角度)\n"正弦图"——每个点源呈正弦曲线')
axes[1].set_xlabel('投影角度 $\\theta$ ($^{\\circ}$)')
axes[1].set_ylabel('探测器位置 s')

axes[2].imshow(sinogram_sparse, cmap='gray', aspect='auto',
               extent=[0, 180, sinogram_sparse.shape[0], 0])
axes[2].set_title('Sinogram (30角度)\n信息量显著减少')
axes[2].set_xlabel('投影角度 $\\theta$ ($^{\\circ}$)')
axes[2].set_ylabel('探测器位置 s')

plt.suptitle('步骤1：Radon变换——从图像到Sinogram', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_Sinogram.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"  Sinogram(180角度): {sinogram_full.shape}")
print(f"  Sinogram(30角度):  {sinogram_sparse.shape}")


# ========================================================================
# 步骤2：反投影 vs FBP——为什么需要滤波？（16.1.4节）
# ========================================================================
print("\n" + "=" * 60)
print("步骤2：反投影 vs FBP——为什么需要滤波？（16.1.4节）")
print("=" * 60)

recon_bp = iradon(sinogram_full, theta=theta_full, circle=True, filter_name=None)
recon_fbp = iradon(sinogram_full, theta=theta_full, circle=True, filter_name='ramp')

psnr_bp = psnr(phantom, recon_bp, data_range=1.0)
psnr_fbp = psnr(phantom, recon_fbp, data_range=1.0)
ssim_bp = ssim(phantom, recon_bp, data_range=1.0)
ssim_fbp = ssim(phantom, recon_fbp, data_range=1.0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(phantom, cmap='gray')
axes[0].set_title('原始幻影')
axes[0].axis('off')

axes[1].imshow(recon_bp, cmap='gray')
axes[1].set_title(f'反投影 $\\mathcal{{R}}^* f$（无滤波）\nPSNR={psnr_bp:.1f}dB, SSIM={ssim_bp:.3f}\n模糊——$\\mathcal{{R}}^*\\mathcal{{R}}$是平滑算子')
axes[1].axis('off')

axes[2].imshow(recon_fbp, cmap='gray')
axes[2].set_title(f'FBP $\\mathcal{{R}}^* \\mathcal{{F}}^{{-1}}[|\\omega| \\hat{{f}}]$\nPSNR={psnr_fbp:.1f}dB, SSIM={ssim_fbp:.3f}\n清晰——斜坡滤波器补偿平滑')
axes[2].axis('off')

plt.suptitle('步骤2：反投影≠逆投影——为什么FBP需要滤波？（16.1.4节）', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_反投影vs_FBP.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"  反投影 PSNR={psnr_bp:.1f}dB, SSIM={ssim_bp:.3f}")
print(f"  FBP    PSNR={psnr_fbp:.1f}dB, SSIM={ssim_fbp:.3f}")


# ========================================================================
# 步骤3：FBP滤波器选择——分辨率与噪声的权衡（16.1.4节）
# ========================================================================
print("\n" + "=" * 60)
print("步骤3：FBP滤波器选择——分辨率与噪声的权衡（16.1.4节）")
print("=" * 60)

noise_sigma = 0.05
sinogram_noisy = sinogram_full + noise_sigma * np.random.randn(*sinogram_full.shape)

filter_names = ['ramp', 'shepp-logan', 'cosine', 'hann']
filter_labels = ['Ramp $|\\omega|$', 'Shepp-Logan $|\\omega|\\mathrm{sinc}$',
                 'Cosine $|\\omega|\\cos$', 'Hann $|\\omega| \\cdot$ Hann窗']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, (fname, flabel) in enumerate(zip(filter_names, filter_labels)):
    recon_clean = iradon(sinogram_full, theta=theta_full, circle=True, filter_name=fname)
    p_clean = psnr(phantom, recon_clean, data_range=1.0)
    recon_noisy = iradon(sinogram_noisy, theta=theta_full, circle=True, filter_name=fname)
    p_noisy = psnr(phantom, recon_noisy, data_range=1.0)

    axes[0, i].imshow(recon_clean, cmap='gray')
    axes[0, i].set_title(f'{flabel}\n无噪声 PSNR={p_clean:.1f}dB')
    axes[0, i].axis('off')

    axes[1, i].imshow(recon_noisy, cmap='gray')
    axes[1, i].set_title(f'含噪($\\sigma$={noise_sigma})\nPSNR={p_noisy:.1f}dB')
    axes[1, i].axis('off')

plt.suptitle('步骤3：FBP滤波器选择——Ramp高频放大噪声，Hann更平滑但分辨率低', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_滤波器对比.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n实验16.1-1完成！")
