# -*- coding: utf-8 -*-
"""
实验16.2-1 CT不适定性与正则化重建

实验目的：理解CT逆问题的两种不适定性（稀疏角度vs有限角度），
          验证迭代正则化重建方法（Tikhonov/TV）的效果，
          量化不适定程度与重建质量的关系

素材来源：基于16.1.py步骤4-6
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
    SAVE_DIR = os.path.join(_gdrive, '实验16.2-1')
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
print(f"实验16.2-1: CT不适定性与正则化重建")
print(f"{'='*60}")

# ---- 准备Shepp-Logan幻影 ----
n = 128
phantom = resize(shepp_logan_phantom(), (n, n), order=0, preserve_range=True, anti_aliasing=False)
phantom = phantom / phantom.max()

# 先做全角FBP作为参考
theta_full = np.linspace(0, 180, 180, endpoint=False)
sinogram_full = radon(phantom, theta=theta_full, circle=True)
recon_fbp_full = iradon(sinogram_full, theta=theta_full, circle=True, filter_name='ramp')
psnr_fbp_full = psnr(phantom, recon_fbp_full, data_range=1.0)
noise_sigma = 0.05


# ========================================================================
# 步骤1：稀疏角度与有限角度CT——两种不同的不适定性
# ========================================================================
print("\n" + "=" * 60)
print("步骤1：稀疏角度与有限角度CT——两种不同的不适定性")
print("=" * 60)

# 稀疏角度：30个均匀分布在[0,180)
theta_sparse30 = np.linspace(0, 180, 30, endpoint=False)
sino_sparse = radon(phantom, theta=theta_sparse30, circle=True)
recon_sparse = iradon(sino_sparse, theta=theta_sparse30, circle=True, filter_name='ramp')

# 有限角度：0°-120°（缺失60°-180°）
theta_limited = np.linspace(0, 120, 120, endpoint=False)
sino_limited = radon(phantom, theta=theta_limited, circle=True)
recon_limited = iradon(sino_limited, theta=theta_limited, circle=True, filter_name='ramp')

p_sparse = psnr(phantom, recon_sparse, data_range=1.0)
p_limited = psnr(phantom, recon_limited, data_range=1.0)
s_sparse = ssim(phantom, recon_sparse, data_range=1.0)
s_limited = ssim(phantom, recon_limited, data_range=1.0)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes[0, 0].imshow(phantom, cmap='gray')
axes[0, 0].set_title('原始幻影')
axes[0, 0].axis('off')

axes[0, 1].imshow(sino_sparse, cmap='gray', aspect='auto',
                   extent=[0, 180, sino_sparse.shape[0], 0])
axes[0, 1].set_title('稀疏Sinogram (30角度)\n角度覆盖[0$^{\\circ}$, 180$^{\\circ}$)但间隔大')
axes[0, 1].set_xlabel('$\\theta$ ($^{\\circ}$)')

axes[0, 2].imshow(sino_limited, cmap='gray', aspect='auto',
                   extent=[0, 120, sino_limited.shape[0], 0])
axes[0, 2].set_title('有限角Sinogram (0$^{\\circ}$-120$^{\\circ}$)\n缺失60$^{\\circ}$-180$^{\\circ}$角度范围')
axes[0, 2].set_xlabel('$\\theta$ ($^{\\circ}$)')

axes[1, 0].imshow(recon_fbp_full, cmap='gray')
_deg = '$^{\\circ}$'  # 度数符号，预定义以避免f-string中反斜杠语法错误
axes[1, 0].set_title(f'FBP 全角(180{_deg})参考\nPSNR={psnr_fbp_full:.1f}dB')
axes[1, 0].axis('off')

axes[1, 1].imshow(recon_sparse, cmap='gray')
axes[1, 1].set_title(f'FBP 稀疏30角度\nPSNR={p_sparse:.1f}dB, SSIM={s_sparse:.3f}\n条纹伪影（混叠）')
axes[1, 1].axis('off')

axes[1, 2].imshow(recon_limited, cmap='gray')
axes[1, 2].set_title(f'FBP 有限角0{_deg}-120{_deg}\nPSNR={p_limited:.1f}dB, SSIM={s_limited:.3f}\n条纹伪影+不可见边缘丢失')
axes[1, 2].axis('off')

plt.suptitle('步骤1：稀疏角度 vs 有限角度——两种不同的不适定性', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_稀疏vs有限角.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"  全角 FBP:     PSNR={psnr_fbp_full:.1f}dB")
print(f"  稀疏30角度:   PSNR={p_sparse:.1f}dB, SSIM={s_sparse:.3f}")
print(f"  有限角0-120°: PSNR={p_limited:.1f}dB, SSIM={s_limited:.3f}")


# ========================================================================
# 步骤2：迭代正则化重建——Tikhonov/TV正则化
# ========================================================================
print("\n" + "=" * 60)
print("步骤2：迭代正则化重建")
print("=" * 60)

def forward_radon(x, theta):
    """正向Radon变换"""
    return radon(x, theta=theta, circle=True)

def backward_radon(sino, theta):
    """伴随Radon变换（无滤波反投影）"""
    return iradon(sino, theta=theta, circle=True, filter_name=None)

def gradient_descent_recon(y, theta, n_iter=100, lam=0.1, reg_type='tikhonov'):
    """
    迭代正则化重建 (Landweber迭代 + 正则化)
    x_{k+1} = x_k - alpha * (A^T(Ax_k - y) + lambda * R'(x_k))

    reg_type: 'tikhonov' -> R'(x) = x
              'tv'       -> R'(x) = -div(grad(x) / |grad(x)|)
    """
    x = iradon(y, theta=theta, circle=True, filter_name='shepp-logan')
    alpha = 0.01  # 小步长，避免发散

    for k in tqdm(range(n_iter), desc=f'  {reg_type}重建', leave=False):
        residual = forward_radon(x, theta) - y
        grad_data = backward_radon(residual, theta)

        if reg_type == 'tikhonov':
            grad_reg = lam * x
        elif reg_type == 'tv':
            eps = 1e-8
            dx = np.diff(x, axis=1, prepend=x[:, :1])
            dy = np.diff(x, axis=0, prepend=x[:1, :])
            mag = np.sqrt(dx**2 + dy**2 + eps)
            grad_tv_x = dx / mag
            grad_tv_y = dy / mag
            div_x = np.diff(grad_tv_x, axis=1, append=grad_tv_x[:, -1:])
            div_y = np.diff(grad_tv_y, axis=0, append=grad_tv_y[-1:, :])
            grad_reg = -lam * (div_x + div_y)
        else:
            grad_reg = 0

        x = x - alpha * (grad_data + grad_reg)
        x = np.clip(x, 0, None)

    return x

sino_sparse_noisy = radon(phantom, theta=theta_sparse30, circle=True)
sino_sparse_noisy = sino_sparse_noisy + noise_sigma * np.random.randn(*sino_sparse_noisy.shape)

print("  正在执行Tikhonov正则化重建...")
recon_tikh = gradient_descent_recon(sino_sparse_noisy, theta_sparse30, n_iter=100, lam=0.1, reg_type='tikhonov')
print("  正在执行TV正则化重建...")
recon_tv = gradient_descent_recon(sino_sparse_noisy, theta_sparse30, n_iter=100, lam=0.1, reg_type='tv')

recon_sparse_noisy = iradon(sino_sparse_noisy, theta=theta_sparse30, circle=True, filter_name='ramp')

p_fbp = psnr(phantom, recon_sparse_noisy, data_range=1.0)
p_tikh = psnr(phantom, recon_tikh, data_range=1.0)
p_tv = psnr(phantom, recon_tv, data_range=1.0)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

axes[0].imshow(phantom, cmap='gray')
axes[0].set_title('原始幻影')
axes[0].axis('off')

axes[1].imshow(recon_sparse_noisy, cmap='gray')
axes[1].set_title(f'FBP (无正则化)\nPSNR={p_fbp:.1f}dB\n噪声放大')
axes[1].axis('off')

axes[2].imshow(recon_tikh, cmap='gray')
axes[2].set_title(f'Tikhonov正则化 ($\\lambda$=0.1)\nPSNR={p_tikh:.1f}dB\n过度平滑')
axes[2].axis('off')

axes[3].imshow(recon_tv, cmap='gray')
axes[3].set_title(f'TV正则化 ($\\lambda$=0.1)\nPSNR={p_tv:.1f}dB\n保持边缘')
axes[3].axis('off')

plt.suptitle('步骤2：正则化重建对比——FBP噪声放大 vs Tikhonov平滑 vs TV保边缘', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_正则化对比.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"  FBP(稀疏含噪):   PSNR={p_fbp:.1f}dB")
print(f"  Tikhonov(λ=0.1): PSNR={p_tikh:.1f}dB")
print(f"  TV(λ=0.1):       PSNR={p_tv:.1f}dB")


# ========================================================================
# 步骤3：角度数量 vs 重建质量——不适定性的量化
# ========================================================================
print("\n" + "=" * 60)
print("步骤3：角度数量 vs 重建质量——不适定性的量化")
print("=" * 60)

n_angles_list = [10, 15, 20, 30, 45, 60, 90, 120, 180]
psnr_clean = []
psnr_noisy_curve = []

for na in tqdm(n_angles_list, desc='  稀疏角度扫描'):
    theta = np.linspace(0, 180, na, endpoint=False)
    sino = radon(phantom, theta=theta, circle=True)
    recon = iradon(sino, theta=theta, circle=True, filter_name='ramp')
    psnr_clean.append(psnr(phantom, recon, data_range=1.0))

    sino_n = sino + noise_sigma * np.random.randn(*sino.shape)
    recon_n = iradon(sino_n, theta=theta, circle=True, filter_name='shepp-logan')
    psnr_noisy_curve.append(psnr(phantom, recon_n, data_range=1.0))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(n_angles_list, psnr_clean, 'bo-', label='FBP (无噪声)')
ax1.plot(n_angles_list, psnr_noisy_curve, 'rs-', label=f'FBP+Shepp-Logan滤波 ($\\sigma$={noise_sigma})')
ax1.set_xlabel('投影角度数')
ax1.set_ylabel('PSNR (dB)')
ax1.set_title('重建质量 vs 投影角度数\n角度越少→信息越少→不适定性越严重')
ax1.legend()
ax1.grid(True)

limited_ranges = [30, 60, 90, 120, 150, 180]
psnr_limited = []
for angle_range in tqdm(limited_ranges, desc='  有限角度扫描'):
    theta = np.linspace(0, angle_range, min(angle_range, 180), endpoint=False)
    sino = radon(phantom, theta=theta, circle=True)
    recon = iradon(sino, theta=theta, circle=True, filter_name='ramp')
    psnr_limited.append(psnr(phantom, recon, data_range=1.0))

ax2.plot(limited_ranges, psnr_limited, 'gs-')
ax2.set_xlabel('角度覆盖范围 ($^{\\circ}$)')
ax2.set_ylabel('PSNR (dB)')
ax2.set_title('有限角CT：角度范围 vs 重建质量\n缺失角度→频域信息缺口→条纹伪影')
ax2.grid(True)

plt.suptitle('步骤3：CT不适定性的量化', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_角度vs质量.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n实验16.2-1完成！")
