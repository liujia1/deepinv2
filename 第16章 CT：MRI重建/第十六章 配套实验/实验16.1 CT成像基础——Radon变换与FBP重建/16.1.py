# -*- coding: utf-8 -*-
"""
实验16.1 CT成像基础——Radon变换与FBP重建
对应知识点：16.1.1-16.1.4节（Beer-Lambert/Radon/Fourier切片/FBP）
           16.2.1-16.2.2节（全角vs有限角不适定性）

素材来源：基于1.4.py扩展，新增滤波器对比、有限角CT、迭代正则化重建
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib as mpl
import warnings
import logging

# ====== 解决中文乱码的核心代码（Windows + Linux 自动适配）======
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

import platform
from matplotlib.font_manager import FontManager, FontProperties

def _find_chinese_font():
    """自动检测系统中可用的中文字体，兼容 Windows / Linux"""
    candidates = []
    if platform.system() == 'Windows':
        candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
    else:
        candidates = [
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
            'Noto Sans CJK SC', 'Noto Sans CJK',
            'Source Han Sans SC', 'AR PL UMing CN',
            'SimHei',
        ]
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available:
            return font
    import os, re
    cjk_patterns = ['cjk', 'wqy', 'noto.*cjk', 'wenquan', 'chinese', 'simhei']
    for f in fm.ttflist:
        name_lower = f.name.lower()
        fname_lower = (os.path.basename(f.fname) if hasattr(f, 'fname') else '').lower()
        for pat in cjk_patterns:
            if re.search(pat, name_lower) or re.search(pat, fname_lower):
                return f.name
    return None

_cn_font = _find_chinese_font()
if _cn_font:
    plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['font.family'] = 'sans-serif'
    print(f"[Font] 已检测到中文字体: {_cn_font}")
else:
    print("[Font] 未找到中文字体，中文可能显示为方框")
# ========================================================

np.random.seed(42)

# ---- 准备Shepp-Logan幻影 ----
n = 128
phantom = resize(shepp_logan_phantom(), (n, n), order=0, preserve_range=True, anti_aliasing=False)
phantom = phantom / phantom.max()


# ========================================================================
# 步骤1：Shepp-Logan幻影与Sinogram（16.1.2节 Radon变换与sinogram）
# ========================================================================
print("=" * 60)
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
axes[1].set_xlabel('投影角度 θ (°)')
axes[1].set_ylabel('探测器位置 s')

axes[2].imshow(sinogram_sparse, cmap='gray', aspect='auto',
               extent=[0, 180, sinogram_sparse.shape[0], 0])
axes[2].set_title(f'Sinogram (30角度)\n信息量显著减少')
axes[2].set_xlabel('投影角度 θ (°)')
axes[2].set_ylabel('探测器位置 s')

plt.suptitle('步骤1：Radon变换——从图像到Sinogram（16.1.2节）', fontsize=13)
plt.tight_layout()
plt.savefig('步骤1_Sinogram.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"  Sinogram(180角度): {sinogram_full.shape}")
print(f"  Sinogram(30角度):  {sinogram_sparse.shape}")


# ========================================================================
# 步骤2：反投影 vs FBP——为什么需要滤波？（16.1.4节）
# ========================================================================
print("\n" + "=" * 60)
print("步骤2：反投影 vs FBP——为什么需要滤波？（16.1.4节）")
print("=" * 60)

# 反投影（无滤波）= R^T
recon_bp = iradon(sinogram_full, theta=theta_full, circle=True, filter_name=None)
# FBP（斜坡滤波器）
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
plt.savefig('步骤2_反投影vs_FBP.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"  反投影 PSNR={psnr_bp:.1f}dB, SSIM={ssim_bp:.3f}")
print(f"  FBP    PSNR={psnr_fbp:.1f}dB, SSIM={ssim_fbp:.3f}")


# ========================================================================
# 步骤3：FBP滤波器选择——分辨率与噪声的权衡（16.1.4节）
# ========================================================================
print("\n" + "=" * 60)
print("步骤3：FBP滤波器选择——分辨率与噪声的权衡（16.1.4节）")
print("=" * 60)

# 添加噪声
noise_sigma = 0.05
sinogram_noisy = sinogram_full + noise_sigma * np.random.randn(*sinogram_full.shape)

filter_names = ['ramp', 'shepp-logan', 'cosine', 'hann']
filter_labels = ['Ramp $|\\omega|$', 'Shepp-Logan $|\\omega|\\mathrm{sinc}$',
                 'Cosine $|\\omega|\\cos$', 'Hann $|\\omega| \\cdot$ Hann窗']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, (fname, flabel) in enumerate(zip(filter_names, filter_labels)):
    # 无噪声重建
    recon_clean = iradon(sinogram_full, theta=theta_full, circle=True, filter_name=fname)
    p_clean = psnr(phantom, recon_clean, data_range=1.0)
    # 有噪声重建
    recon_noisy = iradon(sinogram_noisy, theta=theta_full, circle=True, filter_name=fname)
    p_noisy = psnr(phantom, recon_noisy, data_range=1.0)

    axes[0, i].imshow(recon_clean, cmap='gray')
    axes[0, i].set_title(f'{flabel}\n无噪声 PSNR={p_clean:.1f}dB')
    axes[0, i].axis('off')

    axes[1, i].imshow(recon_noisy, cmap='gray')
    axes[1, i].set_title(f'含噪声(σ={noise_sigma})\nPSNR={p_noisy:.1f}dB')
    axes[1, i].axis('off')

plt.suptitle('步骤3：FBP滤波器选择——Ramp高频放大噪声，Hann更平滑但分辨率低（16.1.4节）', fontsize=12)
plt.tight_layout()
plt.savefig('步骤3_滤波器对比.png', dpi=150, bbox_inches='tight')
plt.show()


# ========================================================================
# 步骤4：稀疏角度与有限角度CT（16.2.1/16.2.2节）
# ========================================================================
print("\n" + "=" * 60)
print("步骤4：稀疏角度与有限角度CT（16.2.1/16.2.2节）")
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
axes[0, 1].set_title('稀疏Sinogram (30角度)\n角度覆盖[0°, 180°)但间隔大')
axes[0, 1].set_xlabel('θ (°)')

axes[0, 2].imshow(sino_limited, cmap='gray', aspect='auto',
                   extent=[0, 120, sino_limited.shape[0], 0])
axes[0, 2].set_title('有限角Sinogram (0°-120°)\n缺失60°-180°角度范围')
axes[0, 2].set_xlabel('θ (°)')

axes[1, 0].imshow(recon_fbp, cmap='gray')
axes[1, 0].set_title(f'FBP 全角(180°)参考\nPSNR={psnr_fbp:.1f}dB')
axes[1, 0].axis('off')

axes[1, 1].imshow(recon_sparse, cmap='gray')
axes[1, 1].set_title(f'FBP 稀疏30角度\nPSNR={p_sparse:.1f}dB, SSIM={s_sparse:.3f}\n条纹伪影（混叠）')
axes[1, 1].axis('off')

axes[1, 2].imshow(recon_limited, cmap='gray')
axes[1, 2].set_title(f'FBP 有限角0°-120°\nPSNR={p_limited:.1f}dB, SSIM={s_limited:.3f}\n条纹伪影+不可见边缘丢失')
axes[1, 2].axis('off')

plt.suptitle('步骤4：稀疏角度 vs 有限角度——两种不同的不适定性（16.2.1/16.2.2节）', fontsize=13)
plt.tight_layout()
plt.savefig('步骤4_稀疏vs有限角.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"  全角 FBP:     PSNR={psnr_fbp:.1f}dB")
print(f"  稀疏30角度:   PSNR={p_sparse:.1f}dB, SSIM={s_sparse:.3f}")
print(f"  有限角0-120°: PSNR={p_limited:.1f}dB, SSIM={s_limited:.3f}")


# ========================================================================
# 步骤5：迭代正则化重建（16.2.4节 Tikhonov/TV正则化）
# ★原创设计：基于skimage的Radon/iradon实现迭代梯度下降
# ========================================================================
print("\n" + "=" * 60)
print("步骤5：迭代正则化重建（16.2.4节）")
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
    x = iradon(y, theta=theta, circle=True, filter_name='shepp-logan')  # 用FBP初始化
    alpha = 0.01  # 小步长，避免发散

    for k in range(n_iter):
        # 数据一致性梯度: A^T(Ax - y)
        residual = forward_radon(x, theta) - y
        grad_data = backward_radon(residual, theta)

        # 正则化梯度
        if reg_type == 'tikhonov':
            grad_reg = lam * x
        elif reg_type == 'tv':
            # TV梯度近似（简化版）
            eps = 1e-8
            dx = np.diff(x, axis=1, prepend=x[:, :1])
            dy = np.diff(x, axis=0, prepend=x[:1, :])
            mag = np.sqrt(dx**2 + dy**2 + eps)
            # 散度 = div(grad/|grad|)
            grad_tv_x = dx / mag
            grad_tv_y = dy / mag
            div_x = np.diff(grad_tv_x, axis=1, append=grad_tv_x[:, -1:])
            div_y = np.diff(grad_tv_y, axis=0, append=grad_tv_y[-1:, :])
            grad_reg = -lam * (div_x + div_y)
        else:
            grad_reg = 0

        x = x - alpha * (grad_data + grad_reg)
        x = np.clip(x, 0, None)  # 非负约束

    return x

# 对稀疏角度含噪sinogram做正则化
sino_sparse_noisy = radon(phantom, theta=theta_sparse30, circle=True)
sino_sparse_noisy = sino_sparse_noisy + noise_sigma * np.random.randn(*sino_sparse_noisy.shape)

print("  正在执行Tikhonov正则化重建...")
recon_tikh = gradient_descent_recon(sino_sparse_noisy, theta_sparse30, n_iter=100, lam=0.1, reg_type='tikhonov')
print("  正在执行TV正则化重建...")
recon_tv = gradient_descent_recon(sino_sparse_noisy, theta_sparse30, n_iter=100, lam=0.1, reg_type='tv')

# 稀疏角度FBP（无正则化参考）
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
axes[2].set_title(f'Tikhonov正则化 (λ=0.01)\nPSNR={p_tikh:.1f}dB\n过度平滑')
axes[2].axis('off')

axes[3].imshow(recon_tv, cmap='gray')
axes[3].set_title(f'TV正则化 (λ=0.05)\nPSNR={p_tv:.1f}dB\n保持边缘')
axes[3].axis('off')

plt.suptitle('步骤5：正则化重建对比——FBP噪声放大 vs Tikhonov平滑 vs TV保边缘（16.2.4节）', fontsize=12)
plt.tight_layout()
plt.savefig('步骤5_正则化对比.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"  FBP(稀疏含噪):    PSNR={p_fbp:.1f}dB")
print(f"  Tikhonov(λ=0.01): PSNR={p_tikh:.1f}dB")
print(f"  TV(λ=0.05):       PSNR={p_tv:.1f}dB")


# ========================================================================
# 步骤6：角度数量 vs 重建质量——不适定性的量化（16.2.1节）
# ========================================================================
print("\n" + "=" * 60)
print("步骤6：角度数量 vs 重建质量（16.2.1节）")
print("=" * 60)

n_angles_list = [10, 15, 20, 30, 45, 60, 90, 120, 180]
psnr_clean = []
psnr_noisy_curve = []

for na in n_angles_list:
    theta = np.linspace(0, 180, na, endpoint=False)
    sino = radon(phantom, theta=theta, circle=True)
    recon = iradon(sino, theta=theta, circle=True, filter_name='ramp')
    psnr_clean.append(psnr(phantom, recon, data_range=1.0))

    sino_n = sino + noise_sigma * np.random.randn(*sino.shape)
    recon_n = iradon(sino_n, theta=theta, circle=True, filter_name='shepp-logan')
    psnr_noisy_curve.append(psnr(phantom, recon_n, data_range=1.0))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(n_angles_list, psnr_clean, 'bo-', label='FBP (无噪声)')
ax1.plot(n_angles_list, psnr_noisy_curve, 'rs-', label=f'FBP+Shepp-Logan滤波 (σ={noise_sigma})')
ax1.set_xlabel('投影角度数')
ax1.set_ylabel('PSNR (dB)')
ax1.set_title('重建质量 vs 投影角度数\n角度越少→信息越少→不适定性越严重')
ax1.legend()
ax1.grid(True)

# 有限角度范围 vs 重建质量
limited_ranges = [30, 60, 90, 120, 150, 180]
psnr_limited = []
for angle_range in limited_ranges:
    theta = np.linspace(0, angle_range, min(angle_range, 180), endpoint=False)
    sino = radon(phantom, theta=theta, circle=True)
    recon = iradon(sino, theta=theta, circle=True, filter_name='ramp')
    psnr_limited.append(psnr(phantom, recon, data_range=1.0))

ax2.plot(limited_ranges, psnr_limited, 'gs-')
ax2.set_xlabel('角度覆盖范围 (°)')
ax2.set_ylabel('PSNR (dB)')
ax2.set_title('有限角CT：角度范围 vs 重建质量\n缺失角度→频域信息缺口→条纹伪影')
ax2.grid(True)

plt.suptitle('步骤6：CT不适定性的量化（16.2.1/16.2.2节）', fontsize=13)
plt.tight_layout()
plt.savefig('步骤6_角度vs质量.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n实验16.1完成！")
