import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize
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
from matplotlib.font_manager import FontManager

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

# ---- 1. 准备幻影图像 ----
n = 128
phantom = resize(shepp_logan_phantom(), (n, n), order=0, preserve_range=True, anti_aliasing=False)
phantom = phantom / phantom.max()

# ---- 2. 三种角度配置的 Radon 变换 ----
# (a) 完整角度：180个均匀分布
theta_full = np.linspace(0, 180, 180, endpoint=False)
# (b) 稀疏角度：30个均匀分布（sparse-view）
theta_sparse = np.linspace(0, 180, 30, endpoint=False)
# (c) 有限角度：仅0-60度范围（limited-angle / missing wedge）
theta_limited = np.linspace(0, 60, 60, endpoint=False)

sinogram_full = radon(phantom, theta=theta_full, circle=True)
sinogram_sparse = radon(phantom, theta=theta_sparse, circle=True)
sinogram_limited = radon(phantom, theta=theta_limited, circle=True)

# ---- 3. FBP 重建 ----
recon_full = iradon(sinogram_full, theta=theta_full, circle=True, filter_name='ramp')
recon_sparse = iradon(sinogram_sparse, theta=theta_sparse, circle=True, filter_name='ramp')
recon_limited = iradon(sinogram_limited, theta=theta_limited, circle=True, filter_name='ramp')

# 全局归一化：使用真值最大值作为参考，保持物理意义的强度比例
global_max = phantom.max()
recon_full = np.clip(recon_full / global_max, 0, 1)
recon_sparse = np.clip(recon_sparse / global_max, 0, 1)
recon_limited = np.clip(recon_limited / global_max, 0, 1)

# ---- 4. 质量评估 ----
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

psnr_full = peak_signal_noise_ratio(phantom, recon_full, data_range=1.0)
psnr_sparse = peak_signal_noise_ratio(phantom, recon_sparse, data_range=1.0)
psnr_limited = peak_signal_noise_ratio(phantom, recon_limited, data_range=1.0)
ssim_full = structural_similarity(phantom, recon_full, data_range=1.0)
ssim_sparse = structural_similarity(phantom, recon_sparse, data_range=1.0)
ssim_limited = structural_similarity(phantom, recon_limited, data_range=1.0)

# ---- 5. 频域采样可视化（傅里叶切片定理）----
def plot_fourier_sampling(ax, theta, title, highlight_missing=False, missing_range=None):
    """
    可视化Radon变换的频域采样模式（傅里叶切片定理）
    
    傅里叶切片定理：每个投影对应频域中的一条完整直线（双向）
    - 投影角度θ对应频域直线：(ωcosθ, ωsinθ)，ω∈[-∞, +∞]
    - 由于θ≡θ+180°对称性，每条直线覆盖两个"相反"的投影方向
    
    三种采样模式：
    - 完整角度：频域被完整采样（圆形覆盖）
    - 稀疏角度：频域呈"星形"采样，存在大量缺失区域
    - 有限角度：频域呈"楔形"缺失（missing wedge）
    
    缺失的频域区域 = 不可观测空间（null space）
    这解释了为什么会产生条纹伪影及其方向性
    """
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    
    # 绘制频域边界圆
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', linewidth=1)
    ax.add_patch(circle)
    
    # 如果需要高亮缺失区域（limited-angle）
    if highlight_missing and missing_range is not None:
        # 绘制missing wedge区域（红色扇形）
        # 注意：由于θ≡θ+180°对称性，缺失区域是对称的两个楔形
        # limited-angle只覆盖某些方向，缺失的是其"法向量"方向
        # Wedge参数: (center, r, theta1, theta2) 从theta1逆时针画到theta2
        # missing_range=(60,180) 表示缺失60°到180°的区域
        wedge1 = plt.matplotlib.patches.Wedge((0, 0), 1, missing_range[0], missing_range[1], 
                                               alpha=0.3, color='red', label='缺失区域')
        wedge2 = plt.matplotlib.patches.Wedge((0, 0), 1, missing_range[0]+180, missing_range[1]+180, 
                                               alpha=0.3, color='red')
        ax.add_patch(wedge1)
        ax.add_patch(wedge2)
    
    # 绘制每个角度的频域采样线（双向直线，从-1到1）
    for angle in theta:
        angle_rad = np.deg2rad(angle)
        # 双向直线：从边界到边界，穿过圆心
        x = [-np.cos(angle_rad), np.cos(angle_rad)]
        y = [-np.sin(angle_rad), np.sin(angle_rad)]
        ax.plot(x, y, 'b-', linewidth=0.8, alpha=0.7)
    
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('频域 u')
    ax.set_ylabel('频域 v')
    ax.grid(True, alpha=0.3)

# ---- 6. 可视化 ----
fig, axes = plt.subplots(3, 4, figsize=(18, 13))

# 第一行：正弦图
axes[0, 0].imshow(phantom, cmap='gray')
axes[0, 0].set_title('原始幻影 x')
axes[0, 0].axis('off')

axes[0, 1].imshow(sinogram_full, cmap='gray', aspect='auto',
                   extent=[0, 180, sinogram_full.shape[0], 0])
axes[0, 1].set_title(f'正弦图 (完整角度)\n{sinogram_full.shape[1]} 次投影')
axes[0, 1].set_xlabel('投影角度 θ (°)')
axes[0, 1].set_ylabel('探测器位置 t')

axes[0, 2].imshow(sinogram_sparse, cmap='gray', aspect='auto',
                   extent=[0, 180, sinogram_sparse.shape[0], 0])
axes[0, 2].set_title(f'正弦图 (稀疏角度)\n{sinogram_sparse.shape[1]} 次投影')
axes[0, 2].set_xlabel('投影角度 θ (°)')
axes[0, 2].set_ylabel('探测器位置 t')

axes[0, 3].imshow(sinogram_limited, cmap='gray', aspect='auto',
                   extent=[0, 60, sinogram_limited.shape[0], 0])
axes[0, 3].set_title(f'正弦图 (有限角度)\n{sinogram_limited.shape[1]} 次投影 (0-60°)')
axes[0, 3].set_xlabel('投影角度 θ (°)')
axes[0, 3].set_ylabel('探测器位置 t')

# 第二行：频域采样模式
plot_fourier_sampling(axes[1, 0], theta_full, 
                       f'频域采样 (完整)\n圆形覆盖')
plot_fourier_sampling(axes[1, 1], theta_sparse, 
                       f'频域采样 (稀疏)\n星形缺失')
plot_fourier_sampling(axes[1, 2], theta_limited, 
                       f'频域采样 (有限)\n楔形缺失',
                       highlight_missing=True, missing_range=(60, 180))

# 频域缺失与不可观测空间说明
axes[1, 3].text(0.5, 0.88, '傅里叶切片定理', fontsize=11, fontweight='bold', 
                ha='center', transform=axes[1, 3].transAxes)
axes[1, 3].text(0.5, 0.73, '每个投影 = 频域一条双向直线', fontsize=9, 
                ha='center', transform=axes[1, 3].transAxes)
axes[1, 3].text(0.5, 0.58, 'θ≡θ+180°对称性', fontsize=9, 
                ha='center', transform=axes[1, 3].transAxes)
axes[1, 3].text(0.5, 0.43, '缺失区域 = 不可观测空间', fontsize=9, 
                ha='center', transform=axes[1, 3].transAxes, color='red')
axes[1, 3].text(0.5, 0.28, '稀疏角度 → 星形缺失 → 各方向条纹', fontsize=9, 
                ha='center', transform=axes[1, 3].transAxes)
axes[1, 3].text(0.5, 0.13, '有限角度 → 楔形缺失 → 方向性条纹', fontsize=9, 
                ha='center', transform=axes[1, 3].transAxes)
axes[1, 3].axis('off')

# 第三行：FBP重建结果
axes[2, 0].imshow(recon_full, cmap='gray')
axes[2, 0].set_title(f'FBP重建 (完整)\nPSNR={psnr_full:.1f}dB, SSIM={ssim_full:.3f}')
axes[2, 0].axis('off')

axes[2, 1].imshow(recon_sparse, cmap='gray')
axes[2, 1].set_title(f'FBP重建 (稀疏)\nPSNR={psnr_sparse:.1f}dB, SSIM={ssim_sparse:.3f}')
axes[2, 1].axis('off')

axes[2, 2].imshow(recon_limited, cmap='gray')
axes[2, 2].set_title(f'FBP重建 (有限)\nPSNR={psnr_limited:.1f}dB, SSIM={ssim_limited:.3f}')
axes[2, 2].axis('off')

# 误差对比：同时显示sparse和limited-angle
error_sparse = np.abs(recon_sparse - phantom)
error_limited = np.abs(recon_limited - phantom)

# 创建拼接的误差对比图
error_combined = np.zeros((n, n*2 + 10))  # 中间留10像素间隔
error_combined[:, :n] = error_sparse
error_combined[:, n+10:] = error_limited

axes[2, 3].imshow(error_combined, cmap='hot', extent=[0, 2, n, 0])
axes[2, 3].axvline(x=1, color='white', linewidth=2, linestyle='--')
# 使用 transAxes，x 坐标控制在 [0,1] 内，y 放在图格下方，避免被裁剪
axes[2, 3].text(0.25, -0.05, '稀疏角度误差', fontsize=9, ha='center', transform=axes[2, 3].transAxes)
axes[2, 3].text(0.25, -0.10, '(全方向条纹)', fontsize=8, ha='center', color='gray', transform=axes[2, 3].transAxes)
axes[2, 3].text(0.75, -0.05, '有限角度误差', fontsize=9, ha='center', transform=axes[2, 3].transAxes)
axes[2, 3].text(0.75, -0.10, '(方向性拉伸)', fontsize=8, ha='center', color='gray', transform=axes[2, 3].transAxes)
axes[2, 3].set_title('误差对比：条纹伪影类型差异', fontsize=10)
axes[2, 3].axis('off')

plt.suptitle('Radon变换：三种角度配置的频域采样与重建对比\n（缺失区域 = 不可观测空间 → 条纹伪影）', fontsize=14)
plt.tight_layout()
plt.savefig('实验1_4_Radon变换.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- 7. 打印数值结果 ----
print("="*60)
print("三种角度配置的重建质量对比")
print("="*60)
print(f"{'配置':<12} {'投影数':<8} {'PSNR(dB)':<10} {'SSIM':<8}")
print("-"*60)
print(f"{'完整角度':<12} {len(theta_full):<8} {psnr_full:<10.2f} {ssim_full:<8.4f}")
print(f"{'稀疏角度':<12} {len(theta_sparse):<8} {psnr_sparse:<10.2f} {ssim_sparse:<8.4f}")
print(f"{'有限角度':<12} {len(theta_limited):<8} {psnr_limited:<10.2f} {ssim_limited:<8.4f}")
print("="*60)

# ---- 8. 角度数量 vs 重建质量曲线 ----
n_angles_list = [10, 15, 20, 30, 45, 60, 90, 120, 180]
psnr_curve = []
for na in n_angles_list:
    theta = np.linspace(0, 180, na, endpoint=False)
    sino = radon(phantom, theta=theta, circle=True)
    recon = iradon(sino, theta=theta, circle=True, filter_name='ramp')
    recon = np.clip(recon / global_max, 0, 1)
    psnr_curve.append(peak_signal_noise_ratio(phantom, recon, data_range=1.0))

plt.figure(figsize=(8, 4))
plt.plot(n_angles_list, psnr_curve, 'o-')
plt.xlabel('投影角度数')
plt.ylabel('FBP 重建 PSNR (dB)')
plt.title('重建质量 vs 投影角度数\n（角度越少→信息越少→不适定性越严重）')
plt.ylim(bottom=0)
plt.grid(True)
plt.savefig('实验1_4_角度vs质量.png', dpi=150, bbox_inches='tight')
plt.show()
