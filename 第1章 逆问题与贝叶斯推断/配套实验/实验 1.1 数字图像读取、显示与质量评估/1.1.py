import numpy as np
import matplotlib.pyplot as plt
from skimage.data import astronaut, shepp_logan_phantom
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)
from skimage.util import random_noise
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


# ---- 1. 加载并预处理图像 ----
x_color = astronaut()
x = rgb2gray(x_color)  # 转为灰度，值域 [0, 1]

# ---- 2. 添加不同水平的高斯噪声 ----
noise_levels = [0.01, 0.03, 0.05, 0.10, 0.20, 0.40]
noisy_images = []
for sigma in noise_levels:
    y = x + sigma * np.random.randn(*x.shape)
    y = np.clip(y, 0, 1)
    noisy_images.append(y)

# ---- 3. 计算三种质量度量 ----
mse_vals, psnr_vals, ssim_vals = [], [], []
for y in noisy_images:
    mse_vals.append(mean_squared_error(x, y))
    psnr_vals.append(peak_signal_noise_ratio(x, y))
    ssim_vals.append(structural_similarity(x, y, data_range=1.0))

# ---- 4. 可视化 ----
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 第一行：原始图像 + 3 个噪声示例
axes[0, 0].imshow(x, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

for i, idx in enumerate([0, 2, 5]):
    axes[0, i + 1].imshow(noisy_images[idx], cmap='gray', vmin=0, vmax=1)
    axes[0, i + 1].set_title(f'σ={noise_levels[idx]:.2f}\nPSNR={psnr_vals[idx]:.1f}dB, SSIM={ssim_vals[idx]:.3f}')
    axes[0, i + 1].axis('off')

# 第二行：度量随噪声水平的变化曲线

axes[1, 0].plot(noise_levels, mse_vals, 'o-')
axes[1, 0].set_xlabel('噪声标准差 σ')
axes[1, 0].set_ylabel('MSE')
axes[1, 0].set_title('MSE vs 噪声水平')
axes[1, 0].grid(True)

axes[1, 1].plot(noise_levels, psnr_vals, 's-', color='orange')
axes[1, 1].set_xlabel('噪声标准差 σ')
axes[1, 1].set_ylabel('PSNR (dB)')
axes[1, 1].set_title('PSNR vs 噪声水平')
axes[1, 1].grid(True)

axes[1, 2].plot(noise_levels, ssim_vals, '^-', color='green')
axes[1, 2].set_xlabel('噪声标准差 σ')
axes[1, 2].set_ylabel('SSIM')
axes[1, 2].set_title('SSIM vs 噪声水平')
axes[1, 2].grid(True)

# 对比三种度量的归一化趋势
mse_norm = (np.array(mse_vals) - min(mse_vals)) / (max(mse_vals) - min(mse_vals) + 1e-10)
psnr_norm = (np.array(psnr_vals) - min(psnr_vals)) / (max(psnr_vals) - min(psnr_vals) + 1e-10)
ssim_norm = (np.array(ssim_vals) - min(ssim_vals)) / (max(ssim_vals) - min(ssim_vals) + 1e-10)
axes[1, 3].plot(noise_levels, mse_norm, 'o-', label='MSE (归一化)')
axes[1, 3].plot(noise_levels, psnr_norm, 's-', label='PSNR (归一化)')
axes[1, 3].plot(noise_levels, ssim_norm, '^-', label='SSIM (归一化)')
axes[1, 3].set_xlabel('噪声标准差 σ')
axes[1, 3].set_ylabel('归一化度量值')
axes[1, 3].set_title('三种度量归一化对比')
axes[1, 3].legend()
axes[1, 3].grid(True)

plt.tight_layout()
plt.savefig('实验1_1_质量评估.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- 5. 打印数值结果 ----
print(f"{'σ':>6s}  {'MSE':>10s}  {'PSNR(dB)':>10s}  {'SSIM':>8s}")
print("-" * 42)
for i, sigma in enumerate(noise_levels):
    print(f"{sigma:6.2f}  {mse_vals[i]:10.6f}  {psnr_vals[i]:10.2f}  {ssim_vals[i]:8.4f}")