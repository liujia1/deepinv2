import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
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
    import os
    import re
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

# ---- 1. 准备图像和模糊算子 ----
n = 64
phantom = resize(shepp_logan_phantom(), (n, n), order=3, preserve_range=True, anti_aliasing=True)
x = phantom / phantom.max()

def gaussian_psf(size, sigma):
    ax = np.concatenate((np.arange(0, size // 2), np.arange(-size // 2, 0)))
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return h / h.sum()

def blur(x, h):
    return np.real(np.fft.ifft2(np.fft.fft2(h) * np.fft.fft2(x)))

h = gaussian_psf(n, sigma=3.0)

# ---- 2. 构造"幽灵"图像 ----
# 高频振荡图案落在模糊算子的近似零空间中
# 物理解释：高斯模糊在傅里叶域是低通滤波器，高频分量被强烈衰减
# - 高频分量对应模糊算子的小奇异值方向（与实验 1.7 呼应）
# - 频率越高 → 傅里叶域衰减越强 → 越接近零空间

xx, yy = np.meshgrid(np.arange(n), np.arange(n))

# 扫描不同频率，展示衰减规律
frequencies = [5, 10, 15, 20, 25, 30, 32]
attenuations = []

for freq in frequencies:
    d_test = np.sin(2 * np.pi * freq * xx / n)
    Ad_test = blur(d_test, h)
    attenuation = np.linalg.norm(Ad_test) / np.linalg.norm(d_test)
    attenuations.append(attenuation)

# 使用频率 15 作为幽灵分量（中等衰减，视觉效果明显）
freq = 15
d = 0.5 * np.sin(2 * np.pi * freq * xx / n)  # 纯水平频率

# 构造"幽灵"图像
x_prime = x + d * x.max()

# ---- 3. 验证"几乎幽灵"性质 ----
Ax = blur(x, h)
Ax_prime = blur(x_prime, h)
Ad = blur(d * x.max(), h)

rel_meas_err = np.linalg.norm(Ad) / np.linalg.norm(Ax)
rel_img_err = np.linalg.norm(d * x.max()) / np.linalg.norm(x)

# ---- 4. 可视化 ----
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

axes[0, 0].imshow(x, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('图像 x')
axes[0, 0].axis('off')

axes[0, 1].imshow(x_prime, cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title(f"图像 x'（x + 幽灵分量）\n‖x-x'‖/‖x‖ = {rel_img_err:.1%}")
axes[0, 1].axis('off')

axes[0, 2].imshow(d * x.max(), cmap='RdBu_r', vmin=-0.3, vmax=0.3)
axes[0, 2].set_title('差异 d = x\' - x\n（高频分量，落在近似零空间）')
axes[0, 2].axis('off')

axes[0, 3].plot(frequencies, attenuations, 'o-', linewidth=2, markersize=8)
axes[0, 3].axvline(freq, color='red', linestyle='--', alpha=0.8,
                   label=f'选用频率={freq}')
axes[0, 3].set_xlabel('空间频率')
axes[0, 3].set_ylabel('衰减因子 ‖Ad‖/‖d‖')
axes[0, 3].set_title('频率-衰减规律\n频率越高→越接近零空间')
axes[0, 3].legend(fontsize=9)
axes[0, 3].grid(True)

ax_vmin, ax_vmax = Ax.min(), Ax.max()
axes[1, 0].imshow(Ax, cmap='gray', vmin=ax_vmin, vmax=ax_vmax)
axes[1, 0].set_title('Ax（x 的观测）')
axes[1, 0].axis('off')

axes[1, 1].imshow(Ax_prime, cmap='gray', vmin=ax_vmin, vmax=ax_vmax)
axes[1, 1].set_title("Ax'（x' 的观测）")
axes[1, 1].axis('off')

Ax_diff = Ax_prime - Ax
vmax_diff = max(np.abs(Ax_diff).max(), 1e-6)
axes[1, 2].imshow(Ax_diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
axes[1, 2].set_title(f"Ax' - Ax（观测差异）\n‖Ax'-Ax‖/‖Ax‖ = {rel_meas_err:.3%}\n（colorscale max={vmax_diff:.2e}）")
axes[1, 2].axis('off')

axes[1, 3].text(0.5, 0.5, 
    f'验证线性性:\n‖(Ax\'−Ax) − Ad‖\n= {np.linalg.norm(Ax_diff - Ad):.2e}',
    ha='center', va='center', fontsize=13, transform=axes[1, 3].transAxes)
axes[1, 3].set_title('线性性验证')
axes[1, 3].axis('off')

plt.suptitle('"几乎幽灵"现象\n两个视觉截然不同的图像，其观测几乎相同', fontsize=14)
plt.tight_layout()
plt.savefig('实验1_8_几乎幽灵.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n=== 几乎幽灵验证 ===")
print(f"图像相对差异: ‖x - x'‖/‖x‖ = {rel_img_err:.1%}")
print(f"测量相对差异: ‖Ax - Ax'‖/‖Ax‖ = {rel_meas_err:.3%}")
print(f"放大因子: 图像差异/测量差异 = {rel_img_err / rel_meas_err:.0f} 倍")
print("\n=== 物理解释 ===")
print("高斯模糊在傅里叶域是低通滤波器，高频分量被强烈衰减")
print("高频幽灵分量对应模糊算子的小奇异值方向（与实验 1.7 呼应）")
print("→ 算子 A 抹去了幽灵方向上的信息，不适定性是问题本身的内在性质！")
print("\n=== 频率 vs 衰减 ===")
for f, a in zip(frequencies, attenuations):
    print(f"频率 {f:2d}: 衰减因子 = {a:.4f}")
