import numpy as np
import matplotlib.pyplot as plt
from skimage.data import astronaut
from skimage.color import rgb2gray
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

# ---- 1. 加载图像 ----
x_color = astronaut()
x = rgb2gray(x_color)
n = x.shape[0]

# ---- 2. 构建高斯 PSF 和模糊算子 ----
def gaussian_psf(size, sigma):
    """生成频域中心的高斯 PSF"""
    ax = np.concatenate((np.arange(0, size // 2), np.arange(-size // 2, 0)))
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    h = h / h.sum()
    return h

def blur_operator(x, h):
    """频域卷积模糊：y = Ax，A 由 PSF h 定义"""
    return np.real(np.fft.ifft2(np.fft.fft2(h) * np.fft.fft2(x)))

def blur_adjoint(y, h):
    """模糊算子的伴随：A^T y"""
    return np.real(np.fft.ifft2(np.conj(np.fft.fft2(h)) * np.fft.fft2(y)))

sigma_psf = 5.0
h = gaussian_psf(n, sigma_psf)

# ---- 3. 下采样算子 ----
def downsample_average(x, factor):
    """平均下采样：将 factor x factor 的块取平均"""
    h, w = x.shape
    h_new, w_new = h // factor, w // factor
    x_reshaped = x[:h_new * factor, :w_new * factor].reshape(h_new, factor, w_new, factor)
    return x_reshaped.mean(axis=(1, 3))

# ---- 4. 掩码算子 ----
def apply_mask(x, keep_ratio=0.5, seed=42):
    """随机掩码：保留 keep_ratio 比例的像素"""
    rng = np.random.RandomState(seed)
    mask = rng.rand(*x.shape) < keep_ratio
    y = x * mask
    return y, mask

# ---- 5. 施加三种退化 ----
y_blur = blur_operator(x, h)
y_ds = downsample_average(x, factor=4)
y_mask, mask = apply_mask(x, keep_ratio=0.5)

# ---- 6. DFT 域直接反卷积（去模糊）----
# 教学说明：为什么无噪声和含噪声需要不同的正则化策略？
#
# 无噪声情况：y_blur = H * X（精确的频域关系）
#   - 即使 |H(k)| ≈ 0（高频衰减），分子 F(y_blur)(k) 也同时趋近于 0
#   - 所以 0/0 的极限有意义，添加 1e-15 仅为了避免除以零错误
#   - 结果：反卷积能完美恢复原始图像
#
# 含噪声情况：y_blur_noisy = H * X + noise
#   - 在高频处：|H(k)| ≈ 0，但 noise(k) 不为零
#   - 导致 noise(k) / (H(k) + ε) ≈ noise(k) / ε → ∞ （如果 ε 太小）
#   - 结果：噪声被极度放大，重建失败
#   - 解决方案：增大分母稳定项或使用 Wiener 滤波

y_blur_noisy = y_blur + 0.01 * np.random.randn(*y_blur.shape)

# 无噪声直接反卷积：分母极小也没问题，因为 y_blur 本身就在 H 的值域内
H = np.fft.fft2(h)
x_deconv_clean = np.real(np.fft.ifft2(np.fft.fft2(y_blur) / (H + 1e-15)))

# 含噪直接反卷积：同样的分母会让高频噪声被 1/H 极度放大
# 这里故意使用相同的 1e-15，以展示"为什么需要正则化"
x_deconv_noisy = np.real(np.fft.ifft2(np.fft.fft2(y_blur_noisy) / (H + 1e-15)))

# ---- 7. 可视化 ----
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

axes[0, 0].imshow(x, cmap='gray')
axes[0, 0].set_title('原始图像 x')
axes[0, 0].axis('off')

axes[0, 1].imshow(y_blur, cmap='gray')
axes[0, 1].set_title('模糊图像 y = Ax\n(高斯PSF, σ=5)')
axes[0, 1].axis('off')

axes[0, 2].imshow(y_ds, cmap='gray')
axes[0, 2].set_title('下采样图像 (4x)\nm < n, A不可逆')
axes[0, 2].axis('off')

axes[0, 3].imshow(y_mask, cmap='gray')
axes[0, 3].set_title('掩码图像 (50%丢失)\ny = Mx')
axes[0, 3].axis('off')

axes[1, 0].imshow(x_deconv_clean, cmap='gray')
axes[1, 0].set_title('无噪声反卷积\nF⁻¹(F(y)/F(h)) ✓')
axes[1, 0].axis('off')

axes[1, 1].imshow(np.clip(x_deconv_noisy, 0, 1), cmap='gray')
axes[1, 1].set_title('含噪声反卷积\n噪声被1/H(k)放大 ✗')
axes[1, 1].axis('off')

# 频谱图：展示 H 的衰减
axes[1, 2].imshow(np.log10(np.abs(np.fft.fftshift(H)) + 1e-15), cmap='hot')
axes[1, 2].set_title('PSF 频谱 |H(k)|\n高频趋零→1/H爆炸')
axes[1, 2].axis('off')

# 掩码可视化：展示哪些位置的像素丢失了（教学意义更强）
axes[1, 3].imshow(mask.astype(float), cmap='gray')
axes[1, 3].set_title('掩码图案 (50%保留)\n白色=保留, 黑色=丢失')
axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig('实验1_2_正向模型.png', dpi=150, bbox_inches='tight')
plt.show()