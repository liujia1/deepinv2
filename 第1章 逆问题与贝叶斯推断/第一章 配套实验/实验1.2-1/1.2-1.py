import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.data import astronaut
from skimage.color import rgb2gray
import sys
import os

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验1.2-1', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验1.2-1')
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _chinese_path)
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=_chinese_path)

np.random.seed(42)

# ---- 1. 加载图像 ----
x_color = astronaut()
x = rgb2gray(x_color)
n = x.shape[0]

# ---- 2. 构建高斯 PSF 和模糊算子 ----
def gaussian_psf(size, sigma, radius):
    """生成截断高斯 PSF，并按循环卷积约定放在数组原点 (0,0)。

    注意：核必须截断到有限支撑（此处 radius=3σ），不能铺满整个 512×512 网格。
    若在整个网格上按高斯取值，|H(k)| 的动态范围会一路衰减到 1e-160 并下溢为 0，
    使绝大多数频点落入"数值阻带"（Y 被浮点舍入误差支配），即便无噪声也谈不上精确可逆。
    """
    s = 2 * radius + 1
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    h = np.zeros((size, size))
    h[:s, :s] = k          # 峰在 (0,0)；数组末尾对应负偏移，符合 DFT 循环卷积约定
    return h

def blur_operator(x, h):
    """频域循环卷积模糊：y = Ax，A 由 PSF h 定义"""
    return np.real(np.fft.ifft2(np.fft.fft2(h) * np.fft.fft2(x)))

sigma_psf = 5.0
psf_radius = int(round(3 * sigma_psf))     # 截断半径 3σ，核尺寸 31×31
h = gaussian_psf(n, sigma_psf, psf_radius)

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
# 说明：为什么无噪声和含噪声需要不同的处理？
#
# 无噪声情况：Y(k) = H(k)·X(k) 是精确的频域关系
#   - 高斯核的 H(k) 处处非零（截断后 min|H| ≈ 2e-10，远大于浮点精度），
#     所以逐元素除法 X = Y/H 在数学上和数值上都成立，能恢复到机器精度（~1e-8）
#   - 即：无噪声时这一步不需要任何正则化
#
# 含噪声情况：Y(k) = H(k)·X(k) + N(k)
#   - 反卷积给出 X̂ = X + N(k)/H(k)：噪声不被压制，反而在每个频点被 1/|H(k)| 放大
#   - 高频处 |H(k)| 最小仅 ~2e-10，σ_n = 0.01 的白噪声被放大到 std ~7e5 量级，重建完全失败
#   - 解决方案：截断/稳定分母（如 Tikhonov）或维纳滤波，这正是 1.3 节正则化的入口

y_blur_noisy = y_blur + 0.01 * np.random.randn(*y_blur.shape)

H = np.fft.fft2(h)
# 无噪声：直接精确除法，不使用任何正则化项
x_deconv_clean = np.real(np.fft.ifft2(np.fft.fft2(y_blur) / H))

# 含噪直接反卷积：与左图算法完全相同，唯一的区别是数据里多了 σ_n=0.01 的噪声，
# 这里故意不加任何正则化，以展示"为什么需要正则化"
x_deconv_noisy = np.real(np.fft.ifft2(np.fft.fft2(y_blur_noisy) / H))

# ---- 7. 可视化 ----
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

axes[0, 0].imshow(x, cmap='gray')
axes[0, 0].set_title('原始图像 x')
axes[0, 0].axis('off')

axes[0, 1].imshow(y_blur, cmap='gray')
axes[0, 1].set_title('模糊图像 y = Ax\n(高斯PSF, σ=5)')
axes[0, 1].axis('off')

# 下采样图真实尺寸只有 128×128，这里不拉升、保持真实像素尺寸，
# 用 extent 把它居中放在 512×512 坐标系的正中，四周留白，
# 让它在其子图框内明显比其它 512×512 的子图小，直观体现"信息不可逆丢失"
half = y_ds.shape[0] / 2.0
cx = n / 2.0
axes[0, 2].imshow(y_ds, cmap='gray',
                  extent=[cx - half, cx + half, cx + half, cx - half],
                  aspect='equal')
axes[0, 2].set_xlim(0, n)
axes[0, 2].set_ylim(n, 0)          # imshow 原点在左上，故 y 轴反向
axes[0, 2].set_title('下采样图像 (4x)\nm < n, A不可逆')
axes[0, 2].set_xticks([])
axes[0, 2].set_yticks([])

axes[0, 3].imshow(y_mask, cmap='gray')
axes[0, 3].set_title('掩码图像 (50%丢失)\ny = Mx')
axes[0, 3].axis('off')

axes[1, 0].imshow(x_deconv_clean, cmap='gray')
axes[1, 0].set_title('无噪声反卷积 $F^{-1}(F(y)/F(h))$\n恢复到机器精度')
axes[1, 0].axis('off')

axes[1, 1].imshow(np.clip(x_deconv_noisy, 0, 1), cmap='gray')
axes[1, 1].set_title('含噪声反卷积\n噪声被 $1/H(k)$ 放大')
axes[1, 1].axis('off')

# 频谱图：展示 H 的高斯衰减。用 log10 尺度 + colorbar（线性尺度下有效通带会缩成一个小点，
# 看不出衰减层次；log 尺度能展示跨 10 个数量级的衰减，这正是 1/H 爆炸的根源）
logH = np.log10(np.abs(np.fft.fftshift(H)))
im = axes[1, 2].imshow(logH, cmap='hot', vmin=-11, vmax=0)
axes[1, 2].set_title('PSF 频谱 $\\log_{10}|H(k)|$\n高频趋零→$1/H$ 爆炸')
axes[1, 2].axis('off')
fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

# 掩码可视化：展示哪些位置的像素丢失了
axes[1, 3].imshow(mask.astype(float), cmap='gray')
axes[1, 3].set_title('掩码图案 (50%保留)\n白色=保留, 黑色=丢失')
axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_2_1_正向模型.png'), dpi=150, bbox_inches='tight')
plt.show()

# ===== 保存数值结果 =====
import json

# 相对误差保留足够位数：无噪声情形是机器精度量级（~1e-8），round(...,6) 会被压成 0
_rel_clean = float(np.linalg.norm(x_deconv_clean - x) / np.linalg.norm(x))
_rel_noisy = float(np.linalg.norm(np.clip(x_deconv_noisy, 0, 1) - x) / np.linalg.norm(x))

def _psnr(img):
    mse = float(np.mean((np.clip(img, 0, 1) - x) ** 2))
    return float('inf') if mse == 0 else float(10 * np.log10(1.0 / mse))

results_summary = {
    'image_size': n,
    'psf_sigma': float(sigma_psf),
    'psf_size': int(2 * psf_radius + 1),
    'psf_transfer_min_abs_H': float(np.abs(np.fft.fft2(h)).min()),
    'downsample_factor': 4,
    'mask_keep_ratio': 0.5,
    'noise_level_deconv': 0.01,
    'reconstruction_error_clean': _rel_clean,
    'psnr_clean_db': _psnr(x_deconv_clean),
    'reconstruction_error_noisy': _rel_noisy,
    'psnr_noisy_db': _psnr(x_deconv_noisy),
}

def _to_native(obj):
    import numpy as np
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return _to_native(obj.tolist())
    try:
        import torch
        if isinstance(obj, torch.Tensor): return _to_native(obj.detach().cpu().tolist())
    except: pass
    return obj

results_summary = {k: _to_native(v) for k, v in results_summary.items()}
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")