import numpy as np
import matplotlib.pyplot as plt
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)
import os
import sys

# ====== 中文字体配置（兼容本地和 Google Colab）======
_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '附录1A', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '附录1A')
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _chinese_path)
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=_chinese_path)
# ========================================================

np.random.seed(42)


# ---- 1. 加载并预处理图像 ----
x_color = astronaut()
x = rgb2gray(x_color)  # 转为灰度，值域 [0, 1]

# ---- 2. 添加不同水平的高斯噪声 ----
# 注意：图像值域为 [0, 1]，σ=0.40 表示噪声幅度达到信号幅度的 40%，属于极强噪声
noise_levels = [0.01, 0.03, 0.05, 0.10, 0.20, 0.40]
print(f"图像值域: [{x.min():.2f}, {x.max():.2f}]，噪声标准差 σ 相对于信号幅度的比例:")
for sigma in noise_levels:
    print(f"  σ={sigma:.2f} → 噪声占信号幅度的 {sigma*100:.0f}%")

noisy_images = []
for sigma in noise_levels:
    y = x + sigma * np.random.randn(*x.shape)
    y = np.clip(y, 0, 1)
    noisy_images.append(y)

def safe_normalize(arr):
    """最小-最大归一化，范围过小时显式报错而非静默返回魔数。"""
    a = np.asarray(arr, dtype=float)
    r = a.max() - a.min()
    if r < 1e-10:
        raise ValueError("归一化范围过小（<1e-10），无法进行有意义的归一化")
    return (a - a.min()) / r

# ---- 3. 计算三种质量度量 ----
mse_vals, psnr_vals, ssim_vals = [], [], []
for y in noisy_images:
    mse_vals.append(mean_squared_error(x, y))
    psnr_vals.append(peak_signal_noise_ratio(x, y))
    ssim_vals.append(structural_similarity(x, y, data_range=1.0, win_size=7))

# ---- 4. 可视化 ----
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 第一行：原始图像 + 3 个噪声示例
# 注意：MSE=0 → PSNR=∞ 是理论极限，此处仅为字符串标注，未实际计算
# （PSNR 定义的分母为 0 时数学上无定义，仅作为"完美图像"的语义化标签）
axes[0, 0].imshow(x, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('原始图像\nPSNR=∞ dB, SSIM=1.000')
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
# 注意：MSE 越大图像越差，PSNR/SSIM 越大图像越好
# 为统一语义（值越高=质量越好），对 MSE 做反转：1 - 归一化MSE
mse_norm = safe_normalize(mse_vals)
mse_norm = 1 - mse_norm  # 反转后语义与 PSNR/SSIM 一致：越高越好
psnr_norm = safe_normalize(psnr_vals)
ssim_norm = safe_normalize(ssim_vals)
axes[1, 3].plot(noise_levels, mse_norm, 'o-', label='1 − 归一化 MSE')
axes[1, 3].plot(noise_levels, psnr_norm, 's-', label='归一化 PSNR')
axes[1, 3].plot(noise_levels, ssim_norm, '^-', label='归一化 SSIM')
axes[1, 3].set_xlabel('噪声标准差 σ')
axes[1, 3].set_ylabel('归一化度量值（越高越好）')
axes[1, 3].set_title('三种度量归一化对比\n（纵轴越高表示图像质量越好）')
axes[1, 3].legend()
axes[1, 3].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_a_1_质量评估.png'), dpi=150, bbox_inches='tight')
plt.show()

# ---- 5. 打印数值结果 ----
print(f"{'噪声σ':>8s}  {'MSE':>12s}  {'PSNR(dB)':>10s}  {'SSIM':>8s}")
print("-" * 42)
for i, sigma in enumerate(noise_levels):
    print(f"{sigma:8.2f}  {mse_vals[i]:12.6f}  {psnr_vals[i]:10.2f}  {ssim_vals[i]:8.4f}")