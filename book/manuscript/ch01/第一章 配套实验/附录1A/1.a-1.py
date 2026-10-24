import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    # 必须用绝对路径：相对路径依赖运行时工作目录，从脚本目录外启动会导入失败
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
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
noisy_images_raw = []
for sigma in noise_levels:
    y_raw = x + sigma * np.random.randn(*x.shape)
    noisy_images_raw.append(y_raw)          # 未裁剪，用于补充口径对照
    y = np.clip(y_raw, 0, 1)
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
    # 显式声明 data_range：图像值域为 [0,1]，峰值 MAX=1（与教材 PSNR 定义一致，避免依赖库按 dtype 推断）
    psnr_vals.append(peak_signal_noise_ratio(x, y, data_range=1.0))
    ssim_vals.append(structural_similarity(x, y, data_range=1.0, win_size=7))

# ---- 3b. 补充对照口径（主表仍用 clip + win7）----
# 1) 未裁剪口径：纯加性高斯噪声下的真实退化，用于说明 clip 会"美化"指标；
# 2) win_size=11（原文标准）：用于说明窗口参数对 SSIM 的影响；
# 3) 反相图 x→1-x 的 SSIM：验证 SSIM 可以取负值。
mse_raw_vals, psnr_raw_vals, ssim_win11_vals, frac_clipped_vals = [], [], [], []
for y_raw, y in zip(noisy_images_raw, noisy_images):
    mse_raw_vals.append(mean_squared_error(x, y_raw))
    psnr_raw_vals.append(peak_signal_noise_ratio(x, y_raw, data_range=1.0))
    ssim_win11_vals.append(structural_similarity(x, y, data_range=1.0, win_size=11))
    frac_clipped_vals.append(float(np.mean((y_raw < 0) | (y_raw > 1))))

ssim_inverted_win7 = structural_similarity(x, 1.0 - x, data_range=1.0, win_size=7)
ssim_inverted_win11 = structural_similarity(x, 1.0 - x, data_range=1.0, win_size=11)

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
axes[1, 3].plot(noise_levels, mse_norm, 'o-', label='$1-$归一化 MSE')  # U+2212 在 SimHei 缺字会渲染成豆腐块，改用 mathtext
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

# ===== 保存数值结果 =====
import json
results_summary = {
    'noise_levels': noise_levels,
    'metrics_by_noise': {f'sigma_{sigma:.2f}': {
        'mse': float(round(mse_vals[i], 6)),
        'psnr_dB': float(round(psnr_vals[i], 2)),
        'ssim': float(round(ssim_vals[i], 4)),
    } for i, sigma in enumerate(noise_levels)},
    'supplementary': {
        'note': '主表口径为 clip 到 [0,1] + SSIM win_size=7；本块给出对照口径',
        'uncropped_by_noise': {f'sigma_{sigma:.2f}': {
            'mse_raw': float(round(mse_raw_vals[i], 6)),
            'psnr_raw_dB': float(round(psnr_raw_vals[i], 2)),
            'frac_pixels_clipped': float(round(frac_clipped_vals[i], 4)),
        } for i, sigma in enumerate(noise_levels)},
        'ssim_win11_by_noise': {f'sigma_{sigma:.2f}': float(round(ssim_win11_vals[i], 4))
                                for i, sigma in enumerate(noise_levels)},
        'inverted_image_ssim': {
            'definition': 'SSIM(1-x, x)，data_range=1.0，用于验证 SSIM 可取负值',
            'win_size_7': float(round(ssim_inverted_win7, 4)),
            'win_size_11': float(round(ssim_inverted_win11, 4)),
        },
    },
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