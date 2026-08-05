import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import sys

# ====== 中文字体配置（兼容本地和 Google Colab）======
_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验1.4-1', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验1.4-1')
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _chinese_path)
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=_chinese_path)
# ========================================================

np.random.seed(42)

# ---- 1. 加载图像（使用 Shepp-Logan 幻影，含亮区和暗区）----
n = 256
# 最近邻插值（order=0）保留幻影的硬边缘，但锯齿会在噪声图中产生结构性残差
# 使得直方图尾部略微偏离纯理论分布
x = resize(shepp_logan_phantom(), (n, n), order=0, preserve_range=True, anti_aliasing=False)
x = x / x.max()  # 归一化到 [0, 1]

# ---- 2. 添加三种噪声 ----
# (a) 高斯噪声：ε ~ N(0, σ²I)，信号无关
sigma_gauss = 0.1
y_gauss = x + sigma_gauss * np.random.randn(n, n)
y_gauss = np.clip(y_gauss, 0, 1)

# (b) Poisson 噪声：y_i ~ Poisson((Ax)_i / gain) * gain，信号依赖
# 调节 gain 使 PSNR 与高斯噪声大致相当
gain = 0.01
y_poisson = np.random.poisson(x / gain).astype(np.float64) * gain
# 注意：clip 操作会截断高亮度区域的噪声，导致实测 σ 在高亮度处低于理论值
# 这是因为当 x 接近 1 时，λ = x/gain = 100，方差较大，部分样本会被 clip 到 1
y_poisson = np.clip(y_poisson, 0, 1)

# (c) 脉冲噪声（椒盐噪声）：随机像素被替换为 0 或 1
amount_sp = 0.05
y_sp = random_noise(x, mode='s&p', amount=amount_sp)

# ---- 3. 计算质量度量 ----
psnr_gauss = peak_signal_noise_ratio(x, y_gauss, data_range=1.0)
psnr_poisson = peak_signal_noise_ratio(x, y_poisson, data_range=1.0)
psnr_sp = peak_signal_noise_ratio(x, y_sp, data_range=1.0)

ssim_gauss = structural_similarity(x, y_gauss, data_range=1.0, win_size=7)
ssim_poisson = structural_similarity(x, y_poisson, data_range=1.0, win_size=7)
ssim_sp = structural_similarity(x, y_sp, data_range=1.0, win_size=7)

# ---- 4. 可视化 ----
fig, axes = plt.subplots(3, 3, figsize=(15, 14))
noisy_list = [y_gauss, y_poisson, y_sp]
names = ['高斯噪声', 'Poisson 噪声', '脉冲噪声']
psnrs = [psnr_gauss, psnr_poisson, psnr_sp]
ssims = [ssim_gauss, ssim_poisson, ssim_sp]

for i, (y, name) in enumerate(zip(noisy_list, names)):
    # 含噪图像
    axes[i, 0].imshow(y, cmap='gray', vmin=0, vmax=1)
    axes[i, 0].set_title(f'{name}\nPSNR={psnrs[i]:.1f}dB, SSIM={ssims[i]:.3f}')
    axes[i, 0].axis('off')

    # 噪声图像 = y - x
    noise = y - x
    # 使用完整范围显示噪声，避免截断导致的误解
    vmax = max(np.abs(noise).max(), 0.1)
    axes[i, 1].imshow(noise, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[i, 1].set_title(f'噪声分量 (y - x)\nmax|ε|={np.abs(noise).max():.3f}')
    axes[i, 1].axis('off')

    # 噪声直方图
    residual = (y - x).ravel()
    axes[i, 2].hist(residual, bins=100, density=True, alpha=0.7, color=f'C{i}')
    axes[i, 2].set_title(f'噪声直方图\nμ={residual.mean():.4f}, σ={residual.std():.4f}')
    axes[i, 2].set_xlabel('残差值')
    axes[i, 2].set_ylabel('概率密度')

plt.suptitle('三种噪声模型对比', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_4_1_噪声建模.png'), dpi=150, bbox_inches='tight')
plt.show()

# ---- 5. Poisson 噪声的信号依赖性验证 ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 将图像按亮度分桶，计算每桶的噪声标准差
n_bins = 20
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
noise_poisson = y_poisson - x
bin_stds = []
for j in range(n_bins):
    if j == n_bins - 1:
        mask_bin = (x >= bin_edges[j]) & (x <= bin_edges[j + 1])  # 最后一桶包含右端点 1.0
    else:
        mask_bin = (x >= bin_edges[j]) & (x < bin_edges[j + 1])
    if mask_bin.sum() > 10:
        bin_stds.append(noise_poisson[mask_bin].std())
    else:
        bin_stds.append(np.nan)

axes[0].plot(bin_centers, bin_stds, 'o-', label='Poisson 噪声实际 σ')
axes[0].plot(bin_centers, np.sqrt(bin_centers * gain), 's--', label='理论 √(signal·gain)', alpha=0.7)
axes[0].set_xlabel('信号亮度')
axes[0].set_ylabel('噪声标准差')
axes[0].set_title(f'Poisson 噪声：信号依赖性 (gain={gain})\nσ ∝ √(信号强度)')
axes[0].legend()
axes[0].grid(True)

noise_gauss_map = y_gauss - x
bin_stds_g = []
for j in range(n_bins):
    if j == n_bins - 1:
        mask_bin = (x >= bin_edges[j]) & (x <= bin_edges[j + 1])
    else:
        mask_bin = (x >= bin_edges[j]) & (x < bin_edges[j + 1])
    if mask_bin.sum() > 10:
        bin_stds_g.append(noise_gauss_map[mask_bin].std())
    else:
        bin_stds_g.append(np.nan)

axes[1].plot(bin_centers, bin_stds_g, 'o-', label='高斯噪声实际 σ', color='C0')
axes[1].axhline(y=sigma_gauss, color='C0', linestyle='--', alpha=0.7, label=f'理论 σ={sigma_gauss}')
axes[1].set_xlabel('信号亮度')
axes[1].set_ylabel('噪声标准差')
axes[1].set_title('高斯噪声：信号无关性\n(两端偏低源于 clip 截断，非信号依赖)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_4_1_噪声信号依赖性.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"{'噪声类型':>12s}  {'PSNR(dB)':>10s}  {'SSIM':>8s}")
print("-" * 36)
for name, psnr_val, ssim_val in zip(names, psnrs, ssims):
    print(f"{name:>12s}  {psnr_val:10.2f}  {ssim_val:8.4f}")

# ===== 保存数值结果 =====
import json
results_summary = {
    'image_size': n,
    'gaussian_noise': {
        'sigma': float(sigma_gauss),
        'psnr_dB': float(round(psnr_gauss, 2)),
        'ssim': float(round(ssim_gauss, 4)),
    },
    'poisson_noise': {
        'gain': float(gain),
        'psnr_dB': float(round(psnr_poisson, 2)),
        'ssim': float(round(ssim_poisson, 4)),
    },
    'impulse_noise': {
        'amount': float(amount_sp),
        'psnr_dB': float(round(psnr_sp, 2)),
        'ssim': float(round(ssim_sp, 4)),
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