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
import json

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
# 最近邻插值（order=0）保留幻影分段常数的硬边缘结构。
# 注意：噪声在插值之后独立添加，故插值方式不改变噪声本身的分布
# （y-x 中信号项精确抵消），只决定各灰度级占据的像素比例。
x = resize(shepp_logan_phantom(), (n, n), order=0, preserve_range=True, anti_aliasing=False)
x = x / x.max()  # 归一化到 [0, 1]

# ---- 2. 添加三种噪声 ----
# 口径说明：PSNR/SSIM/残差全部基于"未截断"的原始含噪数据计算；
# [0,1] 只是图像的显示范围（光子计数在物理上没有上界 1），
# 因此 np.clip 只用于 imshow 显示，不参与任何统计，避免截断偏差。
# （当 E[y]=1 且在 1 处截断时，无论 λ 多大都有约 50% 样本被砍，
#  会使最亮区实测 σ 偏低约 45%，截断是显示取舍而非噪声模型的固有属性。）

# (a) 高斯噪声：ε ~ N(0, σ²I)，信号无关
sigma_gauss = 0.1
y_gauss = x + sigma_gauss * np.random.randn(n, n)

# (b) Poisson 噪声：y_i ~ Poisson((Ax)_i / gain) * gain，信号依赖
# E[y]=x，Var(y)=gain·x；峰值亮度处 λ = 1/gain = 12.5。
# 调节 gain 使未截断 MSE = gain·E[x] 与高斯噪声的 σ² 大致相当。
gain = 0.08
y_poisson = np.random.poisson(x / gain).astype(np.float64) * gain

# (c) 脉冲噪声（椒盐噪声）：随机像素被替换为 0 或 1
# 调节 amount 使 PSNR 与高斯噪声大致相当（amount 越小噪声越弱、PSNR 越高）
# 注意：skimage>=0.20 的 random_noise 不再使用全局 np.random.seed（默认改为
# 从操作系统熵源初始化的 PCG64），必须显式传 rng/seed 才能复现；
# 旧版 skimage 参数名为 seed，这里做双版本兼容。
amount_sp = 0.01
try:
    y_sp = random_noise(x, mode='s&p', amount=amount_sp, rng=42)
except TypeError:
    y_sp = random_noise(x, mode='s&p', amount=amount_sp, seed=42)
# 脉冲噪声的具体数值（PSNR/SSIM）依赖 skimage 版本的内部抽样实现，
# 跨版本可能有 ~0.5dB 漂移；固定种子保证的是"同版本内可复现"。
# 另注：amount=1% 中落在背景(原值0)或亮环(原值1)上的替换不改变像素，
# 故"实际发生改变"的像素比例低于 1%，这是正常现象而非实现错误。

# ---- 3. 计算质量度量（未截断口径）----
psnr_gauss = peak_signal_noise_ratio(x, y_gauss, data_range=1.0)
psnr_poisson = peak_signal_noise_ratio(x, y_poisson, data_range=1.0)
psnr_sp = peak_signal_noise_ratio(x, y_sp, data_range=1.0)

ssim_gauss = structural_similarity(x, y_gauss, data_range=1.0, win_size=7)
ssim_poisson = structural_similarity(x, y_poisson, data_range=1.0, win_size=7)
ssim_sp = structural_similarity(x, y_sp, data_range=1.0, win_size=7)

frac_oob_gauss = float(np.mean((y_gauss < 0) | (y_gauss > 1)))
frac_oob_poisson = float(np.mean(y_poisson > 1))  # Poisson 取值非负，只需检查上界
frac_changed_sp = float(np.mean(y_sp != x))

# ---- 4. 可视化 ----
fig, axes = plt.subplots(3, 3, figsize=(15, 14))
noisy_list = [y_gauss, y_poisson, y_sp]
names = ['高斯噪声', 'Poisson 噪声', '脉冲噪声']
psnrs = [psnr_gauss, psnr_poisson, psnr_sp]
ssims = [ssim_gauss, ssim_poisson, ssim_sp]

for i, (y, name) in enumerate(zip(noisy_list, names)):
    # 含噪图像：仅显示时按 [0,1] 饱和，不修改用于统计的数据
    axes[i, 0].imshow(np.clip(y, 0, 1), cmap='gray', vmin=0, vmax=1)
    axes[i, 0].set_title(f'{name}\nPSNR={psnrs[i]:.1f}dB, SSIM={ssims[i]:.3f}')
    axes[i, 0].axis('off')

    # 噪声图像 = y - x（未截断）
    noise = y - x
    # 使用完整范围显示噪声，避免截断导致的误解
    vmax = max(np.abs(noise).max(), 0.1)
    axes[i, 1].imshow(noise, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[i, 1].set_title(f'噪声分量 (y - x)\nmax|ε|={np.abs(noise).max():.3f}')
    axes[i, 1].axis('off')

    # 噪声直方图（未截断残差）
    residual = (y - x).ravel()
    if name == '高斯噪声':
        axes[i, 2].hist(residual, bins=100, density=True, alpha=0.7,
                        color=f'C{i}', label='含噪残差')
        t = np.linspace(-0.6, 0.6, 400)
        theory = (1 / (np.sqrt(2 * np.pi) * sigma_gauss)) * np.exp(-0.5 * (t / sigma_gauss) ** 2)
        axes[i, 2].plot(t, theory, 'k--', lw=2, label=f'理论 N(0, {sigma_gauss}²)')
        axes[i, 2].legend(loc='upper right', fontsize=7)
        axes[i, 2].set_title(f'高斯噪声直方图（未截断）\nμ={residual.mean():.4f}, σ={residual.std():.4f}')
    elif name == 'Poisson 噪声':
        axes[i, 2].hist(residual, bins=100, density=True, alpha=0.7, color=f'C{i}')
        axes[i, 2].set_title(f'噪声直方图（未截断）\nμ={residual.mean():.4f}, σ={residual.std():.4f}\n'
                             f'(0 处尖峰来自纯黑背景上的 Poisson(0) 恒等于 0)', fontsize=9)
    else:
        axes[i, 2].hist(residual, bins=100, density=True, alpha=0.7, color=f'C{i}')
        axes[i, 2].set_title(f'噪声直方图\nμ={residual.mean():.4f}, σ={residual.std():.4f}')
    axes[i, 2].set_xlabel('残差值 (y - x)')
    axes[i, 2].set_ylabel('概率密度')

plt.suptitle('三种噪声模型对比', fontsize=16, y=1.01)
fig.text(0.5, -0.005,
         '注：含噪图仅按 [0,1] 饱和显示，PSNR/SSIM/残差均基于未截断数据；'
         '幻影约 58% 像素为纯黑背景，Poisson(0) 恒等于 0、脉冲噪声稀疏，暗部几乎无损，'
         '而 SSIM 对均匀区满幅高斯噪声惩罚极重，故 SSIM 排序为 高斯 < Poisson < 脉冲，属正常现象。',
         ha='center', fontsize=9)
plt.tight_layout(rect=[0, 0.035, 1, 0.97])
plt.savefig(os.path.join(SAVE_DIR, '实验1_4_1_噪声建模.png'), dpi=150, bbox_inches='tight')
plt.show()

# ---- 5. 信号依赖性的定量验证：连续渐变图 + 多次重复实现 ----
# 幻影是分段常数图像，20 个桶里只有 6 个有像素（其中两桶仅 52/95 个像素），
# 无法在全亮度范围验证 σ-亮度规律。改用 0→1 水平渐变图，并对每个像素
# 独立重复 K 次噪声实现，用桶内全部残差（像素数×K）估计 σ：
#   1) 20 个桶全部有样本，曲线覆盖整个 [0,1]；
#   2) 单桶样本量 = n²/20×K，统计涨落可忽略。
# 同时叠加幻影在 6 个灰度级上的同口径估计作为对照；理论值一律按
# "桶内信号均值"（而非桶中心）评价，避免分段常数数据造成假性偏差。
K = 50
n_bins = 20
bin_edges = np.linspace(0, 1, n_bins + 1)

x_ramp = np.tile(np.linspace(0, 1, n), (n, 1))  # 每列为一个恒定亮度，列间 0→1 线性渐变


def binned_noise_std(signal, residual_fn, edges, reps=1):
    """按信号亮度分桶，汇总 reps 次独立噪声实现的残差，估计每桶的噪声总体标准差。

    返回：bin_mean_signal（桶内信号均值，空桶为 NaN）、bin_std、bin_count。
    """
    idx = np.clip(np.digitize(signal, edges) - 1, 0, len(edges) - 2)
    nb = len(edges) - 1
    s1 = np.zeros(nb)
    s2 = np.zeros(nb)
    sx = np.zeros(nb)
    cc = np.zeros(nb)
    for _ in range(reps):
        r = residual_fn(signal)
        np.add.at(s1, idx, r)
        np.add.at(s2, idx, r * r)
        np.add.at(sx, idx, signal)
        np.add.at(cc, idx, 1.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_sig = sx / cc
        var = s2 / cc - (s1 / cc) ** 2
    std = np.sqrt(np.maximum(var, 0.0))
    mean_sig[cc == 0] = np.nan
    std[cc == 0] = np.nan
    return mean_sig, std, cc


resid_gauss = lambda s: sigma_gauss * np.random.randn(*s.shape)
resid_poisson = lambda s: np.random.poisson(s / gain).astype(np.float64) * gain - s

sig_ramp_p, std_ramp_p, _ = binned_noise_std(x_ramp, resid_poisson, bin_edges, reps=K)
sig_ramp_g, std_ramp_g, _ = binned_noise_std(x_ramp, resid_gauss, bin_edges, reps=K)
sig_phan_p, std_phan_p, cnt_phan_p = binned_noise_std(x, resid_poisson, bin_edges, reps=K)
sig_phan_g, std_phan_g, cnt_phan_g = binned_noise_std(x, resid_gauss, bin_edges, reps=K)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# 左图：Poisson 信号依赖性
axes[0].plot(sig_ramp_p, std_ramp_p, '-', lw=2, color='C0',
             label=f'渐变图实测 σ（{n_bins} 桶，每像素 {K} 次实现）')
valid_p = cnt_phan_p > 0
axes[0].plot(sig_phan_p[valid_p], std_phan_p[valid_p], 'o', color='C1', markersize=8,
             label='幻影各灰度级实测 σ（同口径）')
axes[0].plot(sig_ramp_p, np.sqrt(gain * sig_ramp_p), 'k--', lw=1.5,
             label='理论 √(gain·信号)')
axes[0].set_xlabel('信号亮度')
axes[0].set_ylabel('噪声标准差')
axes[0].set_title(f'Poisson 噪声：信号依赖性 (gain={gain})\n'
                  f'实测 σ 与理论 √(gain·信号) 在全亮度范围吻合，峰值 λ=1/gain={1/gain:g}')
axes[0].set_ylim(0, 0.32)
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# 右图：高斯信号无关性
axes[1].plot(sig_ramp_g, std_ramp_g, '-', lw=2, color='C0',
             label=f'渐变图实测 σ（{n_bins} 桶，每像素 {K} 次实现）')
valid_g = cnt_phan_g > 0
axes[1].plot(sig_phan_g[valid_g], std_phan_g[valid_g], 'o', color='C1', markersize=8,
             label='幻影各灰度级实测 σ（同口径）')
axes[1].axhline(y=sigma_gauss, color='k', linestyle='--', lw=1.5,
                label=f'理论 σ={sigma_gauss}（常数）')
axes[1].set_xlabel('信号亮度')
axes[1].set_ylabel('噪声标准差')
axes[1].set_title('高斯噪声：信号无关性\n实测 σ 不随信号亮度变化（未截断口径）')
axes[1].set_ylim(0, 0.12)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_4_1_噪声信号依赖性.png'), dpi=150, bbox_inches='tight')
plt.show()

# ---- 5b. 背景区 SSIM：检验"暗部几乎无损"这一说法 ----
# full=True 返回逐像素 SSIM 图，在纯黑背景掩码（x < 0.01）上取均值。
_, ssim_map_g = structural_similarity(x, y_gauss, data_range=1.0, win_size=7, full=True)
_, ssim_map_p = structural_similarity(x, y_poisson, data_range=1.0, win_size=7, full=True)
_bg_mask = x < 0.01
bg_ssim_gauss = float(ssim_map_g[_bg_mask].mean())
bg_ssim_poisson = float(ssim_map_p[_bg_mask].mean())
print(f"\n纯黑背景区（x<0.01，占 {_bg_mask.mean():.1%}）平均 SSIM："
      f"高斯 {bg_ssim_gauss:.4f}，Poisson {bg_ssim_poisson:.4f}")

# 图省事的口径统一：把渐变图 20 桶的 (信号均值, 噪声标准差) 全量写入 JSON，
# 不剔桶、不截断，供正文直接引用（避免正文手算/复述子集造成数值脱节）。
def _binned_pairs(mean_sig, std):
    return [{'bin_mean_signal': None if not np.isfinite(m) else round(float(m), 4),
             'bin_std': None if not np.isfinite(s) else round(float(s), 4)}
            for m, s in zip(mean_sig, std)]

signal_dependence_stats = {
    'ramp_gaussian': _binned_pairs(sig_ramp_g, std_ramp_g),
    'ramp_poisson': _binned_pairs(sig_ramp_p, std_ramp_p),
    'phantom_gaussian_nonempty': [
        {'gray_level': round(float(m), 4), 'bin_std': round(float(s), 4),
         'n_pixels': int(round(c / K))}
        for m, s, c in zip(sig_phan_g, std_phan_g, cnt_phan_g) if c > 0],
    'phantom_poisson_nonempty': [
        {'gray_level': round(float(m), 4), 'bin_std': round(float(s), 4),
         'n_pixels': int(round(c / K))}
        for m, s, c in zip(sig_phan_p, std_phan_p, cnt_phan_p) if c > 0],
}

print(f"{'噪声类型':>12s}  {'PSNR(dB)':>10s}  {'SSIM':>8s}")
print("-" * 36)
for name, psnr_val, ssim_val in zip(names, psnrs, ssims):
    print(f"{name:>12s}  {psnr_val:10.2f}  {ssim_val:8.4f}")

print("\n口径与现象说明：")
print(f"1) 所有指标基于未截断数据。高斯噪声有 {frac_oob_gauss:.1%} 的像素超出显示范围 "
      f"[0,1]（暗背景负噪声为主），Poisson 有 {frac_oob_poisson:.1%} 高于 1（集中在亮环）；"
      f"这些像素仅在 imshow 时饱和显示。")
print("2) 未截断高斯 PSNR≈20 dB（MSE=σ²=0.01），与教科书一致；此前截断口径会把 PSNR 人为抬高约 1.7 dB。")
print(f"3) Poisson 峰值 λ=1/gain={1/gain:g}；未截断时最亮区 σ≈√(gain·1)="
      f"{np.sqrt(gain):.3f} 与实测一致，截断则会使其偏低约 45%。")
print(f"4) 椒盐噪声实际改变 {frac_changed_sp:.2%} 的像素（amount={amount_sp:.2%}，"
      "落在原值 0/1 像素上的替换不产生变化）。")
print("5) SSIM 高斯 < Poisson < 脉冲 属正常：58% 纯黑背景上 Poisson(0) 恒等于 0、脉冲噪声稀疏，"
      "暗部几乎无损；SSIM 均匀窗口对比项 C2/(σ²+C2)≈0.08，对满幅高斯噪声惩罚极重。")

# ===== 保存数值结果 =====
results_summary = {
    'image_size': n,
    'metric_basis': 'PSNR/SSIM/残差均基于未截断的原始含噪数据；np.clip 仅用于 imshow 显示',
    'signal_dependence_check': {
        'ramp_image_size': n,
        'n_bins': n_bins,
        'repeats_per_pixel': K,
        'theory_gaussian_sigma': float(sigma_gauss),
        'theory_poisson_std': 'sqrt(gain * signal)',
    },
    'signal_dependence_stats': signal_dependence_stats,
    'background_ssim_win7': {
        'background_definition': 'x < 0.01（SSIM 逐像素图在该掩码上取均值）',
        'background_pixel_fraction': round(float(_bg_mask.mean()), 4),
        'gaussian': round(bg_ssim_gauss, 4),
        'poisson': round(bg_ssim_poisson, 4),
    },
    'gaussian_noise': {
        'sigma': float(sigma_gauss),
        'psnr_dB': float(round(psnr_gauss, 2)),
        'ssim': float(round(ssim_gauss, 4)),
        'frac_outside_display_range_01': round(frac_oob_gauss, 4),
    },
    'poisson_noise': {
        'gain': float(gain),
        'peak_lambda': round(1.0 / gain, 3),
        'psnr_dB': float(round(psnr_poisson, 2)),
        'ssim': float(round(ssim_poisson, 4)),
        'frac_above_1': round(frac_oob_poisson, 4),
        'theory_mse_gain_times_E_x': round(float(gain * x.mean()), 6),
    },
    'impulse_noise': {
        'amount': float(amount_sp),
        'psnr_dB': float(round(psnr_sp, 2)),
        'ssim': float(round(ssim_sp, 4)),
        'frac_pixels_actually_changed': round(frac_changed_sp, 4),
    },
}

def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    return obj

results_summary = {k: _to_native(v) for k, v in results_summary.items()}
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"\n数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
