"""
实验2.2-3 TV先验的局限：阶梯效应
对应章节：2.2 经典先验族 - TV先验
知识点：TV保边但产生阶梯效应；TV偏好分段常数；与Sobolev先验的对比

素材来源：
  - 2.5.py: TV去噪与阶梯效应展示

修改说明：
  - 将简单Tikhonov改为Sobolev（smoothness prior），使"平滑vs保边"对比更有意义
  - 步骤1图补充原始无噪图像
  - 正则项图同时画t²和|t|对比
  - 统一配色方案
  - 补充参数选参依据说明
  - 修正：Sobolev λ 由 0.15（欠正则、几乎未去噪）调至 2.5（其 SSIM 最优点），
    使"Sobolev 过度平滑磨边 vs TV 保边"的定性对比名副其实，配图不再自相矛盾
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import sys

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.2-3', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.2-3')
    os.makedirs(SAVE_DIR, exist_ok=True)
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
    print("请确保 .chinese 文件夹已上传到 Google Drive 的正确位置")

np.random.seed(42)

DATA_RANGE = 1.0
COLOR_ORIGINAL = 'black'
COLOR_NOISY = 'gray'
COLOR_SOBOLEV = 'steelblue'
COLOR_TV = 'crimson'

n = 128
camera = resize(data.camera(), (n, n))

sigma = 0.1
camera_noisy = random_noise(camera, mode='gaussian', var=sigma**2, rng=42)

def sobolev_denoise(y, lam):
    """
    Sobolev去噪闭式解（周期边界，与 denoise_tv_chambolle 保持一致）
    
    目标函数: J(x) = 0.5||y-x||² + 0.5λ||∇x||²
    频域解: X(ω) = Y(ω) / (1 - λ·Δ̂(ω))
    
    其中 Δ̂(ω) = 2(cos(ωx) + cos(ωy) - 2) 是离散Laplacian的特征值。
    
    Args:
        y: 输入图像，必须为方形（行数=列数）
        lam: 正则化参数
    """
    assert y.shape[0] == y.shape[1], "sobolev_denoise 仅支持方形图像"
    n = y.shape[0]
    freq = np.fft.fftfreq(n) * 2 * np.pi
    fx, fy = np.meshgrid(freq, freq)
    lap_eig = 2 * (np.cos(fx) + np.cos(fy) - 2)
    denom = 1 - lam * lap_eig
    return np.real(np.fft.ifft2(np.fft.fft2(y) / denom))

def find_staircase_position(tv_img, sob_img, margin=20, window_size=7):
    """
    找到阶梯效应最明显的位置
    
    方法：阶梯效应的特征是"TV有梯度但Sobolev平滑"的区域。
    使用 (grad_tv - grad_sob) 作为核心指标，排除真实边缘（两者梯度都大）。
    
    Returns:
        tuple: (stair_x, stair_y) - matplotlib坐标系下的位置，x=列，y=行
    """
    from scipy.ndimage import uniform_filter
    
    grad_tv_x = np.roll(tv_img, -1, axis=1) - tv_img
    grad_tv_y = np.roll(tv_img, -1, axis=0) - tv_img
    grad_tv_mag = np.sqrt(grad_tv_x**2 + grad_tv_y**2)
    
    grad_sob_x = np.roll(sob_img, -1, axis=1) - sob_img
    grad_sob_y = np.roll(sob_img, -1, axis=0) - sob_img
    grad_sob_mag = np.sqrt(grad_sob_x**2 + grad_sob_y**2)
    
    tv_excess = np.maximum(grad_tv_mag - grad_sob_mag, 0)
    
    diff = np.abs(tv_img - sob_img)
    
    staircase_score = tv_excess * uniform_filter(diff, size=window_size)
    
    staircase_score[:margin, :] = 0
    staircase_score[-margin:, :] = 0
    staircase_score[:, :margin] = 0
    staircase_score[:, -margin:] = 0
    
    max_idx = np.unravel_index(np.argmax(staircase_score), staircase_score.shape)
    
    return (max_idx[1], max_idx[0])

lam_sob = 2.5   # Sobolev 去噪最优工作点（SSIM≈0.70），此时才真正体现其"磨平边缘"
camera_sob = sobolev_denoise(camera_noisy, lam_sob)

alpha_tv = 0.15
camera_tv = denoise_tv_chambolle(camera_noisy, weight=alpha_tv)

staircase_pos = find_staircase_position(camera_tv, camera_sob)

psnr_noisy = peak_signal_noise_ratio(camera, camera_noisy, data_range=DATA_RANGE)
psnr_sob = peak_signal_noise_ratio(camera, camera_sob, data_range=DATA_RANGE)
psnr_tv = peak_signal_noise_ratio(camera, camera_tv, data_range=DATA_RANGE)

ssim_noisy = structural_similarity(camera, camera_noisy, data_range=DATA_RANGE)
ssim_sob = structural_similarity(camera, camera_sob, data_range=DATA_RANGE)
ssim_tv = structural_similarity(camera, camera_tv, data_range=DATA_RANGE)

print("=" * 70)
print("实验2.2-3 TV先验的局限：阶梯效应")
print("=" * 70)

print("\n【参数设置】")
print(f"  图像尺寸: {n}×{n}")
print(f"  噪声标准差 σ = {sigma}")
print(f"\n  参数选择说明:")
print(f"    - Sobolev λ = {lam_sob}：调至其 SSIM 最优的去噪工作点")
print(f"    - TV α = {alpha_tv}：调至保边与去噪平衡的工作点")
print(f"    注：两者量纲不同，λ惩罚||∇x||²，α惩罚||∇x||₁，不能直接比较数值。")
print(f"    本实验将各自调至接近最优的工作点，以便在公平的噪声抑制水平下")
print(f"    对比二者的边缘保持行为（Sobolev 磨边 vs TV 保边）。")

print("\n【去噪结果对比】")
print(f"  含噪图像: PSNR = {psnr_noisy:.2f} dB, SSIM = {ssim_noisy:.3f}")
print(f"  Sobolev (smoothness prior): PSNR = {psnr_sob:.2f} dB, SSIM = {ssim_sob:.3f}")
print(f"  TV先验: PSNR = {psnr_tv:.2f} dB, SSIM = {ssim_tv:.3f}")

print("\n【TV先验特点】")
print("  优势: 保边 - 边缘锐利")
print("  局限: 阶梯效应 - 渐变区域出现阶梯状伪影")
print("  原因: TV偏好分段常数函数")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(camera, cmap='gray')
axes[0].set_title('原始图像\n(无噪)')
axes[0].axis('off')

axes[1].imshow(camera_noisy, cmap='gray')
axes[1].set_title(f'含噪图像\nPSNR={psnr_noisy:.2f}dB, SSIM={ssim_noisy:.3f}')
axes[1].axis('off')

axes[2].imshow(camera_sob, cmap='gray')
axes[2].set_title(f'Sobolev去噪 (λ={lam_sob})\nPSNR={psnr_sob:.2f}dB, SSIM={ssim_sob:.3f}\n边缘模糊')
axes[2].axis('off')

axes[3].imshow(camera_tv, cmap='gray')
axes[3].set_title(f'TV去噪 (α={alpha_tv})\nPSNR={psnr_tv:.2f}dB, SSIM={ssim_tv:.3f}\n边缘锐利')
axes[3].axis('off')

plt.suptitle('Sobolev先验 vs TV先验：边缘保持的代价', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_TV去噪对比.png'), dpi=150, bbox_inches='tight')
plt.close()

# ===== 合成一维信号：无歧义地展示阶梯效应 =====
def sobolev_1d(y, lam):
    """一维 Sobolev 去噪闭式解（周期边界）：X = Y / (1 - lam*2(cos f - 1))"""
    m = len(y)
    f = np.fft.fftfreq(m) * 2 * np.pi
    lap = 2 * (np.cos(f) - 1)
    return np.real(np.fft.ifft(np.fft.fft(y) / (1 - lam * lap)))

m = 128
tt = np.arange(m)
x_true = np.zeros(m)
x_true[:32] = 0.20
x_true[32:80] = 0.20 + 0.55 * (tt[32:80] - 32) / (80 - 32)   # 线性渐变区(ramp)
x_true[80:96] = 0.75
x_true[96:] = 0.30                                            # 阶跃边缘(step)
rng_synth = np.random.default_rng(0)
x_noisy = x_true + rng_synth.normal(0, sigma, m)

lam_synth = 3.0
x_sob = sobolev_1d(x_noisy, lam_synth)
x_tv = denoise_tv_chambolle(x_noisy, weight=alpha_tv)

ramp = slice(32, 80)
step = slice(88, 104)
def _mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))
step_mse_tv = _mse(x_tv[step], x_true[step])
step_mse_sob = _mse(x_sob[step], x_true[step])
# 阶梯效应客观度量：真实斜坡应有恒定非零斜率，TV 却把大部分斜率压成零(平台)
_tv_d = np.diff(x_tv[ramp]); _sob_d = np.diff(x_sob[ramp])
ramp_flat_frac_tv = float(np.mean(np.abs(_tv_d) < 0.005))
ramp_flat_frac_sob = float(np.mean(np.abs(_sob_d) < 0.005))

print("\n【合成信号：阶梯效应定量验证】")
print(f"  阶跃边缘 MSE: TV={step_mse_tv:.5f}  Sobolev={step_mse_sob:.5f}  (TV 保边更优)")
print(f"  渐变区平台占比(近零梯度): TV={ramp_flat_frac_tv*100:.1f}%  Sobolev={ramp_flat_frac_sob*100:.1f}%  (TV 阶梯效应)")

# ===== 步骤2图：真实剖面 + 合成信号阶梯效应 =====
stair_x, stair_y = staircase_pos
row = stair_y

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(camera[row, :], color=COLOR_ORIGINAL, linewidth=2, label='真实信号')
axes[0].plot(camera_noisy[row, :], color=COLOR_NOISY, linewidth=0.5, alpha=0.5, label='含噪')
axes[0].plot(camera_sob[row, :], color=COLOR_SOBOLEV, linewidth=1.5, label='Sobolev (边缘模糊)')
axes[0].plot(camera_tv[row, :], color=COLOR_TV, linewidth=1.5, label='TV (边缘锐利)')
axes[0].axvline(x=stair_x, color='red', linestyle='--', linewidth=1, alpha=0.5, label='检测位置')
axes[0].set_title(f'真实图像第{row}行剖面\n(Sobolev 磨边 vs TV 保边)')
axes[0].legend(fontsize=8)
axes[0].set_xlabel('像素索引')
axes[0].set_ylabel('灰度值')
axes[0].grid(True, alpha=0.3)

axes[1].plot(tt, x_true, color=COLOR_ORIGINAL, linewidth=2, label='真实信号')
axes[1].plot(tt, x_noisy, color=COLOR_NOISY, linewidth=0.5, alpha=0.5, label='含噪')
axes[1].plot(tt, x_sob, color=COLOR_SOBOLEV, linewidth=1.5, label=f'Sobolev (λ={lam_synth})')
axes[1].plot(tt, x_tv, color=COLOR_TV, linewidth=1.5, label=f'TV (α={alpha_tv})')
axes[1].axvspan(32, 80, color='gold', alpha=0.15)
axes[1].text(56, 0.90, '渐变区', ha='center', fontsize=9, color='darkgoldenrod')
axes[1].axvline(x=96, color='green', linestyle=':', linewidth=1.2)
axes[1].text(96, 0.55, ' 阶跃', color='green', fontsize=9)
axes[1].set_title('合成斜坡+阶跃：全信号对比\n(阶跃处 TV 更陡、Sobolev 被磨平)')
axes[1].legend(fontsize=8)
axes[1].set_xlabel('像素索引')
axes[1].set_ylabel('信号值')
axes[1].grid(True, alpha=0.3)

zr = slice(30, 82)
axes[2].plot(tt[zr], x_true[zr], color=COLOR_ORIGINAL, linewidth=2, label='真实(线性斜坡)')
axes[2].plot(tt[zr], x_sob[zr], color=COLOR_SOBOLEV, linewidth=1.5, label='Sobolev(平滑跟随)')
axes[2].plot(tt[zr], x_tv[zr], color=COLOR_TV, linewidth=1.5, label='TV(阶梯效应)')
axes[2].set_title(f'渐变区放大：TV 阶梯效应\n(TV平台占比{ramp_flat_frac_tv*100:.0f}% vs Sobolev {ramp_flat_frac_sob*100:.0f}%)')
axes[2].legend(fontsize=8)
axes[2].set_xlabel('像素索引')
axes[2].set_ylabel('信号值')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_阶梯效应分析.png'), dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

t = np.linspace(-2, 2, 400)
axes[0].plot(t, t**2, color=COLOR_SOBOLEV, linewidth=2, label='t² (Sobolev/平滑先验)')
axes[0].plot(t, np.abs(t), color=COLOR_TV, linewidth=2, label='|t| (TV先验)')
axes[0].set_title('正则项形态对比\nL2均匀惩罚 vs L1促稀疏')
axes[0].legend()
axes[0].set_xlabel('t (梯度值)')
axes[0].set_ylabel('惩罚值')
axes[0].set_ylim(-0.2, 4)
axes[0].grid(True, alpha=0.3)

grad_orig_x = np.roll(camera, -1, axis=1) - camera
grad_orig_y = np.roll(camera, -1, axis=0) - camera
grad_orig_mag = np.sqrt(grad_orig_x**2 + grad_orig_y**2).flatten()

grad_sob_x = np.roll(camera_sob, -1, axis=1) - camera_sob
grad_sob_y = np.roll(camera_sob, -1, axis=0) - camera_sob
grad_sob_mag = np.sqrt(grad_sob_x**2 + grad_sob_y**2).flatten()

grad_tv_x = np.roll(camera_tv, -1, axis=1) - camera_tv
grad_tv_y = np.roll(camera_tv, -1, axis=0) - camera_tv
grad_tv_mag = np.sqrt(grad_tv_x**2 + grad_tv_y**2).flatten()

bins = np.linspace(0, 0.5, 50)
axes[1].hist(grad_orig_mag, bins=bins, alpha=0.5, color=COLOR_ORIGINAL, label='原始', density=True)
axes[1].hist(grad_sob_mag, bins=bins, alpha=0.6, color=COLOR_SOBOLEV, label='Sobolev', density=True)
axes[1].hist(grad_tv_mag, bins=bins, alpha=0.6, color=COLOR_TV, label='TV', density=True)
axes[1].set_title('梯度幅值直方图\nTV产生大量零梯度（分段常数）')
axes[1].legend()
axes[1].set_xlabel('|∇x|')
axes[1].set_ylabel('密度')
axes[1].grid(True, alpha=0.3)

tv_zero_grad = np.sum(grad_tv_mag < 0.01) / len(grad_tv_mag) * 100
sob_zero_grad = np.sum(grad_sob_mag < 0.01) / len(grad_sob_mag) * 100
axes[1].text(0.62, 0.88, f'TV零梯度比例: {tv_zero_grad:.1f}%\nSobolev零梯度比例: {sob_zero_grad:.1f}%',
             transform=axes[1].transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

methods = ['含噪', 'Sobolev\n(smoothness)', 'TV先验']
psnrs = [psnr_noisy, psnr_sob, psnr_tv]
ssims_scaled = [ssim_noisy * 100, ssim_sob * 100, ssim_tv * 100]

COLOR_PSNR = '#4C72B0'
COLOR_SSIM = '#DD8452'

x_pos = np.arange(len(methods))
width = 0.35

ax1 = axes[2]
ax1.bar(x_pos - width/2, psnrs, width, color=COLOR_PSNR, alpha=0.8, label='PSNR (dB)')
ax1.axhline(y=psnr_noisy, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_ylabel('PSNR (dB)', color=COLOR_PSNR)
ax1.tick_params(axis='y', labelcolor=COLOR_PSNR)
ax1.set_ylim([min(psnrs) - 2, max(psnrs) + 3])

ax2 = ax1.twinx()
ax2.bar(x_pos + width/2, ssims_scaled, width, color=COLOR_SSIM, alpha=0.7, label='SSIM×100')
ax2.set_ylabel('SSIM×100', color=COLOR_SSIM)
ax2.tick_params(axis='y', labelcolor=COLOR_SSIM)
ax2.set_ylim([min(ssims_scaled) - 5, max(ssims_scaled) + 5])

ax1.set_xticks(x_pos)
ax1.set_xticklabels(methods, fontsize=9)
ax1.set_title('去噪性能对比\n(注: PSNR高不代表视觉自然)')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_正则项与性能对比.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 70)
print("【TV先验总结】")
print("=" * 70)
print("\nTV正则项: ||∇x||₁")
print("\n优势:")
print("  1. 保边性: 边缘不会被过度平滑")
print("  2. 稀疏性: 梯度稀疏假设对自然图像合理")
print("  3. 凸性: 优化问题可高效求解")
print("\n局限:")
print("  1. 阶梯效应: 渐变区域出现阶梯状伪影")
print("  2. 分段常数假设: 无法自然表示渐变")
print("  3. 参数敏感: α需要仔细调节")
print("\n改进方向:")
print("  - TGV (附录2B): 引入二阶信息，允许分段仿射")
print("  - 隐式先验 (2.4节): 从数据中学习先验")

print(f"\n【实验完成】结果已保存至: {SAVE_DIR}")

# ===== 保存数值结果 =====
import json
results_summary = {
    'image_size': n,
    'noise_sigma': float(sigma),
    'sobolev_lambda': float(lam_sob),
    'tv_alpha': float(alpha_tv),
    'psnr_noisy_dB': float(round(psnr_noisy, 2)),
    'psnr_sobolev_dB': float(round(psnr_sob, 2)),
    'psnr_tv_dB': float(round(psnr_tv, 2)),
    'ssim_noisy': float(round(ssim_noisy, 4)),
    'ssim_sobolev': float(round(ssim_sob, 4)),
    'ssim_tv': float(round(ssim_tv, 4)),
    'tv_zero_gradient_pct': float(round(tv_zero_grad, 2)),
    'sobolev_zero_gradient_pct': float(round(sob_zero_grad, 2)),
    'synth_sobolev_lambda': float(lam_synth),
    'synth_step_mse_tv': float(round(step_mse_tv, 5)),
    'synth_step_mse_sobolev': float(round(step_mse_sob, 5)),
    'synth_ramp_flat_frac_tv': float(round(ramp_flat_frac_tv, 3)),
    'synth_ramp_flat_frac_sobolev': float(round(ramp_flat_frac_sob, 3)),
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
