import numpy as np
import matplotlib.pyplot as plt
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio
import os
import sys

# ====== 中文字体配置（兼容本地和 Google Colab）======
_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验1.3-4', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验1.3-4')
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _chinese_path)
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=_chinese_path)
# ========================================================

np.random.seed(42)

# ---- 1. 准备图像和模糊算子 ----
x_color = astronaut()
x = rgb2gray(x_color)
assert x.shape[0] == x.shape[1], "当前实现仅支持正方形图像"
n = x.shape[0]

def gaussian_psf(size: int, sigma: float) -> np.ndarray:
    """生成高斯 PSF（点扩散函数），频率排列与 np.fft.fft2 对齐。

    使用 np.fft.fftfreq 生成正确的 FFT 频率坐标，
    确保 PSF 在频域卷积时位于正确位置。
    
    Args:
        size: 图像尺寸（正方形边长）。
        sigma: 高斯核标准差，控制模糊程度。
    
    Returns:
        size x size 的归一化高斯 PSF，sum(h) == 1。
    """
    ax = np.fft.fftfreq(size) * size  # 正确的 FFT 频率排列：[0,1,...,N/2-1,-N/2,...,-1]
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return h / h.sum()

def blur(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """频域卷积：y = x * h（循环卷积）。
    
    注意：频域卷积等价于循环卷积，图像边缘可能出现环绕伪影。
    若要减轻此效应，可在卷积前对图像做边缘填充（padding）。
    
    Args:
        x: 输入图像。
        h: 卷积核（PSF）。
    
    Returns:
        模糊后的图像。
    """
    return np.real(np.fft.ifft2(np.fft.fft2(h) * np.fft.fft2(x)))

# 两个不同的 PSF：用于 Inverse Crime 对比
sigma_A = 5.0   # 重建模型使用的 PSF 宽度
sigma_B = 5.3   # 数据生成时使用的 PSF 宽度（微有差异，模拟建模误差）

h_A = gaussian_psf(n, sigma_A)  # 重建模型
h_B = gaussian_psf(n, sigma_B)  # "真实"模型

# ---- 2. 生成观测数据 ----
noise_sigma = 0.01  # 噪声水平

# 固定噪声实现，确保 IC/非IC 对比仅受模型差异影响
noise = noise_sigma * np.random.randn(n, n)

# (a) Inverse Crime 数据：用 h_A 生成
y_IC = blur(x, h_A) + noise

# (b) 非 Inverse Crime 数据：用 h_B 生成（使用相同噪声）
y_noIC = blur(x, h_B) + noise

# ---- 3. 朴素逆重建（频域直接除法）----
H_A = np.fft.fft2(h_A)

def naive_deconv(y: np.ndarray, H: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """频域直接反卷积：x = F^{-1}(F(y)/H)。

    注意：H 的模值 < eps 时会被截断为 eps，避免除零；
    直接对复数加实数 eps（如 H + eps）会改变相位，此处改用模截断。
    
    Args:
        y: 观测图像（频域含噪）。
        H: 模糊核的 FFT。
        eps: 模值下界，防止除零。
    
    Returns:
        重建图像。
    """
    # 保相位模截断：模值放大到至少 eps，相位严格不变
    abs_H = np.abs(H)
    abs_H_clamped = np.maximum(abs_H, eps)
    # 安全缩放：避免 H=0 时 0/0 产生 NaN
    scale = np.ones_like(abs_H)
    mask = abs_H > 0
    scale[mask] = abs_H_clamped[mask] / abs_H[mask]
    H_reg = H * scale
    H_reg[~mask] = eps  # H=0 时相位无意义，直接用 eps
    return np.real(np.fft.ifft2(np.fft.fft2(y) / H_reg))

# ---- 4. Tikhonov 正则化反卷积 ----
def tikhonov_deconv(y: np.ndarray, H: np.ndarray, lam: float) -> np.ndarray:
    """Tikhonov 正则化反卷积（频域）：x = F^{-1}(H*/(|H|^2+λ) F(y))。

    Args:
        y: 观测图像（频域含噪）。
        H: 模糊核的 FFT。
        lam: 正则化参数，控制平滑程度。典型范围 [1e-4, 1e-1]，
            需根据图像内容和噪声水平调整（可用 L-curve 或 GCV 选取）。
    
    Returns:
        正则化重建图像。
    """
    H_star = np.conj(H)
    H_sq = np.abs(H) ** 2
    return np.real(np.fft.ifft2(H_star / (H_sq + lam) * np.fft.fft2(y)))

# ---- 5. 计算重建结果 ----
# 无噪声直接反卷积（理想情况）
y_clean = blur(x, h_A)
x_naive_clean = naive_deconv(y_clean, H_A)

# 含噪直接反卷积
x_naive_noisy = naive_deconv(y_IC, H_A)

# Inverse Crime：同模型生成+重建
x_tikh_IC = tikhonov_deconv(y_IC, H_A, lam=1e-2)  # lam=1e-2 为经验值，对应 SNR≈40dB；
                                                     # 最优 λ 需根据图像和噪声水平调整
                                                     # （可用 L-curve 或 GCV 选取）

# 非 Inverse Crime：异模型生成+重建
x_tikh_noIC = tikhonov_deconv(y_noIC, H_A, lam=1e-2)

# ---- 6. PSNR vs 建模偏差曲线 ----
# 首元素 sigma=5.0 对应 Δσ=0（即 IC 条件），与主实验变量 sigma_A 相同
# 后续递增模拟不同建模偏差，观察重建质量退化
# 注意：曲线通常不关于 Δσ=0 对称——
# 低估模糊（Δσ<0）因过度锐化导致噪声放大，PSNR 下降更陡；
# 高估模糊（Δσ>0）因过度平滑导致细节损失，下降相对平缓。
sigma_B_list = [3.0, 4.0, 4.5, 5.0, 5.3, 5.5, 6.0, 7.0, 8.0, 10.0]
psnr_curve = []
for sb in sigma_B_list:
    h_tmp = gaussian_psf(n, sb)
    y_tmp = blur(x, h_tmp) + noise
    x_tmp = tikhonov_deconv(y_tmp, H_A, lam=1e-2)
    psnr_curve.append(peak_signal_noise_ratio(x, np.clip(x_tmp, 0, 1), data_range=1.0))

# ---- 7. 可视化：图像对比 ----
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 上排：逆问题与不适定性
axes[0, 0].imshow(x, cmap='gray')
axes[0, 0].set_title('原始图像 x')
axes[0, 0].axis('off')

axes[0, 1].imshow(y_IC, cmap='gray')
axes[0, 1].set_title('观测 y = Ax + ε\n(噪声 σ=0.01)')
axes[0, 1].axis('off')

psnr_naive = peak_signal_noise_ratio(x, np.clip(x_naive_noisy, 0, 1), data_range=1.0)
axes[0, 2].imshow(np.clip(x_naive_noisy, 0, 1), cmap='gray')
axes[0, 2].set_title(f'含噪直接反卷积\nPSNR={psnr_naive:.1f}dB ✗')
axes[0, 2].axis('off')

# 下排：正则化与 Inverse Crime
psnr_clean = peak_signal_noise_ratio(x, np.clip(x_naive_clean, 0, 1), data_range=1.0)
psnr_tikh_IC = peak_signal_noise_ratio(x, np.clip(x_tikh_IC, 0, 1), data_range=1.0)
psnr_tikh_noIC = peak_signal_noise_ratio(x, np.clip(x_tikh_noIC, 0, 1), data_range=1.0)

axes[1, 0].imshow(np.clip(x_naive_clean, 0, 1), cmap='gray')
axes[1, 0].set_title(f'无噪声直接反卷积\nPSNR={psnr_clean:.1f}dB ✓')
axes[1, 0].axis('off')

axes[1, 1].imshow(np.clip(x_tikh_IC, 0, 1), cmap='gray')
axes[1, 1].set_title(f'Tikhonov (IC: σ={sigma_A})\nPSNR={psnr_tikh_IC:.1f}dB')
axes[1, 1].axis('off')

axes[1, 2].imshow(np.clip(x_tikh_noIC, 0, 1), cmap='gray')
axes[1, 2].set_title(f'Tikhonov (非IC: σ={sigma_B})\nPSNR={psnr_tikh_noIC:.1f}dB')
axes[1, 2].axis('off')

plt.suptitle('朴素逆重建与 Inverse Crime 对比', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_3_4_不适定性与InverseCrime.png'), dpi=150, bbox_inches='tight')
plt.show()

# ---- 8. 可视化：PSNR vs 建模偏差曲线 ----
fig2, ax2 = plt.subplots(figsize=(8, 5))
deviations = [sb - sigma_A for sb in sigma_B_list]
ax2.plot(deviations, psnr_curve, 'bo-', markersize=8, linewidth=2, label='Tikhonov 重建 PSNR')
ax2.axvline(x=0, color='r', linestyle='--', alpha=0.7, label=f'IC 点 (σ_真实=σ_模型={sigma_A})')
ax2.set_xlabel('建模偏差 Δσ = σ_真实 − σ_模型', fontsize=12)
ax2.set_ylabel('PSNR (dB)', fontsize=12)
ax2.set_title('Inverse Crime 警示：重建质量随建模偏差的系统退化', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# 设置 y 轴范围，给数据标注留出上边距
psnr_max = max(psnr_curve)
ax2.set_ylim(bottom=min(psnr_curve) - 1, top=psnr_max + 2.5)

# 标注每个数据点的 PSNR 值
for d, p in zip(deviations, psnr_curve):
    ax2.annotate(f'{p:.1f}dB', (d, p), textcoords="offset points",
                 xytext=(0, 15), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_3_4_PSNR_vs_建模偏差.png'), dpi=150, bbox_inches='tight')
plt.show()

# ---- 9. 打印结果 ----
print("\n=== Inverse Crime 警示 ===")
print(f"同模型(σ={sigma_A})生成 + 重建 → PSNR={psnr_tikh_IC:.1f}dB")
print(f"异模型(σ={sigma_B})生成 + 重建 → PSNR={psnr_tikh_noIC:.1f}dB")
print(f"PSNR 差异: {psnr_tikh_IC - psnr_tikh_noIC:.2f} dB")
print("\n--- PSNR 随建模偏差变化 ---")
for sb, p in zip(sigma_B_list, psnr_curve):
    marker = " ← IC" if abs(sb - sigma_A) < 1e-9 else ""
    print(f"  σ_真实={sb:.1f} (偏差 Δσ={sb-sigma_A:.1f}) → PSNR={p:.2f} dB{marker}")
print("\n建模偏差越大，重建质量系统性地越低——IC 条件下测得的性能不可靠！")
print("\n注：以上结论基于 astronaut 单幅图像和固定正则化参数 λ=1e-2。")
print("  定量数值会因图像内容、噪声水平和 λ 选择而变化。")
print("  Tikhonov 正则化对小幅建模误差有一定鲁棒性，但当建模偏差超过")
print("  正则化的容忍范围时，PSNR 系统性下降的趋势具有一般性。")

# ===== 保存数值结果 =====
import json
results_summary = {
    'image_size': n,
    'sigma_model': float(sigma_A),
    'sigma_true': float(sigma_B),
    'noise_sigma': float(noise_sigma),
    'tikhonov_lambda': 1e-2,
    'psnr_naive_noisy_dB': float(round(psnr_naive, 2)),
    'psnr_naive_clean_dB': float(round(psnr_clean, 2)),
    'psnr_tikhonov_IC_dB': float(round(psnr_tikh_IC, 2)),
    'psnr_tikhonov_noIC_dB': float(round(psnr_tikh_noIC, 2)),
    'psnr_IC_vs_noIC_diff_dB': float(round(psnr_tikh_IC - psnr_tikh_noIC, 2)),
    'psnr_vs_modeling_deviation': {f'dsigma_{sb - sigma_A:.1f}': float(round(p, 2)) for sb, p in zip(sigma_B_list, psnr_curve)},
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