import numpy as np
import matplotlib
matplotlib.use('Agg')
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

# 两组模糊尺度：
#   sigma=5.0：主实验，严重病态（谱跨约54个数量级，大量频点下溢为0）
#   sigma=1.5：良性对照（最小谱幅值约2.3e-10 > 1e-15，频域除法不截断）
# 两个不同的 PSF：用于 Inverse Crime 对比
sigma_A = 5.0     # 重建模型使用的 PSF 宽度
sigma_B = 5.3     # 数据生成时使用的 PSF 宽度（微有差异，模拟建模误差）
sigma_well = 1.5  # 良性条件数对照实验使用的 PSF 宽度

h_A = gaussian_psf(n, sigma_A)    # 重建模型
h_B = gaussian_psf(n, sigma_B)    # "真实"模型（失配演示）
h_well = gaussian_psf(n, sigma_well)  # 良性算子

H_A = np.fft.fft2(h_A)
H_well = np.fft.fft2(h_well)

# ---- 2. 生成观测数据 ----
noise_sigma = 0.01  # 噪声水平

# 固定噪声实现，确保 IC/非IC 对比仅受模型差异影响（配对设计）
noise = noise_sigma * np.random.randn(n, n)

# (a) Inverse Crime 数据：用 h_A 生成
y_IC = blur(x, h_A) + noise

# (b) 非 Inverse Crime 数据：用 h_B 生成（使用相同噪声）
y_noIC = blur(x, h_B) + noise

# 无噪声观测（病态算子与良性算子各一）
y_clean_A = blur(x, h_A)
y_clean_well = blur(x, h_well)

# ---- 3. 朴素逆重建（频域直接除法）----
def naive_deconv(y: np.ndarray, H: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """频域直接反卷积：x = F^{-1}(F(y)/H)。

    注意：H 的模值 < eps 时会被截断为 eps，避免除零；
    直接对复数加实数 eps（如 H + eps）会改变相位，此处改用模截断。

    重要的数值细节（本实验图2左专门演示）：模截断本质上是一个"硬低通"——
    |H|<eps 的高频箱被替换后，1/H 不再作用于这些频率。eps 取得过小也不行：
    即使输入无噪声，FFT 的双精度舍入误差（约1e-16）也会在病态频点被 1/H
    放大到与真实噪声同量级。因此无噪声病态问题的朴素反演结果对 eps 极其敏感，
    PSNR 可在数 dB 到数十 dB 之间变化——这本身就是条件数过大的体现，
    而不是"反演失败"或"反演成功"的简单二分。

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
    """Tikhonov 正则化反卷积（频域）：x = F^{-1}(H*/(|H|^2+lambda) F(y))。

    对应优化问题  min_x ||Hx - y||^2 + lambda ||x||^2（零阶 Tikhonov）。
    滤波因子 w_k = |H_k|^2/(|H|^2+lambda) 逐频点收缩：lambda 越大越平滑。

    Args:
        y: 观测图像。
        H: 重建所用模型的 FFT。
        lam: 正则化参数。

    Returns:
        正则化重建图像。
    """
    H_star = np.conj(H)
    H_sq = np.abs(H) ** 2
    return np.real(np.fft.ifft2(H_star / (H_sq + lam) * np.fft.fft2(y)))

# ---- 5. GCV 数据驱动选择正则化参数 ----
# 不使用真值 x 调参（避免"用答案选参数"），改用广义交叉验证：
#   GCV(lambda) = (1/N) sum_k |(1-w_k) Y_k|^2 / (1 - (1/N) sum_k w_k)^2
# 其中 w_k = |H_k|^2/(|H_k|^2+lambda)。该式完全由观测 y 和模型 H 计算，
# 频域对角化使留一交叉验证可闭式计算，无需重复求解。
def gcv_select_lambda(y: np.ndarray, H: np.ndarray,
                      lambda_grid: np.ndarray) -> tuple[float, np.ndarray]:
    """用广义交叉验证（GCV）为 Tikhonov 反卷积选择正则化参数。

    Args:
        y: 观测图像。
        H: 模型 FFT。
        lambda_grid: 候选 lambda 数组（建议对数等距）。

    Returns:
        (最优 lambda, 各候选对应的 GCV 曲线)。
    """
    Y = np.fft.fft2(y)
    N = y.size
    H_sq = np.abs(H) ** 2
    gcv_curve = np.empty(lambda_grid.size)
    for i, lam in enumerate(lambda_grid):
        w = H_sq / (H_sq + lam)
        residual = np.abs((1.0 - w) * Y) ** 2
        numerator = residual.sum() / N
        denominator = (1.0 - w.mean()) ** 2
        gcv_curve[i] = numerator / denominator
    return float(lambda_grid[np.argmin(gcv_curve)]), gcv_curve

lambda_grid = np.logspace(-6, 1, 200)
# 模拟真实流程：只凭 IC 观测和模型 H_A 调一次参，之后固定部署
lam_gcv, _ = gcv_select_lambda(y_IC, H_A, lambda_grid)
lam_fixed = 1e-2  # 仅作对照：未经调参的"拍脑袋"固定值（原实验设置）

# ---- 6. 计算重建结果 ----
# (a) 病态算子 + 含噪：朴素反演（预期噪声爆炸）
x_naive_noisy = naive_deconv(y_IC, H_A)

# (b) 良性算子 + 无噪声：朴素反演可精确恢复
x_well_naive = naive_deconv(y_clean_well, H_well)

# (c) 病态算子：Tikhonov（GCV 选参），IC 与非 IC
x_tikh_IC = tikhonov_deconv(y_IC, H_A, lam_gcv)
x_tikh_noIC = tikhonov_deconv(y_noIC, H_A, lam_gcv)

# 固定 lambda=1e-2 的对照重建（用于说明次优调参如何扭曲偏差曲线）
x_tikh_IC_fixed = tikhonov_deconv(y_IC, H_A, lam_fixed)
x_tikh_noIC_fixed = tikhonov_deconv(y_noIC, H_A, lam_fixed)

# ---- 7. 无噪声病态反演对截断阈值 eps 的敏感性 ----
# 演示三件事：
#  eps 极小 -> FFT 双精度舍入误差被 1/H 放大，PSNR 崩到含噪水平；
#  eps 适中 -> 截断等价硬低通，高频细节丢失（PSNR 约20dB，绝非"精确恢复"）；
#  eps 极大 -> 过度低通，进一步模糊。
eps_list = np.logspace(-30, -3, 28)
psnr_vs_eps = [peak_signal_noise_ratio(
    x, np.clip(naive_deconv(y_clean_A, H_A, eps=e), 0, 1), data_range=1.0)
    for e in eps_list]

# ---- 8. PSNR vs 建模偏差曲线：GCV 选参 vs 固定 lambda ----
# 结论预期（数据驱动正确调参后）：
#   * IC 点（Δσ=0）位于曲线最高平台——IC 设定给出的是"过于乐观"的性能；
#   * 足够大的偏差无论正负都使质量显著退化：
#       Δσ>0（真实模糊比模型重）-> 残留模糊、细节丢失，正则化无法挽回；
#       Δσ<0（真实模糊比模型轻）-> 逆滤波器过冲，产生振铃；
#   * 机理：重建 = G(H_模型,λ) H_真实 x + G(H_模型,λ) ε。
#     噪声项 G·ε 只依赖模型与 λ，与真实算子无关（实测不同 σ_真实 下
#     重建噪声场完全相同，相关系数=1）；失配只改变信号传递项 G H_真实。
#   * 对照：固定过强的 λ=1e-2 会使 IC 点本身成为"洼地"，伪造出
#     "轻微负偏差反而更好"的假象（那是过强平滑被轻微过锐补偿所致），
#     不能据此声称"偏差有益"。
sigma_B_list = [3.0, 4.0, 4.5, 5.0, 5.3, 5.5, 6.0, 7.0, 8.0, 10.0]
deviations = [sb - sigma_A for sb in sigma_B_list]
psnr_curve_gcv = []
psnr_curve_fixed = []
for sb in sigma_B_list:
    h_tmp = gaussian_psf(n, sb)
    y_tmp = blur(x, h_tmp) + noise
    psnr_curve_gcv.append(peak_signal_noise_ratio(
        x, np.clip(tikhonov_deconv(y_tmp, H_A, lam_gcv), 0, 1), data_range=1.0))
    psnr_curve_fixed.append(peak_signal_noise_ratio(
        x, np.clip(tikhonov_deconv(y_tmp, H_A, lam_fixed), 0, 1), data_range=1.0))

# ---- 9. Monte Carlo：IC vs 非IC 差异的噪声实现波动 ----
# 固定使用 IC 观测上 GCV 选出的 lambda（部署后不再调参），
# 更换噪声实现，检验"IC 偏乐观"是否为单一噪声实现的偶然现象。
n_mc = 100
rng_mc = np.random.RandomState(123)
mc_ic, mc_noic = [], []
for _ in range(n_mc):
    nn = noise_sigma * rng_mc.randn(n, n)
    mc_ic.append(peak_signal_noise_ratio(
        x, np.clip(tikhonov_deconv(blur(x, h_A) + nn, H_A, lam_gcv), 0, 1),
        data_range=1.0))
    mc_noic.append(peak_signal_noise_ratio(
        x, np.clip(tikhonov_deconv(blur(x, h_B) + nn, H_A, lam_gcv), 0, 1),
        data_range=1.0))
mc_ic = np.array(mc_ic)
mc_noic = np.array(mc_noic)
mc_diff = mc_ic - mc_noic

# ---- 10. 汇总 PSNR 数值 ----
psnr_naive_noisy = peak_signal_noise_ratio(
    x, np.clip(x_naive_noisy, 0, 1), data_range=1.0)
psnr_well_naive = peak_signal_noise_ratio(
    x, np.clip(x_well_naive, 0, 1), data_range=1.0)
psnr_tikh_IC = peak_signal_noise_ratio(
    x, np.clip(x_tikh_IC, 0, 1), data_range=1.0)
psnr_tikh_noIC = peak_signal_noise_ratio(
    x, np.clip(x_tikh_noIC, 0, 1), data_range=1.0)

# ---- 11. 可视化：图像对比 ----
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 上排：逆问题与不适定性
axes[0, 0].imshow(x, cmap='gray')
axes[0, 0].set_title('原始图像 x')
axes[0, 0].axis('off')

axes[0, 1].imshow(y_IC, cmap='gray')
axes[0, 1].set_title(f'观测 y = Ax + ε\n(病态算子 σ={sigma_A}, 噪声 σ={noise_sigma})')
axes[0, 1].axis('off')

axes[0, 2].imshow(np.clip(x_naive_noisy, 0, 1), cmap='gray')
axes[0, 2].set_title(f'含噪直接反卷积（失败）\nPSNR={psnr_naive_noisy:.1f} dB')
axes[0, 2].axis('off')

# 下排：可解性、正则化与 Inverse Crime
axes[1, 0].imshow(np.clip(x_well_naive, 0, 1), cmap='gray')
axes[1, 0].set_title(f'无噪声直接反卷积（精确恢复）\n良性算子 σ={sigma_well}: '
                     f'PSNR={psnr_well_naive:.0f} dB')
axes[1, 0].axis('off')

axes[1, 1].imshow(np.clip(x_tikh_IC, 0, 1), cmap='gray')
axes[1, 1].set_title(f'Tikhonov+GCV (IC: σ={sigma_A})\n'
                     f'λ={lam_gcv:.1e}, PSNR={psnr_tikh_IC:.1f} dB')
axes[1, 1].axis('off')

axes[1, 2].imshow(np.clip(x_tikh_noIC, 0, 1), cmap='gray')
axes[1, 2].set_title(f'Tikhonov+GCV (非IC: σ={sigma_B})\nPSNR={psnr_tikh_noIC:.1f} dB')
axes[1, 2].axis('off')

plt.suptitle('朴素逆重建、正则化与 Inverse Crime 对比', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_3_4_不适定性与InverseCrime.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ---- 12. 可视化：eps 敏感性 + PSNR vs 建模偏差 ----
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

# 左图：无噪声病态反演对 eps 的敏感性
ax1.semilogx(eps_list, psnr_vs_eps, 's-', color='darkgreen', markersize=5, linewidth=1.8)
ax1.axhline(psnr_naive_noisy, color='r', linestyle=':', alpha=0.8,
            label=f'含噪直接反卷积 = {psnr_naive_noisy:.1f} dB（地板）')
idx_default = int(np.argmin(np.abs(eps_list - 1e-15)))
ax1.scatter([1e-15], [psnr_vs_eps[idx_default]], color='orange', zorder=5, s=90,
            label=f'默认 eps=1e-15: {psnr_vs_eps[idx_default]:.1f} dB（截断=硬低通）')
ax1.set_xlabel('截断阈值 eps（模值下界）', fontsize=12)
ax1.set_ylabel('PSNR (dB)', fontsize=12)
ax1.set_title('无噪声病态反演仍不"免费":\n数值舍入放大 vs 截断低通', fontsize=13)
# 手动刻度：semilogx 默认的 10^{-x} mathtext 上标减号在中文字体下显示为豆腐块
_xt = [1e-30, 1e-25, 1e-20, 1e-15, 1e-10, 1e-5]
ax1.set_xticks(_xt)
ax1.set_xticklabels(['1e-30', '1e-25', '1e-20', '1e-15', '1e-10', '1e-5'])
ax1.set_xlim(3e-31, 3e-3)
ax1.legend(fontsize=10, loc='upper center')
ax1.grid(True, which='both', alpha=0.3)
ax1.annotate('eps 过小: FFT 舍入误差(~1e-16)\n被 1/H 放大',
             xy=(1e-25, 6.5), fontsize=9.5, ha='center',
             bbox=dict(boxstyle='round', fc='#ffe0e0', alpha=0.8))
ax1.annotate('eps 过大: 高频被整体截断\n等价强低通',
             xy=(8e-4, 23.8), fontsize=9.5, ha='center',
             bbox=dict(boxstyle='round', fc='#e0e8ff', alpha=0.8))

# 右图：偏差曲线，GCV 选参 vs 固定 lambda
ax2.plot(deviations, psnr_curve_gcv, 'o-', color='blue', markersize=8, linewidth=2,
         label=f'Tikhonov, GCV 选参 (λ={lam_gcv:.1e})')
ax2.plot(deviations, psnr_curve_fixed, '--', color='gray', markersize=6, linewidth=1.8,
         marker='s', label=f'对照: 固定 λ={lam_fixed:.0e}（次优调参）')
ax2.axvline(x=0, color='r', linestyle='--', alpha=0.7,
            label=f'IC 点 (σ_真实=σ_模型={sigma_A:g})')
ax2.set_xlabel('建模偏差 Δσ = σ_真实 - σ_模型', fontsize=12)
ax2.set_ylabel('PSNR (dB)', fontsize=12)
ax2.set_title('Inverse Crime 警示：IC 点偏乐观，\n大偏差（正负两侧）系统性退化', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
for d, p in zip(deviations, psnr_curve_gcv):
    ax2.annotate(f'{p:.1f}', (d, p), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=8.5, color='blue')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_3_4_PSNR_vs_建模偏差.png'), dpi=150, bbox_inches='tight')
plt.close(fig2)

# ---- 13. 打印结果 ----
print("\n=== 实验 1.3-4：不适定性、正则化与 Inverse Crime ===")
print(f"[1] 不适定性：病态算子(sigma={sigma_A})含噪直接反卷积 -> "
      f"PSNR={psnr_naive_noisy:.1f} dB（噪声被 1/H 爆炸式放大）")
print(f"[2] 可解性取决于条件数：良性算子(sigma={sigma_well})无噪声直接反卷积 -> "
      f"PSNR={psnr_well_naive:.0f} dB（精确恢复）；")
print(f"    但病态算子即使无噪声，朴素反演也受 FFT 舍入与截断制约（见左图 eps 敏感性）")
print(f"[3] 正则化：GCV 数据驱动选 lambda={lam_gcv:.2e}（未使用真值 x），"
      f"IC 重建 PSNR={psnr_tikh_IC:.1f} dB")
print(f"[4] Inverse Crime：同模型(sigma={sigma_A}) -> {psnr_tikh_IC:.1f} dB；"
      f"失配(sigma={sigma_B}) -> {psnr_tikh_noIC:.1f} dB")
print(f"    Monte Carlo({n_mc} 个噪声实现, 固定 lambda={lam_gcv:.2e})："
      f"IC={mc_ic.mean():.2f}+/-{mc_ic.std():.3f} dB, "
      f"非IC={mc_noic.mean():.2f}+/-{mc_noic.std():.3f} dB")
print(f"    配对差 = {mc_diff.mean():.3f}+/-{mc_diff.std():.3f} dB，"
      f"其中 {100.0 * np.mean(mc_diff > 0):.0f}% 的实现 IC 更乐观")
print("\n--- PSNR 随建模偏差变化（GCV 选参 / 固定1e-2 对照）---")
for sb, d, pg, pf in zip(sigma_B_list, deviations, psnr_curve_gcv, psnr_curve_fixed):
    marker = " <- IC" if abs(sb - sigma_A) < 1e-9 else ""
    print(f"  sigma_真实={sb:5.1f} (delta_sigma={d:+.1f}) -> "
          f"GCV: {pg:6.2f} dB | 固定: {pf:6.2f} dB{marker}")
print("\n机理：重建 x_hat = G(H_模型,lambda) H_真实 x + G(H_模型,lambda) epsilon。")
print("  噪声项只依赖模型与 lambda，与真实算子无关；失配只改变信号传递——")
print("  正偏差残留模糊，负偏差过冲振铃。大偏差两侧都退化，IC 设定掩盖这一风险。")
print("  固定过强 lambda 会使 IC 点成为洼地并伪造'负偏差有益'，调参必须数据驱动。")
print("\n注：定量数值基于 astronaut 单幅图像与噪声水平 sigma=0.01；")
print("  PSNR 随图像内容变化，但'IC 偏乐观、大偏差系统退化'的趋势由线性滤波")
print("  理论保证，不依赖于特定图像（噪声实现波动已由 Monte Carlo 给出）。")

# ===== 保存数值结果 =====
import json
results_summary = {
    'experiment': '1.3-4 不适定性、正则化与 Inverse Crime',
    'image': 'skimage.data.astronaut (rgb2gray)',
    'image_size': int(n),
    'sigma_model_illconditioned': float(sigma_A),
    'sigma_true_mismatch_demo': float(sigma_B),
    'sigma_well_conditioned_demo': float(sigma_well),
    'noise_sigma': float(noise_sigma),
    'lambda_selection': 'GCV (仅用观测y与模型H，未使用真值x；在IC观测上选定后固定)',
    'tikhonov_lambda_gcv': float(lam_gcv),
    'tikhonov_lambda_fixed_for_comparison': float(lam_fixed),
    'psnr_naive_noisy_dB': float(round(psnr_naive_noisy, 2)),
    'psnr_naive_clean_well_conditioned_dB': float(round(psnr_well_naive, 2)),
    'psnr_tikhonov_IC_gcv_dB': float(round(psnr_tikh_IC, 2)),
    'psnr_tikhonov_noIC_gcv_dB': float(round(psnr_tikh_noIC, 2)),
    'psnr_IC_vs_noIC_diff_dB_single_realization': float(round(psnr_tikh_IC - psnr_tikh_noIC, 2)),
    'monte_carlo_paired': {
        'n_noise_realizations': int(n_mc),
        'lambda_used': float(lam_gcv),
        'lambda_fixed_after_IC_GCV_selection': True,
        'IC_psnr_mean_dB': float(round(mc_ic.mean(), 3)),
        'IC_psnr_std_dB': float(round(mc_ic.std(), 4)),
        'noIC_psnr_mean_dB': float(round(mc_noic.mean(), 3)),
        'noIC_psnr_std_dB': float(round(mc_noic.std(), 4)),
        'paired_diff_mean_dB': float(round(mc_diff.mean(), 3)),
        'paired_diff_std_dB': float(round(mc_diff.std(), 4)),
        'fraction_realizations_IC_better': float(round(np.mean(mc_diff > 0), 3)),
    },
    'eps_sensitivity_noiseless_naive_sigma5': [
        {'eps': float(e), 'psnr_dB': float(round(p, 2))}
        for e, p in zip(eps_list, psnr_vs_eps)],
    'psnr_vs_modeling_deviation': {
        f'dsigma_{d:+.1f}': {
            'gcv_lambda': float(round(pg, 2)),
            'fixed_1e-2': float(round(pf, 2)),
        } for d, pg, pf in zip(deviations, psnr_curve_gcv, psnr_curve_fixed)},
}

def _to_native(obj):
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
