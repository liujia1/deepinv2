import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import os
import sys

# ====== 中文字体配置（兼容本地和 Google Colab）======
_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验1.5-2', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验1.5-2')
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _chinese_path)
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=_chinese_path)
# ========================================================

np.random.seed(42)

# ---- 1. 准备小尺寸问题 ----
# 警告：n=32 时 N=1024，矩阵求逆 O(N³) 约 10⁹ 次运算，勉强可行
# n>32（如 n=64 → N=4096）会导致 ~1GB 内存和数分钟求逆时间
n = 32
assert n <= 32, "n>32 时矩阵求逆不可行，请改用迭代法（如共轭梯度）"
phantom = resize(shepp_logan_phantom(), (n, n), order=3,
                 preserve_range=True, anti_aliasing=True)
x = phantom / phantom.max()
x_vec = x.ravel()
N = n * n

# ---- 2. 构造模糊算子矩阵 A ----
def gaussian_psf(size, sigma):
    ax = np.concatenate((np.arange(0, size // 2), np.arange(-size // 2, 0)))
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return h / h.sum()

h = gaussian_psf(n, sigma=2.0)
H_fft = np.fft.fft2(h)

A = np.zeros((N, N))
for j in range(N):
    e_j = np.zeros(N)
    e_j[j] = 1.0
    A[:, j] = np.real(np.fft.ifft2(H_fft * np.fft.fft2(e_j.reshape(n, n)))).ravel()
# 注：以上使用 FFT 实现循环卷积，隐含周期边界条件假设
# 实际图像通常不满足周期边界，边缘处会产生振铃伪影

# ---- 3. 生成含噪观测 ----
sigma_noise = 0.05
y = A @ x_vec + sigma_noise * np.random.randn(N)

# ---- 4. 贝叶斯框架：高斯噪声 + 高斯先验 → Tikhonov ----
# 先验: x ~ N(0, σ_x² I)
# 似然: y|x ~ N(Ax, σ² I)
# 后验: x|y ~ N(μ_post, Σ_post)
# μ_post = (A^T A + λ I)^{-1} A^T y, λ = σ²/σ_x²

sigma_prior = 0.5
lam = sigma_noise ** 2 / sigma_prior ** 2

# 闭式解：直接矩阵求逆（小规模问题可行）
AtA = A.T @ A
Aty = A.T @ y
mu_post = np.linalg.solve(AtA + lam * np.eye(N), Aty)

# ---- 5. 后验不确定性：后验方差 ----
# Σ_post = σ² (A^T A + λ I)^{-1}
# 用 Cholesky 分解计算对角线
# 注：此处仍构造 N×N 矩阵 Z，内存与 inv 相当
# 优势在于数值稳定性更好，且可扩展为只取对角线的迭代版本
L = np.linalg.cholesky(AtA + lam * np.eye(N))
Z = np.linalg.solve(L, np.eye(N))
post_var = sigma_noise**2 * np.sum(Z**2, axis=0).reshape(n, n)

# ---- 6. λ 扫描：验证 λ=σ²/σ_x² 附近最优 ----
# 注意：λ* = σ²/σ_x² 最小化的是期望 MSE（对噪声样本取平均），
# 而非保证单次实验 PSNR 最高。单次结果因噪声实现不同会有波动。
# 扫描时直接算 MSE（不 clip），避免 clip 掩盖负值导致的误差低估
# 注：λ<1e-4 时系统极度病态，PSNR 下降同时包含数值误差的贡献
lambdas_sweep = np.logspace(-5, 1, 50)  # 峰值在 λ≈0.01，1e1 已足够
psnrs = []
for l in lambdas_sweep:
    mu_l = np.linalg.solve(AtA + l * np.eye(N), Aty)
    mse = np.mean((x - mu_l.reshape(n, n))**2)
    psnrs.append(10 * np.log10(1.0 / mse) if mse > 1e-12 else float('-inf'))

# 选取三个代表性 λ 进行可视化对比
lambdas_demo = [1e-4, lam, 10.0]
recons_demo = {}
for l in lambdas_demo:
    mu_l = np.linalg.solve(AtA + l * np.eye(N), Aty)
    recons_demo[l] = mu_l.reshape(n, n)

# ---- 7. 可视化 ----
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# 原始图像
axes[0, 0].imshow(x, cmap='gray')
axes[0, 0].set_title('原始图像 x')
axes[0, 0].axis('off')

# 模糊含噪观测
axes[0, 1].imshow(y.reshape(n, n), cmap='gray')
axes[0, 1].set_title(f'模糊含噪观测 y\n$\\sigma$={sigma_noise}')
axes[0, 1].axis('off')

# 贝叶斯闭式解
mse_closed = np.mean((x - mu_post.reshape(n, n))**2)
psnr_closed = 10 * np.log10(1.0 / mse_closed) if mse_closed > 1e-12 else float('-inf')
axes[0, 2].imshow(np.clip(mu_post.reshape(n, n), 0, 1), cmap='gray')
axes[0, 2].set_title(f'贝叶斯后验均值 $\\mu_{{post}}$\n$\\lambda=\\sigma^2/\\sigma_x^2$={lam:.4f}\nPSNR={psnr_closed:.1f}dB')
axes[0, 2].axis('off')

# 后验不确定性图（直接显示方差，不叠加原图）
im = axes[0, 3].imshow(post_var, cmap='hot')
axes[0, 3].set_title('后验方差 $\\mathrm{diag}(\\Sigma_{{post}})$\n不确定性量化')
axes[0, 3].axis('off')
plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

# λ 扫描展示
lambda_labels = []
for l in lambdas_demo:
    if np.isclose(l, lam, rtol=1e-3):
        lambda_labels.append(f'$\\lambda$={lam:.4f}（贝叶斯最优）')
    elif l < lam:
        lambda_labels.append(f'$\\lambda$={l:.1g}（过小）')
    else:
        lambda_labels.append(f'$\\lambda$={l:.1g}（过大）')

for i, l in enumerate(lambdas_demo):
    img = np.clip(recons_demo[l], 0, 1)
    mse_l = np.mean((x - recons_demo[l])**2)
    psnr_l = 10 * np.log10(1.0 / mse_l) if mse_l > 1e-12 else float('-inf')
    axes[1, i].imshow(img, cmap='gray')
    axes[1, i].set_title(f'{lambda_labels[i]}\nPSNR={psnr_l:.1f}dB')
    axes[1, i].axis('off')

# 合并 λ 扫描曲线到第4个位置
axes[1, 3].semilogx(lambdas_sweep, psnrs, 'b-', linewidth=2)
axes[1, 3].axvline(x=lam, color='r', linestyle='--',
                   label=f'$\\lambda=\\sigma^2/\\sigma_x^2$={lam:.4f}')
axes[1, 3].set_xlabel('正则化参数 $\\lambda$')
axes[1, 3].set_ylabel('PSNR (dB)')
axes[1, 3].set_title('$\\lambda$ 扫描：$\\lambda=\\sigma^2/\\sigma_x^2$ 附近最优\n左：$\\lambda$过小（欠正则化） 右：$\\lambda$过大（过正则化）')
axes[1, 3].legend()
axes[1, 3].grid(True)

plt.suptitle('Tikhonov 正则化的贝叶斯验证\n后验 = 似然 × 先验 → 后验能量 = 数据项 + 正则项',
             fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_5_2_Tikhonov贝叶斯验证.png'),
            dpi=150, bbox_inches='tight')
plt.show()

print("\n=== 贝叶斯验证 ===")
print(f"后验均值 PSNR: {psnr_closed:.2f} dB")
print(f"贝叶斯最优 λ = σ²/σ_x² = {sigma_noise**2:.4f}/{sigma_prior**2:.4f} = {lam:.4f}")
print(f"λ 扫描中 PSNR 最大值: {max(psnrs):.2f} dB")
print(f"对应 λ: {lambdas_sweep[np.argmax(psnrs)]:.4f} (期望最优 {lam:.4f})")
print(f"注：λ*={lam:.4f}=σ²/σ_x² 最小化期望 MSE，单次实验峰值可能略有偏移")
print("\n结论：λ*=σ²/σ_x² 使 PSNR 接近峰值，验证了贝叶斯-正则化等价性（期望最优性）。")

# ===== 保存数值结果 =====
import json
results_summary = {
    'image_size': n,
    'noise_sigma': float(sigma_noise),
    'prior_sigma': float(sigma_prior),
    'bayesian_lambda': float(round(lam, 6)),
    'psnr_posterior_mean_dB': float(round(psnr_closed, 2)),
    'lambda_sweep_max_psnr_dB': float(round(max(psnrs), 2)),
    'lambda_sweep_best_lambda': float(round(lambdas_sweep[np.argmax(psnrs)], 6)),
    'lambda_demo_psnr': {f'lambda_{l:.4g}': float(round(10 * np.log10(1.0 / np.mean((x - recons_demo[l])**2)), 2)) if np.mean((x - recons_demo[l])**2) > 1e-12 else float('-inf') for l in lambdas_demo},
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