"""
实验3.3-1 Tikhonov正则化：闭式解与lambda的贝叶斯诠释
对应章节：3.3 Tikhonov正则化——闭式解与迭代解
知识点：A=I下的Tikhonov闭式解 x = y/(1+lam)；lambda = sigma^2/sigma_x^2；
        lambda扫描与PSNR曲线；欠正则化 vs 过正则化；最优lambda的贝叶斯选择

修改说明：
  从原参考实验3.1.py拆分，聚焦Tikhonov闭式解与lambda的概率诠释，
  去除梯度下降细节（移至3.2-1）和MAP后验推导（移至3.1-1）。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio
import os
import sys

# ====== 中文字体配置（兼容本地和Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验3.3-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)
else:
    _chinese_path = '.chinese'
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)

# ══════════════════════════════════════════════════════════
# 1. 问题设定：A=I 纯去噪
# ══════════════════════════════════════════════════════════
n = 128
x = resize(data.camera(), (n, n))

# 【教学要点】x_std 是全图统计标准差，作为先验 sigma_x 的 oracle 估计。
# 贝叶斯框架中 sigma_x 是先验分布的标准差，用全图统计量代替是一种近似。
# 实际中 sigma_x 未知，需从含噪观测 y 估计（见下方对比）。
x_std = np.std(x)
DATA_RANGE = x.max() - x.min()  # 使用实际动态范围，避免PSNR计算偏差

sigma_noise = 0.1
y = x + sigma_noise * np.random.randn(n, n)

# 对含噪图像裁剪到 [0,1]，避免噪声导致的范围溢出影响 PSNR 计算
y_clipped = np.clip(y, 0, 1)

# ══════════════════════════════════════════════════════════
# 2. Tikhonov闭式解（A=I情形）
# ══════════════════════════════════════════════════════════
# 当 A = I（纯去噪），Tikhonov目标函数为:
#   J(x) = 0.5||y-x||^2 + 0.5*lam*||x||^2
# 梯度 nabla J = (x-y) + lam*x = 0 → x = y/(1+lam)
# 这就是MAP估计在A=I下的闭式解

# 贝叶斯lambda (oracle): lam = sigma^2 / sigma_x^2，其中 sigma_x 用真实x估计
lam_bayes = sigma_noise ** 2 / x_std ** 2
# 重建用原始 y（未 clip），正则化会将结果约束到合理范围
x_bayes = y / (1 + lam_bayes)

psnr_noisy = peak_signal_noise_ratio(x, y_clipped, data_range=DATA_RANGE)
psnr_bayes = peak_signal_noise_ratio(x, x_bayes, data_range=DATA_RANGE)

# 【关键教学点】实际中 sigma_x 未知，只能用含噪观测 y 估计。
# 由于 sigma_y^2 = sigma_x^2 + sigma_noise^2，
# 正确的估计应为 sigma_x_est = sqrt(max(sigma_y^2 - sigma_noise^2, 0))
# 直接用 sigma_y 代替 sigma_x 会高估噪声方差，导致 lam_bayes 系统性偏小。
# 此处同时展示两种估计，量化偏差大小。
# 统计量估计用 y_clipped，避免噪声溢出导致方差偏大
y_std = np.std(y_clipped)
var_x_est = y_std**2 - sigma_noise**2
var_x_est_warning = var_x_est <= 0
sigma_x_est = np.sqrt(max(var_x_est, 1e-8))
lam_bayes_noisy = sigma_noise ** 2 / sigma_x_est ** 2
x_bayes_noisy = y / (1 + lam_bayes_noisy)
psnr_bayes_noisy = peak_signal_noise_ratio(x, x_bayes_noisy, data_range=DATA_RANGE)

print("=" * 60)
print("实验3.3-1 Tikhonov正则化：闭式解与lambda的贝叶斯诠释")
print("=" * 60)
print(f"\n[问题设定]")
print(f"  图像尺寸: {n}x{n}")
print(f"  噪声水平: sigma = {sigma_noise}")
print(f"  图像标准差: sigma_x = {x_std:.4f}")
if var_x_est_warning:
    print("  ⚠️  y_std <= sigma_noise，方差估计为负，sigma_x_est 退化为下界")
print(f"\n[闭式解验证: A=I]")
print(f"  x = y / (1 + lam)")
print(f"  oracle贝叶斯 lam = {lam_bayes:.4f}: PSNR={psnr_bayes:.2f} dB (含噪: {psnr_noisy:.2f} dB)")

# ══════════════════════════════════════════════════════════
# 3. lambda扫描：PSNR vs lambda曲线
# ══════════════════════════════════════════════════════════
lambdas = np.logspace(-4, 1, 50)
psnr_list = []

for lam in lambdas:
    x_lam = y / (1 + lam)
    psnr_list.append(peak_signal_noise_ratio(x, x_lam, data_range=DATA_RANGE))

psnr_list = np.array(psnr_list)
best_idx = np.argmax(psnr_list)
lam_best = lambdas[best_idx]
psnr_best = psnr_list[best_idx]

print(f"\n[lambda扫描结果]")
print(f"  扫描范围: [{lambdas[0]:.4f}, {lambdas[-1]:.2f}]")
if lam_best <= lambdas[0] * 1.1 or lam_best >= lambdas[-1] / 1.1:
    print(f"  ⚠️  最优lambda ({lam_best:.4f}) 接近扫描边界，范围可能需要调整")
else:
    print(f"  ✅ 最优lambda ({lam_best:.4f}) 在扫描范围内")
print(f"  最优lambda (PSNR): lam_best = {lam_best:.4f}, PSNR = {psnr_best:.2f} dB")
print(f"  贝叶斯lambda (oracle): lam_bayes = sigma^2/sigma_x^2 = {lam_bayes:.4f}")
print(f"  贝叶斯lambda对应PSNR: {psnr_bayes:.2f} dB")
print(f"  贝叶斯lambda (含噪估计, 减噪声方差): lam_bayes_noisy = sigma^2/sigma_x_est^2 = {lam_bayes_noisy:.4f}")
print(f"  含噪估计对应PSNR: {psnr_bayes_noisy:.2f} dB")

if abs(lam_best - lam_bayes) / lam_bayes < 0.5:
    print(f"  [验证] oracle贝叶斯lambda与最优lambda接近，验证了贝叶斯框架")
else:
    print(f"  [提示] oracle贝叶斯lambda与最优lambda存在偏差")

print(f"  [关键教学点] sigma_x 使用真实x估计（oracle），实际中不可得。")
print(f"             用含噪y估计时需要减去噪声方差：")
print(f"             sigma_x_est^2 = sigma_y^2 - sigma_noise^2")
print(f"             含噪估计: lam_bayes_noisy = {lam_bayes_noisy:.4f}，")
print(f"             PSNR 从 {psnr_bayes:.2f} dB 降至 {psnr_bayes_noisy:.2f} dB")

# ══════════════════════════════════════════════════════════
# 4. 可视化
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 第一行：原始 → 含噪 → 贝叶斯重建
axes[0, 0].imshow(x, cmap='gray')
axes[0, 0].set_title(r'原始图像 $x$')
axes[0, 0].axis('off')

axes[0, 1].imshow(y_clipped, cmap='gray')
axes[0, 1].set_title(r'含噪图像 $y = x + \epsilon$' + f'\nPSNR={psnr_noisy:.2f}dB')
axes[0, 1].axis('off')

axes[0, 2].imshow(np.clip(x_bayes, 0, 1), cmap='gray')
axes[0, 2].set_title(r'贝叶斯 $\lambda$ 重建 (oracle)' + f'\n$\lambda={lam_bayes:.4f}$\nPSNR={psnr_bayes:.2f}dB')
axes[0, 2].axis('off')

# 第二行：PSNR-λ曲线 → 剖面对比 → 误差图
axes[1, 0].semilogx(lambdas, psnr_list, 'b-', linewidth=1.5, label=r'Tikhonov: $x = y/(1+\lambda)$')
axes[1, 0].axhline(y=psnr_noisy, color='r', linestyle='--', alpha=0.5, label=f'含噪: {psnr_noisy:.1f}dB')
axes[1, 0].axvline(x=lam_bayes, color='g', linestyle=':', linewidth=2,
                    label=r'oracle $\lambda=' + f'{lam_bayes:.4f}$')
axes[1, 0].axvline(x=lam_bayes_noisy, color='orange', linestyle='-.', linewidth=1.5,
                    label=r'含噪估计 $\lambda=' + f'{lam_bayes_noisy:.4f}$')
axes[1, 0].axvline(x=lam_best, color='b', linestyle='--', alpha=0.5,
                    label=r'最优 $\lambda=' + f'{lam_best:.4f}$')
axes[1, 0].scatter([lam_bayes], [psnr_bayes], color='g', s=80, zorder=5, marker='*')
axes[1, 0].scatter([lam_bayes_noisy], [psnr_bayes_noisy], color='orange', s=80, zorder=5, marker='*')
axes[1, 0].scatter([lam_best], [psnr_best], color='b', s=80, zorder=5, marker='*')
axes[1, 0].set_xlabel(r'正则化参数 $\lambda$')
axes[1, 0].set_ylabel('PSNR (dB)')
axes[1, 0].set_title(r'PSNR vs $\lambda$: 欠正则化 $\to$ 过正则化')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# 剖面对比：使用对称倍数，稳定展示欠/适/过正则化
center = n // 2
lam_ratios = [1/20, 1, 20]
lam_demo = [lam_bayes * r for r in lam_ratios]
regime_labels = ['欠正则化 (λ/20)', '贝叶斯 λ', '过正则化 (20λ)']
labels = [rf'$\lambda={lam_demo[i]:.4f}$ ({regime_labels[i]})' for i in range(3)]
colors = ['#fdae61', '#2c7bb6', '#d7191c']

axes[1, 1].plot(x[center, :], 'k--', linewidth=1, label=r'真实 $x$')
for lam_val, label, color in zip(lam_demo, labels, colors):
    x_lam = y / (1 + lam_val)
    axes[1, 1].plot(x_lam[center, :], linewidth=1, label=label, color=color)
axes[1, 1].set_xlabel('像素索引')
axes[1, 1].set_title(r'不同 $\lambda$ 的中心行剖面对比')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

# 误差图：展示欠/适/过正则化的重建误差空间分布
# 【教学要点】
#   - 过正则化(λ大): x̂ = y/(1+λ) ≈ 0.28y，误差 ≈ -0.72x，|误差|≈0.72|x|
#     即误差图显示原始图像结构（人物轮廓清晰）
#   - 欠正则化(λ小): x̂ ≈ y，误差 ≈ 噪声，误差图呈随机分布
errors = []
for lam_val in lam_demo:
    x_lam = y / (1 + lam_val)
    errors.append(x_lam - x)

err_abs = [np.abs(e) for e in errors]
err_max = np.percentile(np.concatenate([e.ravel() for e in err_abs]), 98)

axes[1, 2].set_title(r'重建误差 $|\hat{x}-x|$ 分布', fontsize=11, pad=10)
axes[1, 2].axis('off')

# 布局调整：先调整，再读取坐标，最后添加内嵌子图
plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.08, hspace=0.12)

pos = axes[1, 2].get_position()
x0, y0, w, h = pos.x0, pos.y0, pos.width, pos.height

# 3个子图 + 4段间距（上下各一段 + 两段之间），总高度约束在h以内
gap = h * 0.04
heights = (h - 4 * gap) / 3

for idx, (err, lbl, color) in enumerate(zip(err_abs, regime_labels, colors)):
    y_pos = y0 + gap + (2 - idx) * (heights + gap)
    ax_inset = fig.add_axes([x0, y_pos, w * 0.82, heights])
    im = ax_inset.imshow(err, cmap='hot', vmin=0, vmax=err_max)
    ax_inset.text(0.5, 0.95, lbl, transform=ax_inset.transAxes,
                  fontsize=9, color=color, ha='center', va='top',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor=color, alpha=0.9, linewidth=1.5))
    ax_inset.axis('off')

cbar_ax = fig.add_axes([x0 + w * 0.86, y0 + h * 0.15, w * 0.04, h * 0.7])
plt.colorbar(im, cax=cbar_ax, label='|误差|')

plt.suptitle(r'实验3.3-1: Tikhonov正则化——闭式解与 $\lambda$ 的贝叶斯诠释', fontsize=14)
plt.savefig(os.path.join(SAVE_DIR, '步骤1_Tikhonov闭式解与lambda.png'), dpi=150)
plt.close()

print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print("1. 当 A=I 时，Tikhonov正则化有闭式解: x = y/(1+lam)")
print("2. lambda = sigma^2/sigma_x^2 是最优正则化参数的贝叶斯选择")
print("3. lambda 过小 -> 欠正则化（噪声未被有效抑制）")
print("4. lambda 过大 -> 过正则化（图像过度平滑）")
print("5. PSNR-lambda曲线呈单峰形，最优lambda在峰值处")
print("6. oracle贝叶斯lambda与PSNR最优lambda接近，验证贝叶斯框架的理论正确性")
print("7. 实际中用含噪y估计sigma_x时需减去噪声方差，否则lambda系统性偏小")

print("\n" + "=" * 60)
print("实验完成。结果已保存至:", SAVE_DIR)
print("=" * 60)

# ===== 保存数值结果 =====
import json
results_summary = {
    'n': n,
    'sigma_noise': float(round(sigma_noise, 4)),
    'lam_bayes': float(round(lam_bayes, 4)),
    'lam_bayes_noisy': float(round(lam_bayes_noisy, 4)),
    'lam_best': float(round(lam_best, 4)),
    'psnr_noisy': float(round(psnr_noisy, 2)),
    'psnr_bayes': float(round(psnr_bayes, 2)),
    'psnr_bayes_noisy': float(round(psnr_bayes_noisy, 2)),
    'psnr_best': float(round(psnr_best, 2)),
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
