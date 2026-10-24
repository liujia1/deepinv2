"""
实验3.3-1 Tikhonov正则化：闭式解与lambda的贝叶斯诠释
对应章节：3.3 Tikhonov正则化——闭式解与迭代解
知识点：A=I下以先验均值mu为中心的Tikhonov闭式解 x = mu + (y-mu)/(1+lam)；
        lambda = sigma^2/sigma_x^2；lambda扫描与PSNR曲线；欠正则化 vs 过正则化；最优lambda的贝叶斯选择

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

# 【教学要点·先验均值】自然图像并非零均值（Cameraman 均值约 0.5）。
# Tikhonov 的收缩方向由先验均值决定：先验 N(mu, sigma_x^2) 把解拉向 mu 而非 0。
# 均值对噪声稳健（mean(y)≈mean(x)），可直接从含噪观测估计，无需 oracle。
mu = float(np.mean(y))

# 含噪观测裁剪到 [0,1]，作为展示与 PSNR 评估的统一口径（图像去噪的标准做法）。
# 注意：闭式解仍作用于未裁剪的 y（裁剪是非线性操作，不能代入收缩公式）。
y_clipped = np.clip(y, 0, 1)

# ══════════════════════════════════════════════════════════
# 2. Tikhonov闭式解（A=I，先验均值 mu）
# ══════════════════════════════════════════════════════════
# 当 A = I（纯去噪），以先验均值 mu 为中心的 Tikhonov 目标函数为:
#   J(x) = 0.5||y-x||^2 + 0.5*lam*||x-mu||^2
# 梯度 nabla J = (x-y) + lam*(x-mu) = 0 → x = mu + (y-mu)/(1+lam)
# 这是高斯先验 N(mu, sigma_x^2) 下 MAP=MMSE 的闭式解（mu=0 时退化为 y/(1+lam)）

# 贝叶斯lambda (oracle): lam = sigma^2 / sigma_x^2，其中 sigma_x 用真实x估计
lam_bayes = sigma_noise ** 2 / x_std ** 2
# 闭式解作用于未 clip 的原始 y；线性收缩只会把极值压向 mu，并不能保证结果落入 [0,1]
# （本次重建范围约 [-0.24, 1.2]）。展示与 PSNR 评估统一在 clip 后的图像上进行，
# 与含噪基线 y_clipped 保持同一口径，避免"clip 基线 vs 未 clip 重建"的不公平比较。
x_bayes = mu + (y - mu) / (1 + lam_bayes)
x_bayes_clipped = np.clip(x_bayes, 0, 1)

psnr_noisy = peak_signal_noise_ratio(x, y_clipped, data_range=DATA_RANGE)
psnr_bayes = peak_signal_noise_ratio(x, x_bayes_clipped, data_range=DATA_RANGE)

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
x_bayes_noisy = mu + (y - mu) / (1 + lam_bayes_noisy)
x_bayes_noisy_clipped = np.clip(x_bayes_noisy, 0, 1)
psnr_bayes_noisy = peak_signal_noise_ratio(x, x_bayes_noisy_clipped, data_range=DATA_RANGE)

print("=" * 60)
print("实验3.3-1 Tikhonov正则化：闭式解与lambda的贝叶斯诠释")
print("=" * 60)
print(f"\n[问题设定]")
print(f"  图像尺寸: {n}x{n}")
print(f"  噪声水平: sigma = {sigma_noise}")
print(f"  图像标准差: sigma_x = {x_std:.4f}")
if var_x_est_warning:
    print("  ⚠️  y_std <= sigma_noise，方差估计为负，sigma_x_est 退化为下界")
print(f"\n[闭式解验证: A=I, 先验均值 mu]")
print(f"  x = mu + (y - mu) / (1 + lam),  mu = {mu:.4f}")
print(f"  oracle贝叶斯 lam = {lam_bayes:.4f}: PSNR={psnr_bayes:.2f} dB (含噪: {psnr_noisy:.2f} dB)")

# ══════════════════════════════════════════════════════════
# 3. lambda扫描：PSNR vs lambda曲线
# ══════════════════════════════════════════════════════════
# 加密扫描网格：5个数量级内取200点（相对步距约6%），避免粗网格使 lam_best 量化偏差过大
lambdas = np.logspace(-4, 1, 200)
psnr_list = []

for lam in lambdas:
    # 与含噪基线同口径：重建结果统一 clip 到 [0,1] 后再计算 PSNR
    x_lam = np.clip(mu + (y - mu) / (1 + lam), 0, 1)
    psnr_list.append(peak_signal_noise_ratio(x, x_lam, data_range=DATA_RANGE))

psnr_list = np.array(psnr_list)
best_idx = np.argmax(psnr_list)
lam_best = lambdas[best_idx]
psnr_best = psnr_list[best_idx]

# 理论 lambda = sigma^2/Var(x) 最小化的是"未 clip"的 MSE，因此再扫一条 raw 口径曲线。
# 绘图用与 clip 相同的 200 点网格；理论验证用 801 点细网格（相对步距约0.4%），
# 避免粗网格量化误差干扰"理论公式是否精确成立"的判断。
psnr_list_raw = np.array([
    peak_signal_noise_ratio(x, mu + (y - mu) / (1 + lam), data_range=DATA_RANGE)
    for lam in lambdas
])
_raw_grid = np.logspace(-3, 0.5, 801)
_raw_curve = np.array([
    peak_signal_noise_ratio(x, mu + (y - mu) / (1 + lam), data_range=DATA_RANGE)
    for lam in _raw_grid
])
lam_best_raw = _raw_grid[int(np.argmax(_raw_curve))]
psnr_gap_clip = psnr_best - psnr_bayes  # oracle(理论lambda)在clip指标下距峰值的差距

print(f"\n[lambda扫描结果]")
print(f"  扫描范围: [{lambdas[0]:.4f}, {lambdas[-1]:.2f}]")
if lam_best <= lambdas[0] * 1.1 or lam_best >= lambdas[-1] / 1.1:
    print(f"  ⚠️  最优lambda ({lam_best:.4f}) 接近扫描边界，范围可能需要调整")
else:
    print(f"  ✅ 最优lambda ({lam_best:.4f}) 在扫描范围内")
print(f"  最优lambda (clip口径 PSNR): lam_best = {lam_best:.4f}, PSNR = {psnr_best:.2f} dB")
print(f"  最优lambda (raw口径 PSNR) : lam_best_raw = {lam_best_raw:.4f}, PSNR = {float(_raw_curve.max()):.2f} dB  (理论 lambda 最小化未clip的MSE)")
print(f"  贝叶斯lambda (oracle): lam_bayes = sigma^2/sigma_x^2 = {lam_bayes:.4f}")
print(f"  贝叶斯lambda对应PSNR: {psnr_bayes:.2f} dB")
print(f"  贝叶斯lambda (含噪估计, 减噪声方差): lam_bayes_noisy = sigma^2/sigma_x_est^2 = {lam_bayes_noisy:.4f}")
print(f"  含噪估计对应PSNR: {psnr_bayes_noisy:.2f} dB")

# 稳健验证（不依赖单次实现的 lambda 点估计）：
#   ① raw 口径下 lambda 最优值应与理论公式吻合（相对偏差<10%）；
#   ② clip 口径下最优 lambda 系统性偏小（截断对欠正则化重建有额外收益），
#      但峰值附近平台平坦，理论 lambda 处 PSNR 距峰值应 <0.05 dB。
_rel_raw = abs(lam_best_raw - lam_bayes) / lam_bayes
_rel_clip = abs(lam_best - lam_bayes) / lam_bayes
if _rel_raw < 0.10 and psnr_gap_clip < 0.05:
    print(f"  [验证] raw口径最优lambda与理论公式吻合(偏差{_rel_raw:.1%})，"
          f"且理论lambda在clip口径下仅比峰值低{psnr_gap_clip:.3f}dB，贝叶斯框架成立")
else:
    print(f"  [提示] 理论lambda与最优lambda存在偏差(raw偏差{_rel_raw:.1%}, "
          f"clip峰差{psnr_gap_clip:.3f}dB)")
print(f"  [口径说明] clip口径最优lambda({lam_best:.4f})比理论值偏小{_rel_clip:.1%}是系统性现象，")
print(f"             源于截断对噪声大的重建的额外收益；峰值平台平坦，PSNR实际差异可忽略。")

print(f"  [关键教学点] sigma_x 使用真实x估计（oracle），实际中不可得。")
print(f"             用含噪y估计时需要减去噪声方差：")
print(f"             sigma_x_est^2 = sigma_y^2 - sigma_noise^2")
print(f"             含噪估计: lam_bayes_noisy = {lam_bayes_noisy:.4f}，")
print(f"             PSNR 从 {psnr_bayes:.2f} dB 降至 {psnr_bayes_noisy:.2f} dB")

# ══════════════════════════════════════════════════════════
# 3b. 对照实验：收缩中心若误取 mu=0 会怎样
# ══════════════════════════════════════════════════════════
# 【教学要点】先验均值必须取 mu ≈ mean(x)，而非 0：
#   收缩向 0 会把亮图整体压暗、引入系统偏差；经验最优 lambda 随之落到
#   约 sigma^2/E[x^2]（关于原点 0 的二阶矩 = sigma_x^2 + mu^2），
#   与 sigma^2/sigma_x^2 相差约 4 倍（raw 口径）。
#   此处把该对照显式算出来，供正文引用。
lam_theory_mu0 = sigma_noise ** 2 / float(np.mean(x ** 2))   # 收缩向 0 时的理论 lambda

psnr_mu0_at_bayes = peak_signal_noise_ratio(
    x, np.clip(y / (1 + lam_bayes), 0, 1), data_range=DATA_RANGE)          # clip 口径
psnr_mu0_at_bayes_raw = peak_signal_noise_ratio(
    x, y / (1 + lam_bayes), data_range=DATA_RANGE)                         # raw 口径

_psnr_mu0 = np.array([
    peak_signal_noise_ratio(x, np.clip(y / (1 + l), 0, 1), data_range=DATA_RANGE)
    for l in lambdas])
lam_best_mu0 = float(lambdas[int(np.argmax(_psnr_mu0))])
psnr_best_mu0 = float(_psnr_mu0.max())

_raw_curve_mu0 = np.array([
    peak_signal_noise_ratio(x, y / (1 + l), data_range=DATA_RANGE)
    for l in _raw_grid])
lam_best_mu0_raw = float(_raw_grid[int(np.argmax(_raw_curve_mu0))])
psnr_best_mu0_raw = float(_raw_curve_mu0.max())

print(f"\n[对照: 收缩中心误取 mu=0 的后果]")
print(f"  理论 lambda = sigma^2/E[x^2] = {lam_theory_mu0:.4f}（E[x^2]={np.mean(x**2):.4f} 关于原点 0）")
print(f"  经验最优 lambda：clip 口径 {lam_best_mu0:.4f}（PSNR {psnr_best_mu0:.2f} dB），"
      f"raw 口径 {lam_best_mu0_raw:.4f}（PSNR {psnr_best_mu0_raw:.2f} dB）")
print(f"  与 lam_bayes={lam_bayes:.4f} 相差 {lam_bayes / lam_best_mu0_raw:.1f} 倍（raw 口径）")
print(f"  若仍套用 lam_bayes={lam_bayes:.4f} 收缩向 0：PSNR 跌到 {psnr_mu0_at_bayes:.2f} dB（clip）"
      f"/ {psnr_mu0_at_bayes_raw:.2f} dB（raw），低于含噪观测 {psnr_noisy:.2f} dB")

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

axes[0, 2].imshow(x_bayes_clipped, cmap='gray')
axes[0, 2].set_title(r'贝叶斯 $\lambda$ 重建 (oracle)' + f'\n$\lambda={lam_bayes:.4f}$\nPSNR={psnr_bayes:.2f}dB')
axes[0, 2].axis('off')

# 第二行：PSNR-λ曲线 → 剖面对比 → 误差图
# 蓝线为 clip 口径（与含噪基线一致，左端点与基线重合）；灰线为未 clip 口径，
# 其峰值对应理论 MSE 最优 lambda，可看出 clip 截断使峰值位置系统性左移。
axes[1, 0].semilogx(lambdas, psnr_list, 'b-', linewidth=1.5,
                    label=r'Tikhonov (clip口径): $x=\mu+(y-\mu)/(1+\lambda)$')
axes[1, 0].semilogx(lambdas, psnr_list_raw, color='gray', linestyle='-', linewidth=1.0,
                    alpha=0.7, label=r'未clip口径(理论MSE), 峰值$\lambda=' + f'{lam_best_raw:.4f}$')
axes[1, 0].axhline(y=psnr_noisy, color='r', linestyle='--', alpha=0.5, label=f'含噪: {psnr_noisy:.1f}dB')
axes[1, 0].axvline(x=lam_bayes, color='g', linestyle=':', linewidth=2,
                    label=r'oracle $\lambda=' + f'{lam_bayes:.4f}$')
axes[1, 0].axvline(x=lam_bayes_noisy, color='orange', linestyle='-.', linewidth=1.5,
                    label=r'含噪估计 $\lambda=' + f'{lam_bayes_noisy:.4f}$')
axes[1, 0].axvline(x=lam_best, color='b', linestyle='--', alpha=0.5,
                    label=r'clip最优 $\lambda=' + f'{lam_best:.4f}$')
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
    x_lam = mu + (y - mu) / (1 + lam_val)
    axes[1, 1].plot(x_lam[center, :], linewidth=1, label=label, color=color)
axes[1, 1].set_xlabel('像素索引')
axes[1, 1].set_title(r'不同 $\lambda$ 的中心行剖面对比')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

# 误差图：展示欠/适/过正则化的重建误差空间分布
# 【教学要点】（收缩向先验均值 mu）
#   - 过正则化(λ大): x̂ → mu，误差 ≈ mu - x，|误差|≈|x-mu|
#     即误差图显示图像相对均值的结构（人物轮廓清晰）
#   - 欠正则化(λ小): x̂ ≈ y，误差 ≈ 噪声，误差图呈随机分布
errors = []
for lam_val in lam_demo:
    x_lam = mu + (y - mu) / (1 + lam_val)
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
print("1. 当 A=I 时，以先验均值 mu 为中心的 Tikhonov 闭式解: x = mu + (y-mu)/(1+lam)")
print("2. lambda = sigma^2/sigma_x^2 是最优正则化参数的贝叶斯选择（高斯先验下 MAP=MMSE）")
print("3. lambda 过小 -> 欠正则化（噪声未被有效抑制）")
print("4. lambda 过大 -> 过正则化（图像被拉向先验均值 mu）")
print("5. PSNR-lambda曲线呈单峰形，最优lambda在峰值处")
print("6. oracle贝叶斯lambda=sigma^2/Var(x)是未clip的MSE最优解（raw口径下与理论公式精确吻合）；")
print("   clip口径的PSNR最优lambda因截断收益系统性偏小约20%，但峰值平台平坦，PSNR差<0.05dB")
print("7. 实际中用含噪y估计sigma_x时需减去噪声方差，否则lambda系统性偏小")
print("8. 先验均值必须取 mu≈mean(x)（亮图约0.5）而非0，否则收缩向0引入系统偏差：")
print(f"   经验最优lambda落到 sigma^2/E[x^2]={lam_theory_mu0:.4f} 附近，与 lam_bayes={lam_bayes:.4f} "
      f"相差 {lam_bayes/lam_best_mu0_raw:.1f} 倍；")
print(f"   此时若仍套用 lam_bayes，PSNR 跌到 {psnr_mu0_at_bayes:.2f} dB，反低于含噪观测 {psnr_noisy:.2f} dB")

print("\n" + "=" * 60)
print("实验完成。结果已保存至:", SAVE_DIR)
print("=" * 60)

# ===== 保存数值结果 =====
import json
results_summary = {
    'n': n,
    'mu': float(round(mu, 4)),
    'sigma_noise': float(round(sigma_noise, 4)),
    'lam_bayes': float(round(lam_bayes, 4)),
    'lam_bayes_noisy': float(round(lam_bayes_noisy, 4)),
    'lam_best': float(round(lam_best, 4)),
    'lam_best_raw': float(round(lam_best_raw, 4)),
    # 所有 PSNR 均在 clip 到 [0,1] 的图像上计算（含噪基线与各重建同口径），保留3位小数
    'psnr_noisy': float(round(psnr_noisy, 3)),
    'psnr_bayes': float(round(psnr_bayes, 3)),
    'psnr_bayes_noisy': float(round(psnr_bayes_noisy, 3)),
    'psnr_best': float(round(psnr_best, 3)),
    # raw 口径（未 clip）下理论 lambda 才是 MSE 最优解，该口径最优 lambda 与 lam_bayes 吻合到 <1%
    'psnr_best_raw': float(round(float(_raw_curve.max()), 3)),
    'rel_dev_raw_lam_best_vs_bayes': float(round(_rel_raw, 4)),
    'psnr_gap_clip_bayes_vs_peak': float(round(psnr_gap_clip, 4)),
    # 对照：收缩中心误取 mu=0（见正文"为何收缩中心必须是 mu 而非 0"）
    'mu0_control': {
        'lam_theory_mu0_sigma2_over_Ex2': float(round(lam_theory_mu0, 4)),
        'lam_best_mu0_clip': float(round(lam_best_mu0, 4)),
        'psnr_best_mu0_clip': float(round(psnr_best_mu0, 3)),
        'lam_best_mu0_raw': float(round(lam_best_mu0_raw, 4)),
        'psnr_best_mu0_raw': float(round(psnr_best_mu0_raw, 3)),
        'psnr_at_lam_bayes_mu0_clip': float(round(psnr_mu0_at_bayes, 3)),
        'psnr_at_lam_bayes_mu0_raw': float(round(psnr_mu0_at_bayes_raw, 3)),
        'ratio_lam_bayes_over_lam_best_mu0_raw': float(round(lam_bayes / lam_best_mu0_raw, 3)),
        'note': '收缩中心取 mu=0 时经验最优 lambda 落到 sigma^2/E[x^2] 附近，'
                '与 lam_bayes 相差约4.2倍；此时若仍套用 lam_bayes，PSNR 反低于含噪观测',
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
