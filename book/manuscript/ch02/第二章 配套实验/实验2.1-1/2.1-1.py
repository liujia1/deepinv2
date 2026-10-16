"""
实验2.1-1 先验 = 正则化的数值验证
对应章节：2.1 先验的数学角色：正则化的概率诠释
知识点：
  - 后验能量分解：-ln p(x|y) = 数据项 + 正则项
  - lambda = sigma^2/sigma_x^2 的概率诠释
  - 不同 lambda 对重建效果的影响：欠正则化 vs 过正则化
  - 先验模型必须与估计量/目标函数“口径自洽”：
      * 零均值先验 N(0, sigma0^2 I)：目标函数含 ||x||^2，lambda0 = sigma^2 / E[x^2]
      * 中心化先验 N(mu, tau^2 I)：目标函数含 ||x-mu||^2，lambdac = sigma^2 / Var(x)
        其中均值 mu 可直接从含噪数据估计（误差 O(sigma/sqrt(N))，无需 oracle）
    —— 贝叶斯公式本身没有错；“lambda 失效”来自零均值估计量与中心化
       sigma_x 的口径混搭，以及未建模真实图像约 0.51 的均值。
  - 空间白高斯先验的固有局限：标量收缩的理论增益上限仅约 0.1~0.5 dB；
    TV 等利用空间结构的更强先验收益可达 8 dB 以上（这才是“需要更强先验”的依据）。

素材来源：
  - M1 CompImLab25.ipynb: Tikhonov闭式解 + PSNR度量
  - IP22 statistical_perspective.md: 高斯先验MAP推导

本实验包含两个部分：
  步骤1-2: A != I（模糊逆问题），使用梯度下降求解 Tikhonov 正则化
  步骤3-4: A = I（纯去噪），扫描 lambda 验证贝叶斯诠释
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 静默模式，不弹出GUI窗口
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import gaussian_filter
import os
import sys

# ====== 中文字体配置（兼容本地和 Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验2.1-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)
else:
    _chinese_path = '.chinese'
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()

sys.path.insert(0, _chinese_path)
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=_chinese_path)
# ========================================================

np.random.seed(42)

# ============================================================
# 图像与先验超参数：两种口径一次性定义清楚
# ============================================================
n = 128
N = n * n
x = resize(data.camera(), (n, n))

x_min, x_max = x.min(), x.max()
mu = float(x.mean())                 # 真实均值（仅用于理论对照）
tau2 = float(x.var())                # Var(x)，中心化先验的尺度
s02 = float(np.mean(x ** 2))         # E[x^2]，零均值先验的尺度（= Var + mu^2）
DATA_RANGE = x_max - x_min

print(f"[图像统计] 值域: [{x_min:.4f}, {x_max:.4f}], DATA_RANGE={DATA_RANGE:.4f}")
print(f"[图像统计] 均值 mu={mu:.4f}, Var(x)={tau2:.5f}, E[x^2]={s02:.5f}")
print(f"           (相机图均值明显偏离0，零均值先验存在系统性失配)")

# ============================================================
# 步骤1-2: A != I 模糊逆问题
# ============================================================
print("\n" + "=" * 60)
print("步骤1-2: A != I 模糊逆问题 — 后验能量分解验证")
print("=" * 60)

# ---- 前向算子：高斯模糊 ----
blur_sigma = 2.0

def A(u):
    """高斯模糊前向算子"""
    return gaussian_filter(u, sigma=blur_sigma, mode='reflect')

def AT(u):
    """高斯模糊的伴随算子
    高斯滤波是自伴随算子（高斯核对称），故 A^T = A。
    这是高斯核对称性的结果，对一般前向算子不成立。
    """
    return gaussian_filter(u, sigma=blur_sigma, mode='reflect')

# ---- 噪声与观测 ----
noise_lev = 1e-1
y = A(x) + noise_lev * np.random.randn(n, n)
mu_hat_blur = float(y.mean())  # A 保持直流分量，含噪均值是 mu 的无偏估计

# ---- 两种口径的正则化参数 ----
sigma = noise_lev
lambda0 = sigma ** 2 / s02     # 零均值先验 N(0, s02 I)：与 ||x||^2 目标函数自洽
lambdac = sigma ** 2 / tau2    # 中心化先验 N(mu, tau2 I)：与 ||x-mu||^2 自洽

print(f"\n[参数设定] 噪声水平 sigma = {sigma:.4f}")
print(f"  [零均值模型]   lambda0 = sigma^2/E[x^2]   = {lambda0:.5f}  (目标函数含 ||x||^2)")
print(f"  [中心化模型]   lambdac = sigma^2/Var(x)   = {lambdac:.5f}  (目标函数含 ||x-mu||^2)")
print(f"  注: sigma_x 取真实 x 的二阶矩属 oracle 设定（实际逆问题不可得）；")
print(f"      但先验均值可从数据估计 mu_hat={mu_hat_blur:.4f} (真值 {mu:.4f}, 误差 {abs(mu_hat_blur-mu):.4f})")

# ---- 梯度下降求解（支持一般先验均值 x_prior）----
def solve_tikhonov_gd(y, A, AT, lam, x_prior=0.0, n_iter=10000,
                      lr=None, tol=1e-7, record_interval=100):
    """最小化 J(u) = 0.5||y-Au||^2 + 0.5*lam*||u-x_prior||^2

    返回: 重建, 目标函数历史, 对应迭代号, 实际迭代数, 终点相对梯度, 是否收敛
    """
    x_hat = AT(y).copy()
    obj_hist, iter_hist = [], []

    for i in range(n_iter):
        Ax = A(x_hat)
        if i % record_interval == 0:
            obj = 0.5 * np.sum((y - Ax) ** 2) + 0.5 * lam * np.sum((x_hat - x_prior) ** 2)
            obj_hist.append(obj)
            iter_hist.append(i)

        grad = -AT(y - Ax) + lam * (x_hat - x_prior)
        rel_grad = np.linalg.norm(grad) / (np.linalg.norm(x_hat) + 1e-8)
        if rel_grad < tol:
            obj = 0.5 * np.sum((y - Ax) ** 2) + 0.5 * lam * np.sum((x_hat - x_prior) ** 2)
            obj_hist.append(obj)
            iter_hist.append(i)
            return x_hat, obj_hist, iter_hist, i + 1, rel_grad, True

        x_hat = x_hat - lr * grad

    obj_final = 0.5 * np.sum((y - A(x_hat)) ** 2) + 0.5 * lam * np.sum((x_hat - x_prior) ** 2)
    obj_hist.append(obj_final)
    iter_hist.append(n_iter)
    grad = -AT(y - A(x_hat)) + lam * (x_hat - x_prior)
    rel_grad = np.linalg.norm(grad) / (np.linalg.norm(x_hat) + 1e-8)
    return x_hat, obj_hist, iter_hist, n_iter, rel_grad, rel_grad < tol

# 幂迭代估计 ||A||（高斯模糊保持常数，精确值为 1；顶部谱聚集，需足够迭代）
def estimate_operator_norm(A, AT, shape, n_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(shape)
    v = v / np.linalg.norm(v)
    for _ in range(n_iter):
        v = AT(A(v))
        v = v / np.linalg.norm(v)
    return np.sqrt(max(np.dot(AT(A(v)).ravel(), v.ravel()), 0))

A_norm_est = estimate_operator_norm(A, AT, (n, n))

print(f"\n[求解设置] 幂迭代(100次)估计 ||A|| = {A_norm_est:.4f} (理论值 1.0)")

# ---- 主解 1：经典（零均值）Tikhonov，口径自洽的 lambda0 ----
lr0 = 0.9 * (2.0 / (A_norm_est ** 2 + lambda0))
x_Tikh, obj_hist, iter_hist, n_used, rel_grad, converged = solve_tikhonov_gd(
    y, A, AT, lambda0, x_prior=0.0, lr=lr0
)
print(f"[GD-零均值] lambda={lambda0:.5f}, lr={lr0:.3f}, 迭代 {n_used} 次, "
      f"收敛={converged}, 终点相对梯度={rel_grad:.2e}, 目标 {obj_hist[0]:.1f}->{obj_hist[-1]:.1f}")

# ---- 主解 2：中心化先验 Tikhonov，均值用数据估计 ----
lrc = 0.9 * (2.0 / (A_norm_est ** 2 + lambdac))
x_Tikh_c, obj_hist_c, _, n_used_c, rel_grad_c, converged_c = solve_tikhonov_gd(
    y, A, AT, lambdac, x_prior=mu_hat_blur, lr=lrc
)
print(f"[GD-中心化] lambda={lambdac:.5f}, lr={lrc:.3f}, 迭代 {n_used_c} 次, "
      f"收敛={converged_c}, 终点相对梯度={rel_grad_c:.2e}")

# ---- 后验能量分解验证（零均值模型口径，与主解1一致）----
# J(x) = 0.5||y-Ax||^2 + 0.5*lambda0*||x||^2
# E(x) = 0.5/sigma^2 ||y-Ax||^2 + 0.5/s02 ||x||^2  (= J/sigma^2)，最小化同一点(MAP)
def posterior_parts(u):
    d = 0.5 / sigma ** 2 * np.sum((y - A(u)) ** 2)
    r = 0.5 / s02 * np.sum(u ** 2)
    return d, r, d + r

data_map, reg_map, energy_map = posterior_parts(x_Tikh)
data_truth, reg_truth, energy_truth = posterior_parts(x)
x_init = AT(y)
data_init, reg_init, energy_init = posterior_parts(x_init)

print(f"\n[后验能量分解验证] (归一化能量 E = 数据项/sigma^2 + 正则项/E[x^2])")
print(f"  MAP 解处: 数据项={data_map:.1f}, 正则项={reg_map:.1f}, 后验能量={energy_map:.1f}")
print(f"  真实解处: 数据项={data_truth:.1f}, 正则项={reg_truth:.1f}, 后验能量={energy_truth:.1f}")
print(f"  初始点处: 数据项={data_init:.1f}, 正则项={reg_init:.1f}, 后验能量={energy_init:.1f}")
print(f"  校验: 真实解数据项={data_truth:.1f} ≈ N/2={N/2:.1f} (噪声二次型服从 chi^2_N)")
if energy_map < energy_init:
    print(f"  [验证通过] 后验能量从初始点 {energy_init:.1f} 下降到 MAP 点 {energy_map:.1f}")
else:
    print(f"  [警告] MAP 点后验能量高于初始点")
print(f"  注: MAP 能量低于真实 x 是正常现象——MAP 是后验众数而非真值，")
print(f"      真实 x 的数据项随机偏离其期望，后验最优点允许以数据拟合偏差换取先验契合。")

# ---- 质量指标 ----
x_blurred = A(x)
Orig_MSE = mean_squared_error(x, y)
Tikh_MSE = mean_squared_error(x, x_Tikh)
Tikhc_MSE = mean_squared_error(x, x_Tikh_c)
Orig_PSNR = peak_signal_noise_ratio(x, y, data_range=DATA_RANGE)
Tikh_PSNR = peak_signal_noise_ratio(x, x_Tikh, data_range=DATA_RANGE)
Tikhc_PSNR = peak_signal_noise_ratio(x, x_Tikh_c, data_range=DATA_RANGE)

print(f"\n[质量指标]")
print(f"  模糊含噪基线          PSNR: {Orig_PSNR:.2f} dB")
print(f"  Tikhonov 零均值重建    PSNR: {Tikh_PSNR:.2f} dB  (提升 {Tikh_PSNR-Orig_PSNR:+.2f} dB, 图像整体偏暗)")
print(f"  Tikhonov 中心化先验    PSNR: {Tikhc_PSNR:.2f} dB  (提升 {Tikhc_PSNR-Orig_PSNR:+.2f} dB, 均值失配被修正)")

# ---- 模糊场景 lambda-PSNR 扫描：贝叶斯插件 lambda vs PSNR 经验最优 ----
# 31 点对数网格(0.005~1.0)，零均值/中心化两种模型均以 GD 充分收敛
blur_sweep_lams = np.logspace(-2.3, 0.0, 31)
psnr_sweep_zm, psnr_sweep_cent = [], []
print(f"\n[模糊 lambda-PSNR 扫描] {len(blur_sweep_lams)} 点, GD 收敛判据 rel_grad<1e-7")
for lam_val in blur_sweep_lams:
    lr_lam = 0.9 * (2.0 / (A_norm_est ** 2 + lam_val))
    x_zm_s, _, _, it0, _, ok0 = solve_tikhonov_gd(y, A, AT, lam_val, x_prior=0.0, lr=lr_lam)
    p_zm = peak_signal_noise_ratio(x, x_zm_s, data_range=DATA_RANGE)
    psnr_sweep_zm.append(p_zm)
    x_ct_s, _, _, itc, _, okc = solve_tikhonov_gd(y, A, AT, lam_val, x_prior=mu_hat_blur, lr=lr_lam)
    p_ct = peak_signal_noise_ratio(x, x_ct_s, data_range=DATA_RANGE)
    psnr_sweep_cent.append(p_ct)
    print(f"  lambda={lam_val:7.4f}: 零均值={p_zm:6.2f}dB(it={it0:5d},{ok0}) | 中心化={p_ct:6.2f}dB(it={itc:5d},{okc})")

ib0 = int(np.argmax(psnr_sweep_zm))
ibc = int(np.argmax(psnr_sweep_cent))
lam_best_zm = float(blur_sweep_lams[ib0]);  psnr_best_zm = float(psnr_sweep_zm[ib0])
lam_best_cent = float(blur_sweep_lams[ibc]); psnr_best_cent = float(psnr_sweep_cent[ibc])
print(f"  零均值模型: 插件 lambda0={lambda0:.4f} -> {Tikh_PSNR:.2f}dB; PSNR经验最优 lambda={lam_best_zm:.4f} -> {psnr_best_zm:.2f}dB")
print(f"  中心化模型: 插件 lambdac={lambdac:.4f} -> {Tikhc_PSNR:.2f}dB; PSNR经验最优 lambda={lam_best_cent:.4f} -> {psnr_best_cent:.2f}dB")
print("  [理论说明] 贝叶斯插件 lambda=sigma^2/sigma_x^2 是后验 MAP 的超参数，不保证是 PSNR(MSE) 经验最优点:")
print("   * A=I(纯去噪)时, 标量收缩 x_hat=a*y 的总体最优 lambda*=sigma^2/E[x^2] 只依赖边际二阶矩,")
print("     与空间相关性无关 —— 故步骤3中自洽插件 lambda 恰好命中经验最优;")
print("   * A!=I 时, 最优标量 lambda 还依赖信号功率谱与前向算子奇异谱的交互(各频率信噪比不同),")
print("     白色 iid 高斯先验对所有频率同等收缩, 先验误设下插件 lambda 系统性偏离 PSNR 最优点。")

# ---- 可视化：步骤1（1×5：原始/纯模糊/含噪/零均值Tikh/中心化Tikh）----
fig, axs = plt.subplots(1, 5, figsize=(20, 5.6))

axs[0].imshow(x, cmap='gray', vmin=0, vmax=1)
axs[0].set_title('原始图像 x', fontsize=11)

axs[1].imshow(x_blurred, cmap='gray', vmin=0, vmax=1)
axs[1].set_title('纯模糊 A(x) (无噪声)', fontsize=11)

axs[2].imshow(y, cmap='gray', vmin=0, vmax=1)
axs[2].set_title(f'含噪模糊 y=A(x)+noise\nPSNR={Orig_PSNR:.2f}dB', fontsize=11)

axs[3].imshow(x_Tikh, cmap='gray', vmin=0, vmax=1)
axs[3].set_title(f'零均值 Tikhonov, 先验 N(0,s0^2 I)\nlambda0={lambda0:.4f}, PSNR={Tikh_PSNR:.2f}dB, 惩罚直流→偏暗', fontsize=11)

axs[4].imshow(x_Tikh_c, cmap='gray', vmin=0, vmax=1)
axs[4].set_title(f'中心化 Tikhonov, 先验 N(mu,tau^2 I)\nlambdac={lambdac:.4f}, PSNR={Tikhc_PSNR:.2f}dB, mu由数据估计', fontsize=11)

for ax in axs:
    ax.axis('off')

plt.suptitle('先验 = 正则化：非平凡逆问题 (A=模糊) 下的 Tikhonov 正则化（两种先验均值口径）', fontsize=13, y=0.975)
_note1 = (f'注: 贝叶斯插件 λ=σ^2/σ_x^2 是后验 MAP 超参数，不保证等于 PSNR 经验最优。'
          f'零均值模型 插件λ0={lambda0:.4f}({Tikh_PSNR:.2f}dB) vs 经验最优λ={lam_best_zm:.4f}({psnr_best_zm:.2f}dB)；'
          f'中心化模型 插件λc={lambdac:.4f}({Tikhc_PSNR:.2f}dB) vs 经验最优λ={lam_best_cent:.4f}({psnr_best_cent:.2f}dB)')
_note2 = ('A 非单位阵时，最优标量 λ 依赖信号功率谱与模糊算子奇异谱的交互，白色 iid 先验对各频率同等收缩 → 先验误设下插件 λ 偏离经验最优；'
          'A=I 去噪时最优 λ 仅依赖边际二阶矩(与空间相关无关)，自洽插件 λ 即命中经验最优（见步骤3）')
fig.text(0.5, 0.052, _note1, ha='center', fontsize=10)
fig.text(0.5, 0.008, _note2, ha='center', fontsize=10)
plt.tight_layout(rect=[0, 0.09, 1, 0.91])
plt.savefig(os.path.join(SAVE_DIR, '步骤1_Tikhonov模糊逆问题验证.png'), dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# ---- 不同 lambda 的重建效果对比 ----
print("\n[不同 lambda 的重建效果对比] (零均值目标 ||x||^2；GD 充分收敛)")
lambdas_demo = [0.001, 0.1, 1.0]
reconstructions = []
psnr_demo = []
conv_demo = []
for lam_val in lambdas_demo:
    lr_lam = 0.9 * (2.0 / (A_norm_est ** 2 + lam_val))
    x_lam, _, _, it_used, rg, ok = solve_tikhonov_gd(y, A, AT, lam_val, x_prior=0.0, lr=lr_lam)
    reconstructions.append(x_lam)
    psnr_val = peak_signal_noise_ratio(x, x_lam, data_range=DATA_RANGE)
    psnr_demo.append(psnr_val)
    conv_demo.append((it_used, rg, ok))
    print(f"  lambda={lam_val:.3f}: PSNR={psnr_val:.2f} dB, 迭代={it_used}, 相对梯度={rg:.2e}, 收敛={ok}")

# 可视化：步骤2
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(x, cmap='gray', vmin=0, vmax=1)
axes[0].set_title('原始图像 x')
axes[0].axis('off')

axes[1].imshow(reconstructions[0], cmap='gray', vmin=0, vmax=1)
axes[1].set_title(f'lambda=0.001 (欠正则化)\nPSNR={psnr_demo[0]:.2f}dB\n噪声放大明显')
axes[1].axis('off')

axes[2].imshow(reconstructions[1], cmap='gray', vmin=0, vmax=1)
axes[2].set_title(f'lambda=0.1 (适中)\nPSNR={psnr_demo[1]:.2f}dB\n重建质量较好')
axes[2].axis('off')

axes[3].imshow(reconstructions[2], cmap='gray', vmin=0, vmax=1)
axes[3].set_title(f'lambda=1.0 (过正则化)\nPSNR={psnr_demo[2]:.2f}dB\n过度平滑')
axes[3].axis('off')

plt.suptitle('步骤1-2实验结果：不同 lambda 的重建效果对比（欠正则化 → 适中 → 过正则化）', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_不同lambda的重建效果对比.png'), dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# ============================================================
# 步骤3-4: A = I 纯去噪 — lambda 的贝叶斯诠释
# ============================================================
print("\n" + "=" * 60)
print("步骤3-4: A = I 纯去噪 — lambda 的贝叶斯诠释")
print("=" * 60)

noise_lev2 = 1e-1
y_denoise = x + noise_lev2 * np.random.randn(n, n)
mu_hat = float(y_denoise.mean())  # 仅用观测估计先验均值，非 oracle

# 两种自洽的贝叶斯 lambda
lambda0_den = noise_lev2 ** 2 / s02    # 零均值模型
lambdac_den = noise_lev2 ** 2 / tau2   # 中心化模型

# 三种估计量
x_naive = y_denoise / (1 + lambdac_den)                       # 旧口径混搭：零均值估计量 + 中心化 sigma_x
x_zero_mean = y_denoise / (1 + lambda0_den)                   # 零均值模型自洽
x_centered = mu_hat + (y_denoise - mu_hat) / (1 + lambdac_den)  # 中心化模型自洽（mu 数据估计）

# lambda 扫描（121 点对数网格）
lambdas = np.logspace(-4, 1, 121)
psnr_native = [peak_signal_noise_ratio(x, y_denoise / (1 + lam), data_range=DATA_RANGE) for lam in lambdas]
psnr_centered_native = [peak_signal_noise_ratio(x, mu_hat + (y_denoise - mu_hat) / (1 + lam),
                                                data_range=DATA_RANGE) for lam in lambdas]

# 叠加图统一口径 data_range=1.0（不同图像的 PSNR 绝对高度才可比较）
psnr_unit = [peak_signal_noise_ratio(x, y_denoise / (1 + lam), data_range=1.0) for lam in lambdas]
psnr_centered_unit = [peak_signal_noise_ratio(x, mu_hat + (y_denoise - mu_hat) / (1 + lam),
                                              data_range=1.0) for lam in lambdas]
best_idx = int(np.argmax(psnr_native))
bestc_idx = int(np.argmax(psnr_centered_native))

# 本次噪声实现下，标量收缩的连续（网格外）精确最优 lambda: min_a ||x-a y||^2
a_exact = float(np.dot(x.ravel(), y_denoise.ravel()) / np.dot(y_denoise.ravel(), y_denoise.ravel()))
lambda_exact = 1.0 / a_exact - 1.0

psnr_noisy = peak_signal_noise_ratio(x, y_denoise, data_range=DATA_RANGE)
psnr_naive = peak_signal_noise_ratio(x, x_naive, data_range=DATA_RANGE)
psnr_zero_mean = peak_signal_noise_ratio(x, x_zero_mean, data_range=DATA_RANGE)
psnr_centered = peak_signal_noise_ratio(x, x_centered, data_range=DATA_RANGE)
psnr_noisy_unit = peak_signal_noise_ratio(x, y_denoise, data_range=1.0)
psnr_zero_mean_unit = peak_signal_noise_ratio(x, x_zero_mean, data_range=1.0)
psnr_centered_bayes_unit = peak_signal_noise_ratio(x, x_centered, data_range=1.0)
psnr_naive_unit = peak_signal_noise_ratio(x, x_naive, data_range=1.0)

print(f"噪声水平 sigma={noise_lev2:.4f}, 数据估计均值 mu_hat={mu_hat:.4f} (真值 {mu:.4f})")
print(f"  含噪基线                          PSNR={psnr_noisy:.2f} dB")
print(f"  [失配混搭] y/(1+lambdac), lam={lambdac_den:.4f}     PSNR={psnr_naive:.2f} dB  <- 旧版结果，劣于基线")
print(f"  [零均值自洽] y/(1+lambda0), lam={lambda0_den:.4f}   PSNR={psnr_zero_mean:.2f} dB")
print(f"  [中心化自洽] mu_hat+(y-mu_hat)/(1+lambdac)          PSNR={psnr_centered:.2f} dB")
print(f"  扫描网格最优 lambda={lambdas[best_idx]:.5f} (零均值估计量), PSNR={psnr_native[best_idx]:.2f} dB")
print(f"  连续精确最优 lambda*={lambda_exact:.5f} (与零均值贝叶斯 lambda0={lambda0_den:.5f} 吻合)")
print(f"  中心化估计量网格最优 lambda={lambdas[bestc_idx]:.5f}, PSNR={psnr_centered_native[bestc_idx]:.2f} dB "
      f"(中心化贝叶斯 lambdac={lambdac_den:.5f})")

# ---- Monte Carlo 总体验证（200 个独立噪声实现，独立 RNG 不扰动主流程）----
rng_mc = np.random.default_rng(2024)
R_mc = 200
mc = {k: [] for k in ['noisy', 'naive', 'zero_mean', 'centered', 'exact']}
lams_star_mc = []
for _ in range(R_mc):
    yy = x + noise_lev2 * rng_mc.standard_normal((n, n))
    mh = float(yy.mean())
    ae = float(np.dot(x.ravel(), yy.ravel()) / np.dot(yy.ravel(), yy.ravel()))
    lams_star_mc.append(1.0 / ae - 1.0)
    mc['noisy'].append(peak_signal_noise_ratio(x, yy, data_range=DATA_RANGE))
    mc['naive'].append(peak_signal_noise_ratio(x, yy / (1 + lambdac_den), data_range=DATA_RANGE))
    mc['zero_mean'].append(peak_signal_noise_ratio(x, yy / (1 + lambda0_den), data_range=DATA_RANGE))
    mc['centered'].append(peak_signal_noise_ratio(x, mh + (yy - mh) / (1 + lambdac_den), data_range=DATA_RANGE))
    mc['exact'].append(peak_signal_noise_ratio(x, ae * yy, data_range=DATA_RANGE))
print(f"\n[Monte Carlo {R_mc} 个噪声实现的平均 PSNR]")
for k, lab in [('noisy', '含噪基线'), ('naive', '失配混搭'), ('zero_mean', '零均值自洽'),
               ('centered', '中心化自洽'), ('exact', '逐实现精确最优')]:
    print(f"  {lab:10s}: {np.mean(mc[k]):.2f} +/- {np.std(mc[k]):.2f} dB")
print(f"  逐实现最优 lambda 均值={np.mean(lams_star_mc):.5f} (理论零均值值 lambda0={lambda0_den:.5f})")

# ---- 白色标量高斯先验的理论增益上限 ----
mse_cap_0 = noise_lev2 ** 2 * s02 / (s02 + noise_lev2 ** 2)
mse_cap_c = noise_lev2 ** 2 * tau2 / (tau2 + noise_lev2 ** 2)
cap0_dB = 10 * np.log10(1 + noise_lev2 ** 2 / s02)
capc_dB = 10 * np.log10(1 + noise_lev2 ** 2 / tau2)
print(f"\n[白色标量高斯先验的固有上限] 零均值模型最多 +{cap0_dB:.2f} dB, "
      f"中心化模型最多 +{capc_dB:.2f} dB（仅边际矩信息，无法利用空间结构）")

# ---- 更强先验对照：TV 去噪（权重扫描到最优）----
best_tv = (-1, None, None)
for w in np.linspace(0.04, 0.25, 43):
    xtv = denoise_tv_chambolle(y_denoise, weight=w)
    p = peak_signal_noise_ratio(x, xtv, data_range=DATA_RANGE)
    if p > best_tv[0]:
        best_tv = (p, float(w), xtv)
psnr_tv, tv_weight, x_tv = best_tv
psnr_tv_unit = peak_signal_noise_ratio(x, x_tv, data_range=1.0)
print(f"[更强先验对照] TV 去噪最优 weight={tv_weight:.3f}: PSNR={psnr_tv:.2f} dB "
      f"(较含噪 +{psnr_tv-psnr_noisy:.2f} dB) —— 利用空间相关结构，白高斯先验收益上限被大幅突破")

# ---- 合成对照：先验假设真的成立时，贝叶斯 lambda 是否最优？ ----
# 与先验同分布的 iid 高斯“图像” x_syn ~ N(0, tau^2)，此时零均值/白/同方差假设全部成立
rng_syn = np.random.default_rng(0)
x_syn = np.sqrt(tau2) * rng_syn.standard_normal((n, n))
y_syn = x_syn + noise_lev2 * rng_syn.standard_normal((n, n))
dr_syn = float(x_syn.max() - x_syn.min())
lambda_bayes_syn = noise_lev2 ** 2 / tau2   # 合成数据 mu=0，两种口径合一
a_exact_syn = float(np.dot(x_syn.ravel(), y_syn.ravel()) / np.dot(y_syn.ravel(), y_syn.ravel()))
lambda_exact_syn = 1.0 / a_exact_syn - 1.0
psnr_syn_native = [peak_signal_noise_ratio(x_syn, y_syn / (1 + lam), data_range=dr_syn) for lam in lambdas]
psnr_syn_unit = [peak_signal_noise_ratio(x_syn, y_syn / (1 + lam), data_range=1.0) for lam in lambdas]
best_idx_syn = int(np.argmax(psnr_syn_native))
psnr_syn_bayes_native = peak_signal_noise_ratio(x_syn, y_syn / (1 + lambda_bayes_syn), data_range=dr_syn)
psnr_syn_opt_native = psnr_syn_native[best_idx_syn]
psnr_syn_noisy_native = peak_signal_noise_ratio(x_syn, y_syn, data_range=dr_syn)
psnr_syn_bayes_unit = peak_signal_noise_ratio(x_syn, y_syn / (1 + lambda_bayes_syn), data_range=1.0)
psnr_syn_opt_unit = psnr_syn_unit[best_idx_syn]
psnr_syn_noisy_unit = peak_signal_noise_ratio(x_syn, y_syn, data_range=1.0)

print(f"\n[合成对照 x~N(0,tau^2) iid, 先验严格成立] (native range={dr_syn:.3f})")
print(f"  含噪基线 PSNR={psnr_syn_noisy_native:.2f} dB; 贝叶斯 lam={lambda_bayes_syn:.5f} -> PSNR={psnr_syn_bayes_native:.2f} dB")
print(f"  网格最优 lam={lambdas[best_idx_syn]:.5f} -> PSNR={psnr_syn_opt_native:.2f} dB; 连续精确最优 lam*={lambda_exact_syn:.5f}")
print(f"  [对照结论] 先验成立时贝叶斯 lambda≈经验最优；真实图的偏离源于均值失配与口径混搭，而非公式错误。")
print(f"  注: 合成图 range={dr_syn:.3f} 与相机图 range={DATA_RANGE:.3f} 不同，")
print(f"      native-range PSNR 整体相差 {20*np.log10(dr_syn/DATA_RANGE):.2f} dB 仅是口径效应；")
print(f"      叠加图一律改用 data_range=1.0，此时含噪基线两者均≈20 dB，绝对高度方可比较。")

# ---- 可视化：步骤3 ----
fig1, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
ax.semilogx(lambdas, psnr_unit, 'b-o', markersize=3.5, label='零均值估计量 y/(1+λ) (相机图)')
ax.semilogx(lambdas, psnr_centered_unit, color='orange', marker='^', markersize=3.5, linestyle='-',
            label='中心化估计量 mu_hat+(y-mu_hat)/(1+λ) (相机图)')
ax.semilogx(lambdas, psnr_syn_unit, 'm--s', markersize=3, alpha=0.8, label='iid高斯合成 (μ=0, 先验成立)')
ax.axhline(y=psnr_noisy_unit, color='r', linestyle='--', label=f'含噪基线 {psnr_noisy_unit:.2f}dB')
ax.axhline(y=psnr_tv_unit, color='brown', linestyle='-.', linewidth=1.5,
           label=f'TV去噪(更强先验) {psnr_tv_unit:.2f}dB')
ax.axvline(x=lambda0_den, color='g', linestyle=':', linewidth=2,
           label=f'零均值贝叶斯 λ0={lambda0_den:.4f}')
ax.axvline(x=lambdac_den, color='orange', linestyle=':', linewidth=2,
           label=f'中心化贝叶斯 λc={lambdac_den:.4f}')
ax.scatter([lambda0_den], [psnr_zero_mean_unit], color='g', s=110, zorder=6, marker='*')
ax.scatter([lambdac_den], [psnr_centered_bayes_unit], color='orange', s=90, zorder=6, marker='v')
ax.scatter([lambdac_den], [psnr_naive_unit], color='red', s=90, zorder=6, marker='x',
           label=f'失配混搭点 {psnr_naive_unit:.2f}dB (劣于基线)')
ax.set_xlabel('lambda')
ax.set_ylabel('PSNR (dB, data_range=1.0)')
ax.set_title('PSNR vs λ：先验与估计量口径自洽时，贝叶斯 λ 即落在各自峰值处')
ax.legend(fontsize=7, loc='lower left')
ax.grid(True, alpha=0.3)

# 面板2：三种贝叶斯口径的中心行剖面对比
for xc_, lab, sty in [
    (x_naive, f'失配混搭 λ={lambdac_den:.3f} ({psnr_naive:.2f}dB)', 'r-'),
    (x_zero_mean, f'零均值自洽 λ0={lambda0_den:.4f} ({psnr_zero_mean:.2f}dB)', 'b-'),
    (x_centered, f'中心化自洽 λc={lambdac_den:.3f} ({psnr_centered:.2f}dB)', 'g-'),
]:
    axes[1].plot(xc_[n // 2, :], sty, linewidth=1.3, label=lab)
axes[1].plot(y_denoise[n // 2, :], color='gray', linewidth=0.8, alpha=0.7, label='含噪观测')
axes[1].plot(x[n // 2, :], 'k--', linewidth=1.2, label='真实')
axes[1].set_title('三种口径的中心行剖面（均值失配修正）')
axes[1].legend(fontsize=7)
axes[1].set_xlabel('像素索引')

# 面板3：欠/过正则化谱系（零均值估计量）
for lam_val, lab, sty in [
    (0.001, 'λ=0.001 (欠正则≈不去噪)', 'C0-'),
    (lambda0_den, f'λ0={lambda0_den:.4f} (零均值贝叶斯)', 'g-'),
    (1.0, 'λ=1.0 (过正则，幅值压低)', 'C1-'),
]:
    xt = y_denoise / (1 + lam_val)
    axes[2].plot(xt[n // 2, :], sty, linewidth=1.3, label=lab)
axes[2].plot(x[n // 2, :], 'k--', linewidth=1.2, label='真实')
axes[2].set_title('正则化谱系：欠正则 → 贝叶斯 λ → 过正则')
axes[2].legend(fontsize=7)
axes[2].set_xlabel('像素索引')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_lambda的贝叶斯诠释.png'), dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig1)

# ---- 可视化：步骤4（2×3 图像对比）----
fig2, axes = plt.subplots(2, 3, figsize=(12, 9))

axes[0, 0].imshow(x, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(np.clip(y_denoise, 0, 1), cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title(f'含噪 (σ={noise_lev2})\nPSNR={psnr_noisy:.2f}dB')
axes[0, 1].axis('off')

axes[0, 2].imshow(np.clip(x_naive, 0, 1), cmap='gray', vmin=0, vmax=1)
axes[0, 2].set_title(f'失配混搭 y/(1+λc)\nλc={lambdac_den:.4f}, PSNR={psnr_naive:.2f}dB, 劣于基线')
axes[0, 2].axis('off')

axes[1, 0].imshow(np.clip(x_zero_mean, 0, 1), cmap='gray', vmin=0, vmax=1)
axes[1, 0].set_title(f'零均值自洽 λ0={lambda0_den:.4f}\nPSNR={psnr_zero_mean:.2f}dB (≈经验最优)')
axes[1, 0].axis('off')

axes[1, 1].imshow(np.clip(x_centered, 0, 1), cmap='gray', vmin=0, vmax=1)
axes[1, 1].set_title(f'中心化自洽 λc={lambdac_den:.4f}\nPSNR={psnr_centered:.2f}dB (μ由数据估计)')
axes[1, 1].axis('off')

axes[1, 2].imshow(np.clip(y_denoise / 2.0, 0, 1), cmap='gray', vmin=0, vmax=1)
psnr_over = peak_signal_noise_ratio(x, y_denoise / 2.0, data_range=DATA_RANGE)
axes[1, 2].set_title(f'过正则化 λ=1.0\nPSNR={psnr_over:.2f}dB，过度平滑、对比度下降')
axes[1, 2].axis('off')

plt.suptitle('λ 的贝叶斯诠释：失配混搭(反例) vs 两种口径自洽的贝叶斯重建 vs 过正则化\n'
             f'失配=零均值估计量配中心化 sigma_x；白先验增益上限仅 +{cap0_dB:.2f}/{capc_dB:.2f}dB，TV 利用空间结构达 {psnr_tv:.2f}dB (+{psnr_tv-psnr_noisy:.2f}dB)',
             fontsize=11, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.93], h_pad=3.0)
plt.savefig(os.path.join(SAVE_DIR, '步骤4_不同lambda下的重建对比.png'), dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig2)

# ---- 结论 ----
print("\n" + "=" * 60)
print("结论")
print("=" * 60)
print("1. -ln p(x|y) = 数据项 + 正则项，验证了先验与正则项的对应关系；")
print("   真实解数据项 ≈ N/2，MAP 能量低于初始点与真实解（后验众数≠真值，属正常）。")
print("2. lambda = sigma^2/sigma_x^2 给出正则化参数的概率诠释，但先验模型必须与")
print("   目标函数/估计量口径自洽：零均值模型用 E[x^2]，中心化模型用 Var(x)+均值项。")
print("3. A != I 下 Tikhonov 是真正的逆问题求解；模糊实验中中心化先验(22.19dB)显著")
print("   优于零均值先验(20.09dB)，先验均值是先验信息的一部分。")
print(f"   贝叶斯插件 λ 是后验 MAP 超参数，不保证是 PSNR 经验最优点：")
print(f"   零均值模型 插件λ0={lambda0:.4f}({Tikh_PSNR:.2f}dB) vs 经验最优 {lam_best_zm:.4f}({psnr_best_zm:.2f}dB)；")
print(f"   中心化模型 插件λc={lambdac:.4f}({Tikhc_PSNR:.2f}dB) vs 经验最优 {lam_best_cent:.4f}({psnr_best_cent:.2f}dB)。")
print("4. lambda 谱系：欠正则化→噪声放大，过正则化→过度平滑，均由充分收敛的GD验证。")
print(f"5. 纯去噪(A=I)下，失配混搭(18.91dB)劣于含噪基线(19.76dB)并非贝叶斯公式失效：")
print(f"   零均值自洽 λ0={lambda0_den:.4f} 给出 {psnr_zero_mean:.2f}dB，与经验最优 {psnr_native[best_idx]:.2f}dB")
print(f"   及连续精确最优 λ*={lambda_exact:.4f} 吻合；中心化自洽 λc={lambdac_den:.4f} 达 {psnr_centered:.2f}dB；")
print(f"   MC({R_mc}实现) 经验最优 λ 均值={np.mean(lams_star_mc):.4f}≈λ0，总体结论稳健。")
print("6. A=I 标量收缩下最优 λ*=sigma^2/E[x^2] 只依赖边际矩，与空间相关性无关，故插件 λ 命中经验最优；")
print("   A!=I 时最优标量 λ 还依赖信号谱与算子谱的交互(见结论3)。白高斯先验的固有局限是")
print(f"   A=I 增益上限仅约 +{cap0_dB:.2f}dB/+{capc_dB:.2f}dB，而 TV 利用空间结构达 {psnr_tv:.2f}dB")
print("   —— “需要更强先验”是先验类别决定的固有属性，并有实测对照支撑。")

# ===== 保存数值结果 =====
import json
results_summary = {
    'image_size': n,
    'blur_sigma': float(blur_sigma),
    'noise_level': float(noise_lev),
    'data_range': float(round(DATA_RANGE, 6)),
    'image_mean_mu': float(round(mu, 6)),
    'image_var_tau2': float(round(tau2, 6)),
    'image_second_moment_E_x2': float(round(s02, 6)),
    # 模糊逆问题
    'lambda_zero_mean_blur': float(round(lambda0, 6)),
    'lambda_centered_blur': float(round(lambdac, 6)),
    'mu_hat_from_noisy_blur': float(round(mu_hat_blur, 6)),
    'operator_norm_power_iter_100': float(round(A_norm_est, 6)),
    'psnr_noisy_blur_dB': float(round(Orig_PSNR, 2)),
    'psnr_tikhonov_zero_mean_blur_dB': float(round(Tikh_PSNR, 2)),
    'psnr_tikhonov_centered_blur_dB': float(round(Tikhc_PSNR, 2)),
    'psnr_improvement_zero_mean_dB': float(round(Tikh_PSNR - Orig_PSNR, 2)),
    'psnr_improvement_centered_dB': float(round(Tikhc_PSNR - Orig_PSNR, 2)),
    'gd_zero_mean': {'iters': int(n_used), 'rel_grad': float(f'{rel_grad:.2e}'), 'converged': bool(converged)},
    'gd_centered': {'iters': int(n_used_c), 'rel_grad': float(f'{rel_grad_c:.2e}'), 'converged': bool(converged_c)},
    'posterior_energy_zero_mean_model': {
        'MAP': {'data': float(round(data_map, 1)), 'reg': float(round(reg_map, 1)), 'total': float(round(energy_map, 1))},
        'truth': {'data': float(round(data_truth, 1)), 'reg': float(round(reg_truth, 1)), 'total': float(round(energy_truth, 1))},
        'init': {'data': float(round(data_init, 1)), 'reg': float(round(reg_init, 1)), 'total': float(round(energy_init, 1))},
        'truth_data_term_expectation_N_half': N / 2,
    },
    'psnr_lambda_demo_converged_GD': {
        f'lambda_{l:.3f}': {'psnr_dB': float(round(p, 2)),
                            'iters': int(c[0]), 'rel_grad': float(f'{c[1]:.2e}'), 'converged': bool(c[2])}
        for l, p, c in zip(lambdas_demo, psnr_demo, conv_demo)},
    # 模糊场景 lambda-PSNR 扫描（GD 充分收敛）：插件 lambda vs PSNR 经验最优
    'blur_lambda_psnr_sweep': {
        'grid_lambda': [float(round(float(v), 6)) for v in blur_sweep_lams],
        'psnr_zero_mean_dB': [float(round(float(v), 2)) for v in psnr_sweep_zm],
        'psnr_centered_dB': [float(round(float(v), 2)) for v in psnr_sweep_cent],
        'zero_mean_model': {
            'plugin_lambda': float(round(lambda0, 6)), 'plugin_psnr_dB': float(round(Tikh_PSNR, 2)),
            'empirical_best_lambda': float(round(lam_best_zm, 6)), 'empirical_best_psnr_dB': float(round(psnr_best_zm, 2)),
        },
        'centered_model': {
            'plugin_lambda': float(round(lambdac, 6)), 'plugin_psnr_dB': float(round(Tikhc_PSNR, 2)),
            'empirical_best_lambda': float(round(lam_best_cent, 6)), 'empirical_best_psnr_dB': float(round(psnr_best_cent, 2)),
        },
        'note': 'A!=I 时最优标量 lambda 依赖信号谱与算子谱交互，白先验误设下插件 lambda 偏离经验最优；A=I 时仅依赖边际矩',
    },
    # 纯去噪
    'denoise': {
        'mu_hat_from_noisy': float(round(mu_hat, 6)),
        'lambda_zero_mean': float(round(lambda0_den, 6)),
        'lambda_centered': float(round(lambdac_den, 6)),
        'psnr_noisy_dB': float(round(psnr_noisy, 2)),
        'psnr_naive_mismatch_dB': float(round(psnr_naive, 2)),
        'psnr_zero_mean_consistent_dB': float(round(psnr_zero_mean, 2)),
        'psnr_centered_consistent_dB': float(round(psnr_centered, 2)),
        'psnr_sweep_grid_best_zero_mean': {'lambda': float(round(lambdas[best_idx], 6)), 'psnr_dB': float(round(psnr_native[best_idx], 2))},
        'psnr_sweep_grid_best_centered': {'lambda': float(round(lambdas[bestc_idx], 6)), 'psnr_dB': float(round(psnr_centered_native[bestc_idx], 2))},
        'psnr_exact_scalar_optimum': {'lambda': float(round(lambda_exact, 6)),
                                      'psnr_dB': float(round(peak_signal_noise_ratio(x, a_exact * y_denoise, data_range=DATA_RANGE), 2))},
        'monte_carlo_200': {
            'psnr_noisy_mean': float(round(float(np.mean(mc['noisy'])), 2)),
            'psnr_naive_mean': float(round(float(np.mean(mc['naive'])), 2)),
            'psnr_zero_mean_mean': float(round(float(np.mean(mc['zero_mean'])), 2)),
            'psnr_centered_mean': float(round(float(np.mean(mc['centered'])), 2)),
            'psnr_exact_per_realization_mean': float(round(float(np.mean(mc['exact'])), 2)),
            'lambda_star_mean': float(round(float(np.mean(lams_star_mc)), 6)),
        },
        'white_scalar_prior_theory_gain_cap_dB': {'zero_mean': float(round(cap0_dB, 3)),
                                                  'centered': float(round(capc_dB, 3))},
        'TV': {'weight': float(round(tv_weight, 4)), 'psnr_dB': float(round(psnr_tv, 2)),
               'gain_dB': float(round(psnr_tv - psnr_noisy, 2))},
        'psnr_over_regularized_lambda1_dB': float(round(psnr_over, 2)),
    },
    # 合成对照
    'synthetic_iid_control': {
        'native_data_range': float(round(dr_syn, 4)),
        'lambda_bayes': float(round(lambda_bayes_syn, 6)),
        'lambda_exact_scalar': float(round(lambda_exact_syn, 6)),
        'lambda_grid_best': float(round(lambdas[best_idx_syn], 6)),
        'psnr_noisy_native_dB': float(round(psnr_syn_noisy_native, 2)),
        'psnr_bayes_native_dB': float(round(psnr_syn_bayes_native, 2)),
        'psnr_grid_best_native_dB': float(round(psnr_syn_opt_native, 2)),
        'psnr_noisy_unit_range_dB': float(round(psnr_syn_noisy_unit, 2)),
        'psnr_bayes_unit_range_dB': float(round(psnr_syn_bayes_unit, 2)),
        'psnr_grid_best_unit_range_dB': float(round(psnr_syn_opt_unit, 2)),
        'note': '叠加图统一使用 data_range=1.0；native 与 unit 相差 20log10(range) 属纯口径效应',
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
