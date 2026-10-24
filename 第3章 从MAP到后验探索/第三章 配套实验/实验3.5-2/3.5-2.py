"""
实验3.5-2 Chambolle-Pock算法TV去噪
对应章节：3.5 TV正则化与原始-对偶算法
知识点：TV正则化（ROF模型）的保边性；有限差分算子与离散梯度；
         Chambolle-Pock原始-对偶混合梯度算法；H1-Tikhonov vs TV对比；
         Fenchel对偶与鞍点迭代

素材来源：winter_school/BolognaWinterSchool2023-main/Matlab/tomo_tv.m (Python翻译，简化为ROF去噪)
          winter_school/BolognaWinterSchool2023-main/Matlab/dxp.m, dyp.m, dxm_ad.m, dym_ad.m (Python翻译)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio
import os
import sys

# Windows GBK 控制台下打印 ‖/²/τ 等符号会 UnicodeEncodeError，统一改 UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ====== 中文字体配置（兼容本地和Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验3.5-2')
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
# 1. 有限差分算子
# ══════════════════════════════════════════════════════════

def dxp(u):
    """前向差分(水平): (D_h x)_{i,j} = x_{i,j+1} - x_{i,j}
    对应 dxp.m: dx = [u(:,2:end) u(:,end)] - u
    注: Matlab 原版边界为复制填充(replicate)，此处改用 np.roll 周期边界，
        目的是与 dxm_ad/dym_ad 严格互为伴随算子、并与 H1 的 FFT 闭式解一致；
        phantom 边界处近似为 0，回绕影响可忽略。
    """
    return np.roll(u, -1, axis=1) - u

def dyp(u):
    """前向差分(垂直): (D_v x)_{i,j} = x_{i+1,j} - x_{i,j}
    翻译自 dyp.m: dy = [u(2:end,:); u(end,:)] - u
    """
    return np.roll(u, -1, axis=0) - u

def dxm_ad(p):
    """水平差分的伴随: <dxp(u), p> = <u, dxm_ad(p)>
    翻译自 dxm_ad.m: dx = [u(:,1:end-1) zeros(M,1)] - [zeros(M,1) u(:,1:end-1)]
    注: Matlab 原版用零填充配对 dxp.m 的复制填充边界；此处与 dxp/dyp 一样
        统一改为周期边界（np.roll），使 dxm_ad 严格是 dxp 的伴随算子，
        且与 H1 的 FFT 闭式解所用的谱一致。phantom 边界处近似为 0，影响可忽略。
    """
    return p - np.roll(p, 1, axis=1)

def dym_ad(p):
    """垂直差分的伴随: <dyp(u), p> = <u, dym_ad(p)>
    翻译自 dym_ad.m: dy = [u(1:end-1,:);zeros(1,N)] - [zeros(1,N);u(1:end-1,:)]
    注: 同 dxm_ad——边界由 Matlab 的零填充统一改为周期边界（np.roll）。
    """
    return p - np.roll(p, 1, axis=0)

# ══════════════════════════════════════════════════════════
# 2. Chambolle-Pock算法（TV去噪）
# ══════════════════════════════════════════════════════════

def chambolle_pock_tv_denoise(y, lam, maxiter=500, step_scale=1.0):
    """
    Chambolle-Pock求解ROF模型:
        min_x 0.5||x - y||^2 + lambda * ||Dx||_{2,1}

    本实现用的是【各向同性】TV：||Dx||_{2,1} = sum_像素 ||(D_h x)_像素, (D_v x)_像素||_2
    （每个像素先把水平/垂直梯度合成一个向量再取 ℓ2 范数，最后对像素求和），
    其共轭 g*(p) = iota_{||p||_{2,infty} <= lambda}（逐像素的 ℓ2 球约束）。
    注意：各向异性 TV（||Dx||_{1,2}，逐分量绝对值求和）对应的对偶约束才是
    ℓ_∞ 球（逐分量裁剪）——本实现不是那一种。

    翻译自 tomo_tv.m (Kristian Bredies, 2012; modified by Samuli Siltanen)
    原代码用于CT重建 (min 0.5||Ax - m||^2 + lambda * ||Dx||_1)
    此处简化 A = I（去噪），保留算法核心结构

    鞍点形式:
        min_x max_p <Dx, p> + 0.5||x - y||^2 - iota_{||p||_{2,infty} <= lambda}(p)

    三步迭代:
        (1) 对偶步: 梯度上升 + 投影到逐像素 ℓ2 球 (prox of 指示函数)
        (2) 原始步: 近端梯度下降 (prox of 0.5||·-y||^2)
        (3) 外推步: x_bar = 2x_{k+1} - x_k (类似FISTA动量)
    """
    n1, n2 = y.shape
    x = np.zeros((n1, n2))
    x_bar = np.zeros((n1, n2))
    px = np.zeros((n1, n2))
    py = np.zeros((n1, n2))

    L2 = 8.0
    # 步长条件 τσ‖K‖² < 1（严格小于）。本实现取 τ=σ=1/√8，
    # 而周期差分算子的谱范数恰为 ‖K‖²=8，故 τσ‖K‖²=1 正好落在边界上。
    # step_scale<1 用来留出安全裕度（τσ‖K‖² = step_scale²）。
    sigma = step_scale / np.sqrt(L2)
    tau = step_scale / np.sqrt(L2)

    for k in range(maxiter):
        # ---- 对偶步: 梯度上升 + 投影到 L∞ 球 ----
        gx = dxp(x_bar)
        gy = dyp(x_bar)
        px = px + sigma * gx
        py = py + sigma * gy

        norm_p = np.sqrt(px**2 + py**2)
        factor = np.maximum(1.0, norm_p / lam)
        px = px / factor
        py = py / factor

        # ---- 原始步: 近端梯度下降 ----
        x_old = x.copy()
        div_p = dxm_ad(px) + dym_ad(py)
        x = (x_old + tau * div_p + tau * y) / (1 + tau)
        x = np.maximum(0, x)  # 图像非负约束（非算法标准步骤，仅适用于灰度图像）

        # ---- 外推步: x_bar = 2*x_new - x_old（过外推，驱动对偶变量跟上原始更新）----
        x_bar = 2 * x - x_old

    return x

# ══════════════════════════════════════════════════════════
# 3. H1-Tikhonov去噪（DFT域闭式解，作为TV对比基线）
# ══════════════════════════════════════════════════════════

def h1_tikhonov_denoise(y, lam):
    """
    H1 Tikhonov 去噪: min_x 0.5||x - y||^2 + (lam/2) * ||Dx||_2^2

    闭式解: x = (I + lam * D^T D)^{-1} y
    D^T D 在 DFT 域对角化，利用 FFT 高效求解

    与 TV (||Dx||_1) 的关键差异:
        H1 惩罚梯度的 L2 范数 → 均匀平滑，边缘模糊
        TV 惩罚梯度的 L1 范数 → 保边缘，分段光滑
    """
    n = y.shape[0]
    d_hat = 2 - 2 * np.cos(2 * np.pi * np.arange(n) / n)
    DX, DY = np.meshgrid(d_hat, d_hat)
    DtD_hat = DX + DY
    Y_hat = np.fft.fft2(y)
    X_hat = Y_hat / (1 + lam * DtD_hat)
    return np.real(np.fft.ifft2(X_hat))

# ══════════════════════════════════════════════════════════
# 4. 运行实验
# ══════════════════════════════════════════════════════════

print("=" * 60)
print("实验3.5-2 Chambolle-Pock算法TV去噪")
print("=" * 60)

n = 128
x_true = resize(shepp_logan_phantom(), (n, n), order=1,
                preserve_range=True)

sigma = 0.15
y = x_true + sigma * np.random.randn(n, n)

# ---- 正则化参数选择：两种方法严格同口径 ----
# 在同一张含噪图上按 PSNR 对 λ 扫描、各自取 argmax，对比的是
# "TV 与 H1 各自在最优正则强度下的表现"，避免单方调参、另一方不调的不公平。
# 注意：用真解 PSNR 选 λ 属于 oracle 选参，仅用于教学演示（两种方法口径完全
#       一致）；实际应用应在独立的验证图像/噪声实现上选 λ（差异原理、SURE 等）。
SCAN_ITERS_TV = 300   # 选参扫描步数；argmax 对步数不敏感（300→1000 步结论不变）
lam_tv_grid = np.round(np.arange(0.05, 0.251, 0.01), 2)
lam_h1_grid = np.round(np.arange(0.10, 1.501, 0.05), 2)

psnr_tv_curve, psnr_h1_curve = {}, {}
print(f"\n[正则化参数选择] 同一含噪图上按 PSNR 扫描择优"
      f"（TV 每个 λ 跑 {SCAN_ITERS_TV} 步，共 {len(lam_tv_grid)} 个 λ）...")
for L in lam_tv_grid:
    xi = chambolle_pock_tv_denoise(y, float(L), maxiter=SCAN_ITERS_TV)
    psnr_tv_curve[float(L)] = round(float(peak_signal_noise_ratio(
        x_true, np.clip(xi, None, 1))), 3)
for L in lam_h1_grid:
    xi = h1_tikhonov_denoise(y, float(L))
    psnr_h1_curve[float(L)] = round(float(peak_signal_noise_ratio(
        x_true, np.clip(xi, 0, 1))), 3)
lam_tv = float(max(psnr_tv_curve, key=psnr_tv_curve.get))
lam_tikh = float(max(psnr_h1_curve, key=psnr_h1_curve.get))
print(f"  TV : λ* = {lam_tv:.2f}, PSNR = {psnr_tv_curve[lam_tv]:.2f} dB "
      f"（扫描区间 {lam_tv_grid[0]:.2f}–{lam_tv_grid[-1]:.2f}）")
print(f"  H1: λ* = {lam_tikh:.2f}, PSNR = {psnr_h1_curve[lam_tikh]:.2f} dB "
      f"（扫描区间 {lam_h1_grid[0]:.2f}–{lam_h1_grid[-1]:.2f}）")

print(f"\n[参数设定]")
print(f"  图像尺寸: {n} x {n}")
print(f"  噪声标准差: σ = {sigma}")
print(f"  TV 正则化参数: λ_tv = {lam_tv:.2f}（PSNR 扫描择优）")
print(f"  H1-Tikhonov 正则化参数: λ_tikh = {lam_tikh:.2f}（PSNR 扫描择优）")
print(f"  Chambolle-Pock 最大迭代: 500")
print(f"  步长: L2 = 8.0, σ = τ = 1/√(L2) ≈ {1.0/np.sqrt(8.0):.3f}")

# ---- 步长条件自检：实算 ‖K‖² 与 τσ‖K‖² ----
# K = D（水平+垂直周期前向差分堆叠），D^T D 在 DFT 域对角化，
# |1-e^{-iω}|² = 2-2cos(ω)，故最大特征值 = 2·(2-2cos(π)) = 8。
_k = np.arange(n)
_h = 2 - 2 * np.cos(2 * np.pi * _k / n)
_DX, _DY = np.meshgrid(_h, _h)
normK2 = float((_DX + _DY).max())          # = ‖K‖²
step_tau = 1.0 / np.sqrt(8.0)
step_sigma = 1.0 / np.sqrt(8.0)
tau_sigma_normK2 = step_tau * step_sigma * normK2
print(f"\n[步长条件自检]")
print(f"  ‖K‖² (DFT 精确谱范数) = {normK2:.6f}")
print(f"  τσ‖K‖² = {step_tau:.6f}×{step_sigma:.6f}×{normK2:.6f} = {tau_sigma_normK2:.6f}"
      f"  → 恰在收敛条件 τσ‖K‖²<1 的边界上（=1）")
print(f"  留 10% 裕度的对照步长 τ=σ=0.9/√8: τσ‖K‖² = "
      f"{(0.9/np.sqrt(8.0))**2*normK2:.6f}（严格 <1）")

# ---- TV去噪 ----
print(f"\n[运行 Chambolle-Pock TV去噪...]")
x_tv = chambolle_pock_tv_denoise(y, lam_tv, maxiter=500)
# 收敛性核验：以 2000 步解为参考，确认 500 步已充分收敛
x_tv_ref = chambolle_pock_tv_denoise(y, lam_tv, maxiter=2000)
tv_rel_change = float(np.linalg.norm(x_tv - x_tv_ref)
                     / max(np.linalg.norm(x_tv_ref), 1e-12))
print(f"  完成（500 步 vs 2000 步参考解: 相对变化 {tv_rel_change:.2e}）")

# ---- 步长敏感性对照：边界步长 vs 留裕度步长（解应基本一致）----
x_tv_margin = chambolle_pock_tv_denoise(y, lam_tv, maxiter=500, step_scale=0.9)
margin_rel_diff = float(np.linalg.norm(x_tv_margin - x_tv)
                        / max(np.linalg.norm(x_tv), 1e-12))
print(f"  步长敏感性: 边界步长(τσ‖K‖²=1) vs 裕度步长(0.81) 解的相对差异 = "
      f"{margin_rel_diff:.2e}")

# ---- H1-Tikhonov去噪（DFT域） ----
print(f"\n[运行 H1-Tikhonov去噪 (DFT域闭式解)...]")
x_tikh = h1_tikhonov_denoise(y, lam_tikh)
print(f"  完成")

# ---- 定量指标（绘图前先算，供标题/JSON 统一引用，避免两处口径漂移） ----
psnr_noisy = peak_signal_noise_ratio(x_true, y)
psnr_tikh = peak_signal_noise_ratio(x_true, np.clip(x_tikh, 0, 1))
psnr_tv = peak_signal_noise_ratio(x_true, np.clip(x_tv, None, 1))
psnr_tv_ref = peak_signal_noise_ratio(x_true, np.clip(x_tv_ref, None, 1))

# 差异图"边缘集中"的定量验证：用真解梯度幅值划分边缘带与平坦区
_gx, _gy = np.gradient(x_true)
_eg = np.sqrt(_gx**2 + _gy**2)
_thr = np.quantile(_eg[_eg > 1e-6], 0.90)
_edge_mask = _eg > max(_thr, 0.02)
_flat_mask = _eg < 0.005
_diff_map = np.abs(x_tv - x_tikh)
diff_edge = float(_diff_map[_edge_mask].mean())
diff_flat = float(_diff_map[_flat_mask].mean())
diff_edge_to_flat_ratio = diff_edge / diff_flat

# 差异图的"边缘集中度"：差异最大的 10% 像素里有多少落在边缘带。
# 若差异完全由边缘引起，该比例应接近 100%；若与随机位置无异，则应等于边缘带自身占全图的比例。
_diff_thr10 = np.quantile(_diff_map, 0.90)
_top10_mask = _diff_map >= _diff_thr10
diff_top10_in_edge_pct = float(_edge_mask[_top10_mask].mean() * 100)
edge_band_pixel_pct = float(_edge_mask.mean() * 100)

# ══════════════════════════════════════════════════════════
# 5. 可视化
# ══════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

axes[0].imshow(x_true, cmap='gray')
axes[0].set_title(r'真解', fontsize=11)

axes[1].imshow(y, cmap='gray')
axes[1].set_title(r'含噪 ($\sigma=' + str(sigma) + r'$)', fontsize=11)

axes[2].imshow(x_tikh, cmap='gray')
axes[2].set_title(
    f'H1-Tikhonov去噪 ($\\lambda={lam_tikh:.2f}$)'
    '\n(均匀平滑，边缘模糊)',
    fontsize=11)

axes[3].imshow(x_tv, cmap='gray')
axes[3].set_title(
    f'TV去噪 ($\\lambda={lam_tv:.2f}$)'
    '\n(保边缘，分段光滑)',
    fontsize=11)

axes[4].imshow(_diff_map, cmap='hot', vmin=0, vmax=0.1)
axes[4].set_title(r'差异 $|\mathrm{TV} - \mathrm{Tikhonov}|$'
                  f'\n(边缘带/平坦区 ≈ {diff_edge_to_flat_ratio:.2f}×，整体仍弥散)',
                  fontsize=11)

for ax in axes:
    ax.axis('off')

plt.suptitle(r'实验3.5-2: Chambolle-Pock TV去噪 vs H1-Tikhonov去噪',
             fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(SAVE_DIR, '步骤1_TV去噪对比.png'), dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════
# 6. 结果输出
# ══════════════════════════════════════════════════════════

# psnr_*、diff_edge/flat 等指标已在第 4 节绘图前统一计算，避免两处口径漂移

print(f"\n[PSNR 对比]")
print(f"  含噪图像:      {psnr_noisy:.2f} dB")
print(f"  H1-Tikhonov:   {psnr_tikh:.2f} dB")
print(f"  TV (Chambolle-Pock, 500步): {psnr_tv:.2f} dB")
print(f"    └ 2000步参考解: {psnr_tv_ref:.2f} dB，500步差距 {psnr_tv - psnr_tv_ref:+.3f} dB，"
      f"相对解变化 {tv_rel_change:.1e}（500 步已充分收敛）")

print(f"\n[算法机制]")
print(f"  Chambolle-Pock 三步迭代（解各向同性 TV: min 0.5||x-y||^2 + λ||Dx||_(2,1)）:")
print(f"    (1) 对偶步: p^{{k+1}} = prox_{{σ g^*}}(p^k + σ D x_bar^k)")
print(f"        g^* = ι_{{‖·‖_(2,∞) ≤ λ}} → 投影到逐像素 ℓ2 球（各向同性 TV 的共轭；"
      f"各向异性 TV 才对应 ℓ∞ 球）")
print(f"    (2) 原始步: x^{{k+1}} = prox_{{τ f}}(x^k - τ D^T p^{{k+1}})")
print(f"        f = 0.5‖x-y‖² → x^{{k+1}} = (x^k + τ D^T p^{{k+1}} + τ y)/(1+τ)")
print(f"    (3) 外推步: x_bar^{{k+1}} = 2x^{{k+1}} - x^k")

print(f"\n[H1-Tikhonov vs TV 对比]")
print(f"  H1-Tikhonov (λ={lam_tikh:.2f}):")
print(f"    - 正则项: (λ/2)‖Dx‖₂² → 惩罚梯度的 L2 范数")
print(f"    - 特点: 均匀平滑，边缘模糊")
print(f"    - 求解: DFT 域闭式解（O(n log n)）")
print(f"  TV (λ={lam_tv:.2f}):")
print(f"    - 正则项: λ‖Dx‖_{{2,1}} → 惩罚梯度的 L1 范数")
print(f"    - 特点: 保边缘，分段光滑（允许跳变）")
print(f"    - 求解: Chambolle-Pock 迭代（O(n) 每步）")
print(f"  差异图定量: 边缘带 mean|diff| = {diff_edge:.4f}, 平坦区 mean|diff| = {diff_flat:.4f}, "
      f"比值 = {diff_edge_to_flat_ratio:.2f}x")
print(f"  边缘集中度: 差异最大的10%像素中 {diff_top10_in_edge_pct:.1f}% 落在边缘带"
      f"（边缘带本身只占全图 {edge_band_pixel_pct:.1f}%，随机基线）")
print(f"  最强差异出现在边缘处，但平坦区差异并不为零（H1 保留平滑噪声纹理、TV 呈分段常数），")
print(f"  验证 TV 保边性的同时说明差异图并非完全'集中'于边缘")

print(f"\n[核心发现]")
print(f"  1. TV正则化 (‖Dx‖_(2,1)) 的保边性: 允许边缘处有不连续的跳变")
print(f"     — 只惩罚梯度的总幅度，不惩罚梯度的平方")
print(f"  2. 当近端算子无闭式解时 (TV = ‖D·‖_(2,1))，Fenchel对偶打开新路径:")
print(f"     g(x) = λ‖Dx‖_(2,1) → g*(y) = ι_{{‖y‖_(2,∞) ≤ λ}} (逐像素 ℓ2 球，简单投影!)")
print(f"  3. Chambolle-Pock 在鞍点结构上交替更新原始/对偶变量")
print(f"     — 每步代价 O(n)，步长条件 τσ‖K‖² < 1 保证收敛")
print(f"     — 本例 τ=σ=1/√8 且 ‖K‖²=8，故 τσ‖K‖²=1 恰在条件边界；")
print(f"       实测边界步长与留裕度步长(0.81)的解相对差异仅 {margin_rel_diff:.1e}，结论不受影响")
print(f"  4. TV vs H1-Tikhonov: 保边 vs 平滑 — 同一问题的不同正则化哲学")
print(f"     — TV更适合分段光滑图像 (如 phantom)，H1更适合平滑图像")

print(f"\n{'=' * 60}")
print(f"实验3.5-2 完成")
print(f"输出图片: {os.path.join(SAVE_DIR, '步骤1_TV去噪对比.png')}")
print(f"{'=' * 60}")

# ===== 保存数值结果 =====
import json
results_summary = {
    'n': n,
    'sigma': float(round(sigma, 2)),
    'random_seed': 42,
    'lam_tv': float(round(lam_tv, 4)),
    'lam_tikh': float(round(lam_tikh, 4)),
    'lam_selection': ('PSNR argmax on the same noisy image (oracle), identical '
                      'protocol for TV and H1; for real use select lambda on '
                      'independent data (discrepancy principle, SURE, CV)'),
    'lam_tv_scan_iters': SCAN_ITERS_TV,
    'lam_tv_scan_psnr': psnr_tv_curve,
    'lam_tikh_scan_psnr': psnr_h1_curve,
    'psnr_noisy': float(round(psnr_noisy, 2)),
    'psnr_tikh': float(round(psnr_tikh, 2)),
    'psnr_tv': float(round(psnr_tv, 2)),
    'psnr_tv_500_iter': float(round(psnr_tv, 2)),
    'psnr_tv_2000_iter': float(round(psnr_tv_ref, 2)),
    'tv_rel_change_500_vs_2000': float(round(tv_rel_change, 6)),
    'cp_norm_K_squared': float(round(normK2, 6)),
    'cp_tau_sigma_normK2': float(round(tau_sigma_normK2, 6)),
    'cp_step_on_convergence_boundary': True,
    'cp_margin_step_tau_sigma_normK2': float(round((0.9/np.sqrt(8.0))**2*normK2, 6)),
    'cp_margin_vs_boundary_rel_diff': float(round(margin_rel_diff, 8)),
    'diff_edge_mean': float(round(diff_edge, 4)),
    'diff_flat_mean': float(round(diff_flat, 4)),
    'diff_edge_to_flat_ratio': float(round(diff_edge_to_flat_ratio, 2)),
    'diff_top10pct_pixels_in_edge_band_pct': float(round(diff_top10_in_edge_pct, 1)),
    'edge_band_pixel_pct_of_image': float(round(edge_band_pixel_pct, 2)),
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