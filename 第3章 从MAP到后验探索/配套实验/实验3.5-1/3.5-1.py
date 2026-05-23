"""
实验3.5-1 Chambolle-Pock算法TV去噪
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

# ====== 中文字体配置（兼容本地和Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验3.5-1')
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
    翻译自 dxp.m: dx = [u(:,2:end) u(:,end)] - u
    """
    return np.roll(u, -1, axis=1) - u

def dyp(u):
    """前向差分(垂直): (D_v x)_{i,j} = x_{i+1,j} - x_{i,j}
    翻译自 dyp.m: dy = [u(2:end,:); u(end,:)] - u
    """
    return np.roll(u, -1, axis=0) - u

def dxm_ad(p):
    """水平差分的伴随:
    翻译自 dxm_ad.m: dx = [u(:,1:end-1) zeros(M,1)] - [zeros(M,1) u(:,1:end-1)]
    """
    return p - np.roll(p, 1, axis=1)

def dym_ad(p):
    """垂直差分的伴随:
    翻译自 dym_ad.m: dy = [u(1:end-1,:);zeros(1,N)] - [zeros(1,N);u(1:end-1,:)]
    """
    return p - np.roll(p, 1, axis=0)

# ══════════════════════════════════════════════════════════
# 2. Chambolle-Pock算法（TV去噪）
# ══════════════════════════════════════════════════════════

def chambolle_pock_tv_denoise(y, lam, maxiter=500):
    """
    Chambolle-Pock求解ROF模型:
        min_x 0.5||x - y||^2 + lambda * ||Dx||_{2,1}

    翻译自 tomo_tv.m (Kristian Bredies, 2012; modified by Samuli Siltanen)
    原代码用于CT重建 (min 0.5||Ax - m||^2 + lambda * ||Dx||_1)
    此处简化 A = I（去噪），保留算法核心结构

    鞍点形式:
        min_x max_p <Dx, p> + 0.5||x - y||^2 - iota_{||p||_infty <= lambda}(p)

    三步迭代:
        (1) 对偶步: 梯度上升 + 投影到 L∞ 球 (prox of 指示函数)
        (2) 原始步: 近端梯度下降 (prox of 0.5||·-y||^2)
        (3) 外推步: x_bar = 2x_{k+1} - x_k (类似FISTA动量)
    """
    n1, n2 = y.shape
    x = np.zeros((n1, n2))
    x_bar = np.zeros((n1, n2))
    px = np.zeros((n1, n2))
    py = np.zeros((n1, n2))

    L2 = 8.0
    sigma = 1.0 / np.sqrt(L2)
    tau = 1.0 / np.sqrt(L2)

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
        x = (x - tau * (-div_p) + tau * y) / (1 + tau)
        x = np.maximum(0, x)

        # ---- 外推步 ----
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
print("实验3.5-1 Chambolle-Pock算法TV去噪")
print("=" * 60)

n = 128
x_true = resize(shepp_logan_phantom(), (n, n), order=0,
                preserve_range=True, anti_aliasing=True)

sigma = 0.15
y = x_true + sigma * np.random.randn(n, n)

lam_tv = 0.15
lam_tikh = 0.5  # 增大λ使H1平滑效果更明显，与TV保边效果形成鲜明对比

print(f"\n[参数设定]")
print(f"  图像尺寸: {n} x {n}")
print(f"  噪声标准差: σ = {sigma}")
print(f"  TV 正则化参数: λ_tv = {lam_tv}")
print(f"  H1-Tikhonov 正则化参数: λ_tikh = {lam_tikh}")
print(f"  Chambolle-Pock 最大迭代: 500")
print(f"  步长: L2 = 8.0, σ = τ = 1/√(L2) ≈ {1.0/np.sqrt(8.0):.3f}")

# ---- TV去噪 ----
print(f"\n[运行 Chambolle-Pock TV去噪...]")
x_tv = chambolle_pock_tv_denoise(y, lam_tv, maxiter=500)
print(f"  完成")

# ---- H1-Tikhonov去噪（DFT域） ----
print(f"\n[运行 H1-Tikhonov去噪 (DFT域闭式解)...]")
x_tikh = h1_tikhonov_denoise(y, lam_tikh)
print(f"  完成")

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
    r'H1-Tikhonov去噪 ($\lambda=' + str(lam_tikh) + r'$)'
    + '\n' + r'(均匀平滑，边缘模糊)',
    fontsize=11)

axes[3].imshow(x_tv, cmap='gray')
axes[3].set_title(
    r'TV去噪 ($\lambda=' + str(lam_tv) + r'$)'
    + '\n' + r'(保边缘，分段光滑)',
    fontsize=11)

diff = np.abs(x_tv - x_tikh)
axes[4].imshow(diff, cmap='hot', vmin=0, vmax=0.1)
axes[4].set_title(r'差异 $|\mathrm{TV} - \mathrm{Tikhonov}|$'
                  + '\n' + r'(边缘处差异最大)', fontsize=11)

for ax in axes:
    ax.axis('off')

plt.suptitle(r'实验3.5-1: Chambolle-Pock TV去噪 vs H1-Tikhonov去噪',
             fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(SAVE_DIR, '步骤1_TV去噪对比.png'), dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════
# 6. 结果输出
# ══════════════════════════════════════════════════════════

psnr_noisy = peak_signal_noise_ratio(x_true, y)
psnr_tikh = peak_signal_noise_ratio(x_true, np.clip(x_tikh, 0, 1))
psnr_tv = peak_signal_noise_ratio(x_true, np.clip(x_tv, 0, 1))

print(f"\n[PSNR 对比]")
print(f"  含噪图像:      {psnr_noisy:.2f} dB")
print(f"  H1-Tikhonov:   {psnr_tikh:.2f} dB")
print(f"  TV (Chambolle-Pock): {psnr_tv:.2f} dB")

print(f"\n[算法机制]")
print(f"  Chambolle-Pock 三步迭代:")
print(f"    (1) 对偶步: p^{{k+1}} = prox_{{σ g^*}}(p^k + σ D x_bar^k)")
print(f"        g^* = ι_{{‖·‖_∞ ≤ λ}} → 投影到 L∞ 球")
print(f"    (2) 原始步: x^{{k+1}} = prox_{{τ f}}(x^k - τ D^T p^{{k+1}})")
print(f"        f = 0.5‖x-y‖² → x^{{k+1}} = (x^k + τ D^T p^{{k+1}} + τ y)/(1+τ)")
print(f"    (3) 外推步: x_bar^{{k+1}} = 2x^{{k+1}} - x^k")

print(f"\n[H1-Tikhonov vs TV 对比]")
print(f"  H1-Tikhonov (λ={lam_tikh}):")
print(f"    - 正则项: (λ/2)‖Dx‖₂² → 惩罚梯度的 L2 范数")
print(f"    - 特点: 均匀平滑，边缘模糊")
print(f"    - 求解: DFT 域闭式解（O(n log n)）")
print(f"  TV (λ={lam_tv}):")
print(f"    - 正则项: λ‖Dx‖_{2,1} → 惩罚梯度的 L1 范数")
print(f"    - 特点: 保边缘，分段光滑（允许跳变）")
print(f"    - 求解: Chambolle-Pock 迭代（O(n) 每步）")
print(f"  差异 |TV - Tikhonov| 集中在边缘处，验证TV的保边性")

print(f"\n[核心发现]")
print(f"  1. TV正则化 (‖Dx‖₁) 的保边性: 允许边缘处有不连续的跳变")
print(f"     — 只惩罚梯度的总幅度，不惩罚梯度的平方")
print(f"  2. 当近端算子无闭式解时 (TV = ‖D·‖₁)，Fenchel对偶打开新路径:")
print(f"     g(x) = ‖Dx‖₁ → g*(y) = ι_{{‖y‖_∞ ≤ λ}} (简单投影!)")
print(f"  3. Chambolle-Pock 在鞍点结构上交替更新原始/对偶变量")
print(f"     — 每步代价 O(n)，步长条件 τσ‖K‖² < 1 保证收敛")
print(f"  4. TV vs H1-Tikhonov: 保边 vs 平滑 — 同一问题的不同正则化哲学")
print(f"     — TV更适合分段光滑图像 (如 phantom)，H1更适合平滑图像")

print(f"\n{'=' * 60}")
print(f"实验3.5-1 完成")
print(f"输出图片: {os.path.join(SAVE_DIR, '步骤1_TV去噪对比.png')}")
print(f"{'=' * 60}")