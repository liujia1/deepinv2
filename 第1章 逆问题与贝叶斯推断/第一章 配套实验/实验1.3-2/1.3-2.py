import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio
import sys
import os

# Jupyter 中渲染 LaTeX 公式
try:
    from IPython.display import display, Markdown
    _in_jupyter = True
except ImportError:
    _in_jupyter = False

# ====== 中文字体配置（兼容本地和 Google Colab）======
_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验1.3-2', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验1.3-2')
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    pass
# ========================================================

np.random.seed(42)

# ---- 1. 构造小尺寸模糊算子的显式矩阵 ----
# 使用小尺寸以使 SVD 计算可行
n = 32
phantom = resize(shepp_logan_phantom(), (n, n), order=0, preserve_range=True, anti_aliasing=False)
x = phantom / phantom.max()
x_vec = x.ravel()  # 向量化
N = n * n

# 构造模糊矩阵 A（通过逐列应用模糊算子）
# 注意：FFT 对应循环卷积（periodic boundary condition）
def gaussian_psf(size, sigma):
    ax = np.concatenate((np.arange(0, size // 2), np.arange(-size // 2, 0)))
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return h / h.sum()

h = gaussian_psf(n, sigma=1.5)
H_fft = np.fft.fft2(h)

# 逐列构建 A 矩阵（小尺寸可行）
A = np.zeros((N, N))
for j in range(N):
    e_j = np.zeros(N)
    e_j[j] = 1.0
    Ae_j = np.real(np.fft.ifft2(H_fft * np.fft.fft2(e_j.reshape(n, n)))).ravel()
    A[:, j] = Ae_j

# ---- 2. SVD 分解 ----
print("计算 SVD（矩阵大小 {}x{}）...".format(N, N))
U, sigma, Vt = np.linalg.svd(A, full_matrices=False)

# ---- 3. 生成含噪观测 ----
y_clean = A @ x_vec
noise_sigma = 0.01
y_noisy = y_clean + noise_sigma * np.random.randn(N)

# ---- 4. 伪逆解 vs 截断 SVD 解 ----
# 注：diag(1/σ) @ v 等价于 v / σ 的逐元素除法，后者避免构造 1024×1024 对角阵
def tsvd_solve(y_vec, k):
    """截断 SVD 解：x_k = V_k diag(1/σ_k) U_k^T y；k=N 时即伪逆。"""
    return Vt[:k, :].T @ ((U[:, :k].T @ y_vec) / sigma[:k])

def psnr_two_conventions(xhat):
    """同一重建的两种 PSNR 口径（data_range 均取 1）：
    - raw     : 直接在重建值上计算，如实保留 1/σ_i 对噪声的放大（本实验主口径）；
    - clipped : 先 clip 到 [0,1] 再计算（与图像显示范围一致；但崩坏重建的误差被
                截断人为封顶，会严重"美化"灾难级结果，故仅作对照口径记录）。
    """
    img = xhat.reshape(n, n)
    raw = peak_signal_noise_ratio(x, img, data_range=1.0)
    clipped = peak_signal_noise_ratio(x, np.clip(img, 0, 1), data_range=1.0)
    return float(raw), float(clipped)

# 伪逆解（使用所有奇异值）
x_pinv = tsvd_solve(y_noisy, len(sigma))
psnr_pinv_raw, psnr_pinv_clipped = psnr_two_conventions(x_pinv)

# 截断 SVD 解（固定候选档位，用于横向对比）
truncation_levels = [50, 100, 200, 500]
x_trunc, psnr_trunc_raw, psnr_trunc_clipped = {}, {}, {}
for k_level in truncation_levels:
    k_eff = min(k_level, len(sigma))  # 用独立变量名，不覆盖 for 循环变量
    x_trunc[k_eff] = tsvd_solve(y_noisy, k_eff)
    _pr, _pc = psnr_two_conventions(x_trunc[k_eff])
    psnr_trunc_raw[k_eff], psnr_trunc_clipped[k_eff] = _pr, _pc

# ---- 4b. 全 k=1..N 扫描：事后(oracle)最优截断 ----
# 利用截断解的嵌套结构增量累加：x_k = x_{k-1} + (<u_k,y>/σ_k) v_k，O(N^2) 扫完所有 k
Uty = U.T @ y_noisy
x_cum = np.zeros(N)
psnr_curve_raw = np.empty(N)
psnr_curve_clipped = np.empty(N)
best_k, best_psnr_raw, x_best = 1, -np.inf, None
for idx in range(N):
    x_cum = x_cum + (Uty[idx] / sigma[idx]) * Vt[idx, :]
    _pr, _pc = psnr_two_conventions(x_cum)
    psnr_curve_raw[idx], psnr_curve_clipped[idx] = _pr, _pc
    if _pr > best_psnr_raw:
        best_psnr_raw, best_k, x_best = _pr, idx + 1, x_cum.copy()
best_psnr_clipped = psnr_curve_clipped[best_k - 1]
# 方法学说明：按真值 PSNR 选 k 是"事后最优"(oracle)，实际问题中没有真值，
# 常用 GCV、L 曲线或 Morozov 偏差原理等数据驱动准则；此处仅供教学展示最优平衡。
# 另：本实验正、反演刻意共用同一离散算子 A，以隔离"噪声×小奇异值"这一单一机制；
# 正反演模型不一致（Inverse Crime）的影响由实验 1.3-4 专门展示。

# ---- 5. Picard 图 ----
coeffs_clean = np.abs(U.T @ y_clean)
coeffs_noisy = np.abs(U.T @ y_noisy)

# ---- 6. 可视化 ----
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# (a) 奇异值衰减
axes[0, 0].semilogy(np.arange(1, len(sigma) + 1), sigma, 'b-', linewidth=1)
axes[0, 0].set_xlabel('奇异值索引 i')
axes[0, 0].set_ylabel(r'$\sigma_i$')
axes[0, 0].set_title(r'奇异值衰减曲线' + '\n$\sigma_i \to 0$ 导致 $1/\sigma_i \to \infty$')
axes[0, 0].grid(True)

# (b) Picard 图
i_range = np.arange(1, min(500, len(sigma)) + 1)
axes[0, 1].semilogy(i_range, sigma[:len(i_range)], 'b-', label=r'$\sigma_i$')
axes[0, 1].semilogy(i_range, coeffs_clean[:len(i_range)], 'g.', markersize=2, alpha=0.5, label=r'$|\langle u_i, y\rangle|$ (无噪)')
axes[0, 1].semilogy(i_range, coeffs_noisy[:len(i_range)], 'r.', markersize=2, alpha=0.5, label=r'$|\langle u_i, y^\delta\rangle|$ (含噪)')
axes[0, 1].set_xlabel(r'奇异值索引 $i$')
axes[0, 1].set_title('Picard 图\n含噪数据违反 Picard 条件')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True)

# (c) 1/σ_i 的爆炸性增长
axes[0, 2].semilogy(i_range, 1.0 / sigma[:len(i_range)], 'r-', linewidth=1)
axes[0, 2].set_xlabel('奇异值索引 i')
axes[0, 2].set_ylabel(r'$1/\sigma_i$')
axes[0, 2].set_title(r'伪逆放大因子 $1/\sigma_i$' + '\n小奇异值→巨大放大')
axes[0, 2].grid(True)

# (d) 原始图像（三张小图统一显示范围 [0,1]，保证灰度可比）
axes[1, 0].imshow(x, cmap='gray', vmin=0, vmax=1)
axes[1, 0].set_title('原始图像 x')
axes[1, 0].axis('off')

# (e) 伪逆重建（含噪数据）
# 显示按 [0,1] 截断（等价于显示器的有效范围），标题报告未截断(raw)口径 PSNR：
# 该口径下噪声被 1/σ_i 放大约 200 万倍，PSNR 约 -114 dB；clip 口径只有 3.6 dB，
# 会把灾难级重建"美化"掉（完整对照见 results_summary.json 与参数曲线图虚线）。
axes[1, 1].imshow(x_pinv.reshape(n, n), cmap='gray', vmin=0, vmax=1)
axes[1, 1].set_title(f'伪逆重建（含噪）\nPSNR={psnr_pinv_raw:.0f} dB（未截断口径）\n噪声被 $1/\\sigma_i$ 完全放大')
axes[1, 1].axis('off')

# (f) 截断 SVD 重建：k 为全 k=1..N 扫描的事后(oracle) PSNR 峰值（raw 主口径）
axes[1, 2].imshow(x_best.reshape(n, n), cmap='gray', vmin=0, vmax=1)
axes[1, 2].set_title(f'截断 SVD（k={best_k}，事后最优）\nPSNR={best_psnr_raw:.1f} dB（未截断口径）\n丢弃小奇异值方向，抑制噪声')
axes[1, 2].axis('off')

plt.suptitle('SVD 视角：奇异值衰减、噪声放大与截断 SVD', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_3_2_SVD视角.png'), dpi=150, bbox_inches='tight')
plt.show()

# ---- 7. 截断参数 k 对重建质量的影响（复用 4b 的全 k=1..N 扫描）----
k_all = np.arange(1, N + 1)
plt.figure(figsize=(8.5, 4.5))
plt.plot(k_all, psnr_curve_raw, '-', lw=1.6, color='C0',
         label='PSNR（未截断口径，主口径）')
plt.plot(k_all, psnr_curve_clipped, '--', lw=1.3, color='C1',
         label='PSNR（先 clip 到 [0,1]，显示口径）')
plt.axvline(best_k, color='gray', ls=':', lw=1.2)
plt.plot(best_k, best_psnr_raw, 'k*', ms=13)
plt.annotate(f'事后最优 k={best_k}\nPSNR={best_psnr_raw:.1f} dB',
             xy=(best_k, best_psnr_raw), xytext=(best_k + 150, best_psnr_raw - 3.2),
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9)
plt.xlabel('截断参数 k')
plt.ylabel('重建 PSNR (dB)')
plt.title('截断 SVD：k 的选择\nk 太大→噪声放大（k→N 时 raw PSNR 跌至 %.0f dB，已出框）；'
          'k 太小→过度平滑' % psnr_curve_raw[-1])
plt.legend(fontsize=9, loc='lower right')
plt.grid(True)
plt.ylim(-5, 17.5)  # 截断纵轴以突出最优区间；崩坏尾部用文字注明
plt.savefig(os.path.join(SAVE_DIR, '实验1_3_2_截断SVD参数.png'), dpi=150, bbox_inches='tight')
plt.show()

# ===== 保存数值结果 =====
import json
condition_number_A = float(sigma[0] / sigma[-1])
psnr_truncated = {}
for k_eff in sorted(psnr_trunc_raw):
    psnr_truncated[str(k_eff)] = {
        'raw_dB': round(psnr_trunc_raw[k_eff], 2),
        'clipped_dB': round(psnr_trunc_clipped[k_eff], 2),
    }

results_summary = {
    'image_size': n,
    'matrix_size': N,
    'noise_sigma': float(noise_sigma),
    'condition_number_A': float(round(condition_number_A, 4)),
    'psnr_convention': {
        'raw_dB': '主口径：直接在重建值上以 data_range=1 计算，如实反映 1/sigma_i 的噪声放大',
        'clipped_dB': '对照口径：先 np.clip 到 [0,1] 再计算，与显示范围一致，但会为崩坏重建的误差封顶',
    },
    'psnr_pseudo_inverse_dB': {
        'raw_dB': round(psnr_pinv_raw, 2),
        'clipped_dB': round(psnr_pinv_clipped, 2),
    },
    'psnr_truncated_SVD': psnr_truncated,
    'best_k_global_scan': int(best_k),
    'psnr_best_global_dB': {
        'raw_dB': round(best_psnr_raw, 2),
        'clipped_dB': round(best_psnr_clipped, 2),
    },
    'selection_rule_note': 'best_k 按真值 PSNR 在 k=1..1024 上事后选峰（oracle），仅供教学；'
                           '实战无真值时用 GCV、L 曲线或 Morozov 偏差原理',
    'inverse_crime_note': '正、反演刻意共用同一离散算子 A 以隔离噪声放大机制；'
                          '建模误差（Inverse Crime）的影响见实验 1.3-4',
}
print(f"全 k 扫描事后最优: k={best_k}, PSNR(raw)={best_psnr_raw:.2f} dB, "
      f"PSNR(clipped)={best_psnr_clipped:.2f} dB")
print(f"伪逆 PSNR(raw)={psnr_pinv_raw:.2f} dB, PSNR(clipped)={psnr_pinv_clipped:.2f} dB")

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