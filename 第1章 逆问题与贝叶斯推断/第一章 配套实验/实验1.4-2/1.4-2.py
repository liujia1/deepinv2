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
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    _chinese_path = os.path.join(_gdrive, '实验1.4-2', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验1.4-2')
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

sys.path.insert(0, _chinese_path)
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=_chinese_path)
# ========================================================

# ---- 1. 自包含合成数据：合成图像 + 程序化构造的病态算子 ----
# 不再依赖任何外部图片或 .npy 文件：图像用 Shepp-Logan 幻影缩放到 32×32，
# 病态算子由 SVD 谱直接构造（U、V 为随机正交阵，稠密且含负元素，非严格卷积核）。
n_side = 32
N = n_side * n_side

# 奇异值谱：前 1000 个从 1（相对值）平缓衰减到 1e-2，末尾 24 个继续衰减到 1/κ。
# 这样 σ < 1%·σ_max 的方向恰好 24 个，κ = 4.8e4（轻度病态）。
N_NEAR_NULL = 24
KAPPA = 4.8e4
s = np.concatenate([
    np.logspace(0, -2, N - N_NEAR_NULL),
    np.logspace(-2.1, -np.log10(KAPPA), N_NEAR_NULL),
])


def _random_orthogonal(rng, n):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


_rng_build = np.random.default_rng(2024)
U = _random_orthogonal(_rng_build, N)
V = _random_orthogonal(_rng_build, N)
A = (U * s) @ V.T          # A = U diag(s) Vᵀ
Vt = V.T
condition_number = float(s[0] / s[-1])
print(f"变换矩阵 A 的形状: {A.shape}")
print(f"\nA 的奇异值分析：")
print(f"  最大奇异值 σ_max: {s[0]:.4e}")
print(f"  最小奇异值 σ_min: {s[-1]:.4e}")
print(f"  条件数 κ(A) = σ_max/σ_min: {condition_number:.4e}")
print(f"  log₁₀(κ): {np.log10(condition_number):.2f}")
print(f"  σ < 1%·σ_max 的方向数: {int(np.sum(s < 0.01 * s[0]))}")

if condition_number > 1e10:
    print(f"\n⚠️ A 是高度病态矩阵（条件数 > 10¹⁰）")
    print("   这意味着 A 严重损失信息，逆问题极度不稳定")
elif condition_number > 1e6:
    print(f"\n⚠️ A 是严重病态矩阵（条件数 > 10⁶）")
    print("   这意味着 A 显著损失信息，逆问题很不稳定")
elif condition_number > 1e3:
    print(f"\nℹ️ A 是轻度病态矩阵（条件数 > 10³）")
    print("   这意味着 A 有一定信息损失，逆问题较不稳定")
else:
    print(f"\nℹ️ A 的条件数较小，不适定性不明显")

# 合成图像：Shepp-Logan 幻影缩放到 32×32，归一化到 [0,1]
x_true_img = resize(shepp_logan_phantom(), (n_side, n_side), order=1,
                    preserve_range=True, anti_aliasing=True)
x_true_img = x_true_img / x_true_img.max()
x_true = x_true_img.ravel()

# 观测：y = A x_true + ε，ε 取 8-bit 量化噪声（远小于信号，与正文一致）
y_clean = A @ x_true
y_obs = np.round(y_clean * 255.0) / 255.0

# 对照图像：x_random = x_true + d，d 完全落在 A 的 24 维近零空间内。
# 用像素级 RMS=0.15 标定 d 的幅度，使其"看起来像噪声"。
_g_null = _rng_build.standard_normal(N_NEAR_NULL)
_d_raw = V[:, -N_NEAR_NULL:] @ _g_null
d = _d_raw / np.sqrt(np.mean(_d_raw ** 2)) * 0.15
x_random = x_true + d

print(f"图像形状: {x_true_img.shape}")
print(f"量化噪声 ||ε||_2 = {np.linalg.norm(y_obs - y_clean):.6f}")
n_pix = x_true.size

# ---- 3. 计算似然（数据拟合误差）----
# 似然函数 p(y|x) ∝ exp(-||Ax - y||₂² / 2σ²)
# 数据拟合误差越小，似然越大

Ax_true = A @ x_true
Ax_random = A @ x_random

error_true = np.linalg.norm(Ax_true - y_obs)
error_random = np.linalg.norm(Ax_random - y_obs)

print("\n" + "="*60)
print("似然的关键洞察：似然只关心数据拟合，不关心 x 是否合理")
print("="*60)
print("\n数据拟合误差 ||Ax - y||_2 (L2范数):")
print(f"  真实图像 x_true:   {error_true:.4f}")
print(f"  对照图像 x_random: {error_random:.4f}")
print(f"\n比值: {error_true / error_random:.2f} 倍")

# ---- 3.5 三项诊断：厘清 x_random 为何能拟合数据 ----
# 诊断一：全新随机噪声并不会“碰巧”拟合更好（固定种子，可复现）
rng = np.random.default_rng(2026)
N_MC = 500
X_mc = rng.uniform(0.0, 1.0, size=(n_pix, N_MC))
E_mc = np.linalg.norm(A @ X_mc - y_obs[:, None], axis=0)
frac_beat = float((E_mc < error_true).mean())
pct_given = float((E_mc < error_random).mean())

# 诊断二：x_random 与 x_true 的差落在 A 的哪些奇异方向上
d_diff = x_random - x_true
Ad_diff = A @ d_diff
coef = Vt @ d_diff
mask_small = s < 0.01 * s[0]
n_small = int(mask_small.sum())
energy_frac_small = float(np.sum(coef[mask_small] ** 2) / np.sum(coef ** 2))

# 诊断三：最小二乘解——拟合完美，却离真值很远
x_ls = np.linalg.lstsq(A, y_obs, rcond=None)[0]
ls_fit_error = float(np.linalg.norm(A @ x_ls - y_obs))
ls_x_error = float(np.linalg.norm(x_ls - x_true))
x_true_norm = float(np.linalg.norm(x_true))

print("\n" + "-"*60)
print("诊断一：重新抽随机噪声，现象不会重现")
print("-"*60)
print(f"  {N_MC} 张 uniform[0,1] 随机噪声图（seed=2026）:")
print(f"    拟合误差中位数 {np.median(E_mc):.4f}，5–95 分位 "
      f"[{np.percentile(E_mc, 5):.4f}, {np.percentile(E_mc, 95):.4f}]")
print(f"    误差小于真实图像的比例: {frac_beat:.1%}")
print(f"  给定 x_random 的误差 {error_random:.4f} 处于该分布的 {pct_given:.1%} 分位")
print(f"  → 它不是“运气好的随机噪声”，而是被构造出来的对照样本")

print("\n" + "-"*60)
print("诊断二：构造原理——差值藏在 A 的近零空间里")
print("-"*60)
print(f"  ||x_random - x_true||   = {np.linalg.norm(d_diff):.4f}   （像素级看是噪声）")
print(f"  ||A(x_random - x_true)|| = {np.linalg.norm(Ad_diff):.6f} （经 A 后几乎消失）")
print(f"  差值能量落在 σ < 1%·σ_max 的 {n_small} 个方向上的比例: {energy_frac_small:.2%}")
print(f"  若差值是独立随机噪声，||A·diff|| 应为 {np.median(E_mc):.2f} 量级——相差约 "
      f"{np.median(E_mc) / max(np.linalg.norm(Ad_diff), 1e-12):.0f} 倍")

print("\n" + "-"*60)
print("诊断三：病态性的直接后果——完美拟合 ≠ 接近真值")
print("-"*60)
print(f"  最小二乘解 ||A x_ls - y|| = {ls_fit_error:.2e} （几乎完美拟合）")
print(f"  但 ||x_ls - x_true|| = {ls_x_error:.4f}，为 ||x_true|| = {x_true_norm:.4f} 的 "
      f"{ls_x_error / x_true_norm:.0%}")
print(f"  → 数据端微小误差经 A⁻¹ 放大为解端巨大偏差，这才是 κ(A) 的意义")

print("\n💡 正确的归因：")
print(f"   - κ(A) = {condition_number:.2e}，按脚本自身判据属“轻度病态”")
print(f"     A 并未“严重损失高频”：{n_pix} 个奇异值中仅 {n_small} 个衰减到 1% 以下")
print(f"   - 但只要存在这些近零方向，就有无穷多个“像素级像噪声、")
print(f"     数据上却与真值同样好”的解，似然对它们一视同仁")
print(f"   - 局限的根源是 A 的近零空间（不适定性），而非随机噪声的运气")

# ---- 4. 可视化 ----
fig, axes = plt.subplots(3, 3, figsize=(14, 13.5))

# 第一行：原始图像
axes[0, 0].imshow(x_true_img, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('真实图像 $x_{true}$\n（Shepp-Logan 合成幻影，32×32）')
axes[0, 0].axis('off')

axes[0, 1].imshow(y_obs.reshape(n_side, n_side), cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title('观测数据 $y = Ax_{true}+\\varepsilon$\n（8-bit 量化观测）')
axes[0, 1].axis('off')

axes[0, 2].imshow(np.clip(x_random.reshape(n_side, n_side), 0, 1), cmap='gray', vmin=0, vmax=1)
axes[0, 2].set_title('对照图像 $x_{random}$\n（像素级似噪声，实为构造样本）')
axes[0, 2].axis('off')

# 第二行：变换结果和矩阵A可视化
axes[1, 0].imshow(Ax_true.reshape(n_side, n_side), cmap='gray')
axes[1, 0].set_title(f'$A \\cdot x_{{true}}$\n$||Ax - y||_2 = {error_true:.4f}$')
axes[1, 0].axis('off')

# 中间：可视化矩阵 A（将 1024x1024 缩小到 32x32，与其他子图保持一致）
from scipy.ndimage import zoom
# 使用双线性插值将 1024x1024 缩小到 32x32（缩放因子 32/1024 = 1/32）
A_resized = zoom(A, 32/1024, order=1)  # order=1 表示双线性插值
im = axes[1, 1].imshow(A_resized, cmap='RdBu_r', aspect='auto')
axes[1, 1].set_title(f'变换矩阵 $A$（1024×1024 缩放到 32×32）\n$\\kappa(A) = {condition_number:.2e}$（轻度病态）')
axes[1, 1].axis('off')
# 添加颜色条
cbar = plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
cbar.set_label('矩阵元素值', fontsize=8)

axes[1, 2].imshow(Ax_random.reshape(n_side, n_side), cmap='gray')
axes[1, 2].set_title(f'$A \\cdot x_{{random}}$\n$||Ax - y||_2 = {error_random:.4f}$（与 $A x_{{true}}$ 几乎不可分辨）')
axes[1, 2].axis('off')

# 第三行：三项诊断
# (a) 全新随机噪声的拟合误差分布 vs 真值/对照样本
axes[2, 0].hist(E_mc, bins=40, color='lightsteelblue', edgecolor='steelblue')
axes[2, 0].axvline(error_true, color='tab:green', ls='--', lw=2,
                   label=f'$x_{{true}}$: {error_true:.4f}')
axes[2, 0].axvline(error_random, color='tab:red', ls='--', lw=2,
                   label=f'$x_{{random}}$（构造）: {error_random:.4f}')
axes[2, 0].set_yscale('log')
axes[2, 0].set_xlabel('$||A x - y||_2$')
axes[2, 0].set_ylabel('频数（对数轴）')
axes[2, 0].set_title(f'诊断一：{N_MC} 张全新随机噪声图的拟合误差（seed=2026）\n'
                     f'最小也达 {E_mc.min():.2f}，小于真值误差的比例 {frac_beat:.0%}')
axes[2, 0].legend(fontsize=9, loc='upper right')
axes[2, 0].annotate(f'两虚线在左端 ≈ {max(error_true, error_random):.3f}，\n'
                    f'比新抽噪声（中位数 {np.median(E_mc):.2f}）小约 '
                    f'{np.median(E_mc) / max(error_random, 1e-12):.0f} 倍\n（构造样本才能做到）',
                    xy=(0.35, 0.32), xycoords='axes fraction', fontsize=9)

# (b) 差值 d = x_random - x_true 的能量在奇异方向上的分布
axes[2, 1].scatter(s, coef ** 2, s=6, c=np.where(mask_small, 'tab:red', 'tab:blue'))
axes[2, 1].set_xscale('log')
axes[2, 1].set_yscale('log')
axes[2, 1].set_xlabel('奇异值 $\\sigma_j$（对数轴）')
axes[2, 1].set_ylabel('$|v_j^T d|^2$（对数轴）')
axes[2, 1].set_title('诊断二：差值能量谱\n'
                     f'红色（{n_small} 个近零方向）占能量 {energy_frac_small:.2%}')

# (c) 最小二乘解：完美拟合但远离真值
axes[2, 2].imshow(x_ls.reshape(n_side, n_side), cmap='gray')
axes[2, 2].set_title('诊断三：最小二乘解 $x_{ls}$（无裁剪显示）\n'
                     f'$||Ax_{{ls}}-y|| = {ls_fit_error:.0e}$，而 $||x_{{ls}}-x_{{true}}|| = {ls_x_error:.1f}$')
axes[2, 2].axis('off')

plt.suptitle('似然的局限：数据拟合好 ≠ 图像有意义\n'
             '对照样本由真值加近零空间噪声构造而成；根源是 A 的近零空间（不适定性），而非随机噪声的运气',
             fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_4_2_似然的局限.png'), dpi=150, bbox_inches='tight')
plt.show()

# ---- 5. 结论 ----
print("\n" + "="*60)
print("结论")
print("="*60)
print("似然函数 p(y|x) 只衡量'数据拟合程度'，即 ||Ax - y||_2 的大小。")
print("它无法区分：")
print("  - 与数据拟合好的'有意义'图像（真实合成幻影）")
print("  - 与数据拟合好的'无意义'图像（如本实验中构造的噪声样本）")
print("\n这正是不适定性的后果：")
print(f"  算子 A 存在近零空间（{n_pix} 个方向中 {n_small} 个衰减到 1% 以下）")
print("  真值 + 近零空间扰动 仍与数据拟合同样好，仅靠似然无法从中选出'正确'的 x")
print("\n需要说明的是：")
print("  本实验的 x_random 并非'碰巧拟合好'的随机噪声——重新抽样的")
print(f"  随机噪声无一能接近真值的拟合水平（误差中位数 {np.median(E_mc):.2f} vs {error_true:.4f}）")
print("  它是按'真值 + 近零空间噪声'构造的对照样本，这正说明：")
print("  似然的局限来自 A 的结构（不适定性），而不是噪声的运气")
print("\n解决方案：引入先验 p(x)")
print("  先验编码了我们对'什么样的图像更合理'的知识")
print("  后验 = 似然 × 先验，同时考虑数据拟合和图像合理性")

# ===== 保存数值结果 =====
import json
results_summary = {
    'data_source': '完全自包含合成数据（Shepp-Logan 缩放到 32×32 + 程序化 SVD 谱构造 A）',
    'synthetic_data': {
        'seed': 2024,
        'image': 'shepp_logan_phantom → 32×32 双线性缩放，归一化到 [0,1]',
        'spectrum': '前 1000 个 logspace(0,-2)，末尾 24 个 logspace(-2.1,-log10(4.8e4))',
        'n_near_null': N_NEAR_NULL,
        'target_kappa': KAPPA,
        'observation_noise': 'y = round(A x_true·255)/255（8-bit 量化）',
        'quantization_noise_norm': float(np.linalg.norm(y_obs - y_clean)),
        'd_rms_per_pixel': 0.15,
    },
    'transform_matrix_shape': list(A.shape),
    'condition_number': float(condition_number),
    'max_singular_value': float(s[0]),
    'min_singular_value': float(s[-1]),
    'log10_condition_number': float(np.log10(condition_number)),
    'data_fit_error_true': float(error_true),
    'data_fit_error_random': float(error_random),
    'error_ratio_true_vs_random': float(error_true / error_random),
    'mc_fresh_noise_seed': 2026,
    'mc_fresh_noise_n': N_MC,
    'mc_fresh_noise_median_error': float(np.median(E_mc)),
    'mc_fresh_noise_err_p05': float(np.percentile(E_mc, 5)),
    'mc_fresh_noise_err_p95': float(np.percentile(E_mc, 95)),
    'mc_fresh_noise_beat_true_fraction': frac_beat,
    'given_random_gray_percentile_in_mc': pct_given,
    'diff_norm_x_random_vs_true': float(np.linalg.norm(d_diff)),
    'diff_norm_after_A': float(np.linalg.norm(Ad_diff)),
    'n_near_null_directions_sigma_lt_1pct': n_small,
    'diff_energy_frac_in_near_null': energy_frac_small,
    'lstsq_fit_error': ls_fit_error,
    'lstsq_deviation_from_true': ls_x_error,
    'x_true_norm': x_true_norm,
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
