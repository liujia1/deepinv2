import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging

# ====== 解决中文乱码的核心代码（Windows + Linux 自动适配）======
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

import platform
from matplotlib.font_manager import FontManager

def _find_chinese_font():
    """自动检测系统中可用的中文字体，兼容 Windows / Linux"""
    candidates = []
    if platform.system() == 'Windows':
        candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
    else:
        candidates = [
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
            'Noto Sans CJK SC', 'Noto Sans CJK',
            'Source Han Sans SC', 'AR PL UMing CN',
            'SimHei',
        ]
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available:
            return font
    import os, re
    cjk_patterns = ['cjk', 'wqy', 'noto.*cjk', 'wenquan', 'chinese', 'simhei']
    for f in fm.ttflist:
        name_lower = f.name.lower()
        fname_lower = (os.path.basename(f.fname) if hasattr(f, 'fname') else '').lower()
        for pat in cjk_patterns:
            if re.search(pat, name_lower) or re.search(pat, fname_lower):
                return f.name
    return None

_cn_font = _find_chinese_font()
if _cn_font:
    plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['font.family'] = 'sans-serif'
    print(f"[Font] 已检测到中文字体: {_cn_font}")
else:
    print("[Font] 未找到中文字体，中文可能显示为方框")
# ========================================================

np.random.seed(42)

# ---- 1. 构造不同条件数的对角矩阵 ----
n = 100
x_true = np.random.randn(n)

# 对角矩阵：最大奇异值=1，最小奇异值=1/κ
# 条件数从 1 到 10^12
kappas = [1, 10, 1e2, 1e3, 1e4, 1e6, 1e8, 1e10, 1e12]
noise_level = 1e-6  # 数据中的噪声水平

# 预计算各条件数对应的奇异值，避免重复构造
# 使用对数间隔，更接近真实不适定问题的奇异值衰减特性
sv_cache = {kappa: np.logspace(0, -np.log10(kappa), n) for kappa in kappas}

# 正则化参数（简化处理：实际应用中应使用 L-curve 或 GCV 选择最优参数）
lam = noise_level

rel_errors = []
rel_errors_tikh = []
bias_errors = []
variance_errors = []

for kappa in kappas:
    singular_values = sv_cache[kappa]

    # 正问题：y = Ax
    y = singular_values * x_true

    # 加入噪声
    y_noisy = y + noise_level * np.random.randn(n)

    # 逆问题：x = A^{-1} y_noisy（朴素重建）
    x_recon = y_noisy / singular_values

    # 朴素重建相对误差
    rel_err = np.linalg.norm(x_recon - x_true) / np.linalg.norm(x_true)
    rel_errors.append(rel_err)

    # Tikhonov 正则化重建：x = Σ/(Σ²+λ) * y_noisy
    # 注意：正则化引入了 bias-variance 权衡
    # - 偏差 (bias)：正则化使解偏离真实值（系统性误差）
    # - 方差 (variance)：正则化抑制噪声放大（随机误差）
    filter_factors = singular_values ** 2 / (singular_values ** 2 + lam)
    x_tikh = (filter_factors / singular_values) * y_noisy
    
    # 误差分解
    x_tikh_noise_free = (filter_factors / singular_values) * y  # 无噪声时的正则化解
    bias = x_tikh_noise_free - x_true  # 正则化偏差
    variance = x_tikh - x_tikh_noise_free  # 噪声引起的方差
    
    rel_err_tikh = np.linalg.norm(x_tikh - x_true) / np.linalg.norm(x_true)
    rel_errors_tikh.append(rel_err_tikh)
    
    bias_errors.append(np.linalg.norm(bias) / np.linalg.norm(x_true))
    variance_errors.append(np.linalg.norm(variance) / np.linalg.norm(x_true))

# ---- 2. 可视化 ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 重建误差 vs 条件数
axes[0, 0].loglog(kappas, rel_errors, 'o-', linewidth=2, markersize=8, label='朴素逆重建')
axes[0, 0].loglog(kappas, rel_errors_tikh, 's-', linewidth=2, markersize=8, label='Tikhonov 正则化')

# 理论上界: ‖δx‖/‖x‖ ≤ κ · ‖δy‖/‖y‖
# 注意：噪声向量 δy = noise_level * randn(n)，其期望范数为 noise_level * √n
theory_bound = []
for kappa in kappas:
    sv = sv_cache[kappa]
    y_norm = np.linalg.norm(sv * x_true)
    theory_bound.append(kappa * (noise_level * np.sqrt(n)) / y_norm)
axes[0, 0].loglog(kappas, theory_bound, '--', alpha=0.7, label='理论上界 κ·(‖δy‖/‖y‖)')

axes[0, 0].set_xlabel('条件数 κ(A)')
axes[0, 0].set_ylabel('重建相对误差 ‖x-x̂‖/‖x‖')
axes[0, 0].set_title('条件数与误差放大\nκ 越大→噪声放大越严重')
axes[0, 0].legend()
axes[0, 0].grid(True, which='both')

# 奇异值衰减示意
for kappa in [1, 1e3, 1e6, 1e10]:
    sv = sv_cache[kappa]
    axes[0, 1].semilogy(np.arange(1, n + 1), sv, label=f'κ={kappa:.0e}')
axes[0, 1].set_xlabel('奇异值索引 i')
axes[0, 1].set_ylabel('奇异值 σ_i')
axes[0, 1].set_title('不同条件数的奇异值分布\n小 σ_i → 1/σ_i 爆炸 → 噪声放大')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Bias-Variance vs 条件数 κ（固定 λ）
axes[1, 0].loglog(kappas, bias_errors, '^-', linewidth=2, markersize=8, label='偏差 (Bias)')
axes[1, 0].loglog(kappas, variance_errors, 'v-', linewidth=2, markersize=8, label='方差 (Variance)')
axes[1, 0].loglog(kappas, rel_errors_tikh, 's-', linewidth=2, markersize=8, label='总误差')
axes[1, 0].set_xlabel('条件数 κ(A)')
axes[1, 0].set_ylabel('相对误差')
axes[1, 0].set_title(f'Bias-Variance vs 条件数 κ（固定 λ={lam:.0e}）\n展示：问题难度变化时正则化的效果')
axes[1, 0].legend()
axes[1, 0].grid(True, which='both')

# Bias-Variance vs 正则化参数 λ（固定 κ）—— 经典权衡曲线
kappa_fixed = 1e6  # 固定一个高条件数
singular_values_fixed = sv_cache[kappa_fixed]
y_fixed = singular_values_fixed * x_true
y_noisy_fixed = y_fixed + noise_level * np.random.randn(n)

lambdas = np.logspace(-14, -2, 60)  # 扫描正则化参数，下界延伸到 σ_min² 以下
bias_vs_lambda = []
variance_vs_lambda = []
total_vs_lambda = []

for lam_val in lambdas:
    filter_factors = singular_values_fixed ** 2 / (singular_values_fixed ** 2 + lam_val)
    x_tikh = (filter_factors / singular_values_fixed) * y_noisy_fixed
    x_tikh_noise_free = (filter_factors / singular_values_fixed) * y_fixed
    
    bias = x_tikh_noise_free - x_true
    variance = x_tikh - x_tikh_noise_free
    
    bias_vs_lambda.append(np.linalg.norm(bias) / np.linalg.norm(x_true))
    variance_vs_lambda.append(np.linalg.norm(variance) / np.linalg.norm(x_true))
    total_vs_lambda.append(np.linalg.norm(x_tikh - x_true) / np.linalg.norm(x_true))

axes[1, 1].loglog(lambdas, bias_vs_lambda, '^-', linewidth=2, markersize=6, label='偏差 (Bias)')
axes[1, 1].loglog(lambdas, variance_vs_lambda, 'v-', linewidth=2, markersize=6, label='方差 (Variance)')
axes[1, 1].loglog(lambdas, total_vs_lambda, 's-', linewidth=2, markersize=6, label='总误差')

# 标记最优 λ
optimal_idx = np.argmin(total_vs_lambda)
optimal_lambda = lambdas[optimal_idx]
axes[1, 1].axvline(optimal_lambda, color='red', linestyle=':', alpha=0.7, label=f'最优 λ={optimal_lambda:.1e}')

axes[1, 1].set_xlabel('正则化参数 λ')
axes[1, 1].set_ylabel('相对误差')
axes[1, 1].set_title('Bias-Variance vs 正则化参数 λ（固定 κ=1e6）\n经典权衡：λ↑→偏差↑、方差↓，存在最优点')
axes[1, 1].legend()
axes[1, 1].grid(True, which='both')

plt.tight_layout()
plt.savefig('实验1_6_条件数.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"{'κ(A)':>12s}  {'朴素误差':>12s}  {'Tikhonov误差':>12s}  {'偏差':>12s}  {'方差':>12s}")
print("-" * 68)
for kappa, err, err_tikh, bias, var in zip(kappas, rel_errors, rel_errors_tikh, bias_errors, variance_errors):
    print(f"{kappa:12.0e}  {err:12.4e}  {err_tikh:12.4e}  {bias:12.4e}  {var:12.4e}")

# 验证误差分解的 Pythagorean 关系
# ‖total‖² = ‖bias‖² + 2⟨bias, variance⟩ + ‖variance‖²
# 仅当 bias ⊥ variance 时才等于 ‖bias‖² + ‖variance‖²
print("\n=== Pythagorean 验证（κ=1e6）===")
print(f"（使用主循环的固定 λ={lam:.0e}，κ=1e6 的噪声实现）")
idx = kappas.index(1e6)
total_sq = rel_errors_tikh[idx] ** 2
bias_sq = bias_errors[idx] ** 2
var_sq = variance_errors[idx] ** 2
print(f"总误差² = {total_sq:.4e}")
print(f"偏差² + 方差² = {bias_sq + var_sq:.4e}")
print(f"差异（交叉项）= {total_sq - (bias_sq + var_sq):.4e}")
