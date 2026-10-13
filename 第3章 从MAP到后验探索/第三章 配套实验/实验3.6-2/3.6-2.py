"""
经验贝叶斯参数估计：边际似然最大化
对应章节：3.6 收敛性分析与正则化参数选择（经验贝叶斯部分）
知识点：边际似然 p(y|alpha)、经验贝叶斯策略、高斯共轭闭式解、Fisher恒等式

严谨版说明（复核后升级）：
  - 早期版本用 100 点粗对数网格，经验贝叶斯(EB)与 PSNR 最优(oracle)恰好舍入到同一格点，
    造成"完全相同"的假象。本版改用 400 点网格 + 连续峰值细分，如实呈现二者"接近但不同"；
  - 增加 30 个随机种子的稳健性实验：EB 逐样本并不复现 oracle 的 alpha（相关性弱），
    但重建 PSNR 与 oracle 几乎持平（峰顶平台效应）——这才是经验贝叶斯真正稳健的价值。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
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
    SAVE_DIR = os.path.join(_gdrive, '经验贝叶斯参数估计')
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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")

np.random.seed(42)

# ══════════════════════════════════════════════════════════
# 1. 构造小规模问题
# ══════════════════════════════════════════════════════════
n = 50
m = 40
A = np.random.randn(m, n) / np.sqrt(m)

x_true = np.zeros(n)
x_true[5] = 1.0
x_true[15] = 0.8
x_true[30] = -0.5

sigma2 = 0.01
y = A @ x_true + np.sqrt(sigma2) * np.random.randn(m)
peak = np.max(np.abs(x_true))

# ══════════════════════════════════════════════════════════
# 2. 边际似然（高斯共轭闭式解）
# ══════════════════════════════════════════════════════════
def marginal_likelihood(y, A, alpha, sigma2):
    """
    高斯共轭下的边际似然 log p(y|alpha)
    边际协方差: C = sigma^2 * I + alpha * A * A^T

    高斯共轭情形下 p(y|alpha) 有解析闭式解，无需 EM / MCMC / Monte Carlo gradient。

    Fisher恒等式（标准形式）:
        nabla_alpha log p(y|alpha) = E_{p(x|y,alpha)}[nabla_alpha log p(y,x|alpha)]
    当似然 p(y|x) 不依赖 alpha 时（即 sigma^2 固定、alpha 仅出现在先验中），
    nabla_alpha log p(y|x) = 0，上式右端退化为：
        E_{p(x|y,alpha)}[nabla_alpha log p(x|alpha)]
    Fisher恒等式在一般模型中提供梯度的随机近似途径，
    但本实验有闭式边际似然，可直接对 alpha 优化，无需借助Fisher恒等式。
    """
    m = A.shape[0]
    C = sigma2 * np.eye(m) + alpha * A @ A.T
    sign, logdet = np.linalg.slogdet(C)
    C_inv_y = np.linalg.solve(C, y)
    log_ml = -0.5 * m * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * y @ C_inv_y
    return log_ml

# ══════════════════════════════════════════════════════════
# 3. Tikhonov解（用于评估重建质量）
# ══════════════════════════════════════════════════════════
def tikhonov_solve(y, A, alpha, sigma2):
    return np.linalg.solve(
        A.T @ A / sigma2 + np.eye(A.shape[1]) / alpha,
        A.T @ y / sigma2
    )

def psnr_of(y, A, alpha, x_true, sigma2, peak):
    x_hat = tikhonov_solve(y, A, alpha, sigma2)
    mse = np.mean((x_hat - x_true) ** 2)
    return 10 * np.log10(peak ** 2 / mse)

def refine_peak(fun, center, lo=1 / 4, hi=4, npts=4000):
    """在 center 邻域 [center*lo, center*hi] 用细网格求连续极大值点，消除粗网格舍入。"""
    local = np.logspace(np.log10(center * lo), np.log10(center * hi), npts)
    vals = np.array([fun(a) for a in local])
    return local[np.argmax(vals)], vals.max()

# ══════════════════════════════════════════════════════════
# 4. 对比不同参数选择方法（400 点网格 + 连续细分）
# ══════════════════════════════════════════════════════════
alphas = np.logspace(-4, 2, 400)

# (a) 边际似然（经验贝叶斯）
log_ml = np.array([marginal_likelihood(y, A, alpha, sigma2) for alpha in alphas])
alpha_eb = alphas[np.argmax(log_ml)]
alpha_eb_cont, log_ml_max = refine_peak(
    lambda a: marginal_likelihood(y, A, a, sigma2), alpha_eb)

# (b) PSNR（需知真解，oracle 参考）
psnrs = np.array([psnr_of(y, A, alpha, x_true, sigma2, peak) for alpha in alphas])
alpha_opt = alphas[np.argmax(psnrs)]
alpha_opt_cont, psnr_opt = refine_peak(
    lambda a: psnr_of(y, A, a, x_true, sigma2, peak), alpha_opt)
psnr_eb = psnr_of(y, A, alpha_eb, x_true, sigma2, peak)

print("=" * 60)
print("经验贝叶斯参数估计")
print("=" * 60)
print(f"[400点网格]  经验贝叶斯 alpha_hat = {alpha_eb:.4f}   oracle(PSNR最优) = {alpha_opt:.4f}")
print(f"[连续细分]   经验贝叶斯 alpha_hat = {alpha_eb_cont:.4f}   oracle(PSNR最优) = {alpha_opt_cont:.4f}")
print(f"  两者 alpha 相对差 = {100*(alpha_opt_cont/alpha_eb_cont-1):.2f}%  （非零，说明并非'完全相同'）")
print(f"  但重建 PSNR：EB={psnr_eb:.3f} dB, oracle={psnr_opt:.3f} dB, 差={psnr_opt-psnr_eb:.4f} dB")
print(f"  对数边际似然最大值: log p(y|alpha) = {log_ml_max:.2f}")

# ══════════════════════════════════════════════════════════
# 5. 多种子稳健性：EB 是否可靠地逼近 oracle 的“重建性能”
# ══════════════════════════════════════════════════════════
seeds = list(range(42, 72))
eb_list, opt_list, gap_list = [], [], []
for s in seeds:
    np.random.seed(s)
    A_s = np.random.randn(m, n) / np.sqrt(m)
    y_s = A_s @ x_true + np.sqrt(sigma2) * np.random.randn(m)
    lm = np.array([marginal_likelihood(y_s, A_s, a, sigma2) for a in alphas])
    ps = np.array([psnr_of(y_s, A_s, a, x_true, sigma2, peak) for a in alphas])
    ae = alphas[np.argmax(lm)]
    ao = alphas[np.argmax(ps)]
    eb_list.append(ae)
    opt_list.append(ao)
    gap_list.append(ps.max() - psnr_of(y_s, A_s, ae, x_true, sigma2, peak))
eb_arr = np.array(eb_list)
opt_arr = np.array(opt_list)
gap_arr = np.array(gap_list)
ratio = eb_arr / opt_arr
within2 = int(np.sum((ratio < 2) & (ratio > 0.5)))
corr = np.corrcoef(np.log10(eb_arr), np.log10(opt_arr))[0, 1]

print("\n" + "=" * 60)
print(f"[多种子稳健性 n={len(seeds)}]")
print("=" * 60)
print(f"  alpha 中位数：EB={np.median(eb_arr):.4f}, oracle={np.median(opt_arr):.4f}")
print(f"  EB 落在 oracle 2倍范围内的种子数：{within2}/{len(seeds)}")
print(f"  log10(alpha) 上 EB 与 oracle 的相关系数 = {corr:.3f}  （弱/负 => EB 不复现逐样本最优 alpha）")
print(f"  重建 PSNR 差距 (oracle-EB)：均值 {gap_arr.mean():.3f} dB, 最大 {gap_arr.max():.3f} dB  （≈0 => EB 性能≈oracle）")

# ══════════════════════════════════════════════════════════
# 6. 可视化 1：边际似然与重建质量随 alpha 变化（seed=42）
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].semilogx(alphas, log_ml, 'b-', linewidth=2)
axes[0].axvline(alpha_eb, color='r', linestyle='--',
                label=rf'EB: $\hat{{\alpha}}={alpha_eb:.4f}$')
axes[0].axvline(alpha_opt, color='g', linestyle=':',
                label=rf'最优: $\alpha={alpha_opt:.4f}$')
axes[0].set_xlabel(r'正则化参数 $\alpha$')
axes[0].set_ylabel(r'$\log p(y|\alpha)$')
axes[0].set_title(r'边际似然 $\log p(y|\alpha)$')
axes[0].legend()
axes[0].grid(True)

axes[1].semilogx(alphas, psnrs, 'b-', linewidth=2)
axes[1].axvline(alpha_eb, color='r', linestyle='--',
                label=rf'EB: $\hat{{\alpha}}={alpha_eb:.4f}$')
axes[1].axvline(alpha_opt, color='g', linestyle=':',
                label=rf'最优: $\alpha={alpha_opt:.4f}$')
axes[1].set_xlabel(r'正则化参数 $\alpha$')
axes[1].set_ylabel(r'PSNR (dB)')
axes[1].set_title(r'重建质量 vs $\alpha$')
axes[1].legend()
axes[1].grid(True)

plt.suptitle(r'经验贝叶斯参数估计', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_经验贝叶斯参数估计.png'), dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════
# 7. 可视化 2：多种子稳健性（alpha 对照散点 + PSNR 差距直方图）
# ══════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))

lims = [min(eb_arr.min(), opt_arr.min()) * 0.7,
        max(eb_arr.max(), opt_arr.max()) * 1.4]
axes2[0].loglog(opt_arr, eb_arr, 'o', ms=6, alpha=0.75, label='各随机种子')
axes2[0].plot(lims, lims, 'k--', linewidth=1, label=r'$\alpha_{EB}=\alpha_{oracle}$')
axes2[0].fill_between(lims, [l * 0.5 for l in lims], [l * 2 for l in lims],
                      color='gray', alpha=0.15, label='2× 一致带')
axes2[0].set_xlabel(r'oracle $\alpha$（PSNR 最优，需真解）')
axes2[0].set_ylabel(r'经验贝叶斯 $\hat{\alpha}$（仅需 $y$）')
axes2[0].set_title(rf'多种子：$\alpha$ 对照 (corr={corr:.2f})')
axes2[0].legend(fontsize=9)
axes2[0].grid(True, which='both', alpha=0.3)

axes2[1].hist(gap_arr, bins=12, color='steelblue', edgecolor='white')
axes2[1].axvline(0, color='r', linestyle='--', linewidth=1, label='0 dB（与 oracle 持平）')
axes2[1].set_xlabel(r'重建质量差距 PSNR(oracle) $-$ PSNR(EB) (dB)')
axes2[1].set_ylabel('种子数')
axes2[1].set_title(rf'EB 相对 oracle 的 PSNR 损失 (均值 {gap_arr.mean():.2f} dB)')
axes2[1].legend(fontsize=9)
axes2[1].grid(True, alpha=0.3)

plt.suptitle(r'经验贝叶斯选参的稳健性（30 个种子）', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_多种子稳健性.png'), dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════
# 8. 结果输出与教学说明
# ══════════════════════════════════════════════════════════
print(f"\n[核心发现]")
print(f"  1. EB 与 oracle 的 alpha 接近但不同（连续值 {alpha_eb_cont:.4f} vs {alpha_opt_cont:.4f}，"
      f"差 {100*(alpha_opt_cont/alpha_eb_cont-1):.1f}%）；早期 100 点粗网格的'完全相同'是舍入假象")
print(f"  2. 但 EB 选参的重建 PSNR（{psnr_eb:.2f} dB）与 oracle（{psnr_opt:.2f} dB）几乎持平——"
      f"PSNR 峰顶是平台，alpha 的小偏移不损害重建")
print(f"  3. 多种子：alpha 相关性弱（corr={corr:.2f}），EB 不复现逐样本最优 alpha；"
      f"但 PSNR 损失均值仅 {gap_arr.mean():.2f} dB => EB 稳健地逼近 oracle 的“性能”而非“参数”")
print(f"  4. 结论：经验贝叶斯无需真解即可把重建送到最优平台附近，是可靠的自动选参法")

# ===== 保存数值结果 =====
import json

def _to_native(obj):
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return _to_native(obj.tolist())
    try:
        import torch
        if isinstance(obj, torch.Tensor): return _to_native(obj.detach().cpu().tolist())
    except Exception:
        pass
    return obj

results_summary = {
    'n': n,
    'm': m,
    'sigma2': float(round(sigma2, 4)),
    'grid_points': len(alphas),
    'seed_main': 42,
    'alpha_eb_grid': float(round(alpha_eb, 4)),
    'alpha_opt_grid': float(round(alpha_opt, 4)),
    'alpha_eb_cont': float(round(alpha_eb_cont, 5)),
    'alpha_opt_cont': float(round(alpha_opt_cont, 5)),
    'alpha_rel_gap_pct': float(round(100 * (alpha_opt_cont / alpha_eb_cont - 1), 2)),
    'log_ml_max': float(round(log_ml_max, 2)),
    'psnr_eb': float(round(psnr_eb, 3)),
    'psnr_opt': float(round(psnr_opt, 3)),
    'psnr_gap_db': float(round(psnr_opt - psnr_eb, 4)),
    'multiseed': {
        'n_seeds': len(seeds),
        'median_alpha_eb': float(round(np.median(eb_arr), 4)),
        'median_alpha_opt': float(round(np.median(opt_arr), 4)),
        'within_2x_count': within2,
        'log_alpha_corr': float(round(corr, 3)),
        'psnr_gap_mean_db': float(round(gap_arr.mean(), 3)),
        'psnr_gap_max_db': float(round(gap_arr.max(), 3)),
    },
}

results_summary = {k: _to_native(v) for k, v in results_summary.items()}
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

print(f"\n{'=' * 60}")
print(f"实验完成")
print(f"输出图片: 步骤1_经验贝叶斯参数估计.png, 步骤2_多种子稳健性.png")
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
print(f"{'=' * 60}")
