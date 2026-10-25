# -*- coding: utf-8 -*-
"""
实验2.2-2 Laplace先验的稀疏魔力：为什么它真的能把系数"压成零"
对应章节：2.2 经典先验族 - Laplace先验（稀疏促成的几何解释）
知识点：Laplace先验(双指数)在原点处的尖峰(不可导)→ MAP诱导L1惩罚
          → 对小系数"置零"而非"缩小"；与高斯先验(岭回归)的对比；
          三视角(几何/概率/优化)在一维稀疏恢复上的统一体现

素材来源：
  - 2.2章节 Laplace先验小节：零点奇性、L1促稀疏、稀疏表示框架
  - 实验2.2-1：作为对照（高斯先验→岭回归）

修改说明：
  1. 一维稀疏信号 + 加噪观测，构造可手算的 MAP 问题；SNR 严格按向量能量定标
  2. 同 λ（与噪声水平匹配）下对比 LASSO (L1) 与 Ridge (L2) 的系数估计
  3. 用软阈值(soft-thresholding)闭式解展示 Laplace 的"硬置零"机制，并附 λ 扫描
  4. Part A 输出图1-图3；Part B 输出图4-图5，并新增图6（30 seed 稳健性验证）
  5. Part B：1D 压缩感知下 IHT / 重加权 ℓ¹ / OMP 三种逼近 ℓ⁰ 算法

修订记录（2026-10 复核修正）：
  - 修复 SNR 定标 bug：旧实现 σ = ‖信号‖/10^(SNR/20) 漏除 √维度，
    Part A 实际向量 SNR 仅 3.4dB（标称 20dB）、Part B 仅 11.2dB（标称 30dB）；
    现统一改为 σ = ‖信号‖/(√维度 · 10^(SNR/20))。
  - λ 改为随噪声水平选取（Part A: λ=0.1≈1.9σ；Part B: λ=1.2σ），
    旧的固定 λ=0.6/0.1 在正确 SNR 下属于严重过正则。
  - 支撑集指标从单侧"恢复率(=召回率)"改为双侧 recall/precision/误报/漏检，
    避免 Ridge 靠"全部非零"白拿 100%。
  - 新增 30 个随机种子的稳健性验证（图6 + JSON），替代旧版仅凭单 seed 的文字断言。
"""

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import sys
import io
import json
import warnings
import logging

# 设置控制台输出为 UTF-8 (Windows 下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# 静默 matplotlib 相关警告
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====== 中文字体配置(兼容本地和 Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验2.2-2')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(_chinese_path, exist_ok=True)
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

# ─── 实验参数配置 ───
SIGNAL_SEED = 7        # 稀疏信号随机种子
NOISE_SEED  = 42       # 观测噪声随机种子
N        = 50          # 信号维度
K        = 5           # 真实非零系数个数（稀疏度）
SNR_DB   = 20          # 目标观测信噪比（向量能量口径）
# 正则强度（L1 与 L2 共用同一 λ 以便公平对比）：
# 20dB 下 σ≈0.054，取 λ=0.1（≈1.9σ），既能清掉绝大多数纯噪声分量，
# 又能保住最弱的真峰（其观测量级约 2.6σ）。旧值 0.6≈11σ 属过正则。
LAMBDA   = 0.1
GRID     = 200         # 绘图网格密度

# ─── 构造稀疏真值 x* 与观测 y = x* + 噪声 ───
def make_sparse_signal(n, k, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    idx = rng.choice(n, size=k, replace=False)
    # 正负交替、幅度不一，制造"少数大系数 + 大量零"的稀疏结构
    x[idx] = rng.uniform(-2.5, 2.5, size=k)
    return x, idx

x_true, support = make_sparse_signal(N, K, seed=SIGNAL_SEED)
# 加性高斯噪声，按"向量 SNR"定标：SNR = 20log10(‖x‖/‖e‖)，而 E‖e‖² = N·σ²
# 故 σ = ‖x‖ / (√N · 10^(SNR/20))（旧实现漏除 √N，实际 SNR 只有 3.4dB）
sig = np.linalg.norm(x_true) / (np.sqrt(N) * 10 ** (SNR_DB / 20))
noise = np.random.RandomState(NOISE_SEED).normal(0, sig, size=N)
y = x_true + noise
snr_realized = 20 * np.log10(np.linalg.norm(x_true) / np.linalg.norm(noise))

# ─── 闭式 MAP 估计 ───
# 似然 x ~ N(y,1)：负对数似然 ∝ ½||y - x||^2（数据项统一取 ½，使 L1/L2 在同 λ 下可比）
# 高斯先验 N(0, 1/λ) → L2 惩罚 (λ/2)||x||^2 → Ridge 闭式解 x = y / (1+λ)
# Laplace 先验 → L1 惩罚 λ|x| → 软阈值 x = sign(y)·max(|y|-λ, 0)
def ridge_estimate(y, lam):
    return y / (1.0 + lam)

def lasso_estimate(y, lam):
    return np.sign(y) * np.maximum(np.abs(y) - lam, 0.0)

x_ridge = ridge_estimate(y, LAMBDA)
x_lasso = lasso_estimate(y, LAMBDA)

# ─── 评估：恢复误差 + 支撑集双侧指标（召回/精确率/误报/漏检）───
def support_pr(x_hat, x_true, tol=1e-6):
    """返回 (召回率, 精确率, 真阳TP, 误报FP, 漏检FN)。
    旧版只算召回率：Ridge 因 50 个分量全非零而白拿 100%——单侧指标会误导。"""
    true_sup = np.abs(x_true) > tol
    est_sup = np.abs(x_hat) > tol
    tp = int(np.sum(true_sup & est_sup))
    fp = int(np.sum(~true_sup & est_sup))
    fn = int(np.sum(true_sup & ~est_sup))
    recall = tp / max(int(true_sup.sum()), 1)        # 真支撑中被正确检出的比例
    precision = tp / max(tp + fp, 1)                 # 报为非零的分量中真是支撑的比例
    return recall, precision, tp, fp, fn

def evaluate(x_hat, x_true):
    recall, precision, tp, fp, fn = support_pr(x_hat, x_true)
    return {
        'MSE': float(np.mean((x_hat - x_true) ** 2)),
        'support_recall': float(recall),
        'support_precision': float(precision),
        'support_hits': tp,
        'false_positives': fp,
        'misses': fn,
        'exact_zero_count': int(np.sum(np.abs(x_hat) < 1e-6)),
    }

met_ridge = evaluate(x_ridge, x_true)
met_lasso = evaluate(x_lasso, x_true)
mse_ridge = met_ridge['MSE']
mse_lasso = met_lasso['MSE']
rec_ridge = met_ridge['support_recall']
rec_lasso = met_lasso['support_recall']
n_zero_ridge = met_ridge['exact_zero_count']
n_zero_lasso = met_lasso['exact_zero_count']

print(f"目标SNR={SNR_DB}dB  实际向量SNR={snr_realized:.1f}dB  σ={sig:.4f}  λ={LAMBDA}")
print(f"Ridge(L2): MSE={mse_ridge:.4f}  召回={rec_ridge:.0%}  精确率="
      f"{met_ridge['support_precision']:.0%}  误报={met_ridge['false_positives']}  "
      f"精确置零={n_zero_ridge}")
print(f"LASSO(L1): MSE={mse_lasso:.4f}  召回={rec_lasso:.0%}  精确率="
      f"{met_lasso['support_precision']:.0%}  误报={met_lasso['false_positives']}  "
      f"精确置零={n_zero_lasso}")

# ─── 图1：先验密度对比（几何/概率视角）───
fig, ax = plt.subplots(figsize=(7, 4.5))
t = np.linspace(-3, 3, GRID)
p_gauss = np.exp(-0.5 * t ** 2) / np.sqrt(2 * np.pi)
p_lap = 0.5 * np.exp(-np.abs(t))
ax.plot(t, p_gauss, lw=2, label='高斯先验 N(0,1)：处处光滑、原点平缓')
ax.plot(t, p_lap, lw=2, label='Laplace先验：原点尖峰(不可导)')
ax.axvline(0, color='gray', ls=':', lw=1)
ax.set_title('Laplace vs 高斯：先验密度在原点处的形态差异')
ax.set_xlabel('系数取值 x')
ax.set_ylabel('先验密度 p(x)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '图1_先验密度对比.png'), dpi=120)
plt.close(fig)

# ─── 图2：系数估计对比（恢复效果）───
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
coords = np.arange(N)
for ax, x_, title, color in [
    (axes[0], x_true, '真实稀疏信号 x*', 'black'),
    (axes[1], x_ridge, f'Ridge (L2)\nMSE={mse_ridge:.3f}', 'tab:blue'),
    (axes[2], x_lasso, f'LASSO (L1)\nMSE={mse_lasso:.3f}', 'tab:red'),
]:
    ax.stem(coords, x_, basefmt=' ')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('系数索引')
    ax.axhline(0, color='gray', lw=0.8)
    ax.grid(alpha=0.3)
axes[0].set_ylabel('系数值')
fig.suptitle(f'同 $\\lambda={LAMBDA}$ 下：高斯先验只"缩小"、Laplace先验能"置零"', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '图2_系数估计对比.png'), dpi=120)
plt.close(fig)

# ─── 图3：软阈值函数（优化视角）───
fig, ax = plt.subplots(figsize=(7, 4.5))
u = np.linspace(-3, 3, GRID)
soft = np.sign(u) * np.maximum(np.abs(u) - LAMBDA, 0)
hard_ridge = u / (1 + LAMBDA)
ax.plot(u, u, 'k--', lw=1, label='恒等（无正则，$y=x$）')
ax.plot(u, hard_ridge, lw=2, color='tab:blue', label=f'Ridge 映射 $x=y/(1+\\lambda)$')
ax.plot(u, soft, lw=2, color='tab:red', label=f'软阈值 $S_\\lambda(y)=\\operatorname{{sign}}(y)\\cdot\\max(|y|-\\lambda,0)$')
ax.axvspan(-LAMBDA, LAMBDA, color='tab:red', alpha=0.12, label=f'置零死区 $[-\\lambda, \\lambda]={LAMBDA}$')
ax.set_title('软阈值 vs 岭映射：为什么 $L_1$ 能把小系数真正压成 0')
ax.set_xlabel('观测 $y$')
ax.set_ylabel('估计 $x$')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '图3_软阈值函数.png'), dpi=120)
plt.close(fig)

print("已保存：图1_先验密度对比.png / 图2_系数估计对比.png / 图3_软阈值函数.png")

# ─── λ 扫描：把"置零能力 vs 误杀弱峰"的权衡变成可核验数据 ───
lam_grid = np.linspace(0.01, 0.30, 30)
_fields = ('MSE', 'support_recall', 'support_precision', 'exact_zero_count')
sweep = {'lambda': [], 'LASSO_L1': {f: [] for f in _fields},
                          'Ridge_L2': {f: [] for f in _fields}}
print("\nλ 扫描（LASSO 召回/精确率/置零数 与 Ridge MSE）：")
for L in lam_grid:
    ml = evaluate(lasso_estimate(y, L), x_true)
    mr = evaluate(ridge_estimate(y, L), x_true)
    sweep['lambda'].append(round(float(L), 4))
    for key, met in (('LASSO_L1', ml), ('Ridge_L2', mr)):
        for f in _fields:
            sweep[key][f].append(float(met[f]))
    print(f"  λ={L:5.2f}  LASSO: MSE={ml['MSE']:.5f} 召回={ml['support_recall']:.0%} "
          f"精确率={ml['support_precision']:.0%} 置零={ml['exact_zero_count']:2d} | "
          f"Ridge: MSE={mr['MSE']:.5f} 置零={mr['exact_zero_count']}")

# ─── Part A 结果保存为 JSON ───
part_a_results = {
    'target_snr_db': SNR_DB,
    'realized_snr_db': float(snr_realized),
    'sigma': float(sig),
    'lambda': LAMBDA,
    'N': N,
    'K': K,
    'signal_seed': SIGNAL_SEED,
    'noise_seed': NOISE_SEED,
    'Ridge_L2': met_ridge,
    'LASSO_L1': met_lasso,
    'lambda_sweep': sweep,
}


# ======================================================================
# Part B：从 ℓ¹ 到 ℓ⁰ —— 压缩感知下的迭代硬阈值 / 重加权 ℓ¹ / OMP
# 对应章节：2.2 经典先验族 · "从 ℓ¹ 到 ℓ⁰"说明框（271-281 行）
# 动机：Laplace 先验在"变换域系数 α"上生效（249-265 行）。
#       这里 α 即合成形式 min‖y-Aα‖² + λ‖α‖₁ 中的稀疏系数，
#       用 1D 压缩感知构造一个真正欠定(A 为 M×N, M<N)且 α* 稀疏的问题，
#       对比三种"逼近 ℓ⁰"的实用算法。
# ======================================================================

# ─── Part B 实验参数 ───
N_B      = 256         # 信号维度（系数个数）
K_B      = 10          # 真实非零系数个数（稀疏度）
M_B      = 80          # 观测数（M < N，真正欠定）
SNR_B    = 30          # 观测信噪比
SEED_B   = 7           # 与 Part A 一致，便于复现


def make_cs_problem(n, k, m, snr, seed=0):
    """构造 1D 压缩感知问题：y = A α* + 噪声，α* 仅 k 个分量非零。"""
    rng = np.random.default_rng(seed)
    alpha = np.zeros(n)
    idx = rng.choice(n, size=k, replace=False)
    alpha[idx] = rng.uniform(-3, 3, size=k)
    # 高斯随机测量矩阵（归一化列，近似满足 RIP）
    A = rng.standard_normal((m, n))
    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    # 按"向量 SNR"定标：E‖e‖² = m·σ²，故 σ = ‖Aα‖/(√m·10^(SNR/20))
    # 旧实现漏除 √m（30dB 实际只有约 11dB）
    sigma = np.linalg.norm(A @ alpha) / (np.sqrt(m) * 10 ** (snr / 20))
    noise = rng.standard_normal(m) * sigma
    y = A @ alpha + noise
    return A, y, alpha, idx, sigma


A_b, y_b, alpha_true, support_b, sigma_b = make_cs_problem(N_B, K_B, M_B, SNR_B, seed=SEED_B)
# 实际向量 SNR；重加权 ℓ¹ 的初始正则强度取约 1.2σ，随测量噪声水平自适应
snr_b_realized = 20 * np.log10(np.linalg.norm(A_b @ alpha_true) /
                               np.linalg.norm(y_b - A_b @ alpha_true))
LAM_RW = 1.2 * sigma_b


# ─── 算法 1：迭代硬阈值 IHT ───
def IHT(A, y, K_max, max_iter=500, tol=1e-8, record_err=None, alpha_ref=None, verbose=True):
    """交替做梯度步 + 硬阈值（只保留绝对值最大的 K_max 个分量）。
    步长取 1/L，L=‖A‖₂²（谱范数平方）：RIP 条件下的标准安全步长。
    注意：在测量数偏少（RIP 边界附近）时，固定步长 IHT 可能停滞在错误支撑上，
    并不保证每次都恢复成功——实践中常用归一化 IHT（带线搜）或 HTP。"""
    m, n = A.shape
    At = A.T
    L = np.linalg.norm(A, 2) ** 2
    step = 1.0 / L
    alpha = np.zeros(n)
    for _ in tqdm(range(max_iter), desc='IHT', leave=False, disable=not verbose):
        grad = At @ (y - A @ alpha)
        alpha_new = alpha + step * grad  # 梯度步（步长 1/L）
        # 硬阈值：保留绝对值最大的 K_max 个分量，其余置零
        if K_max < alpha_new.size:
            thresh = np.sort(np.abs(alpha_new))[-K_max]
            alpha_new[np.abs(alpha_new) < thresh] = 0.0
        if record_err is not None and alpha_ref is not None:
            record_err.append(np.linalg.norm(alpha_new - alpha_ref) / np.linalg.norm(alpha_ref))
        if np.linalg.norm(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new
    return alpha


# ─── 算法 2：重加权 ℓ¹（IRL1）───
def Reweighted_L1(A, y, n_outer=12, eps=1e-2, lam=0.1, max_inner=1000,
                  record_err=None, alpha_ref=None, verbose=True):
    """外层迭代更新权重 w_i = 1/(|α_i|+ε)，内层用 ISTA 软阈值求解加权 ℓ¹ 子问题。
    lam 为初始（未加权）正则强度，应随测量噪声 σ 选取（本实验取 1.2σ）。
    max_inner=1000：λ 较小时 ISTA 收缩慢，300 步会提前停滞（与紧解相差约 0.017），
    1000 步经 FISTA 对照验证已收敛到 1e-5 以内。"""
    m, n = A.shape
    At = A.T
    L = np.linalg.norm(A, 2) ** 2  # Lipschitz 常数（用于 ISTA 步长）
    step = 1.0 / L
    alpha = np.zeros(n)
    w = np.ones(n)
    for _ in tqdm(range(n_outer), desc='Reweighted l1', leave=False, disable=not verbose):
        for _ in tqdm(range(max_inner), desc='  ISTA', leave=False, disable=not verbose):
            grad = At @ (y - A @ alpha)
            soft = np.sign(alpha + step * grad) * np.maximum(
                np.abs(alpha + step * grad) - step * lam * w, 0.0)
            if np.linalg.norm(soft - alpha) < 1e-8:
                alpha = soft
                break
            alpha = soft
        w = 1.0 / (np.abs(alpha) + eps)
        if record_err is not None and alpha_ref is not None:
            record_err.append(np.linalg.norm(alpha - alpha_ref) / np.linalg.norm(alpha_ref))
    return alpha


# ─── 算法 3：正交匹配追踪 OMP ───
def OMP(A, y, K_max, tol=1e-6, verbose=True):
    """贪心法：每步挑一个最能降低残差的基础原子，逐步搭起支撑集后做最小二乘。"""
    m, n = A.shape
    residual = y.copy()
    idx_set = []
    alpha = np.zeros(n)
    for _ in tqdm(range(K_max), desc='OMP', leave=False, disable=not verbose):
        corr = A.T @ residual
        new_idx = int(np.argmax(np.abs(corr)))
        if new_idx in idx_set:
            break
        idx_set.append(new_idx)
        Asub = A[:, idx_set]
        coef, *_ = np.linalg.lstsq(Asub, y, rcond=None)
        residual = y - Asub @ coef
        if np.linalg.norm(residual) < tol:
            break
    alpha[idx_set] = coef
    return alpha


# ─── 运行三种算法 ───
alpha_omp  = OMP(A_b, y_b, K_B)
# alpha_iht / alpha_rw 在下方收敛曲线段通过 IHT/Reweighted_L1(..., record_err=...) 一并计算

# 收敛曲线：IHT 每步相对误差 + 重加权 ℓ¹ 外层相对误差
iht_errs = []
alpha_iht = IHT(A_b, y_b, K_B, max_iter=200, record_err=iht_errs, alpha_ref=alpha_true)

rw_errs = []
alpha_rw = Reweighted_L1(A_b, y_b, n_outer=12, eps=1e-2, lam=LAM_RW,
                         record_err=rw_errs, alpha_ref=alpha_true)


# ─── 评估：恢复相对误差 + 非零个数 + 支撑命中/误报/漏检 ───
def rel_err(a_hat, a_true):
    return np.linalg.norm(a_hat - a_true) / np.linalg.norm(a_true)

def n_nonzero(a_hat, tol=1e-6):
    return int(np.sum(np.abs(a_hat) > tol))

def eval_cs(a_hat, a_true):
    """压缩感知恢复的完整指标（复用 Part A 的双侧支撑集统计）。"""
    recall, precision, tp, fp, fn = support_pr(a_hat, a_true)
    return {
        'relative_error': float(rel_err(a_hat, a_true)),
        'recovered_nonzeros': n_nonzero(a_hat),
        'support_recall': float(recall),
        'support_precision': float(precision),
        'support_hits': tp,
        'false_positives': fp,
        'misses': fn,
    }

_alphas_b = {
    'IHT (硬阈值→ℓ⁰)': alpha_iht,
    'Reweighted ℓ¹':   alpha_rw,
    'OMP (贪婪)':      alpha_omp,
}
metrics_b = {name: eval_cs(a, alpha_true) for name, a in _alphas_b.items()}
# 图4 仍按 (估计, 相对误差, 非零数) 三元组使用
results_b = {name: (a, metrics_b[name]['relative_error'],
                          metrics_b[name]['recovered_nonzeros'])
             for name, a in _alphas_b.items()}
print(f"\n[Part B] N={N_B} K={K_B} M={M_B} 目标SNR={SNR_B}dB 实际SNR={snr_b_realized:.1f}dB "
      f"σ={sigma_b:.4f} λ_RW={LAM_RW:.4f} 真支撑大小={len(support_b)}")
for name in _alphas_b:
    m = metrics_b[name]
    print(f"  {name:18s} 相对误差={m['relative_error']:.4f}  非零={m['recovered_nonzeros']:2d}  "
          f"命中={m['support_hits']:2d}  误报={m['false_positives']}  漏检={m['misses']}")

# ─── 30 个随机种子的稳健性验证（旧版只凭单 seed 文字断言，此处固化为可核验数据）───
N_SEEDS = 30
FAIL_THRESH = 0.1   # 相对误差 > 0.1 记为一次恢复失败
_ALGO_NAMES = list(_alphas_b.keys())

def run_multi_seed(n_seeds=N_SEEDS):
    stats = {nm: {'relative_error': [], 'support_hits': [], 'false_positives': [],
                  'recovered_nonzeros': []} for nm in _ALGO_NAMES}
    for s in tqdm(range(n_seeds), desc='多seed验证'):
        A_s, y_s, at_s, _, sig_s = make_cs_problem(N_B, K_B, M_B, SNR_B, seed=s)
        lam_s = 1.2 * sig_s   # 每个 seed 都按自身噪声水平定 λ，保持准则一致
        alphas = [
            IHT(A_s, y_s, K_B, max_iter=200, verbose=False),
            Reweighted_L1(A_s, y_s, n_outer=12, eps=1e-2, lam=lam_s, verbose=False),
            OMP(A_s, y_s, K_B, verbose=False),
        ]
        for nm, a in zip(_ALGO_NAMES, alphas):
            m = eval_cs(a, at_s)
            for key in ('relative_error', 'support_hits', 'false_positives', 'recovered_nonzeros'):
                stats[nm][key].append(m[key])
    return stats

seed_stats = run_multi_seed()
multi_seed_summary = {}
print(f"\n[多seed验证] seeds=0..{N_SEEDS - 1}，真实 SNR≈{SNR_B}dB，失败阈值 rel>{FAIL_THRESH}")
for nm, st in seed_stats.items():
    e = np.array(st['relative_error'])
    h = np.array(st['support_hits'])
    fp = np.array(st['false_positives'])
    summary = {
        'relative_error_mean': float(e.mean()),
        'relative_error_std': float(e.std()),
        'relative_error_min': float(e.min()),
        'relative_error_max': float(e.max()),
        'support_hits_mean': float(h.mean()),
        'zero_fp_seed_ratio': float(np.mean(fp == 0)),
        f'failure_rate_rel_gt_{FAIL_THRESH}': float(np.mean(e > FAIL_THRESH)),
    }
    multi_seed_summary[nm] = summary
    print(f"  {nm:18s} rel={e.mean():.4f}±{e.std():.4f} [{e.min():.4f}~{e.max():.4f}]  "
          f"命中={h.mean():.1f}/10  零误报seed={summary['zero_fp_seed_ratio']:.0%}  "
          f"失败率={summary[f'failure_rate_rel_gt_{FAIL_THRESH}']:.0%}")

# ─── 图4：四合一茎叶图——真值 vs 三算法恢复 ───
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
coords = np.arange(N_B)
for ax, (title, (a, _, _)), color in [
    (axes[0], (r'真实稀疏系数 $\alpha^*$', (alpha_true, 0, 0)), 'black'),
    (axes[1], ('IHT',            results_b['IHT (硬阈值→ℓ⁰)']), 'tab:red'),
    (axes[2], (r'Reweighted $\ell^1$',  results_b['Reweighted ℓ¹']),   'tab:green'),
    (axes[3], ('OMP',            results_b['OMP (贪婪)']),      'tab:blue'),
]:
    ax.stem(coords, a, basefmt=' ')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('系数索引')
    ax.axhline(0, color='gray', lw=0.8)
    ax.grid(alpha=0.3)
axes[0].set_ylabel('系数值')
fig.suptitle(f'压缩感知($N={N_B}, M={M_B}<N$)：三种"逼近 $\\ell^0$"算法对稀疏系数 $\\alpha$ 的恢复', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '图4_1D恢复对比.png'), dpi=120)
plt.close(fig)

# ─── 图5：收敛曲线（IHT 与重加权 ℓ¹ 的相对误差）───
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(np.arange(1, len(iht_errs) + 1), iht_errs, lw=2, color='tab:red',
            label='IHT（每步硬阈值）')
ax.semilogy(np.arange(1, len(rw_errs) + 1), rw_errs, lw=2, color='tab:green',
            label=r'Reweighted $\ell^1$（外层迭代）')
ax.set_xlabel('迭代次数')
ax.set_ylabel('相对恢复误差 $\\|\\hat{\\alpha}-\\alpha^*\\| / \\|\\alpha^*\\|$')
ax.set_title('IHT 与重加权 $\\ell^1$ 的收敛行为：从 $\\ell^1$ 走向更贴近 $\\ell^0$ 的解')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which='both')
# 对数轴默认 LogFormatter 的主刻度标签包在 \mathdefault{} 里，其上标负号（U+2212）会
# 回退到中文字体（SimHei 缺该字形）而渲染成豆腐块；改用纯 mathtext 格式器，
# 负指数由数学字体（DejaVu）渲染，与本图标题/轴标里的其他 mathtext 风格保持一致
ax.yaxis.set_major_formatter(FuncFormatter(
    lambda v, _: f'$10^{{{int(round(np.log10(v)))}}}$'))
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '图5_收敛曲线.png'), dpi=120)
plt.close(fig)

# ─── 图6：30 个随机种子下的稳健性（误差分布 + 支撑命中）───
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
colors_b = ['tab:red', 'tab:green', 'tab:blue']
short_names = ['IHT', 'Reweighted $\\ell^1$', 'OMP']
data_err = [seed_stats[nm]['relative_error'] for nm in _ALGO_NAMES]
bp = axes[0].boxplot(data_err, labels=short_names, patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='yellow', markeredgecolor='black'))
for patch, c in zip(bp['boxes'], colors_b):
    patch.set_facecolor(c)
    patch.set_alpha(0.35)
axes[0].set_yscale('log')
# 对数轴默认 LogFormatter 的标签含 \mathdefault{-1}，负号会回退到中文字体（SimHei 缺 U+2212）
# 渲染成豆腐块；改用纯 mathtext 格式器由数学字体渲染，负指数正常显示
plt.rcParams['mathtext.fontset'] = 'cm'
axes[0].yaxis.set_major_formatter(FuncFormatter(
    lambda v, _: f'$10^{{{int(round(np.log10(v)))}}}$'))
axes[0].set_ylabel('相对恢复误差（30 个 seed 的分布，对数轴）')
axes[0].set_title(f'误差分布（菱形=均值；rel>{FAIL_THRESH} 记为失败）')
axes[0].grid(alpha=0.3, which='both', axis='y')

xs = np.arange(3)
hits_mean = [np.mean(seed_stats[nm]['support_hits']) for nm in _ALGO_NAMES]
hits_std = [np.std(seed_stats[nm]['support_hits']) for nm in _ALGO_NAMES]
axes[1].bar(xs, hits_mean, yerr=hits_std, color=colors_b, alpha=0.6, capsize=5)
axes[1].axhline(K_B, color='gray', ls='--', lw=1, label=f'真支撑大小 K={K_B}')
axes[1].set_xticks(xs)
axes[1].set_xticklabels(short_names)
axes[1].set_ylim(0, 11)
axes[1].set_ylabel('正确命中的支撑数（均值±标准差）')
axes[1].set_title('支撑集命中数（满分 10）')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3, axis='y')
fig.suptitle(f'{N_SEEDS} 个随机种子的稳健性验证（真实 SNR={SNR_B}dB, '
             f'$M={M_B}<N={N_B}$, 每 seed 独立生成问题）', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '图6_多seed稳健性.png'), dpi=120)
plt.close(fig)

print("已保存：图4_1D恢复对比.png / 图5_收敛曲线.png / 图6_多seed稳健性.png")

# ─── Part B 结果保存为 JSON ───
part_b_results = {
    'N': N_B,
    'K': K_B,
    'M': M_B,
    'target_snr_db': SNR_B,
    'realized_snr_db': float(snr_b_realized),
    'sigma': float(sigma_b),
    'rw_lambda': float(LAM_RW),
    'seed': SEED_B,
    'true_support_size': int(len(support_b)),
    'algorithms': metrics_b,
}

# ─── 多 seed 验证结果 ───
multi_seed_results = {
    'n_seeds': N_SEEDS,
    'seeds': list(range(N_SEEDS)),
    'target_snr_db': SNR_B,
    'rw_lambda_rule': '1.2 * sigma（随每个 seed 的噪声水平自适应）',
    'failure_threshold_rel': FAIL_THRESH,
    'summary': multi_seed_summary,
    'raw_per_seed': seed_stats,
}

# ─── 合并 A / B 两 Part 结果，统一写入 results.json ───
all_results = {
    'experiment': '2.2-2',
    'title': 'Laplace先验的稀疏魔力：从 ℓ¹ 到 ℓ⁰',
    'part_A_L1_soft_threshold': part_a_results,
    'part_B_l0_approximation': part_b_results,
    'multi_seed_validation': multi_seed_results,
}
with open(os.path.join(SAVE_DIR, 'results.json'), 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print("已保存：results.json（Part A + Part B 数值结果）")
