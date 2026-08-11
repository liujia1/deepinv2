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
  1. 一维稀疏信号 + 加噪观测，构造可手算的 MAP 问题
  2. 同 λ 下对比 LASSO (L1) 与 Ridge (L2) 的系数估计
  3. 用软阈值(soft-thresholding)闭式解展示 Laplace 的"硬置零"机制
  4. 输出三张图：①先验密度对比 ②系数估计对比 ③软阈值函数
  5. Part B：1D 压缩感知下 IHT / 重加权 ℓ¹ / OMP 三种逼近 ℓ⁰ 算法
"""

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
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

np.random.seed(42)

# ─── 实验参数配置 ───
N        = 50          # 信号维度
K        = 5           # 真实非零系数个数（稀疏度）
LAMBDA   = 0.6         # 正则强度（L1 与 L2 共用同一 λ 以便公平对比）
SNR_DB   = 20          # 观测信噪比
GRID     = 200         # 绘图网格密度

# ─── 构造稀疏真值 x* 与观测 y = x* + 噪声 ───
def make_sparse_signal(n, k, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    idx = rng.choice(n, size=k, replace=False)
    # 正负交替、幅度不一，制造"少数大系数 + 大量零"的稀疏结构
    x[idx] = rng.uniform(-2.5, 2.5, size=k)
    return x, idx

x_true, support = make_sparse_signal(N, K, seed=7)
# 加性高斯噪声，按目标 SNR 定标
sig = np.linalg.norm(x_true) / (10 ** (SNR_DB / 20))
noise = np.random.normal(0, sig, size=N)
y = x_true + noise

# ─── 闭式 MAP 估计 ───
# 似然 ~ N(0,1)：负对数似然 = ||y - x||^2
# 高斯先验 N(0, 1/λ) → L2 惩罚 λ||x||^2/2 → Ridge 闭式解 x = y / (1+λ)
# Laplace 先验 → L1 惩罚 λ|x| → 软阈值 x = sign(y)·max(|y|-λ, 0)
def ridge_estimate(y, lam):
    return y / (1.0 + lam)

def lasso_estimate(y, lam):
    return np.sign(y) * np.maximum(np.abs(y) - lam, 0.0)

x_ridge = ridge_estimate(y, LAMBDA)
x_lasso = lasso_estimate(y, LAMBDA)

# ─── 评估：恢复误差 + 支撑恢复率 ───
def support_recovery(x_hat, x_true, tol=1e-6):
    true_sup = np.abs(x_true) > tol
    est_sup = np.abs(x_hat) > tol
    if true_sup.sum() == 0:
        return 1.0
    return np.mean(est_sup[true_sup])  # 真支撑中被正确检出的比例

mse_ridge = np.mean((x_ridge - x_true) ** 2)
mse_lasso = np.mean((x_lasso - x_true) ** 2)
rec_ridge = support_recovery(x_ridge, x_true)
rec_lasso = support_recovery(x_lasso, x_true)
n_zero_ridge = int(np.sum(np.abs(x_ridge) < 1e-6))
n_zero_lasso = int(np.sum(np.abs(x_lasso) < 1e-6))

print(f"λ={LAMBDA}  SNR={SNR_DB}dB")
print(f"Ridge(L2): MSE={mse_ridge:.4f}  支撑恢复率={rec_ridge:.2%}  精确置零数={n_zero_ridge}")
print(f"LASSO(L1): MSE={mse_lasso:.4f}  支撑恢复率={rec_lasso:.2%}  精确置零数={n_zero_lasso}")

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

# ─── Part A 结果保存为 JSON ───
part_a_results = {
    'lambda': LAMBDA,
    'snr_db': SNR_DB,
    'N': N,
    'K': K,
    'Ridge_L2': {
        'MSE': float(mse_ridge),
        'support_recovery_rate': float(rec_ridge),
        'exact_zero_count': n_zero_ridge,
    },
    'LASSO_L1': {
        'MSE': float(mse_lasso),
        'support_recovery_rate': float(rec_lasso),
        'exact_zero_count': n_zero_lasso,
    },
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
    sigma = np.linalg.norm(A @ alpha) / (10 ** (snr / 20))
    noise = rng.standard_normal(m) * sigma
    y = A @ alpha + noise
    return A, y, alpha, idx


A_b, y_b, alpha_true, support_b = make_cs_problem(N_B, K_B, M_B, SNR_B, seed=SEED_B)


# ─── 算法 1：迭代硬阈值 IHT ───
def IHT(A, y, K_max, max_iter=500, tol=1e-8, record_err=None, alpha_ref=None):
    """交替做梯度步 + 硬阈值（只保留绝对值最大的 K_max 个分量）。
    步长取 1/L，L=‖A‖₂²（谱范数平方），保证在欠定情形下收敛。"""
    m, n = A.shape
    At = A.T
    L = np.linalg.norm(A, 2) ** 2
    step = 1.0 / L
    alpha = np.zeros(n)
    for _ in tqdm(range(max_iter), desc='IHT', leave=False):
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
def Reweighted_L1(A, y, n_outer=12, eps=1e-2, lam=0.1, max_inner=300, record_err=None, alpha_ref=None):
    """外层迭代更新权重 w_i = 1/(|α_i|+ε)，内层用 ISTA 软阈值求解 ℓ¹ 子问题。"""
    m, n = A.shape
    At = A.T
    L = np.linalg.norm(A, 2) ** 2  # Lipschitz 常数（用于 ISTA 步长）
    step = 1.0 / L
    alpha = np.zeros(n)
    w = np.ones(n)
    for _ in tqdm(range(n_outer), desc='Reweighted l1', leave=False):
        for _ in tqdm(range(max_inner), desc='  ISTA', leave=False):
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
def OMP(A, y, K_max, tol=1e-6):
    """贪心法：每步挑一个最能降低残差的基础原子，逐步搭起支撑集后做最小二乘。"""
    m, n = A.shape
    residual = y.copy()
    idx_set = []
    alpha = np.zeros(n)
    for _ in tqdm(range(K_max), desc='OMP', leave=False):
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
alpha_rw   = Reweighted_L1(A_b, y_b, n_outer=12, eps=1e-2, lam=0.1)
alpha_omp  = OMP(A_b, y_b, K_B)
# alpha_iht 在下方收敛曲线段通过 IHT(..., record_err=...) 一并计算

# 收敛曲线：IHT 每步相对误差 + 重加权 ℓ¹ 外层相对误差
iht_errs = []
alpha_iht = IHT(A_b, y_b, K_B, max_iter=200, record_err=iht_errs, alpha_ref=alpha_true)

rw_errs = []
alpha_rw = Reweighted_L1(A_b, y_b, n_outer=12, eps=1e-2, lam=0.1, record_err=rw_errs, alpha_ref=alpha_true)


# ─── 评估：恢复相对误差 + 非零个数 ───
def rel_err(a_hat, a_true):
    return np.linalg.norm(a_hat - a_true) / np.linalg.norm(a_true)

def n_nonzero(a_hat, tol=1e-6):
    return int(np.sum(np.abs(a_hat) > tol))

results_b = {
    'IHT (硬阈值→ℓ⁰)':      (alpha_iht, rel_err(alpha_iht, alpha_true), n_nonzero(alpha_iht)),
    'Reweighted ℓ¹':        (alpha_rw,  rel_err(alpha_rw,  alpha_true), n_nonzero(alpha_rw)),
    'OMP (贪婪)':           (alpha_omp, rel_err(alpha_omp, alpha_true), n_nonzero(alpha_omp)),
}
print(f"\n[Part B] N={N_B} K={K_B} M={M_B} SNR={SNR_B}dB  真支撑大小={len(support_b)}")
for name, (a, e, nz) in results_b.items():
    print(f"  {name:18s} 相对误差={e:.4f}  恢复非零个数={nz}")

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
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '图5_收敛曲线.png'), dpi=120)
plt.close(fig)

print("已保存：图4_1D恢复对比.png / 图5_收敛曲线.png")

# ─── Part B 结果保存为 JSON ───
part_b_results = {
    'N': N_B,
    'K': K_B,
    'M': M_B,
    'snr_db': SNR_B,
    'true_support_size': int(len(support_b)),
    'algorithms': {
        'IHT (硬阈值→ℓ⁰)': {
            'relative_error': float(rel_err(alpha_iht, alpha_true)),
            'recovered_nonzeros': n_nonzero(alpha_iht),
        },
        'Reweighted ℓ¹': {
            'relative_error': float(rel_err(alpha_rw, alpha_true)),
            'recovered_nonzeros': n_nonzero(alpha_rw),
        },
        'OMP (贪婪)': {
            'relative_error': float(rel_err(alpha_omp, alpha_true)),
            'recovered_nonzeros': n_nonzero(alpha_omp),
        },
    },
}

# ─── 合并 A / B 两 Part 结果，统一写入 results.json ───
all_results = {
    'experiment': '2.2-2',
    'title': 'Laplace先验的稀疏魔力：从 ℓ¹ 到 ℓ⁰',
    'part_A_L1_soft_threshold': part_a_results,
    'part_B_l0_approximation': part_b_results,
}
with open(os.path.join(SAVE_DIR, 'results.json'), 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print("已保存：results.json（Part A + Part B 数值结果）")
