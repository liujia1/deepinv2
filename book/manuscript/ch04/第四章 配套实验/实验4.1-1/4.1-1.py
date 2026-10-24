"""
实验4.1-1 Monte Carlo积分：从维数诅咒到后验采样困难
对应章节：4.1 Monte Carlo方法——从积分到采样
知识点：高维积分的困难（维数诅咒）；Monte Carlo估计与收敛速率 O(1/√M)；
        收敛速率与维数无关；归一化常数不可解导致后验采样困难

实验步骤：
  步骤1：确定性积分 vs Monte Carlo —— 维数诅咒演示
  步骤2：Monte Carlo收敛速率 —— 验证 O(1/√M) 与维数无关
  步骤3：后验采样困难 —— 归一化常数不可解演示

复核修订（本版）：
  R1 全流程使用显式种子的独立 RNG（rng1/rng2/rng3），逐步可逐位复现；
     旧版步骤1、2 使用未播种的全局 np.random，表中数值无法复现
  R2 步骤1 的 MC 误差改为 R1=50 次重复的均值（旧版单次抽样，d=5 时恰好落在
     抽样分布 2% 分位的幸运值上，使"MC 误差随维数下降"的图形与理论矛盾）
  R3 步骤1 增加"等评估预算"面板：固定函数评估次数时梯形误差按 B^{-2/d} 爆炸，
     这才是维数诅咒在"精度"上的体现（旧版只对比了时间）
  R4 步骤2 增加多种子斜率统计，量化 -0.5 附近的种子噪声（与维数无关）
  R5 步骤3 修正 log_likelihood 漏掉的归一化常数 -(d/2)log(2πσ²)：
     旧版 MC 估计的目标是 log p(y)-69.18，与"解析解"直接相减会把记账误差
     混进方差误差
  R6 步骤3 增加维数扫描与噪声扫描，把"高维导致失败"从断言变成实测，并用
     D₂(后验‖先验)（可靠估计需 M ~ exp(D₂)）与"后验附近提议"的对照组划清固有属性的边界
  R7 步骤3 改为"固定数据实例 + 多种子误差棒"：旧版 headline、维数扫描、噪声扫描
     各自随机生成问题实例，同一配置（d=50, σ=0.1）竟出现三个不同的解析值与 D₂
     （-46.63/97.05、-45.46/98.05、97.68），读者无从对照；且两个扫描面板
     每点只有单个实例、无误差棒，实例噪声被误读成"维数效应"
     （旧版 d=2 偏差 0.084 > d=5 偏差 0.062，与"随维数急剧恶化"的叙事相悖）。
     现改为：基准实例（seed=456，d=50, σ=0.1）贯穿 headline、维数扫描 d=50、
     噪声扫描 σ=0.1 与对照组；噪声扫描整族共享同一 x_true/A/ε 只改 σ；
     维数扫描每点取 3 个固定种子的独立实例，报告均值 ± 实例间标准差，
     噪声扫描实例固定，误差棒取 MC 试验间标准差
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time
import warnings
import logging

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

# ====== 中文字体配置（兼容本地和Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验4.1-1')
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

# 每个步骤使用独立的、显式种子的 RNG：步骤之间互不消耗随机流，可单独复现
# SEED3 对应步骤3 的基准问题实例（rng3 先抽 x_true/A/eps，再抽 MC 样本）
SEED1, SEED2, SEED3 = 11, 22, 456
rng1 = np.random.default_rng(SEED1)
rng2 = np.random.default_rng(SEED2)
rng3 = np.random.default_rng(SEED3)


def build_problem(d, sigma, seed):
    """构造一个由种子完全决定的问题实例（"固定数据实例"）

    x_true ~ N(0, I)，A ~ N(0, I/d)（每行除以 sqrt(d)），y = A x_true + sigma * eps
    其中 eps ~ N(0, I)。

    关键性质：同一个 seed 永远给出同一个 (x_true, A, eps)。因此"只改 sigma"的
    一族扫描共享同一个 x_true 与 A，只改变噪声水平——这正是固定数据实例的含义；
    而"改维数"的扫描必须换 A，其随机性只能靠多种子实例平均来控制（见 R7）。
    """
    rng = np.random.default_rng(seed)
    x_true = rng.standard_normal(d)
    A = rng.standard_normal((d, d)) / np.sqrt(d)
    eps = rng.standard_normal(d)
    y = A @ x_true + sigma * eps
    return x_true, A, y, eps

# ══════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════

def trapezoidal_integral_1d(func, a, b, n_points):
    """
    一维梯形法则数值积分

    参数:
        func: 被积函数
        a, b: 积分区间
        n_points: 离散点数

    返回:
        积分近似值
    """
    x = np.linspace(a, b, n_points)
    y = func(x)
    h = (b - a) / (n_points - 1)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])


def trapezoidal_integral_nd(func, bounds, n_points_per_dim):
    """
    多维梯形法则数值积分（仅用于低维演示）

    参数:
        func: 被积函数，输入为 d 维向量
        bounds: [(a1, b1), ..., (ad, bd)] 积分边界
        n_points_per_dim: 每维离散点数

    每维的权重向量为 [0.5, 1, 1, ..., 1, 0.5]，总权重为各维权重向量的张量积

    注意: 函数值用 Python 循环逐点求值，因此实测耗时同时反映
          "格点数 n^d 的爆炸"与逐点调用的常数开销；干净的复杂度度量
          是右子图里的函数评估次数 n^d。
    """
    d = len(bounds)
    grids = [np.linspace(b[0], b[1], n_points_per_dim) for b in bounds]
    mesh = np.meshgrid(*grids, indexing='ij')
    points = np.stack([m.flatten() for m in mesh], axis=-1)

    values = np.array([func(p) for p in points])
    values = values.reshape([n_points_per_dim] * d)

    # 各维步长（一般情形各维区间可不同，体积取乘积）
    hs = np.array([(b[1] - b[0]) / (n_points_per_dim - 1) for b in bounds])

    w1d = np.ones(n_points_per_dim)
    w1d[0] = w1d[-1] = 0.5
    W = w1d.copy()
    for _ in range(d - 1):
        W = np.outer(W, w1d).reshape([-1])

    # 张量积梯形：∑ W · f · ∏ h_i
    integral = np.dot(values.flatten(), W.flatten()) * np.prod(hs)
    return integral


def monte_carlo_integral(func, d, M, rng, bounds=None, max_elems=2_000_000):
    """
    Monte Carlo 积分（分块采样，避免高维大样本时一次性分配 (M,d) 数组）

    参数:
        func: 被积函数，输入为 (k, d) 数组，返回 (k,) 向量
        d: 维数
        M: 样本数
        rng: numpy Generator（显式传入以保证可复现）
        bounds: [(a1, b1), ..., (ad, bd)] 积分边界，默认 [0,1]^d
        max_elems: 单次分块的最大元素数

    返回:
        estimate: 积分估计值
        std_error: 标准误差估计
    """
    if bounds is None:
        bounds = [(0, 1)] * d
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    chunk = max(1, min(M, max_elems // max(1, d)))
    n, s1, s2 = 0, 0.0, 0.0
    while n < M:
        k = min(chunk, M - n)
        u = rng.random((k, d)) * (hi - lo) + lo
        v = func(u)
        s1 += v.sum()
        s2 += np.dot(v, v)
        n += k

    estimate = s1 / M
    var_sample = max((s2 - M * estimate ** 2) / (M - 1), 0.0)
    std_error = np.sqrt(var_sample / M)

    volume = np.prod(hi - lo)
    estimate *= volume
    std_error *= volume

    return estimate, std_error


# ══════════════════════════════════════════════════════════
# 步骤1：确定性积分 vs Monte Carlo —— 维数诅咒演示
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("实验4.1-1 Monte Carlo积分：从维数诅咒到后验采样困难")
print("=" * 60)

print("\n[步骤1] 确定性积分 vs Monte Carlo —— 维数诅咒演示")
print("-" * 60)

# 被积函数: h(x) = sum(x_i^2)，在 [0,1]^d 上积分
# 解析解: I_d = d/3
# Var[h] = d * Var[x_1^2] = 4d/45  =>  MC 标准误差 = sqrt(4d/45)/sqrt(M) ~ sqrt(d)

def h_squared(samples):
    """被积函数: sum(x_i^2)，输入为 (M, d) 数组"""
    return np.sum(samples ** 2, axis=1)


def h_squared_point(x):
    """被积函数点版本: sum(x_i^2)，输入为 1D 向量（用于 trapezoidal_integral_nd）"""
    return np.sum(x ** 2)


# 测试维数
dimensions = [1, 2, 3, 4, 5]
n_points = 11      # 梯形法则每维离散点数（固定，与旧版一致）
M_mc = 10000       # Monte Carlo 样本数（= 评估预算）
R1 = 50            # MC 重复次数：单次抽样的误差不具备可比性（见修订 R2）

trap_errors = []
trap_eq_errors = []
mc_errors = []
mc_err_sems = []
mc_theory_se = []
trap_times = []
mc_times = []
trap_n_eq = []

for d in dimensions:
    true_value = d / 3

    # --- 梯形法则（固定每维 n=11 点）---
    start = time.time()
    try:
        trap_result = trapezoidal_integral_nd(h_squared_point, [(0, 1)] * d, n_points)
        trap_error = abs(trap_result - true_value)
        trap_time = time.time() - start
    except MemoryError:
        trap_error, trap_time, trap_result = np.nan, np.nan, np.nan
    trap_errors.append(trap_error)
    trap_times.append(trap_time)

    # --- 梯形法则（等评估预算 B = M_mc：每维 n_eq = B^{1/d} 点）---
    n_eq = max(2, int(round(M_mc ** (1.0 / d))))
    trap_eq_result = trapezoidal_integral_nd(h_squared_point, [(0, 1)] * d, n_eq)
    trap_eq_errors.append(abs(trap_eq_result - true_value))
    trap_n_eq.append(n_eq)

    # --- Monte Carlo：R1 次重复取均值 ---
    start = time.time()
    trial_errs = []
    for _ in range(R1):
        mc_result, mc_std = monte_carlo_integral(h_squared, d, M_mc, rng1)
        trial_errs.append(abs(mc_result - true_value))
    mc_time = (time.time() - start) / R1
    trial_errs = np.array(trial_errs)
    mc_errors.append(trial_errs.mean())
    mc_err_sems.append(trial_errs.std(ddof=1) / np.sqrt(R1))
    mc_times.append(mc_time)

    mc_theory_se.append(np.sqrt(4.0 * d / 45.0) / np.sqrt(M_mc))

    print(f"d={d}: 梯形误差(n={n_points})={trap_error:.6f}, "
          f"梯形误差(等预算 n={n_eq})={trap_eq_errors[-1]:.6f}, "
          f"MC误差({R1}次均值)={trial_errs.mean():.6f}±{mc_err_sems[-1]:.6f}, "
          f"MC理论SE={mc_theory_se[-1]:.6f}")
    print(f"       梯形时间={trap_time:.4f}s ({n_points ** d} 次评估), "
          f"MC时间={mc_time:.4f}s ({M_mc} 次评估)")

# 绘图：误差 / 等预算误差 / 复杂度
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

axes[0].semilogy(dimensions, trap_errors, 'ro-', linewidth=2, markersize=7,
                 label=rf'梯形法则 每维 $n={n_points}$ ($n^d$ 点)')
axes[0].errorbar(dimensions, mc_errors, yerr=mc_err_sems, fmt='bs-', capsize=4,
                 linewidth=2, markersize=7, label=rf'MC ${M_mc}$ 样本 ({R1} 次均值)')
axes[0].plot(dimensions, mc_theory_se, 'b:', linewidth=2,
             label=r'MC 理论 SE $\propto\sqrt{d}/\sqrt{M}$')
axes[0].set_xlabel(r'维数 $d$', fontsize=12)
axes[0].set_ylabel(r'绝对误差', fontsize=12)
axes[0].set_title(r'步骤1: 固定网格下的误差对比', fontsize=12)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3, which='both')

axes[1].semilogy(dimensions, trap_eq_errors, 'r^-', linewidth=2, markersize=7,
                 label=r'梯形法则 等预算 $n^d\leq B$')
axes[1].errorbar(dimensions, mc_errors, yerr=mc_err_sems, fmt='bs-', capsize=4,
                 linewidth=2, markersize=7, label=rf'MC $M=B={M_mc}$')
for dd, ne in zip(dimensions, trap_n_eq):
    axes[1].annotate(rf'$n={ne}$', (dd, trap_eq_errors[dd - 1]),
                     textcoords='offset points', xytext=(0, 8), fontsize=8, color='r')
axes[1].set_xlabel(r'维数 $d$', fontsize=12)
axes[1].set_ylabel(r'绝对误差', fontsize=12)
axes[1].set_ylim(min(trap_eq_errors) * 0.2, max(trap_eq_errors) * 5)
axes[1].set_title(rf'步骤1: 固定评估预算 $B={M_mc}$ 下的误差（真正的诅咒）', fontsize=12)
axes[1].legend(fontsize=9, loc='upper left')
axes[1].grid(True, alpha=0.3, which='both')

n_theory, M_theory = n_points, M_mc
d_range = np.arange(1, 8)
axes[2].semilogy(d_range, n_theory ** d_range, 'r--', linewidth=2,
                 label=rf'梯形法则: $O(n^d)$ ($n={n_theory}$)')
axes[2].semilogy(d_range, np.full_like(d_range, M_theory, dtype=float), 'b--', linewidth=2,
                 label=rf'Monte Carlo: $O(M)$ ($M={M_theory}$)')
axes[2].set_xlabel(r'维数 $d$', fontsize=12)
axes[2].set_ylabel(r'函数评估次数', fontsize=12)
axes[2].set_title(r'步骤1: 计算复杂度对比', fontsize=12)
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step1_curse_of_dimensionality.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  图已保存: step1_curse_of_dimensionality.png")

# ══════════════════════════════════════════════════════════
# 步骤2：Monte Carlo收敛速率 —— 验证 O(1/√M) 与维数无关
# ══════════════════════════════════════════════════════════
print("\n[步骤2] Monte Carlo收敛速率 —— 验证 O(1/√M) 与维数无关")
print("-" * 60)

M_values = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
dimensions_test = [1, 10, 100, 1000]
n_repeats = 20   # 每个 M 的重复次数（用于估计误差的均值）
n_seeds = 5      # 斜率的种子重复次数（量化种子噪声，见修订 R4）

results = {d: {'errors': [], 'std_errors': []} for d in dimensions_test}
slopes = []
slope_spreads = {}


def slope_of(d, rng):
    """给定维数与 RNG，返回 log(误差) vs log(M) 的最小二乘斜率及误差曲线"""
    true_value = d / 3
    errs = []
    for M in M_values:
        trial = [abs(monte_carlo_integral(h_squared, d, M, rng)[0] - true_value)
                 for _ in range(n_repeats)]
        errs.append(np.mean(trial))
    sl = np.polyfit(np.log(M_values), np.log(errs), 1)[0]
    return sl, errs


for d in dimensions_test:
    sl, errs = slope_of(d, rng2)          # 主曲线（绘图用，固定种子 SEED2）
    results[d]['errors'] = errs
    slopes.append(sl)

    sl_all = [slope_of(d, np.random.default_rng(2000 + s))[0] for s in range(n_seeds)]
    slope_spreads[d] = (float(np.mean(sl_all)), float(np.std(sl_all, ddof=1)))

    print(f"d={d}: M={M_values[-1]}, 平均误差={errs[-1]:.6f}, 斜率={sl:+.4f} "
          f"(另 {n_seeds} 个种子: {np.mean(sl_all):+.4f}±{np.std(sl_all, ddof=1):.4f})")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = ['r', 'g', 'b', 'm']
for idx, d in enumerate(dimensions_test):
    axes[0].loglog(M_values, results[d]['errors'], f'{colors[idx]}o-',
                   linewidth=2, markersize=6, label=rf'$d={d}$')

# 理论参考线: O(1/√M)，取 d=1 的方差 Var(h)=4/45
#   Var(x^2)=E[x^4]-E[x^2]^2=1/5-1/9=4/45, Var(h)=d*4/45
var_h_1d = 4.0 / 45.0
ref_error = np.sqrt(var_h_1d / np.array(M_values))
axes[0].loglog(M_values, ref_error, 'k--', linewidth=1.5, label=r'$O(1/\sqrt{M})$ 参考 ($d=1$)')

axes[0].set_xlabel(r'样本数 $M$', fontsize=12)
axes[0].set_ylabel(r'绝对误差', fontsize=12)
axes[0].set_title(r'步骤2: MC收敛速率 (斜率与维数无关, 竖直间距 $\propto\sqrt{d}$)', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, which='both')

err_bars = [slope_spreads[d][1] for d in dimensions_test]
axes[1].bar(range(len(dimensions_test)), slopes, yerr=err_bars, capsize=5,
            color=colors, alpha=0.7)
axes[1].axhline(y=-0.5, color='k', linestyle='--', linewidth=2, label=r'理论值: $-0.5$')
axes[1].set_xticks(range(len(dimensions_test)))
axes[1].set_xticklabels([rf'$d={d}$' for d in dimensions_test])
axes[1].set_xlabel(r'维数 $d$', fontsize=12)
axes[1].set_ylabel(r'$\log$(误差) vs $\log(M)$ 斜率', fontsize=12)
axes[1].set_title(rf'步骤2: 验证 $O(1/\sqrt{{M}})$ 收敛 (误差棒={n_seeds} 个种子的标准差)', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].set_ylim(-0.7, -0.3)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step2_convergence_rate.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  图已保存: step2_convergence_rate.png")

# ══════════════════════════════════════════════════════════
# 步骤3：后验采样困难 —— 归一化常数不可解演示
# ══════════════════════════════════════════════════════════
print("\n[步骤3] 后验采样困难 —— 归一化常数不可解演示")
print("-" * 60)

# 考虑一个简单的贝叶斯逆问题：
# 似然: y|x ~ N(Ax, sigma^2 I)
# 先验: x ~ N(0, sigma_x^2 I)
# 后验: p(x|y) = p(y|x)p(x) / p(y)
#
# 归一化常数 p(y) = integral p(y|x)p(x) dx
# 线性高斯情形 p(y) 有解析解，这里用它作为基准，检验"从先验直接 MC 估计"这条
# 通用路径的可行性；一般（非线性/非高斯）模型连这个基准都不存在。

print("考虑贝叶斯线性逆问题:")
print(r"  似然: p(y|x) = N(Ax, \sigma^2 I)")
print(r"  先验: p(x) = N(0, \sigma_x^2 I)")
print(r"  后验: p(x|y) = p(y|x)p(x) / p(y)")
print("")

d_demo = 50
sigma = 0.1
sigma_x = 1.0

# 基准问题实例（种子固定）：步骤3 的 headline、维数扫描的 d=demo 点、
# 噪声扫描的 σ=0.1 点与"后验附近提议"对照组全部使用这同一个实例，
# 保证全文数字自洽（见修订 R7）。
# 抽样顺序 x_true → A → eps 与 build_problem(d, σ, SEED3) 完全一致，
# 因此本实例与 build_problem(d_demo, sigma, SEED3) 等价，而 rng3 仍处在
# 抽完 eps 之后的状态（headline 的 MC 样本继续从该流抽取）。
x_true = rng3.standard_normal(d_demo)
A = rng3.standard_normal((d_demo, d_demo)) / np.sqrt(d_demo)
eps_demo = rng3.standard_normal(d_demo)
y = A @ x_true + sigma * eps_demo


def log_likelihood(x, y, A, sigma):
    """对数似然 log p(y|x)（含归一化常数，修订 R5）"""
    residual = y - A @ x
    return -0.5 * np.sum(residual ** 2) / sigma ** 2 - 0.5 * x.size * np.log(2 * np.pi * sigma ** 2)


def log_prior(x, sigma_x):
    """对数先验 log p(x)（含归一化常数）"""
    return -0.5 * np.sum(x ** 2) / sigma_x ** 2 - 0.5 * x.size * np.log(2 * np.pi * sigma_x ** 2)


def log_joint(x, y, A, sigma, sigma_x):
    """联合对数密度 log p(y,x) = log p(y|x) + log p(x)"""
    return log_likelihood(x, y, A, sigma) + log_prior(x, sigma_x)


def estimate_log_evidence_prior(d, sig, M, n_trials, rng, sig_x=1.0, problem=None):
    """从先验 N(0, sig_x^2 I) 采样，用 log-sum-exp 估计 log p(y)（含完整常数）"""
    if problem is None:
        xt = rng.standard_normal(d)
        Am = rng.standard_normal((d, d)) / np.sqrt(d)
        yy = Am @ xt + sig * rng.standard_normal(d)
    else:
        Am, yy = problem
    ests = []
    for _ in range(n_trials):
        Xs = sig_x * rng.standard_normal((M, d))
        # 向量化写法，与 log_likelihood(x, y, A, sigma) 逐字对应（含归一化常数）
        ll = (-0.5 * np.sum((yy[None, :] - Xs @ Am.T) ** 2, axis=1) / sig ** 2
              - 0.5 * d * np.log(2 * np.pi * sig ** 2))
        mx = ll.max()
        ests.append(np.log(np.mean(np.exp(ll - mx))) + mx)
    return np.array(ests), Am, yy


def analytic_log_py(Am, yy, sig, sig_x):
    """线性高斯模型的解析 log p(y) = log N(y; 0, sig_x^2 A A^T + sig^2 I)"""
    dd = yy.size
    cov = sig_x ** 2 * (Am @ Am.T) + sig ** 2 * np.eye(dd)
    _, logdet = np.linalg.slogdet(cov)
    return -0.5 * (dd * np.log(2 * np.pi) + logdet + yy @ np.linalg.solve(cov, yy))


def renyi2_post_prior(Am, yy, sig, sig_x):
    """D₂(后验 ‖ 先验) = log E_先验[w²]，w = p(y|x)/p(y) 为归一化重要性权重

    从先验做重要性采样估计 p(y) 时，相对方差 = exp(D₂) - 1，
    因此可靠估计需要 M ≳ exp(D₂)；线性高斯下 D₂ 有闭式（已由 1D 数值积分逐位核验）。
    注意 D₂ ≠ KL：D₂ 是 Rényi-2 散度，高斯闭式中先验精度项取 1/(2σx²)。
    """
    dd = Am.shape[0]
    post_prec = np.eye(dd) / sig_x ** 2 + Am.T @ Am / sig ** 2
    post_cov = np.linalg.inv(post_prec)
    m = post_cov @ (Am.T @ yy / sig ** 2)
    P = post_prec - np.eye(dd) / (2 * sig_x ** 2)      # q² 的精度 - 1/p 的精度 ≻ 0
    _, logdet_S = np.linalg.slogdet(post_cov)
    _, logdet_P = np.linalg.slogdet(P)
    quad_term = (m @ post_prec @ np.linalg.inv(P) @ post_prec @ m - m @ post_prec @ m)
    return (-dd * np.log(2 * np.pi) - logdet_S + 0.5 * dd * np.log(2 * np.pi * sig_x ** 2)
            + 0.5 * dd * np.log(np.pi) - 0.5 * logdet_P + quad_term)


# --- 方法1：直接 Monte Carlo（从先验采样）---
M_test = 100000
X_samples = sigma_x * rng3.standard_normal((M_test, d_demo))
ll_all = (-0.5 * np.sum((y[np.newaxis, :] - X_samples @ A.T) ** 2, axis=1) / sigma ** 2
          - 0.5 * d_demo * np.log(2 * np.pi * sigma ** 2))
mx = ll_all.max()
log_py_estimate = np.log(np.mean(np.exp(ll_all - mx))) + mx

# 标量实现与向量化实现的一致性自检（确保 log_likelihood 未写错）
chk = np.array([log_likelihood(X_samples[i], y, A, sigma) for i in range(3)])
assert np.allclose(chk, ll_all[:3]), "标量/向量化对数似然不一致"

log_py_exact = analytic_log_py(A, y, sigma, sigma_x)
d2_demo = renyi2_post_prior(A, y, sigma, sigma_x)
w_prior = np.exp(ll_all - mx)
ess_prior = w_prior.sum() ** 2 / np.sum(w_prior ** 2)   # 经验有效样本数

print(f"  直接 Monte Carlo 估计 log p(y)（从先验采样）:")
print(f"    样本数 M = {M_test}")
print(f"    估计值 = {log_py_estimate:.4f}")
print(f"    解析解 log p(y) = {log_py_exact:.4f}")
print(f"    偏差 = {abs(log_py_estimate - log_py_exact):.4f}")
print(f"    问题: 方差极大，估计不可靠")
print("")
print(f"  失效机理: D2(后验 || 先验) = {d2_demo:.2f}")
print(f"    从先验做重要性采样的相对方差 = exp(D2)-1 = {np.exp(min(d2_demo, 700)):.2e}")
print(f"    => 要让相对误差 < 10% 需要 M ~ 100 * exp(D2) = {100 * np.exp(min(d2_demo, 700)):.2e} 个先验样本")
print(f"    实测归一化权重的有效样本数 ESS = {ess_prior:.2e}（/ M = {M_test}）")
print(f"    注: M << exp(D2) 时经验方差本身也会低估真实方差（样本看不到重尾），这正是失效的一部分")
print("")

# 演示：即使知道未归一化的后验密度，也无法直接采样
lj_true = log_joint(x_true, y, A, sigma, sigma_x)
lj_rand = log_joint(X_samples[0], y, A, sigma, sigma_x)
print(f"  未归一化后验可逐点计算: log p(y,x_true)={lj_true:.2f}, "
      f"log p(y,某先验样本)={lj_rand:.2f}")
print("  → 两点之差与 p(y) 无关，因此“比较密度”容易，“从密度生成样本”难")
print("关键洞见:")
print("  1. 后验密度 p(x|y) ∝ exp(log_joint(x)) 可计算（到常数）")
print("  2. 但归一化常数 p(y) 难以计算（高维积分）")
print("  3. 没有归一化常数，无法使用逆CDF采样")
print("  → 这正是 MCMC 方法的出发点！")

# --- 面板(a)：不同样本数 M 下的估计（20 次试验）---
M_range = [1000, 5000, 10000, 50000, 100000]
n_trials_a = 20
estimates, stds, trial_lists = [], [], []
for M in M_range:
    ests, _, _ = estimate_log_evidence_prior(d_demo, sigma, M, n_trials_a,
                                             np.random.default_rng(3000 + M),
                                             problem=(A, y))
    estimates.append(ests.mean())
    stds.append(ests.std(ddof=1))
    trial_lists.append(ests.tolist())
    print(f"  M={M}: 估计值={ests.mean():.2f}, 试验标准差={ests.std(ddof=1):.2f}, "
          f"偏差={abs(ests.mean() - log_py_exact):.2f}")

# --- 面板(b)：固定 σ=0.1，扫描维数 d ---
# 维数一变 A 必须重抽，"同一实例"无从谈起，故每个 d 取 n_inst_b 个固定种子的
# 独立实例取均值，并用实例间标准差作误差棒（修订 R7）；
# d=demo 时第一个实例就是基准实例（SEED3），与 headline/对照组/噪声扫描 σ=0.1 同源。
d_scan = [1, 2, 5, 10, 20, 50, 100]
M_scan, n_trials_b = 20000, 5
n_inst_b = 3
d_errors, d_err_stds = [], []
d_kl, d_kl_stds = [], []
d_exact, d_est = [], []
d_inst_seeds = {}
for dd in d_scan:
    seeds = [SEED3] if dd == d_demo else []
    seeds += [4000 + 100 * dd + k for k in range(n_inst_b - len(seeds))]
    d_inst_seeds[dd] = seeds
    inst_bias, inst_d2, inst_ex, inst_est = [], [], [], []
    for k, sd in enumerate(seeds):
        _, Am, yy, _ = build_problem(dd, sigma, sd)
        ex = analytic_log_py(Am, yy, sigma, sigma_x)
        ests, _, _ = estimate_log_evidence_prior(dd, sigma, M_scan, n_trials_b,
                                                 np.random.default_rng(6000 + 100 * dd + k),
                                                 problem=(Am, yy))
        inst_ex.append(ex)
        inst_est.append(ests.mean())
        inst_bias.append(abs(ests.mean() - ex))
        inst_d2.append(renyi2_post_prior(Am, yy, sigma, sigma_x))
    d_errors.append(float(np.mean(inst_bias)))
    d_err_stds.append(float(np.std(inst_bias, ddof=1)))
    d_kl.append(float(np.mean(inst_d2)))
    d_kl_stds.append(float(np.std(inst_d2, ddof=1)))
    d_exact.append(float(np.mean(inst_ex)))
    d_est.append(float(np.mean(inst_est)))
    print(f"  d={dd:4d}: 解析={np.mean(inst_ex):9.2f}, 估计={np.mean(inst_est):9.2f}, "
          f"偏差={np.mean(inst_bias):9.2f}±{np.std(inst_bias, ddof=1):.2f}, "
          f"D2={np.mean(inst_d2):7.2f}±{np.std(inst_d2, ddof=1):.2f} "
          f"({n_inst_b} 个实例, seeds={seeds})")

# --- 面板(c)：固定 d=demo，扫描噪声 σ（每维信息量的作用）---
# 固定数据实例：整族共享基准实例的同一 x_true 与 A，只改噪声水平 σ，
# 因此曲线反映的是"同一问题、逐级加大噪声"的效应，不混入实例噪声；
# 误差棒取 MC 试验间标准差（该面板每点 5 次独立 MC 重复）。
sigma_scan = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
M_sig, n_trials_c = 50000, 5
s_errors, s_err_stds, s_kl, s_exact, s_est = [], [], [], [], []
for sg in sigma_scan:
    _, Am, yy, _ = build_problem(d_demo, sg, SEED3)
    ex = analytic_log_py(Am, yy, sg, sigma_x)
    ests, _, _ = estimate_log_evidence_prior(d_demo, sg, M_sig, n_trials_c,
                                             np.random.default_rng(7000 + int(100 * sg)),
                                             problem=(Am, yy))
    s_errors.append(float(abs(ests.mean() - ex)))
    s_err_stds.append(float(ests.std(ddof=1)))
    s_kl.append(float(renyi2_post_prior(Am, yy, sg, sigma_x)))
    s_exact.append(float(ex))
    s_est.append(float(ests.mean()))
    print(f"  σ={sg:4.2f}: 解析={ex:9.2f}, 估计={ests.mean():9.2f}, "
          f"偏差={abs(ests.mean() - ex):9.2f}±{ests.std(ddof=1):.2f}, D2={s_kl[-1]:7.2f}")

# --- 对照组：改用"后验附近"的高斯提议 N(post_mean, alpha*post_cov) 做重要性采样 ---
# 目的：划清固有属性的边界——失效到底是"先验提议 + 高 D₂"造成的，还是 MC 本身造成的
post_cov = np.linalg.inv(np.eye(d_demo) / sigma_x ** 2 + A.T @ A / sigma ** 2)
post_mean = post_cov @ (A.T @ y / sigma ** 2)
post_prec = np.linalg.inv(post_cov)


def log_py_with_proposal(alpha, M, rng):
    """用提议 q = N(post_mean, alpha*post_cov) 做重要性采样估计 log p(y)"""
    cov_q = alpha * post_cov
    Lq = np.linalg.cholesky(cov_q)
    Xq = post_mean[None, :] + rng.standard_normal((M, d_demo)) @ Lq.T
    _, logdet_q = np.linalg.slogdet(cov_q)
    ll = (-0.5 * np.sum((y[np.newaxis, :] - Xq @ A.T) ** 2, axis=1) / sigma ** 2
          - 0.5 * d_demo * np.log(2 * np.pi * sigma ** 2))
    lp = (-0.5 * np.sum(Xq ** 2, axis=1) / sigma_x ** 2
          - 0.5 * d_demo * np.log(2 * np.pi * sigma_x ** 2))
    dq = Xq - post_mean
    lq = (-0.5 * d_demo * np.log(2 * np.pi) - 0.5 * logdet_q
          - 0.5 * np.einsum('ij,jk,ik->i', dq, post_prec / alpha, dq))
    lw = ll + lp - lq
    mxw = lw.max()
    est = mxw + np.log(np.mean(np.exp(lw - mxw)))
    # 有效样本数 ESS（权重归一化后）
    w = np.exp(lw - mxw)
    ess = w.sum() ** 2 / np.sum(w ** 2)
    return est, ess


ctl_results = []
for alpha in [1.0, 2.0, 5.0]:
    est_ctl, ess_ctl = log_py_with_proposal(alpha, M_test, np.random.default_rng(777))
    ctl_results.append((alpha, float(est_ctl), float(abs(est_ctl - log_py_exact)), float(ess_ctl)))
    print(f"  对照: 提议 q=N(后验均值, {alpha:.0f}×后验协方差), M={M_test}: "
          f"log p(y) 估计={est_ctl:9.4f}, 偏差={abs(est_ctl - log_py_exact):.4f}, ESS={ess_ctl:.3e}")
print("  注: alpha=1 时权重恒等于 p(y)（线性高斯下提议恰为后验），方差恒为 0，")
print("      这是同义反复的基准而非可用方法；alpha=2,5 才是‘提议大致对但欠精确’的现实情形")
print("      而该对照仍需已知后验协方差——一般模型恰恰拿不到，这正是 MCMC/变分要解决的问题")

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

axes[0].errorbar(M_range, estimates, yerr=stds, fmt='bo-', capsize=5, capthick=2,
                 linewidth=2, markersize=7, label=f'{n_trials_a} 次试验的均值±标准差')
axes[0].axhline(y=log_py_exact, color='r', linestyle='--', linewidth=2,
                label=rf'解析解: $\log p(y)={log_py_exact:.2f}$')
axes[0].set_xscale('log')
axes[0].set_xlabel(r'样本数 $M$', fontsize=12)
axes[0].set_ylabel(r'$\log p(y)$ 估计值', fontsize=12)
axes[0].set_title(rf'步骤3(a): 从先验采样估计证据 ($d={d_demo},\,\sigma={sigma}$)', fontsize=12)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axb = axes[1]
# 误差棒 = 3 个固定种子实例之间的标准差；对数纵轴下把下界压到均值的 90%，
# 避免标准差大于均值时下界为负而无法显示
d_err_lo = np.minimum(d_err_stds, 0.9 * np.array(d_errors))
axb.errorbar(d_scan, d_errors, yerr=[d_err_lo, d_err_stds], fmt='bs-',
             capsize=4, capthick=2, linewidth=2, markersize=7,
             label=f'实测偏差 ($M={M_scan}$, {n_inst_b} 个实例均值)')
axb.set_xlabel(r'维数 $d$', fontsize=12)
axb.set_ylabel(r'偏差（对数轴）', fontsize=12)
axb.set_title(rf'步骤3(b): 固定 $M$ 下偏差随维数急剧恶化 ($\sigma={sigma}$)', fontsize=12)
axb.grid(True, alpha=0.3, which='both')
axb2 = axb.twinx()
axb2.errorbar(d_scan, d_kl, yerr=d_kl_stds, fmt='r^--', linewidth=2, markersize=7,
              capsize=3, label='D2(后验 || 先验) 右轴')
axb2.set_ylabel(r'D2(后验 || 先验)', fontsize=12)
lines1, labels1 = axb.get_legend_handles_labels()
lines2, labels2 = axb2.get_legend_handles_labels()
axb.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')

s_err_lo = np.minimum(s_err_stds, 0.9 * np.array(s_errors))
axes[2].errorbar(sigma_scan, s_errors, yerr=[s_err_lo, s_err_stds], fmt='mo-',
                 capsize=4, capthick=2, linewidth=2, markersize=7,
                 label=f'实测偏差 ($d={d_demo}$, 固定实例)')
axes[2].set_xscale('log')
axes[2].set_xlabel(r'噪声水平 $\sigma$', fontsize=12)
axes[2].set_ylabel(r'偏差（对数轴）', fontsize=12)
axes[2].set_title(rf'步骤3(c): 偏差随每维信息量变化 ($d={d_demo}$, $M={M_sig}$)', fontsize=12)
axes[2].legend(fontsize=9, loc='lower left')
axes[2].grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step3_sampling_difficulty.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  图已保存: step3_sampling_difficulty.png")

# ══════════════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print("1. 维数诅咒: 确定性积分复杂度 O(n^d)，Monte Carlo 复杂度 O(M)")
print("   固定评估预算时梯形误差按 B^{-2/d} 爆炸，MC 误差只按 sqrt(d/M) 增长")
print("2. Monte Carlo 收敛速率 O(1/√M) 与维数无关（斜率种子噪声亦与维数无关），")
print("   但方差常数 Var[h]=4d/45 随维数线性增长 -> 误差绝对值按 sqrt(d) 放大")
print("3. 后验采样困难: 从先验直接 MC 估计 p(y) 的相对方差 = exp(D2(后验||先验))-1，")
print("   D2 随维数线性增长 -> 所需样本数指数增长，这是该策略的固有失效")
print("→ MCMC 方法的出发点: 只需后验密度到归一化常数即可采样")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")

# ══════════════════════════════════════════════════════════
# 保存数值结果到JSON文件
# ══════════════════════════════════════════════════════════
import json


def _to_native(obj):
    """递归将numpy/torch类型转换为Python原生类型，便于JSON序列化"""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, 'item') and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except (ValueError, RuntimeError, TypeError):
            return obj
    return obj


def _sig(x, n=6):
    """按有效数字舍入（小量级的误差用 round(x, k) 会被舍成 0）"""
    return float(f'{float(x):.{n}g}')


results_summary = {
    'experiment': '4.1-1',
    'title': 'Monte Carlo积分：从维数诅咒到后验采样困难',
    'seeds': {'step1': SEED1, 'step2': SEED2, 'step3': SEED3},
    'step1_curse_of_dimensionality': {
        'dimensions': dimensions,
        'trap_errors': [_sig(e) for e in trap_errors],
        'trap_errors_equal_budget': [_sig(e) for e in trap_eq_errors],
        'trap_points_equal_budget': trap_n_eq,
        'mc_errors_mean': [_sig(e) for e in mc_errors],
        'mc_error_sems': [_sig(e) for e in mc_err_sems],
        'mc_theory_se': [_sig(e) for e in mc_theory_se],
        'trap_times': [float(round(t, 4)) for t in trap_times],
        'mc_times_per_repeat': [float(round(t, 4)) for t in mc_times],
        'n_points_per_dim': n_points,
        'mc_samples': M_mc,
        'mc_repeats': R1,
    },
    'step2_convergence_rate': {
        'dimensions_test': dimensions_test,
        'M_values': M_values,
        'n_repeats': n_repeats,
        'slopes': [float(round(s, 4)) for s in slopes],
        'slope_mean_over_seeds': {str(d): float(round(v[0], 4)) for d, v in slope_spreads.items()},
        'slope_std_over_seeds': {str(d): float(round(v[1], 4)) for d, v in slope_spreads.items()},
        'n_seeds': n_seeds,
    },
    'step3_sampling_difficulty': {
        'dimension': d_demo,
        'sigma': sigma,
        'sigma_x': sigma_x,
        'M_test': M_test,
        'instance_policy': ('基准实例 seed=456（d=50, σ=0.1）贯穿 headline、维数扫描 d=50、'
                            '噪声扫描 σ=0.1 与对照组；噪声扫描整族共享同一 x_true/A/ε 只改 σ；'
                            '维数扫描每 d 取 n_inst_b 个固定种子实例取均值，误差棒为实例间标准差；'
                            '噪声扫描实例固定，误差棒为 MC 试验间标准差'),
        'base_instance_seed': SEED3,
        'log_py_estimate': float(round(log_py_estimate, 4)),
        'log_py_exact': float(round(log_py_exact, 4)),
        'mc_error': float(round(abs(log_py_estimate - log_py_exact), 4)),
        'renyi2_posterior_prior': float(round(d2_demo, 4)),
        'ess_prior_proposal': float(ess_prior),
        'M_range': M_range,
        'estimates': [float(round(e, 4)) for e in estimates],
        'stds': [float(round(s, 4)) for s in stds],
        'trials_per_M': n_trials_a,
        'd_scan': d_scan,
        'd_scan_instances': n_inst_b,
        'd_scan_instance_seeds': {str(d): list(s) for d, s in d_inst_seeds.items()},
        'd_scan_errors': [_sig(e) for e in d_errors],
        'd_scan_errors_std': [_sig(e) for e in d_err_stds],
        'd_scan_d2': [float(round(k, 4)) for k in d_kl],
        'd_scan_d2_std': [float(round(k, 4)) for k in d_kl_stds],
        'd_scan_exact_log_py': [float(round(e, 4)) for e in d_exact],
        'd_scan_estimates': [float(round(e, 4)) for e in d_est],
        'sigma_scan': sigma_scan,
        'sigma_scan_shared_instance_seed': SEED3,
        'sigma_scan_errors': [_sig(e) for e in s_errors],
        'sigma_scan_errors_std': [_sig(e) for e in s_err_stds],
        'sigma_scan_d2': [float(round(k, 4)) for k in s_kl],
        'sigma_scan_exact_log_py': [float(round(e, 4)) for e in s_exact],
        'sigma_scan_estimates': [float(round(e, 4)) for e in s_est],
        'sigma_scan_trials_per_point': n_trials_c,
        'control_posterior_proposal': [
            {'alpha': a, 'log_py': round(e, 4), 'abs_error': round(err, 6), 'ESS': ess}
            for a, e, err, ess in ctl_results],
    },
}

with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(_to_native(results_summary), f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
