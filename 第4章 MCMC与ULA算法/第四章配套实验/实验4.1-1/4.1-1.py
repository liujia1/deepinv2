"""
实验4.1-1 Monte Carlo积分：从维数诅咒到后验采样困难
对应章节：4.1 Monte Carlo方法——从积分到采样
知识点：高维积分的困难（维数诅咒）；Monte Carlo估计与收敛速率 O(1/√M)；
        收敛速率与维数无关；归一化常数不可解导致后验采样困难

实验步骤：
  步骤1：确定性积分 vs Monte Carlo —— 维数诅咒演示
  步骤2：Monte Carlo收敛速率 —— 验证 O(1/√M) 与维数无关
  步骤3：后验采样困难 —— 归一化常数不可解演示

注意: 本节使用独立的 RNG 以保证步骤间可重复性
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
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

# 为每个步骤使用独立的 RNG
rng1 = np.random.default_rng(42)
rng2 = np.random.default_rng(123)
rng3 = np.random.default_rng(456)

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

    返回:
        积分近似值
    """
    d = len(bounds)
    grids = [np.linspace(b[0], b[1], n_points_per_dim) for b in bounds]
    mesh = np.meshgrid(*grids, indexing='ij')
    points = np.stack([m.flatten() for m in mesh], axis=-1)

    values = np.array([func(p) for p in points])
    values = values.reshape([n_points_per_dim] * d)

    h = (bounds[0][1] - bounds[0][0]) / (n_points_per_dim - 1)

    # 多维梯形法则：边缘点权重为 1，内部点权重为 2^d
    weights = np.ones([n_points_per_dim] * d)
    for axis in range(d):
        edge_mask = np.ones([n_points_per_dim] * d, dtype=bool)
        slices = [slice(None)] * d
        slices[axis] = slice(1, -1)
        edge_mask[tuple(slices)] = False
        weights = np.where(edge_mask, weights, weights * 2)

    integral = (h ** d) * np.sum(values * weights) / (2 ** d)
    return integral


def monte_carlo_integral(func, d, M, bounds=None):
    """
    Monte Carlo 积分

    参数:
        func: 被积函数，输入为 (M, d) 数组
        d: 维数
        M: 样本数
        bounds: [(a1, b1), ..., (ad, bd)] 积分边界，默认 [0,1]^d

    返回:
        estimate: 积分估计值
        std_error: 标准误差估计
    """
    if bounds is None:
        bounds = [(0, 1)] * d

    # 在超立方体中均匀采样
    samples = np.random.rand(M, d)
    for i in range(d):
        samples[:, i] = samples[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

    # 计算函数值
    values = func(samples)

    # Monte Carlo 估计
    estimate = np.mean(values)

    # 标准误差
    std_error = np.std(values, ddof=1) / np.sqrt(M)

    # 体积因子
    volume = np.prod([b[1] - b[0] for b in bounds])
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

def h_squared(samples):
    """被积函数: sum(x_i^2)"""
    return np.sum(samples ** 2, axis=1)

# 测试维数
dimensions = [1, 2, 3, 4, 5]
n_points = 11  # 每维离散点数
M_mc = 10000   # Monte Carlo 样本数

trap_errors = []
mc_errors = []
trap_times = []
mc_times = []

import time

for d in dimensions:
    true_value = d / 3

    # 梯形法则
    start = time.time()
    try:
        trap_result = trapezoidal_integral_nd(lambda x: np.sum(x**2), [(0, 1)] * d, n_points)
        trap_error = abs(trap_result - true_value)
        trap_time = time.time() - start
    except MemoryError:
        trap_error = np.nan
        trap_time = np.nan
        trap_result = np.nan

    trap_errors.append(trap_error)
    trap_times.append(trap_time)

    # Monte Carlo
    start = time.time()
    mc_result, mc_std = monte_carlo_integral(h_squared, d, M_mc)
    mc_error = abs(mc_result - true_value)
    mc_time = time.time() - start

    mc_errors.append(mc_error)
    mc_times.append(mc_time)

    print(f"d={d}: 梯形误差={trap_error:.6f}, MC误差={mc_error:.6f}, "
          f"梯形时间={trap_time:.4f}s, MC时间={mc_time:.4f}s")

# 绘图：误差随维数变化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 误差对比
axes[0].semilogy(dimensions, trap_errors, 'ro-', linewidth=2, markersize=8,
                 label=rf'梯形法则 (每维 $n=11$ 点)')
axes[0].semilogy(dimensions, mc_errors, 'bs-', linewidth=2, markersize=8,
                 label=rf'Monte Carlo ($M={M_mc}$)')
axes[0].set_xlabel(r'维数 $d$', fontsize=12)
axes[0].set_ylabel(r'绝对误差', fontsize=12)
axes[0].set_title(r'步骤1: 维数诅咒', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 计算复杂度对比（理论值）
# 梯形法则: O(n^d) 个函数评估
# Monte Carlo: O(M) 个函数评估
n_theory = 11
M_theory = 10000
d_range = np.arange(1, 8)

trap_complexity = n_theory ** d_range
mc_complexity = np.full_like(d_range, M_theory, dtype=float)

axes[1].semilogy(d_range, trap_complexity, 'r--', linewidth=2, label=rf'梯形法则: $O(n^d)$ ($n={n_theory}$)')
axes[1].semilogy(d_range, mc_complexity, 'b--', linewidth=2, label=rf'Monte Carlo: $O(M)$ ($M={M_theory}$)')
axes[1].set_xlabel(r'维数 $d$', fontsize=12)
axes[1].set_ylabel(r'函数评估次数', fontsize=12)
axes[1].set_title(r'步骤1: 计算复杂度对比', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step1_curse_of_dimensionality.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  图已保存: step1_curse_of_dimensionality.png")

# ══════════════════════════════════════════════════════════
# 步骤2：Monte Carlo收敛速率 —— 验证 O(1/√M) 与维数无关
# ══════════════════════════════════════════════════════════
print("\n[步骤2] Monte Carlo收敛速率 —— 验证 O(1/√M) 与维数无关")
print("-" * 60)

# 不同样本数
M_values = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
dimensions_test = [1, 10, 100, 1000]
n_repeats = 20  # 重复次数以估计误差分布

# 存储结果
results = {d: {'errors': [], 'std_errors': []} for d in dimensions_test}

for d in dimensions_test:
    true_value = d / 3
    errors_for_M = []
    std_errors_for_M = []

    for M in M_values:
        trial_errors = []
        for _ in range(n_repeats):
            mc_result, mc_std = monte_carlo_integral(h_squared, d, M)
            trial_errors.append(abs(mc_result - true_value))
        errors_for_M.append(np.mean(trial_errors))
        std_errors_for_M.append(np.std(trial_errors))

    results[d]['errors'] = errors_for_M
    results[d]['std_errors'] = std_errors_for_M

    print(f"d={d}: M={M_values[-1]}, 平均误差={errors_for_M[-1]:.6f}")

# 绘图：收敛曲线
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 不同维数的收敛曲线
colors = ['r', 'g', 'b', 'm']
for idx, d in enumerate(dimensions_test):
    axes[0].loglog(M_values, results[d]['errors'], f'{colors[idx]}o-',
                   linewidth=2, markersize=6, label=rf'$d={d}$')

# 理论参考线: O(1/√M)
M_ref = np.array(M_values)
ref_error = results[1]['errors'][0] * np.sqrt(M_values[0]) / np.sqrt(M_ref)
axes[0].loglog(M_ref, ref_error, 'k--', linewidth=1.5, label=r'$O(1/\sqrt{M})$')

axes[0].set_xlabel(r'样本数 $M$', fontsize=12)
axes[0].set_ylabel(r'绝对误差', fontsize=12)
axes[0].set_title(r'步骤2: MC收敛速率 (与维数无关)', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, which='both')

# 验证收敛速率：log(error) vs log(M) 的斜率应接近 -0.5
slopes = []
for d in dimensions_test:
    log_M = np.log(M_values)
    log_err = np.log(results[d]['errors'])
    slope = np.polyfit(log_M, log_err, 1)[0]
    slopes.append(slope)
    print(f"  d={d}: 收敛速率斜率 = {slope:.4f} (理论值 = -0.5)")

axes[1].bar(range(len(dimensions_test)), slopes, color=colors, alpha=0.7)
axes[1].axhline(y=-0.5, color='k', linestyle='--', linewidth=2, label=r'理论值: $-0.5$')
axes[1].set_xticks(range(len(dimensions_test)))
axes[1].set_xticklabels([rf'$d={d}$' for d in dimensions_test])
axes[1].set_xlabel(r'维数 $d$', fontsize=12)
axes[1].set_ylabel(r'$\log$(误差) vs $\log(M)$ 斜率', fontsize=12)
axes[1].set_title(r'步骤2: 验证 $O(1/\sqrt{M})$ 收敛', fontsize=12)
axes[1].legend(fontsize=10)
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
# 对于简单情况（A 为单位阵），p(y) 有解析解
# 但对于一般 A，p(y) 是高维积分，无法解析计算

# 演示：计算 p(y) 的困难
print("考虑贝叶斯线性逆问题:")
print(r"  似然: p(y|x) = N(Ax, \sigma^2 I)")
print(r"  先验: p(x) = N(0, \sigma_x^2 I)")
print(r"  后验: p(x|y) = p(y|x)p(x) / p(y)")
print("")

# 设置问题
d_demo = 50  # 维数
sigma = 0.1
sigma_x = 1.0

# 生成真实信号和观测
np.random.seed(123)
x_true = np.random.randn(d_demo)
A = np.random.randn(d_demo, d_demo) / np.sqrt(d_demo)  # 随机矩阵
y = A @ x_true + sigma * np.random.randn(d_demo)

# 尝试用 Monte Carlo 估计 p(y)
# p(y) = integral p(y|x) p(x) dx
#      = integral N(y; Ax, sigma^2 I) N(x; 0, sigma_x^2 I) dx

def log_likelihood(x, y, A, sigma):
    """对数似然 log p(y|x)"""
    residual = y - A @ x
    return -0.5 * np.sum(residual ** 2) / sigma ** 2

def log_prior(x, sigma_x):
    """对数先验 log p(x)"""
    return -0.5 * np.sum(x ** 2) / sigma_x ** 2

def log_joint(x, y, A, sigma, sigma_x):
    """联合对数密度 log p(y,x) = log p(y|x) + log p(x)"""
    return log_likelihood(x, y, A, sigma) + log_prior(x, sigma_x)

# 尝试估计 p(y) 的困难
print("尝试估计归一化常数 p(y):")
print(f"  维数 d = {d_demo}")
print("")

# 方法1：直接 Monte Carlo（效率极低）
M_test = 100000
log_joint_values = np.zeros(M_test)
for i in range(M_test):
    x_sample = sigma_x * np.random.randn(d_demo)
    log_joint_values[i] = log_joint(x_sample, y, A, sigma, sigma_x)

# log p(y) = log integral exp(log_joint(x)) dx
# 使用 log-sum-exp 技巧
log_py_estimate = np.log(np.mean(np.exp(log_joint_values - np.max(log_joint_values)))) + np.max(log_joint_values)

print(f"  直接 Monte Carlo 估计 log p(y):")
print(f"    样本数 M = {M_test}")
print(f"    估计值 = {log_py_estimate:.4f}")
print(f"    问题: 方差极大，估计不可靠")
print("")

# 方法2：对于这个简单问题，p(y) 有解析解
# p(y) = N(y; 0, A A^T sigma_x^2 + sigma^2 I)
Sigma_y = sigma_x ** 2 * (A @ A.T) + sigma ** 2 * np.eye(d_demo)
log_py_exact = -0.5 * (d_demo * np.log(2 * np.pi) + np.log(np.linalg.det(Sigma_y)) + y @ np.linalg.solve(Sigma_y, y))

print(f"  解析解 log p(y) = {log_py_exact:.4f}")
print(f"  Monte Carlo 误差 = {abs(log_py_estimate - log_py_exact):.4f}")
print("")

# 演示：即使知道未归一化的后验密度，也无法直接采样
print("关键洞见:")
print("  1. 后验密度 p(x|y) ∝ exp(log_joint(x)) 可计算（到常数）")
print("  2. 但归一化常数 p(y) 难以计算（高维积分）")
print("  3. 没有归一化常数，无法使用逆CDF采样")
print("  → 这正是 MCMC 方法的出发点！")

# 绘图：log p(y) 估计的困难
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# 不同样本数下的估计方差
M_range = [1000, 5000, 10000, 50000]
estimates = []
stds = []

for M in M_range:
    trial_estimates = []
    for trial in range(5):  # 减少试验次数以加速
        log_vals = np.zeros(M)
        for i in range(M):
            x_sample = sigma_x * np.random.randn(d_demo)
            log_vals[i] = log_joint(x_sample, y, A, sigma, sigma_x)
        log_py = np.log(np.mean(np.exp(log_vals - np.max(log_vals)))) + np.max(log_vals)
        trial_estimates.append(log_py)
    estimates.append(np.mean(trial_estimates))
    stds.append(np.std(trial_estimates))
    print(f"  M={M}: 估计值={estimates[-1]:.2f}, 标准差={stds[-1]:.2f}")

ax.errorbar(M_range, estimates, yerr=stds, fmt='bo-', capsize=5, capthick=2, linewidth=2, markersize=8)
ax.axhline(y=log_py_exact, color='r', linestyle='--', linewidth=2, label=rf'解析解: $\log p(y)={log_py_exact:.2f}$')
ax.set_xlabel(r'样本数 $M$', fontsize=12)
ax.set_ylabel(r'$\log p(y)$ 估计值', fontsize=12)
ax.set_title(r'步骤3: 归一化常数 $p(y)$ 估计的困难', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

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
print("2. Monte Carlo 收敛速率 O(1/√M) 与维数无关")
print("3. 后验采样困难: 归一化常数 p(y) 是高维积分，难以估计")
print("→ MCMC 方法的出发点: 只需后验密度到归一化常数即可采样")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")
