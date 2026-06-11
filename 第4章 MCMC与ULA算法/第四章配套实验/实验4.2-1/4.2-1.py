"""
实验4.2-1 Metropolis-Hastings算法基础演示
对应章节：4.2 Metropolis-Hastings算法
知识点：MH算法基本流程；细致平衡条件数值验证；MH不需要归一化常数

实验步骤：
  步骤1：MH算法基本流程演示
  步骤2：细致平衡条件数值验证（统计经验转移频率）
  步骤3：MH不需要归一化常数的演示
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
    SAVE_DIR = os.path.join(_gdrive, '实验4.2-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)

# ══════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════

def target_density(x):
    """
    未归一化目标密度: p(x) ∝ exp(-x²/2)

    注意：这里故意不使用归一化常数 sqrt(2π)，
    以演示 MH 算法不需要归一化常数。

    参数:
        x: 标量或数组

    返回:
        未归一化密度值
    """
    # 下溢保护：限制指数参数范围
    x_sq = np.clip(x ** 2, None, 1400)
    return np.exp(-0.5 * x_sq)


def log_target_density(x):
    """
    目标密度的对数形式（用于数值稳定的接受概率计算）

    参数:
        x: 标量或数组

    返回:
        log p(x) = -x²/2
    """
    return -0.5 * np.clip(x ** 2, None, 1400)


def mh_sampler(target, log_target=None, n_samples=50000, sigma=1.0, x0=0.0, burn_in=5000):
    """
    Metropolis-Hastings 采样器（纯采样，无验证逻辑耦合）

    参数:
        target: 未归一化目标密度函数
        log_target: 目标密度的对数形式（可选，用于数值稳定计算）
        n_samples: 总采样数（含 burn-in）
        sigma: 随机游走提议的标准差
        x0: 初始状态
        burn_in: 预烧期样本数

    返回:
        samples_full: 完整采样轨迹（含 burn-in）
        samples_post: post-burn-in 样本
        accept_rate_post: post-burn-in 接受率
        accept_rate_full: 完整接受率
    """
    x = x0
    samples_full = []
    n_accept_full = 0
    n_accept_post = 0

    for i in range(n_samples):
        # 提议：随机游走
        proposal = x + sigma * np.random.randn()

        # 接受概率（数值稳定版本：使用 log 域计算）
        if log_target is not None:
            log_alpha = min(0.0, log_target(proposal) - log_target(x))
        else:
            # 回退路径：当 log_target 未提供时使用
            # 注意：此路径在 target(x) 极小时可能产生数值问题，应优先使用 log_target
            log_alpha = min(0.0, np.log(target(proposal)) - np.log(target(x)))
        alpha = np.exp(log_alpha)

        # 接受/拒绝
        if np.random.rand() < alpha:
            x = proposal
            n_accept_full += 1
            if i >= burn_in:
                n_accept_post += 1

        samples_full.append(x)

    samples_full = np.array(samples_full)
    samples_post = samples_full[burn_in:]
    accept_rate_full = n_accept_full / n_samples
    accept_rate_post = n_accept_post / (n_samples - burn_in) if n_samples > burn_in else 0.0

    return samples_full, samples_post, accept_rate_post, accept_rate_full


def verify_detailed_balance_empirical(target, log_target=None, n_samples=100000, sigma=1.0, 
                                       n_bins=20, x_range=(-3, 3), burn_in=5000):
    """
    通过统计经验转移频率验证细致平衡条件

    细致平衡条件：p(x) · P(x→x') = p(x') · P(x'→x)
    
    离散化后验证：统计经验转移频率，验证
    \hat{p}(x_i) · \hat{P}(x_i→x_j) ≈ \hat{p}(x_j) · \hat{P}(x_j→x_i)
    
    其中 \hat{p}(x_i) 来自链的经验访问频率，\hat{P} 来自经验转移频率。

    参数:
        target: 未归一化目标密度函数
        log_target: 目标密度的对数形式（可选，用于数值稳定计算）
        n_samples: 采样数（含 burn-in）
        sigma: 随机游走提议的标准差
        n_bins: 离散化区间数
        x_range: 状态空间范围
        burn_in: 预烧期样本数（丢弃，不参与统计）

    返回:
        results: 包含验证结果的字典
    """
    # 离散化状态空间
    bins = np.linspace(x_range[0], x_range[1], n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 初始化计数
    transition_counts = np.zeros((n_bins, n_bins))
    state_counts = np.zeros(n_bins)  # 各 bin 的访问次数
    
    # 从平稳分布采样并记录转移
    x = 0.0
    
    # Burn-in 阶段：不记录统计
    for _ in range(burn_in):
        proposal = x + sigma * np.random.randn()
        # 使用 log 域计算接受概率（数值稳定）
        if log_target is not None:
            log_alpha = min(0.0, log_target(proposal) - log_target(x))
        else:
            # 回退路径：当 log_target 未提供时使用
            # 注意：此路径在 target(x) 极小时可能产生数值问题，应优先使用 log_target
            log_alpha = min(0.0, np.log(target(proposal)) - np.log(target(x)))
        alpha = np.exp(log_alpha)
        if np.random.rand() < alpha:
            x = proposal
    
    # 统计阶段：记录转移和状态访问
    for _ in range(n_samples - burn_in):
        proposal = x + sigma * np.random.randn()
        
        # 使用 log 域计算接受概率（数值稳定）
        if log_target is not None:
            log_alpha = min(0.0, log_target(proposal) - log_target(x))
        else:
            # 回退路径：当 log_target 未提供时使用
            # 注意：此路径在 target(x) 极小时可能产生数值问题，应优先使用 log_target
            log_alpha = min(0.0, np.log(target(proposal)) - np.log(target(x)))
        alpha = np.exp(log_alpha)
        
        if np.random.rand() < alpha:
            new_x = proposal
        else:
            new_x = x
        
        # 离散化并记录转移（仅统计范围内的转移）
        x_bin = np.digitize(x, bins) - 1
        new_x_bin = np.digitize(new_x, bins) - 1
        
        if 0 <= x_bin < n_bins and 0 <= new_x_bin < n_bins:
            transition_counts[x_bin, new_x_bin] += 1
            state_counts[x_bin] += 1
        
        x = new_x
    
    # 计算经验转移概率
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    empirical_transition = transition_counts / row_sums
    
    # 计算经验平稳分布（来自链的访问频率）
    total_visits = state_counts.sum()
    if total_visits > 0:
        empirical_probs = state_counts / total_visits
    else:
        empirical_probs = np.ones(n_bins) / n_bins
    
    # 验证细致平衡：p(i) * P(i→j) ≈ p(j) * P(j→i)
    balance_errors = []
    balance_pairs = []
    
    for i in range(n_bins):
        for j in range(i + 1, n_bins):  # 只检查上三角，避免重复
            if empirical_transition[i, j] > 0 and empirical_transition[j, i] > 0:
                lhs = empirical_probs[i] * empirical_transition[i, j]
                rhs = empirical_probs[j] * empirical_transition[j, i]
                error = np.abs(lhs - rhs) / (lhs + rhs + 1e-10)
                balance_errors.append(error)
                balance_pairs.append({
                    'i': i, 'j': j,
                    'p_i': empirical_probs[i], 'p_j': empirical_probs[j],
                    'P_ij': empirical_transition[i, j], 'P_ji': empirical_transition[j, i],
                    'lhs': lhs, 'rhs': rhs, 'error': error
                })
    
    return {
        'transition_matrix': empirical_transition,
        'empirical_probs': empirical_probs,
        'balance_errors': np.array(balance_errors),
        'balance_pairs': balance_pairs,
        'bin_centers': bin_centers,
        'mean_error': np.mean(balance_errors) if balance_errors else 0.0,
        'max_error': np.max(balance_errors) if balance_errors else 0.0
    }


# ══════════════════════════════════════════════════════════
# 步骤1：MH算法基本流程演示
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("实验4.2-1 Metropolis-Hastings算法基础演示")
print("=" * 60)

print("\n[步骤1] MH算法基本流程演示")
print("-" * 60)

# 目标分布: 标准正态分布 N(0, 1)
# 未归一化密度: p(x) ∝ exp(-x²/2)
print("目标分布: 标准正态分布 N(0, 1)")
print("未归一化密度: p(x) ∝ exp(-x²/2)")
print("")

# MH 采样参数
n_samples = 50000
sigma = 1.0
burn_in = 5000

print(f"采样参数:")
print(f"  总样本数: {n_samples}")
print(f"  提议标准差 σ = {sigma}")
print(f"  预烧期: {burn_in}")
print("")

# 执行采样（初始点 x0=5.0，远离众数，以展示 burn-in 收敛过程）
samples_full, samples_post, accept_rate_post, accept_rate_full = mh_sampler(
    target=target_density,
    log_target=log_target_density,
    n_samples=n_samples,
    sigma=sigma,
    x0=5.0,
    burn_in=burn_in
)

print(f"采样完成:")
print(f"  有效样本数: {len(samples_post)}")
print(f"  完整接受率: {accept_rate_full:.4f}")
print(f"  稳态接受率 (post-burn-in): {accept_rate_post:.4f}")
print(f"  样本均值: {np.mean(samples_post):.4f} (理论值: 0)")
print(f"  样本方差: {np.var(samples_post):.4f} (理论值: 1)")

# 绘图：轨迹图（含burn-in展示）
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 轨迹图：展示burn-in初段和post-burn-in
burn_in_show = 1000   # 展示 burn-in 的最初 1000 步（从 x0=5.0 出发）
post_show = 1000      # 展示稳态期的 1000 步
n_show = burn_in_show + post_show

# burn-in 阶段（红色）：从初始点出发的游走
axes[0].plot(range(burn_in_show), samples_full[:burn_in_show], 
             'r-', linewidth=0.5, alpha=0.7, label=r'Burn-in 期')
# post-burn-in 阶段（蓝色）
axes[0].plot(range(burn_in_show, n_show), samples_full[burn_in:burn_in + post_show], 
             'b-', linewidth=0.5, alpha=0.7, label=r'稳态期')
axes[0].axvline(x=burn_in_show, color='gray', linestyle='--', linewidth=1.5, 
                label=r'Burn-in 截止')
axes[0].set_xlabel(r'迭代次数（中间省略4000步）', fontsize=12)
axes[0].set_ylabel(r'状态 $x$', fontsize=12)
axes[0].set_title(r'步骤1: MH轨迹 (含Burn-in展示)', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 经验分布 vs 真实分布
x_grid = np.linspace(-4, 4, 500)
true_pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_grid ** 2)

axes[1].hist(samples_post, bins=80, density=True, alpha=0.6, color='blue',
             label=r'MH采样')
axes[1].plot(x_grid, true_pdf, 'r-', linewidth=2, label=r'真实 $N(0,1)$')
axes[1].set_xlabel(r'状态 $x$', fontsize=12)
axes[1].set_ylabel(r'密度', fontsize=12)
axes[1].set_title(r'步骤1: 平稳分布', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step1_mh_basic.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  图已保存: step1_mh_basic.png")

# ══════════════════════════════════════════════════════════
# 步骤2：细致平衡条件数值验证（统计经验转移频率）
# ══════════════════════════════════════════════════════════
print("\n[步骤2] 细致平衡条件数值验证")
print("-" * 60)

print("细致平衡条件:")
print(r"  p(x) · P(x→x') = p(x') · P(x'→x)")
print("")
print("验证方法: 统计经验转移频率，验证离散化后的统计版本")
print(r"  p̂(x_i) · P̂(x_i→x_j) ≈ p̂(x_j) · P̂(x_j→x_i)")
print("")

# 执行细致平衡验证
db_results = verify_detailed_balance_empirical(
    target=target_density,
    log_target=log_target_density,
    n_samples=200000,
    sigma=1.0,
    n_bins=20,
    x_range=(-3, 3),
    burn_in=5000
)

print(f"细致平衡验证结果:")
print(f"  离散化区间数: 20")
print(f"  有效转移对数: {len(db_results['balance_errors'])}")
print(f"  平均相对误差: {db_results['mean_error']:.4f}")
print(f"  最大相对误差: {db_results['max_error']:.4f}")

if db_results['mean_error'] < 0.1:
    print(f"  结论: 经验转移频率满足细致平衡条件（误差 < 10%）")
else:
    print(f"  结论: 需要更多样本以减小统计误差")

# 绘图：细致平衡验证
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 经验转移矩阵热力图
im = axes[0].imshow(db_results['transition_matrix'], cmap='Blues', aspect='auto')

# 设置坐标轴为实际的 bin 位置
n_bins = len(db_results['bin_centers'])
tick_positions = np.linspace(0, n_bins - 1, 5).astype(int)
tick_labels = [f'{db_results["bin_centers"][p]:.1f}' for p in tick_positions]
axes[0].set_xticks(tick_positions)
axes[0].set_xticklabels(tick_labels)
axes[0].set_yticks(tick_positions)
axes[0].set_yticklabels(tick_labels)

axes[0].set_xlabel(r'目标状态 $x_j$', fontsize=12)
axes[0].set_ylabel(r'起始状态 $x_i$', fontsize=12)
axes[0].set_title(r'步骤2: 经验转移概率矩阵', fontsize=12)
plt.colorbar(im, ax=axes[0], label=r'$\hat{P}(x_i \to x_j)$')

# 细致平衡误差分布
if len(db_results['balance_errors']) > 0:
    axes[1].hist(db_results['balance_errors'], bins=30, density=True, 
                  alpha=0.7, color='green', edgecolor='darkgreen')
    axes[1].axvline(x=db_results['mean_error'], color='r', linestyle='--', 
                    linewidth=2, label=rf'平均误差: {db_results["mean_error"]:.3f}')
    axes[1].set_xlabel(r'相对误差 $\frac{|LHS - RHS|}{LHS + RHS}$', fontsize=12)
    axes[1].set_ylabel(r'密度', fontsize=12)
    axes[1].set_title(r'步骤2: 细致平衡误差分布', fontsize=12)
    axes[1].legend(fontsize=10)
# 无论是否有数据，都添加网格（空图也保持格式一致）
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step2_detailed_balance.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  图已保存: step2_detailed_balance.png")

# ══════════════════════════════════════════════════════════
# 步骤3：MH不需要归一化常数的演示
# ══════════════════════════════════════════════════════════
print("\n[步骤3] MH不需要归一化常数的演示")
print("-" * 60)

print("关键洞见:")
print("  MH接受概率 α = min(1, p(x')/p(x))")
print("  若 p(x) = C · f(x)，其中 C 为归一化常数")
print("  则 p(x')/p(x) = f(x')/f(x)，常数 C 自动消去")
print("")

# 使用不同"归一化常数"的目标密度
def unnormalized_density(x, C=1.0):
    """未归一化密度: C · exp(-x²/2)"""
    return C * np.exp(-0.5 * np.clip(x ** 2, None, 1400))


def log_unnormalized_density(x, C=1.0):
    """
    未归一化密度的对数形式: log(C) - x²/2
    
    注意：log(C) 项在 log 域的差值中会被消掉，
    这更好地呼应"归一化常数自动消去"的教学主题。
    """
    return np.log(C) - 0.5 * np.clip(x ** 2, None, 1400)

# 测试不同的归一化常数
constants = [1.0, 2.0, 5.0, 10.0, 100.0]

print("使用不同的归一化常数进行采样:")
print("-" * 40)

results = []
for C in constants:
    # 使用默认参数绑定，避免Lambda捕获问题
    # 传入 log_target 参数，使用 log 域计算（数值稳定）
    _, samples_post_C, accept_rate_post_C, _ = mh_sampler(
        target=lambda x, _C=C: unnormalized_density(x, _C),
        log_target=lambda x, _C=C: log_unnormalized_density(x, _C),
        n_samples=20000,
        sigma=1.0,
        burn_in=2000
    )
    mean_C = np.mean(samples_post_C)
    var_C = np.var(samples_post_C)
    results.append({'C': C, 'mean': mean_C, 'var': var_C, 'accept_rate': accept_rate_post_C})
    print(f"  C = {C:6.1f}: 均值 = {mean_C:7.4f}, 方差 = {var_C:7.4f}, 接受率 = {accept_rate_post_C:.4f}")

print("")
print("结论: 无论归一化常数 C 取何值，采样结果相同（均值≈0，方差≈1）")
print("      这说明 MH 算法不需要知道归一化常数")

# 绘图：不同归一化常数的采样结果
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 均值对比
C_values = [r['C'] for r in results]
means = [r['mean'] for r in results]
var_list = [r['var'] for r in results]  # 避免遮蔽内置vars

axes[0].bar(range(len(C_values)), means, color='steelblue', alpha=0.7)
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2, label=r'理论值: 0')
axes[0].set_xticks(range(len(C_values)))
axes[0].set_xticklabels([rf'$C={int(c)}$' for c in C_values])
axes[0].set_ylabel(r'样本均值', fontsize=12)
axes[0].set_title(r'步骤3: 均值与 $C$ 无关', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

# 方差对比
axes[1].bar(range(len(C_values)), var_list, color='coral', alpha=0.7)
axes[1].axhline(y=1, color='r', linestyle='--', linewidth=2, label=r'理论值: 1')
axes[1].set_xticks(range(len(C_values)))
axes[1].set_xticklabels([rf'$C={int(c)}$' for c in C_values])
axes[1].set_ylabel(r'样本方差', fontsize=12)
axes[1].set_title(r'步骤3: 方差与 $C$ 无关', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step3_normalization_constant.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  图已保存: step3_normalization_constant.png")

# ══════════════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print("1. MH算法通过提议-接受/拒绝机制生成马尔可夫链")
print("2. 细致平衡条件保证链收敛到目标分布（通过经验转移频率验证）")
print("3. MH算法不需要归一化常数，这是其核心优势")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")
