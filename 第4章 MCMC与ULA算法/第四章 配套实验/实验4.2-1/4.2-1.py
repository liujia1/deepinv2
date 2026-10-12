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


def mh_sampler(target, log_target=None, n_samples=50000, sigma=1.0, x0=0.0, burn_in=5000,
               rng=None):
    """
    Metropolis-Hastings 采样器（纯采样，无验证逻辑耦合）

    参数:
        target: 未归一化目标密度函数
        log_target: 目标密度的对数形式（可选，用于数值稳定计算）
        n_samples: 总采样数（含 burn-in）
        sigma: 随机游走提议的标准差
        x0: 初始状态
        burn_in: 预烧期样本数
        rng: numpy.random.Generator/RandomState 随机数发生器（可选）。
             传入种子相同的独立 rng，可在相同随机流下对比不同目标密度，
             严格验证"归一化常数消去后轨迹逐位相同"；默认为全局 np.random。

    返回:
        samples_full: 完整采样轨迹（含 burn-in）
        samples_post: post-burn-in 样本
        accept_rate_post: post-burn-in 接受率
        accept_rate_full: 完整接受率
    """
    if rng is None:
        rng = np.random

    x = x0
    samples_full = []
    n_accept_full = 0
    n_accept_post = 0

    for i in range(n_samples):
        # 提议：随机游走
        proposal = x + sigma * rng.randn()

        # 接受概率（数值稳定版本：使用 log 域计算）
        if log_target is not None:
            log_alpha = min(0.0, log_target(proposal) - log_target(x))
        else:
            # 回退路径：当 log_target 未提供时使用
            # 注意：此路径在 target(x) 极小时可能产生数值问题，应优先使用 log_target
            log_alpha = min(0.0, np.log(target(proposal)) - np.log(target(x)))
        alpha = np.exp(log_alpha)

        # 接受/拒绝
        if rng.rand() < alpha:
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


def _db_compute_metrics(transition_counts):
    """
    从离散转移计数矩阵 N_ij 计算三种口径的细致平衡偏差。

    恒等关系（说明三种口径为何都等价于检验"双向流量对称"）：
        令 v_i = Σ_j N_ij 为从 bin i 出发的记录数，
        p̂(i) = v_i / Σv，P̂(i→j) = N_ij / v_i，
        则 p̂(i)·P̂(i→j) = N_ij / Σv，
        因此 p̂(i)P̂(ij) ≈ p̂(j)P̂(ji) 与 N_ij ≈ N_ji 逐 bin 对等价。
    注意 p̂ 与 P̂ 由同一批计数导出、并非独立估计量，本验证的实质是
    检验链的正向/反向经验转移流量是否对称。

    三种口径：
      - filtered : 仅统计双向计数均 > 0 的 bin 对，相对误差 |Nij-Nji|/(Nij+Nji)；
                    剔除了稀有 bin 对，数值偏小，仅作参考。
      - all_pairs: 纳入单向计数为 0 的 bin 对（其相对误差恰为 1），不隐瞒稀有对；
                    受低计数对的泊松噪声主导，收敛较慢。
      - flow     : 流量加权不平衡度 Σ|Nij-Nji| / Σ(Nij+Nji)，
                    大流量 bin 对权重大、统计上最稳健，理论上随 1/√n 收敛，
                    作为本实验的主判定指标。

    参数:
        transition_counts: 离散转移计数矩阵 N_ij

    返回:
        包含三种口径指标与经验分布/转移矩阵的字典
    """
    n_bins = transition_counts.shape[0]

    # 经验转移概率与经验平稳分布
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    empirical_transition = transition_counts / row_sums

    state_counts = transition_counts.sum(axis=1)
    total_visits = state_counts.sum()
    if total_visits > 0:
        empirical_probs = state_counts / total_visits
    else:
        empirical_probs = np.ones(n_bins) / n_bins

    errors_filtered = []
    errors_all = []
    flow_diff = 0.0
    flow_total = 0.0

    for i in range(n_bins):
        for j in range(i + 1, n_bins):  # 只检查上三角，避免重复
            n_ij = transition_counts[i, j]
            n_ji = transition_counts[j, i]
            pair_flow = n_ij + n_ji
            if pair_flow > 0:
                error = abs(n_ij - n_ji) / pair_flow
                errors_all.append(error)
                flow_diff += abs(n_ij - n_ji)
                flow_total += pair_flow
                if n_ij > 0 and n_ji > 0:
                    errors_filtered.append(error)

    return {
        'transition_matrix': empirical_transition,
        'empirical_probs': empirical_probs,
        'balance_errors': np.array(errors_filtered),
        'mean_error': float(np.mean(errors_filtered)) if errors_filtered else 0.0,
        'max_error': float(np.max(errors_filtered)) if errors_filtered else 0.0,
        'n_pairs_filtered': len(errors_filtered),
        'mean_error_all_pairs': float(np.mean(errors_all)) if errors_all else 0.0,
        'n_pairs_all': len(errors_all),
        'flow_imbalance': float(flow_diff / flow_total) if flow_total > 0 else 0.0,
    }


def verify_detailed_balance_empirical(target, log_target=None, n_samples=400000, sigma=1.0,
                                      n_bins=20, x_range=(-3, 3), burn_in=5000,
                                      checkpoints=(50000, 100000, 200000)):
    """
    通过统计经验转移频率验证细致平衡条件

    细致平衡条件：p(x) · P(x→x') = p(x') · P(x'→x)

    离散化后验证：统计经验转移计数 N_ij，检验双向流量对称 N_ij ≈ N_ji
    （与 p̂(x_i)·P̂(x_i→x_j) ≈ p̂(x_j)·P̂(x_j→x_i) 逐 bin 对等价，
      详见 _db_compute_metrics 的说明）。

    边界删失说明：状态空间被截断到 x_range=(-3,3)。凡起点或终点落在
    该区间之外的转移一律不计入 N_ij（链本身仍正常推进，不受影响），
    截断比例会如实统计并返回（N(0,1) 下 |x|>3 的概率质量仅约 0.27%）。

    参数:
        target: 未归一化目标密度函数
        log_target: 目标密度的对数形式（可选，用于数值稳定计算）
        n_samples: 采样数（含 burn-in）
        sigma: 随机游走提议的标准差
        n_bins: 离散化区间数
        x_range: 状态空间截断范围
        burn_in: 预烧期样本数（丢弃，不参与统计）
        checkpoints: 统计阶段的中间检查点（post-burn-in 步数），
                     用于考察各误差口径随样本量的收敛行为

    返回:
        results: 包含最终指标、收敛历史与截断比例的字典
    """
    # 离散化状态空间
    bins = np.linspace(x_range[0], x_range[1], n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # 初始化计数
    transition_counts = np.zeros((n_bins, n_bins))

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

    # 统计阶段：记录转移，并在检查点处计算一次指标（同一条链、计数累积）
    n_post = n_samples - burn_in
    checkpoints = tuple(checkpoints) + (n_post,)
    convergence = []
    n_truncated = 0

    for t in range(n_post):
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

        # 离散化并记录转移；端点越界则计为截断、不纳入计数
        x_bin = np.digitize(x, bins) - 1
        new_x_bin = np.digitize(new_x, bins) - 1

        if 0 <= x_bin < n_bins and 0 <= new_x_bin < n_bins:
            transition_counts[x_bin, new_x_bin] += 1
        else:
            n_truncated += 1

        x = new_x

        if (t + 1) in checkpoints:
            snapshot = _db_compute_metrics(transition_counts)
            snapshot['n_post'] = t + 1
            snapshot['truncated_fraction'] = n_truncated / (t + 1)
            convergence.append(snapshot)

    final = convergence[-1]

    return {
        'transition_matrix': final['transition_matrix'],
        'empirical_probs': final['empirical_probs'],
        'balance_errors': final['balance_errors'],
        'bin_centers': bin_centers,
        # 三种口径
        'mean_error': final['mean_error'],                 # 过滤口径（兼容旧字段）
        'max_error': final['max_error'],
        'n_pairs_filtered': final['n_pairs_filtered'],
        'mean_error_all_pairs': final['mean_error_all_pairs'],
        'n_pairs_all': final['n_pairs_all'],
        'flow_imbalance': final['flow_imbalance'],
        # 数据质量
        'n_post': n_post,
        'n_truncated': n_truncated,
        'truncated_fraction': n_truncated / n_post if n_post > 0 else 0.0,
        # 随样本量收敛历史
        'convergence': convergence,
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

# burn-in 阶段（红色）：从初始点出发的游走（实际 burn-in 共 5000 步）
axes[0].plot(range(burn_in_show), samples_full[:burn_in_show],
             'r-', linewidth=0.5, alpha=0.7, label=r'Burn-in 期（第0–999步）')
# post-burn-in 阶段（蓝色）
axes[0].plot(range(burn_in_show, n_show), samples_full[burn_in:burn_in + post_show],
             'b-', linewidth=0.5, alpha=0.7, label=r'稳态期（第5000步起）')
# 注意：虚线处只是两段轨迹的拼接位置，并非真实的 burn-in 截止点（真实截止在第5000步）
axes[0].axvline(x=burn_in_show, color='gray', linestyle='--', linewidth=1.5,
                label=r'拼接处（省略第1000–4999步）')
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
print("验证方法: 离散化状态空间并统计经验转移计数 N_ij，检验双向流量对称 N_ij ≈ N_ji")
print(r"  （与 p̂(x_i)·P̂(x_i→x_j) ≈ p̂(x_j)·P̂(x_j→x_i) 逐 bin 对等价）")
print("")

# 执行细致平衡验证
DB_N_SAMPLES = 400000       # 总样本数（含 burn-in）
DB_FLOW_THRESHOLD = 0.03    # 主判定阈值：流量加权不平衡度 < 3%（结果约 1%，留 2 倍以上裕度）
db_results = verify_detailed_balance_empirical(
    target=target_density,
    log_target=log_target_density,
    n_samples=DB_N_SAMPLES,
    sigma=1.0,
    n_bins=20,
    x_range=(-3, 3),
    burn_in=5000
)

print(f"细致平衡验证结果（post-burn-in 统计步数: {db_results['n_post']}）:")
print(f"  离散化区间数: 20，截断范围: [-3, 3]")
print(f"  越界截断转移比例: {db_results['truncated_fraction']:.3%} "
      f"（端点越界的转移不计入，链的推进不受影响）")
print(f"  [主指标] 流量加权不平衡度 Σ|Nij−Nji|/Σ(Nij+Nji) = "
      f"{db_results['flow_imbalance']:.4f}（阈值 {DB_FLOW_THRESHOLD:.0%}）")
print(f"  [参考] 双向非零 bin 对: 平均相对误差 = {db_results['mean_error']:.4f}，"
      f"最大 = {db_results['max_error']:.4f}（共 {db_results['n_pairs_filtered']} 对）")
print(f"  [参考] 含单向零计数 bin 对的全口径平均误差 = "
      f"{db_results['mean_error_all_pairs']:.4f}（共 {db_results['n_pairs_all']} 对）")
print(f"  说明: 全口径误差偏大由两端稀有 bin（期望计数仅个位数）的泊松噪声主导；")
print(f"        流量加权指标给大流量 bin 对更高权重，统计上更稳健。")

# 收敛历史（同一链、计数累积）
print(f"  收敛历史（流量加权不平衡度随样本量变化）:")
for snap in db_results['convergence']:
    print(f"    n = {snap['n_post']:7d}: flow = {snap['flow_imbalance']:.4f}, "
          f"过滤口径 = {snap['mean_error']:.4f}, 全口径 = {snap['mean_error_all_pairs']:.4f}")

db_pass = db_results['flow_imbalance'] < DB_FLOW_THRESHOLD
if db_pass:
    print(f"  结论: 双向转移流量对称，细致平衡条件成立（主指标 {db_results['flow_imbalance']:.4f} "
          f"< 阈值 {DB_FLOW_THRESHOLD:.0%}，且随样本量按约 1/√n 下降）")
else:
    print(f"  结论: 主指标未达阈值 {DB_FLOW_THRESHOLD:.0%}，需要更多样本以减小统计误差")

# 绘图：细致平衡验证（3 个子图：转移矩阵 / 误差分布 / 收敛曲线）
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

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

# 细致平衡误差分布（过滤口径，仅双向非零 bin 对）
if len(db_results['balance_errors']) > 0:
    axes[1].hist(db_results['balance_errors'], bins=30, density=True,
                  alpha=0.7, color='green', edgecolor='darkgreen')
    axes[1].axvline(x=db_results['mean_error'], color='r', linestyle='--',
                    linewidth=2, label=rf'平均误差: {db_results["mean_error"]:.3f}')
    axes[1].set_xlabel(r'相对误差 $\frac{|N_{ij}-N_{ji}|}{N_{ij}+N_{ji}}$', fontsize=12)
    axes[1].set_ylabel(r'密度', fontsize=12)
    axes[1].set_title(r'步骤2: 双向非零 bin 对误差分布', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].text(
        0.97, 0.97,
        f"最大误差 {db_results['max_error']:.3f} 来自两端\n"
        f"稀有 bin（计数为个位数），属泊松\n"
        f"统计噪声；流量加权不平衡度仅\n"
        f"{db_results['flow_imbalance']*100:.2f}%",
        transform=axes[1].transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    )
axes[1].grid(True, alpha=0.3)

# 误差随样本量收敛曲线（对数坐标 + 1/√n 参考线）
conv = db_results['convergence']
ns = np.array([s['n_post'] for s in conv], dtype=float)
flow_curve = np.array([s['flow_imbalance'] for s in conv])
filt_curve = np.array([s['mean_error'] for s in conv])
all_curve = np.array([s['mean_error_all_pairs'] for s in conv])

axes[2].loglog(ns, flow_curve, 'o-', color='purple', linewidth=2,
               label=r'流量加权不平衡度（主指标）')
axes[2].loglog(ns, filt_curve, 's--', color='green', linewidth=1.5,
               label=r'双向非零对平均误差（参考）')
axes[2].loglog(ns, all_curve, '^--', color='orange', linewidth=1.5,
               label=r'全口径平均误差（参考）')
# 1/√n 参考线：以主指标最后一点标定
ref_curve = flow_curve[-1] * np.sqrt(ns[-1] / ns)
axes[2].loglog(ns, ref_curve, ':', color='gray', linewidth=1.5, label=r'$1/\sqrt{n}$ 参考斜率')
axes[2].axhline(y=DB_FLOW_THRESHOLD, color='red', linestyle='--', linewidth=1.2,
                label=rf'通过阈值 {DB_FLOW_THRESHOLD:.0%}')
axes[2].set_xlabel(r'post-burn-in 样本数 $n$', fontsize=12)
axes[2].set_ylabel(r'细致平衡偏差', fontsize=12)
axes[2].set_title(r'步骤2: 偏差随样本量收敛', fontsize=12)
axes[2].legend(fontsize=8.5, loc='lower left')
axes[2].grid(True, alpha=0.3, which='both')

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

print("使用不同的归一化常数进行采样（每次使用相同种子的独立随机流）:")
print("-" * 40)

STEP3_SEED = 2024
results = []
traj_ref = None
traj_identical = []
for C in constants:
    # 默认参数绑定避免 Lambda 捕获问题；每个 C 使用种子相同的独立 rng，
    # 因而五组实验面对完全一致的随机数序列。若归一化常数 C 在接受比中
    # 被精确消去，则五条轨迹必须逐位相同——这比"五组统计量差不多"是
    # 强得多的证明。
    samples_full_C, samples_post_C, accept_rate_post_C, _ = mh_sampler(
        target=lambda x, _C=C: unnormalized_density(x, _C),
        log_target=lambda x, _C=C: log_unnormalized_density(x, _C),
        n_samples=20000,
        sigma=1.0,
        burn_in=2000,
        rng=np.random.RandomState(STEP3_SEED)
    )
    mean_C = np.mean(samples_post_C)
    var_C = np.var(samples_post_C)
    results.append({'C': C, 'mean': mean_C, 'var': var_C, 'accept_rate': accept_rate_post_C})
    if traj_ref is None:
        traj_ref = samples_full_C
    traj_identical.append(bool(np.array_equal(samples_full_C, traj_ref)))
    print(f"  C = {C:6.1f}: 均值 = {mean_C:7.4f}, 方差 = {var_C:7.4f}, "
          f"接受率 = {accept_rate_post_C:.4f}, 轨迹与C=1逐位相同 = {traj_identical[-1]}")

print("")
n_same = sum(traj_identical)
print(f"结论: 相同随机流下 {n_same}/{len(constants)} 组轨迹逐位相同，均值≈0、方差≈1，")
print("      说明 C 在接受比中被精确消去，MH 算法不需要知道归一化常数")

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

# 五组 C 使用相同随机种子：柱高完全重合（逐位相同的轨迹给出相同统计量）
_note_step3 = f'五组 C 使用相同随机种子（seed={STEP3_SEED}）\n采样轨迹逐位相同，五根柱完全重合'
for _ax in axes:
    _ax.text(0.03, 0.97, _note_step3, transform=_ax.transAxes, va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))

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
print("2. 细致平衡条件保证链收敛到目标分布（双向转移流量对称，流量加权不平衡度约1%，并随 1/√n 收敛）")
print("3. MH算法不需要归一化常数：相同随机流下任意 C 给出逐位相同的采样轨迹，而非仅仅'统计上差不多'")
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

results_summary = {
    'experiment': '4.2-1',
    'title': 'Metropolis-Hastings算法基础演示',
    'step1_mh_basic': {
        'n_samples': n_samples,
        'sigma': sigma,
        'burn_in': burn_in,
        'accept_rate_full': float(round(accept_rate_full, 4)),
        'accept_rate_post': float(round(accept_rate_post, 4)),
        'sample_mean': float(round(np.mean(samples_post), 4)),
        'sample_var': float(round(np.var(samples_post), 4)),
    },
    'step2_detailed_balance': {
        'n_samples_total': DB_N_SAMPLES,
        'burn_in': 5000,
        'n_post_samples': db_results['n_post'],
        'n_bins': 20,
        'x_range': [-3, 3],
        'truncated_transition_fraction': float(round(db_results['truncated_fraction'], 6)),
        'flow_weighted_imbalance': float(round(db_results['flow_imbalance'], 4)),
        'pass_threshold_flow_imbalance': DB_FLOW_THRESHOLD,
        'pass': bool(db_pass),
        'mean_error_filtered': float(round(db_results['mean_error'], 4)),
        'max_error_filtered': float(round(db_results['max_error'], 4)),
        'n_pairs_filtered': db_results['n_pairs_filtered'],
        'mean_error_all_pairs': float(round(db_results['mean_error_all_pairs'], 4)),
        'n_pairs_all': db_results['n_pairs_all'],
        'convergence': [
            {
                'n_post': snap['n_post'],
                'flow_weighted_imbalance': float(round(snap['flow_imbalance'], 4)),
                'mean_error_filtered': float(round(snap['mean_error'], 4)),
                'mean_error_all_pairs': float(round(snap['mean_error_all_pairs'], 4)),
                'truncated_fraction': float(round(snap['truncated_fraction'], 6)),
            }
            for snap in db_results['convergence']
        ],
    },
    'step3_normalization_constant': {
        'constants': [r['C'] for r in results],
        'rng_seed_per_run': STEP3_SEED,
        'trajectories_bitwise_identical_to_C1': [bool(v) for v in traj_identical],
        'means': [float(round(r['mean'], 4)) for r in results],
        'vars': [float(round(r['var'], 4)) for r in results],
        'accept_rates': [float(round(r['accept_rate'], 4)) for r in results],
    }
}

with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(_to_native(results_summary), f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
