"""
实验4.2-1 Metropolis-Hastings算法基础演示
对应章节：4.2 Metropolis-Hastings算法
知识点：MH算法基本流程；细致平衡条件数值验证；MH不需要归一化常数

实验步骤：
  步骤1：MH算法基本流程演示
  步骤2：细致平衡条件数值验证
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
    return np.exp(-0.5 * x ** 2)


def mh_sampler(target, n_samples, sigma=1.0, x0=0.0, burn_in=5000):
    """
    Metropolis-Hastings 采样器

    参数:
        target: 未归一化目标密度函数
        n_samples: 总采样数（含 burn-in）
        sigma: 随机游走提议的标准差
        x0: 初始状态
        burn_in: 预烧期样本数

    返回:
        samples: 采样结果（去除 burn-in）
        accept_rate: 接受率
        detailed_balance_data: 细致平衡验证数据
    """
    x = x0
    samples = []
    n_accept = 0

    # 用于细致平衡验证
    lhs_values = []
    rhs_values = []

    for i in range(n_samples):
        # 提议：随机游走
        proposal = x + sigma * np.random.randn()

        # 接受概率
        alpha = min(1.0, target(proposal) / target(x))

        # 接受/拒绝
        if np.random.rand() < alpha:
            # 记录接受转移用于细致平衡验证
            # 细致平衡: p(x) * α(x→x') = p(x') * α(x'→x)
            lhs = target(x) * alpha
            rhs = target(proposal) * min(1.0, target(x) / target(proposal))
            lhs_values.append(lhs)
            rhs_values.append(rhs)

            x = proposal
            n_accept += 1

        samples.append(x)

    samples = np.array(samples)
    samples_post = samples[burn_in:]
    accept_rate = n_accept / n_samples

    detailed_balance_data = {
        'lhs': np.array(lhs_values),
        'rhs': np.array(rhs_values)
    }

    return samples_post, accept_rate, detailed_balance_data


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

# 执行采样
samples, accept_rate, db_data = mh_sampler(
    target=target_density,
    n_samples=n_samples,
    sigma=sigma,
    x0=0.0,
    burn_in=burn_in
)

print(f"采样完成:")
print(f"  有效样本数: {len(samples)}")
print(f"  接受率: {accept_rate:.4f}")
print(f"  样本均值: {np.mean(samples):.4f} (理论值: 0)")
print(f"  样本方差: {np.var(samples):.4f} (理论值: 1)")

# 绘图：轨迹图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 前1000次迭代的轨迹
axes[0].plot(samples[:1000], 'b-', linewidth=0.5, alpha=0.7)
axes[0].set_xlabel(r'迭代次数', fontsize=12)
axes[0].set_ylabel(r'状态 $x$', fontsize=12)
axes[0].set_title(r'步骤1: MH轨迹 (前1000次迭代)', fontsize=12)
axes[0].grid(True, alpha=0.3)

# 经验分布 vs 真实分布
x_grid = np.linspace(-4, 4, 500)
true_pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_grid ** 2)

axes[1].hist(samples, bins=80, density=True, alpha=0.6, color='blue',
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
# 步骤2：细致平衡条件数值验证
# ══════════════════════════════════════════════════════════
print("\n[步骤2] 细致平衡条件数值验证")
print("-" * 60)

print("细致平衡条件:")
print(r"  p(x) · α(x→x') = p(x') · α(x'→x)")
print("")

# 计算细致平衡误差
lhs = db_data['lhs']
rhs = db_data['rhs']

db_error = np.mean(np.abs(lhs - rhs))
db_relative_error = np.mean(np.abs(lhs - rhs) / (np.abs(lhs) + 1e-10))

print(f"细致平衡验证:")
print(f"  样本数: {len(lhs)}")
print(f"  平均绝对误差: {db_error:.6e}")
print(f"  平均相对误差: {db_relative_error:.6e}")
print(f"  结论: 误差接近机器精度，细致平衡条件成立")

# 绘图：细致平衡验证
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# LHS vs RHS 散点图
axes[0].scatter(lhs[::10], rhs[::10], alpha=0.5, s=10)
max_val = max(np.max(lhs), np.max(rhs))
axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label=r'$y=x$')
axes[0].set_xlabel(r'LHS: $p(x) \cdot \alpha(x \to x\')$', fontsize=12)
axes[0].set_ylabel(r'RHS: $p(x\') \cdot \alpha(x\' \to x)$', fontsize=12)
axes[0].set_title(r'步骤2: 细致平衡验证', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 误差分布
errors = np.abs(lhs - rhs)
axes[1].hist(errors, bins=50, density=True, alpha=0.7, color='green')
axes[1].axvline(x=db_error, color='r', linestyle='--', linewidth=2,
                label=rf'平均误差: {db_error:.2e}')
axes[1].set_xlabel(r'绝对误差 $|LHS - RHS|$', fontsize=12)
axes[1].set_ylabel(r'密度', fontsize=12)
axes[1].set_title(r'步骤2: 细致平衡误差分布', fontsize=12)
axes[1].legend(fontsize=10)
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
    return C * np.exp(-0.5 * x ** 2)

# 测试不同的归一化常数
constants = [1.0, 2.0, 5.0, 10.0, 100.0]

print("使用不同的归一化常数进行采样:")
print("-" * 40)

results = []
for C in constants:
    samples_C, accept_rate_C, _ = mh_sampler(
        target=lambda x: unnormalized_density(x, C),
        n_samples=20000,
        sigma=1.0,
        burn_in=2000
    )
    mean_C = np.mean(samples_C)
    var_C = np.var(samples_C)
    results.append({'C': C, 'mean': mean_C, 'var': var_C, 'accept_rate': accept_rate_C})
    print(f"  C = {C:6.1f}: 均值 = {mean_C:7.4f}, 方差 = {var_C:7.4f}, 接受率 = {accept_rate_C:.4f}")

print("")
print("结论: 无论归一化常数 C 取何值，采样结果相同（均值≈0，方差≈1）")
print("      这说明 MH 算法不需要知道归一化常数")

# 绘图：不同归一化常数的采样结果
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 均值对比
C_values = [r['C'] for r in results]
means = [r['mean'] for r in results]
vars = [r['var'] for r in results]

axes[0].bar(range(len(C_values)), means, color='steelblue', alpha=0.7)
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2, label=r'理论值: 0')
axes[0].set_xticks(range(len(C_values)))
axes[0].set_xticklabels([rf'$C={int(c)}$' for c in C_values])
axes[0].set_ylabel(r'样本均值', fontsize=12)
axes[0].set_title(r'步骤3: 均值与 $C$ 无关', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

# 方差对比
axes[1].bar(range(len(C_values)), vars, color='coral', alpha=0.7)
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
print("2. 细致平衡条件保证链收敛到目标分布")
print("3. MH算法不需要归一化常数，这是其核心优势")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")
