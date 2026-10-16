"""
实验4.2-2 提议分布步长与MH采样效率
对应章节：4.2 Metropolis-Hastings算法
知识点：提议分布步长对接受率的影响；接受率与采样效率的关系；自相关分析

实验步骤：
  步骤1：不同步长对接受率的影响
  步骤2：接受率与采样效率的关系（自相关分析）
  步骤3：最优步长的选择原则
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
    SAVE_DIR = os.path.join(_gdrive, '实验4.2-2')
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


def use_plain_log_ticks(ax):
    """对数轴使用纯文本十进制刻度，避免 mathtext 上标负号在中文字体下显示为方块"""
    ax.set_xticks([0.1, 1.0, 10.0])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_xlim(0.08, 12.0)

# ══════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════

def target_log_density(x):
    """
    对数目标密度: log p(x) ∝ -x²/2

    参数:
        x: 标量或数组

    返回:
        对数密度值
    """
    return -0.5 * x ** 2


def mh_sampler(target_log_density, n_samples, sigma=1.0, x0=0.0, burn_in=0):
    """
    Metropolis-Hastings 采样器

    参数:
        target_log_density: 对数目标密度函数
        n_samples: 采样数
        sigma: 随机游走提议的标准差
        x0: 初始状态
        burn_in: 燃烧期（丢弃的初始样本数）

    返回:
        samples: 采样结果（已丢弃burn_in样本）
        accept_rate: 接受率（仅基于采样阶段，不含burn-in）
    """
    x = x0
    samples = []
    n_accept_post = 0
    log_x = target_log_density(x)

    for i in range(n_samples + burn_in):
        proposal = x + sigma * np.random.randn()
        log_proposal = target_log_density(proposal)
        # log 域判定：log(u) < log(alpha)=log_proposal-log_x（u~U(0,1)），
        # 与 "u < min(1, exp(...))" 等价，且可避免极端密度差下 exp 上溢
        accepted = np.log(np.random.rand()) < (log_proposal - log_x)
        if accepted:
            x = proposal
            log_x = log_proposal

        if i >= burn_in:
            samples.append(x)
            if accepted:
                n_accept_post += 1

    samples = np.array(samples)
    accept_rate = n_accept_post / n_samples if n_samples > 0 else 0.0

    return samples, accept_rate


def autocorrelation(x, max_lag=None):
    """
    计算自相关函数（FFT，有偏一致估计，周期图保证正定性）

    参数:
        x: 时间序列
        max_lag: 最大滞后阶数；None 时自适应取 min(N-1, max(100, N//4))，
                 保证慢混合链的自相关在被计入 ESS 前已充分衰减

    返回:
        acf: 自相关函数值 (从 lag=1 到 max_lag)
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    N = len(x)
    if max_lag is None:
        max_lag = min(N - 1, max(100, N // 4))

    denom = np.dot(x, x)
    if denom < 1e-12:
        return np.zeros(max_lag)

    n_fft = 1 << int(np.ceil(np.log2(2 * N)))
    f = np.fft.rfft(x, n=n_fft)
    acf_full = np.fft.irfft(f * np.conj(f), n=n_fft)[:N] / denom

    return acf_full[1:max_lag + 1]


def effective_sample_size(samples):
    """
    计算有效样本数 (ESS)

    使用 Geyer (1992) 初始正序列 (Initial Positive Sequence, IPS) 估计量：
        tau = -1 + 2 * sum_k Gamma_k,  Gamma_k = rho(2k) + rho(2k+1),  rho(0)=1
        ESS = N / tau
    在第一个 Gamma_k <= 0 处截断；滞后阶数随链长自适应。
    相比固定 max_lag=100，可避免慢混合链（如 sigma=0.1 时 rho(100)≈0.63）
    因自相关未衰减完即截断而高估 ESS。

    参数:
        samples: 采样结果

    返回:
        ess: 有效样本数
    """
    N = len(samples)
    rho = np.concatenate(([1.0], autocorrelation(samples)))

    tau = -1.0
    for k in range(len(rho) // 2):
        gamma_k = rho[2 * k] + rho[2 * k + 1]
        if gamma_k <= 0:
            break
        tau += 2.0 * gamma_k

    # 有限样本噪声下钳到理论下界 tau>=1（即 ESS<=N）
    tau = max(tau, 1.0)
    return N / tau


# ══════════════════════════════════════════════════════════
# 步骤1：不同步长对接受率的影响
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("实验4.2-2 提议分布步长与MH采样效率")
print("=" * 60)

print("\n[步骤1] 不同步长对接受率的影响")
print("-" * 60)

# 测试不同步长
sigmas = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
n_samples = 30000

print(f"采样数: {n_samples}, 燃烧期: 1000")
print("")
print("步长 σ 与接受率:")
print("-" * 40)

results = {}
for sigma in sigmas:
    samples, accept_rate = mh_sampler(
        target_log_density=target_log_density,
        n_samples=n_samples,
        sigma=sigma,
        burn_in=1000
    )
    avg_disp = np.mean(np.abs(np.diff(samples)))
    results[sigma] = {
        'samples': samples,
        'accept_rate': accept_rate,
        'avg_displacement': avg_disp
    }
    print(f"  σ = {sigma:4.1f}: 接受率 = {accept_rate:.4f}, 平均位移 = {avg_disp:.4f}")

print("")
print("观察:")
print("  - 步长过小 (σ=0.1): 接受率高，平均位移极小（链'卡'在初始值附近）")
print("  - 步长适中 (σ=1~2): 接受率适中，移动效率高")
print("  - 步长过大 (σ=8): 接受率低，大量提议被拒绝")

# 绘图：步长 vs 接受率
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

accept_rates = [results[s]['accept_rate'] for s in sigmas]
axes[0].plot(sigmas, accept_rates, 'bo-', linewidth=2, markersize=8)
axes[0].axhline(y=0.44, color='r', linestyle='--', linewidth=2,
                label=r'1维RWM理论最优值 ≈ 0.44')
axes[0].set_xlabel(r'提议步长 $\sigma$', fontsize=12)
axes[0].set_ylabel(r'接受率', fontsize=12)
axes[0].set_title(r'步骤1: 步长与接受率', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xscale('log')
use_plain_log_ticks(axes[0])

# 轨迹对比：展示 burn-in 的必要性
sigmas_demo = [0.1, 1.0, 8.0]
for idx, sigma in enumerate(sigmas_demo):
    samples = results[sigma]['samples']
    axes[1].plot(samples[:500], alpha=0.7, linewidth=0.5,
                 label=rf'$\sigma={sigma}$, 接受率={results[sigma]["accept_rate"]:.2f}')

axes[1].set_xlabel(r'迭代次数 (burn-in 后)', fontsize=12)
axes[1].set_ylabel(r'状态 $x$', fontsize=12)
axes[1].set_title(rf'步骤1: 轨迹对比 (前500次迭代, burn-in=1000)', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# 新增：展示 burn-in 阶段，从远端 x0=10 出发
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

# 从 x0=10 开始的小步长轨迹
for sigma, color in zip([0.1, 1.0, 8.0], ['red', 'green', 'blue']):
    samples_bi, _ = mh_sampler(
        target_log_density=target_log_density,
        n_samples=500,
        sigma=sigma,
        x0=10.0,
        burn_in=0
    )
    axes2[0].plot(samples_bi[:500], alpha=0.7, linewidth=0.5,
                  color=color, label=rf'$\sigma={sigma}$')

axes2[0].axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
axes2[0].set_xlabel(r'迭代次数', fontsize=12)
axes2[0].set_ylabel(r'状态 $x$', fontsize=12)
axes2[0].set_title(r'步骤1: burn-in 演示 (x0=10)', fontsize=12)
axes2[0].legend(fontsize=10)
axes2[0].grid(True, alpha=0.3)
axes2[0].set_ylim(-4, 12)

# 平均位移 vs 步长
axes2[1].plot(sigmas, [results[s]['avg_displacement'] for s in sigmas], 'bo-', linewidth=2, markersize=8)
axes2[1].set_xlabel(r'提议步长 $\sigma$', fontsize=12)
axes2[1].set_ylabel(r'平均位移 $E[|x_{t+1}-x_t|]$', fontsize=12)
axes2[1].set_title(r'步骤1: 移动效率 (平均每次迭代位移)', fontsize=12)
axes2[1].set_xscale('log')
use_plain_log_ticks(axes2[1])
axes2[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'step1_step_size_acceptance.png'), dpi=150, bbox_inches='tight')
fig2.tight_layout()
fig2.savefig(os.path.join(SAVE_DIR, 'step1_burnin_demo.png'), dpi=150, bbox_inches='tight')
plt.close('all')

print(f"  图已保存: step1_step_size_acceptance.png, step1_burnin_demo.png")

# ══════════════════════════════════════════════════════════
# 步骤2：接受率与采样效率的关系（自相关分析）
# ══════════════════════════════════════════════════════════
print("\n[步骤2] 接受率与采样效率的关系（自相关分析）")
print("-" * 60)

print("关键洞见:")
print("  接受率高 ≠ 采样效率高")
print("  采样效率由自相关决定：自相关越低，有效样本越多")
print("")

# 计算自相关和 ESS
max_lag = 100
print("步长 σ 与采样效率:")
print("-" * 60)
print(f"{'σ':>6} {'接受率':>12} {'ESS':>12} {'ESS/N':>10}")
print("-" * 60)

for sigma in sigmas:
    samples = results[sigma]['samples']
    ess = effective_sample_size(samples)
    results[sigma]['ess'] = ess
    results[sigma]['ess_ratio'] = ess / n_samples
    print(f"{sigma:6.1f} {results[sigma]['accept_rate']:12.4f} {ess:12.0f} {ess/n_samples:10.4f}")

print("")

# 绘图：自相关函数
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 自相关函数对比
lags = np.arange(1, max_lag + 1)
colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown']

for idx, sigma in enumerate(sigmas):
    samples = results[sigma]['samples']
    acf = autocorrelation(samples, max_lag)
    axes[0].plot(lags, acf, color=colors[idx], linewidth=1.5,
                 label=rf'$\sigma={sigma}$')

axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[0].set_xlabel(r'滞后阶数', fontsize=12)
axes[0].set_ylabel(r'自相关', fontsize=12)
axes[0].set_title(r'步骤2: 自相关函数', fontsize=12)
axes[0].legend(fontsize=9, ncol=2)
axes[0].grid(True, alpha=0.3)

# ESS vs 步长
ess_values = [results[s]['ess'] for s in sigmas]
ess_ratios = [results[s]['ess_ratio'] for s in sigmas]

axes[1].bar(range(len(sigmas)), ess_ratios, color=colors, alpha=0.7)
axes[1].set_xticks(range(len(sigmas)))
axes[1].set_xticklabels([rf'$\sigma={s}$' for s in sigmas], rotation=30, ha='right')
axes[1].set_ylabel(r'ESS / N (效率)', fontsize=12)
axes[1].set_title(r'步骤2: 有效样本数比率', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step2_autocorrelation_ess.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  图已保存: step2_autocorrelation_ess.png")

# ══════════════════════════════════════════════════════════
# 步骤3：最优步长的选择原则
# ══════════════════════════════════════════════════════════
print("\n[步骤3] 最优步长的选择原则")
print("-" * 60)

# 更细致的步长扫描（重复运行取均值，减少随机波动）
# 网格在理论最优区 sigma∈[2,3]（接受率约0.4~0.5）加密，两侧对数稀疏采样
sigmas_fine = np.unique(np.round(np.concatenate([
    np.logspace(-1, np.log10(2.0), 10),  # 0.1 ~ 2.0
    np.linspace(2.0, 3.0, 11),           # 2.0 ~ 3.0 加密区
    np.logspace(np.log10(3.1), 1, 7),    # 3.1 ~ 10
]), 4))
n_repeats = 20
n_samples_fine = 20000

ess_fine_arr = np.zeros((len(sigmas_fine), n_repeats))
accept_fine_arr = np.zeros((len(sigmas_fine), n_repeats))

for i, sigma in enumerate(sigmas_fine):
    for r in range(n_repeats):
        samples, accept_rate = mh_sampler(
            target_log_density=target_log_density,
            n_samples=n_samples_fine,
            sigma=sigma,
            burn_in=1000
        )
        ess = effective_sample_size(samples)
        ess_fine_arr[i, r] = ess
        accept_fine_arr[i, r] = accept_rate

ess_fine = np.mean(ess_fine_arr, axis=1)
accept_fine = np.mean(accept_fine_arr, axis=1)
ess_std = np.std(ess_fine_arr, axis=1)

# 有限样本 argmax：仅作为峰值位置的参考点（会在平坦区内随机跳动）
optimal_idx = np.argmax(ess_fine)
optimal_sigma = sigmas_fine[optimal_idx]
optimal_ess = ess_fine[optimal_idx]
optimal_accept = accept_fine[optimal_idx]

# 平坦最优区：ESS 不低于峰值 97% 的步长范围
# （理论上 1 维 RWM 的扩散速度目标在峰值附近很平坦，单点 argmax 不稳健）
plateau_frac = 0.97
plateau_mask = ess_fine >= plateau_frac * ess_fine.max()
plateau_sigmas = sigmas_fine[plateau_mask]
region_sigma_lo, region_sigma_hi = float(plateau_sigmas.min()), float(plateau_sigmas.max())
region_acc_lo = float(accept_fine[plateau_mask].min())
region_acc_hi = float(accept_fine[plateau_mask].max())

# 接受率 = 0.44 对应的步长（接受率随 sigma 单调下降，反向插值）
sigma_at_044 = float(np.interp(0.44, accept_fine[::-1], sigmas_fine[::-1]))

print(f"ESS 峰值 ({n_repeats}次平均): argmax σ = {optimal_sigma:.3f}, "
      f"ESS = {optimal_ess:.0f} ± {ess_std[optimal_idx]:.0f}, 对应接受率 = {optimal_accept:.3f}")
print(f"平坦最优区 (ESS ≥ {int(plateau_frac * 100)}% 峰值): "
      f"σ ∈ [{region_sigma_lo:.2f}, {region_sigma_hi:.2f}], "
      f"对应接受率 ∈ [{region_acc_lo:.3f}, {region_acc_hi:.3f}]")
print(f"接受率 = 0.44 对应 σ ≈ {sigma_at_044:.3f}（落在平坦区内，与 1 维 RWM 理论一致）")
print(f"  (基于 {n_repeats} 次独立重复，每次 {n_samples_fine} 样本，burn-in=1000)")
print("  说明: ESS 目标在峰值附近很平坦，有限样本 argmax 会在平坦区内随机跳动，")
print("        因此最优步长应表述为一个区间，而非单一数值。")
print("")

print("选择原则:")
print("  1. 步长过小: 高接受率但低效率（自相关高）")
print("  2. 步长过大: 低接受率，大量拒绝导致低效率")
print("  3. 最优步长: 平衡接受率和移动距离，且最优点附近是一段平坦区")
print("  4. 1维RWM理论最优接受率 ≈ 0.44 (Roberts, Gelman & Gilks, 1997)")
print("  5. 高维RWM渐近最优接受率 ≈ 0.234 (Roberts, Gelman & Gilks, 1997)")

# 绘图：ESS 和接受率随步长变化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ESS vs 步长（带误差线）
axes[0].semilogx(sigmas_fine, ess_fine, 'b-', linewidth=2)
axes[0].fill_between(sigmas_fine,
                      np.maximum(ess_fine - ess_std, 0),
                      ess_fine + ess_std,
                      alpha=0.2, color='blue')
axes[0].axvspan(region_sigma_lo, region_sigma_hi, color='red', alpha=0.12,
                label=rf'平坦最优区 ($\sigma\in[{region_sigma_lo:.1f},{region_sigma_hi:.1f}]$)')
axes[0].axvline(x=optimal_sigma, color='r', linestyle='--', linewidth=2,
                label=rf'有限样本 argmax $\sigma={optimal_sigma:.2f}$, 接受率≈{optimal_accept:.2f}')
axes[0].scatter([optimal_sigma], [optimal_ess], color='r', s=100, zorder=5)
axes[0].axhline(y=n_samples_fine, color='gray', linestyle=':', linewidth=1,
                alpha=0.5, label=r'理论上限 ESS = N')
axes[0].set_xlabel(r'提议步长 $\sigma$', fontsize=12)
axes[0].set_ylabel(r'有效样本数 (ESS)', fontsize=12)
axes[0].set_title(rf'步骤3: ESS与步长 ({n_repeats}次重复均值±标准差)', fontsize=12)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
use_plain_log_ticks(axes[0])

# 接受率 vs 步长
axes[1].semilogx(sigmas_fine, accept_fine, 'g-', linewidth=2)
axes[1].axvspan(region_sigma_lo, region_sigma_hi, color='red', alpha=0.12)
axes[1].axvline(x=optimal_sigma, color='r', linestyle='--', linewidth=2,
                label=rf'有限样本 argmax $\sigma={optimal_sigma:.2f}$')
axes[1].axhline(y=0.44, color='purple', linestyle='--', linewidth=2,
                label=r'1维RWM理论最优值 ≈ 0.44')
axes[1].scatter([sigma_at_044], [0.44], color='purple', s=70, zorder=5,
                label=rf'实验交点 $\sigma≈{sigma_at_044:.2f}$')
axes[1].axhline(y=0.234, color='orange', linestyle=':', linewidth=1.5, alpha=0.6,
                label=r'高维RWM渐近值 ≈ 0.234')
axes[1].set_xlabel(r'提议步长 $\sigma$', fontsize=12)
axes[1].set_ylabel(r'接受率', fontsize=12)
axes[1].set_title(r'步骤3: 接受率与步长', fontsize=12)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
use_plain_log_ticks(axes[1])

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step3_optimal_step_size.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  图已保存: step3_optimal_step_size.png")

# ══════════════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print("1. 提议分布步长直接影响 MH 算法的接受率")
print("2. 接受率高不等于采样效率高，需关注自相关")
print("3. 有效样本数 (ESS) 是衡量采样效率的关键指标")
print("4. 最优步长在「接受率适中」和「移动距离足够」之间取得平衡")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")
print(f"\n【注意事项】")
print(f"本实验针对1维标准正态目标分布:")
print(f"  - 1维RWM理论最优接受率 ≈ 0.44")
print(f"  - 高维RWM(Roberts et al. 1997)渐近最优接受率 ≈ 0.234")
print(f"  - 一般经验推荐区间: [0.2, 0.5]")
print(f"  - 实测平坦最优区: σ ∈ [{region_sigma_lo:.2f}, {region_sigma_hi:.2f}], "
      f"接受率 ∈ [{region_acc_lo:.3f}, {region_acc_hi:.3f}]，与理论 0.44 一致")
print(f"  - 有限样本 argmax 会在平坦区内跳动（本次为 σ={optimal_sigma:.2f}），应以区间为准")

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
    'experiment': '4.2-2',
    'title': '提议分布步长与MH采样效率',
    'step1_step_size': {
        'sigmas': sigmas,
        'accept_rates': [float(round(results[s]['accept_rate'], 4)) for s in sigmas],
        'avg_displacements': [float(round(results[s]['avg_displacement'], 4)) for s in sigmas],
        'n_samples': n_samples,
    },
    'step2_autocorrelation_ess': {
        'sigmas': sigmas,
        'ess': [float(round(results[s]['ess'], 1)) for s in sigmas],
        'ess_ratio': [float(round(results[s]['ess_ratio'], 4)) for s in sigmas],
    },
    'step3_optimal_step': {
        'argmax_sigma': float(round(optimal_sigma, 4)),
        'argmax_ess': float(round(optimal_ess, 1)),
        'argmax_accept': float(round(optimal_accept, 4)),
        'plateau_fraction': plateau_frac,
        'plateau_sigma_range': [round(region_sigma_lo, 4), round(region_sigma_hi, 4)],
        'plateau_accept_range': [round(region_acc_lo, 4), round(region_acc_hi, 4)],
        'sigma_at_accept_0.44': round(sigma_at_044, 4),
        'n_repeats': n_repeats,
        'n_samples_fine': n_samples_fine,
    }
}

with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(_to_native(results_summary), f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
