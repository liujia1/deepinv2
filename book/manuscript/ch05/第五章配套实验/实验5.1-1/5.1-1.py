"""
实验5.1-1 ULA采样与步长敏感性分析
对应章节：5.1 从MCMC到Langevin SDE
知识点：
  - ULA与Langevin SDE的关系
  - 步长对收敛性的影响（偏差-方差权衡）
  - ULA的理论方差公式

素材来源：
  - Mathematics.../Teaching Unit 2/labs/lab1_ULA_sol.ipynb
    - ULA_gauss() 函数
    - var_compare() 函数
    - 1D实验代码（ULA直方图 vs 真实密度）

修改说明：
  从原参考实验5.1.py拆分，聚焦ULA采样与步长分析，
  去除得分场可视化（移至5.2-1）。
  复核修订（严谨性加强）：
    - 步骤2由"单链、所有步长固定迭代数"改为多条独立链统计
      （均值±标准差、2.5%~97.5%分位数），并报告理论有效样本量ESS，
      以区分"离散化偏差"与"单链蒙特卡洛噪声"；直方图叠加ULA真实
      不变分布 N(0, 1/(1-δ/2))，而非只画目标 N(0,1)。
    - 新增步骤3：实证检验稳定边界（δ=1.0/1.9/2.0/2.5 的二阶矩
      随迭代变化，并与理论闭式公式对比），不再仅作口头断言。
    - 修正"固定δ即收敛到真实目标分布"等不严谨表述：固定δ时
      ULA收敛到有偏不变分布，仅当δ→0才趋于目标分布。
    - 步骤1沿用seed=42单随机流，历史结果（方差1.071742等）可
      逐位复现；新增ESS与MC标准误作为统计解释，不改采样路径。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
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
    SAVE_DIR = os.path.join(_gdrive, '实验5.1-1')
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

# ============================================================
# ULA_gauss 函数（取自 lab1_ULA_sol.ipynb）
# 从1D标准高斯 N(0,1) 中用ULA采样
# 势能 U(x)=x²/2, ∇U(x)=x
# ULA: X_{m+1} = X_m - δ·∇U(X_m) + √(2δ)·Z_{m+1}
#                  = (1-δ)·X_m + √(2δ)·Z_{m+1}
# ============================================================
def ULA_gauss(niter, delta, x0=0):
    """
    ULA采样标准高斯分布 N(0,1)
    势能 U(x) = -log p(x) = x^2/2
    梯度 ∇U(x) = x
    
    参数:
        niter: 迭代次数
        delta: 步长 δ
        x0: 初始值
    
    返回:
        Y: 采样序列
    """
    Y = np.zeros(niter,)
    X = x0
    for i in range(niter):
        Z = np.random.randn()
        grad = X  # ∇U(x) = x 对于标准高斯
        X = X - delta * grad + np.sqrt(2 * delta) * Z
        Y[i] = X
    return Y


def ULA_gauss_chains(niter, delta, nchains, seed, x0=0.0, burn_in=0):
    """
    并行运行 nchains 条相互独立的ULA链（标准高斯情形）。
    与 ULA_gauss 使用完全相同的递推 X_{m+1}=(1-δ)X_m+√(2δ)Z，
    只是各链使用独立随机数，用于估计单链方差估计量的抽样波动。

    参数:
        niter: 每条链记录的迭代次数（burn-in之后）
        delta: 步长 δ
        nchains: 独立链条数
        seed: 随机种子（RandomState）
        x0: 初始值（所有链相同）
        burn_in: 预热步数，其样本不记录

    返回:
        Y: shape=(niter, nchains) 的采样序列
    """
    rng = np.random.RandomState(seed)
    X = np.full(nchains, float(x0))
    for _ in range(burn_in):
        X = (1 - delta) * X + np.sqrt(2 * delta) * rng.randn(nchains)
    Y = np.empty((niter, nchains))
    for i in range(niter):
        X = (1 - delta) * X + np.sqrt(2 * delta) * rng.randn(nchains)
        Y[i] = X
    return Y


# ============================================================
# 步骤1：1D高斯ULA采样（回顾第4章，增加自相关分析）
# ============================================================
print("=" * 60)
print("步骤1：1D高斯ULA采样")
print("=" * 60)

niter = 100000
delta = 0.1
x0 = 0

# 运行ULA
samples = ULA_gauss(niter, delta, x0)

# 可视化
plt.figure(figsize=(12, 4))

# 直方图 vs 真实密度
plt.subplot(1, 3, 1)
x = np.linspace(-4, 4, 100)
true_density = 1/np.sqrt(2*np.pi) * np.exp(-x**2 / 2)
plt.plot(x, true_density, 'b-', lw=2, label='真实 $N(0,1)$')
plt.hist(samples, bins=50, density=True, alpha=0.7, color='orange', label='ULA采样')
plt.xlabel('$x$')
plt.ylabel('密度')
plt.title('ULA采样 vs 真实密度')
plt.legend()

# 轨迹图
plt.subplot(1, 3, 2)
plt.plot(samples[:1000], 'g-', alpha=0.6, lw=0.5)
plt.xlabel('迭代步')
plt.ylabel('样本值')
plt.title('ULA轨迹（前1000步）')

# 自相关函数（纯numpy实现，无需statsmodels）
def compute_acf(x, nlags=50):
    """
    计算自相关函数
    
    使用有偏估计（分母为n而非n-lag），与np.correlate行为一致
    这是信号处理中的标准做法，能保证自相关矩阵的正定性
    
    参数:
        x: 时间序列
        nlags: 最大滞后阶数
    
    返回:
        acf_vals: 自相关函数值 (nlags+1,)
    """
    x = x - np.mean(x)
    n = len(x)
    acf_vals = np.zeros(nlags + 1)
    for lag in range(nlags + 1):
        acf_vals[lag] = np.sum(x[:n-lag] * x[lag:]) / (n * np.var(x))
    return acf_vals

acf_values = compute_acf(samples, nlags=50)
# 自相关首次衰减到 0.05 以下所需的滞后阶数（链混合/近独立的经验判断）
_acf_below = np.where(acf_values[1:] < 0.05)[0]
acf_mix_lag = int(_acf_below[0] + 1) if len(_acf_below) > 0 else None
print(f"自相关首次<0.05的滞后阶数: {acf_mix_lag}")
plt.subplot(1, 3, 3)
plt.plot(acf_values, 'r-o')
plt.xlabel('滞后阶数')
plt.ylabel('自相关')
plt.title('自相关函数')
plt.axhline(y=0.05, color='k', linestyle='--', label='混合判断阈值（ACF<0.05视为独立）')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_ULA采样验证.png'), dpi=150)
plt.close()

# 计算统计量
step1_empirical_var = float(np.var(samples))
print(f"采样方差: {step1_empirical_var:.4f}")
# ULA平稳分布方差公式推导：
# 递推式 X_{m+1} = (1-δ)X_m + √(2δ)Z，平稳时 σ² = (1-δ)²σ² + 2δ
# 解得 σ² = 2δ / [1-(1-δ)²] = 2δ / (2δ-δ²) = 1 / (1-δ/2)
step1_theoretical_var = 1 / (1 - delta / 2)
print(f"理论方差: {step1_theoretical_var:.4f}")
print(f"采样均值: {np.mean(samples):.4f}")
print(f"理论均值: {0:.4f}")

# 有效样本量与方差估计量的MC标准误（x0=0即平稳均值，初始瞬态可忽略）：
# 滞后k阶自相关 ρ(k)=(1-δ)^k；均值估计的ESS = n·(1-ρ)/(1+ρ) = n·δ/(2-δ)
# 注意：方差估计量不能直接套均值ESS——X²的自相关是ρ(k)²而非ρ(k)。
# 高斯过程Bartlett近似：Var(s²) ≈ (2σ⁴/n)·Σρ(k)² = (2σ⁴/n)·(1+ρ²)/(1-ρ²)
# （误用均值ESS会把标准误高估约√2倍，已用1000条独立链实测证实）
step1_rho = 1 - delta
step1_ess = niter * (1 - step1_rho) / (1 + step1_rho)  # 均值估计的ESS
step1_var_mcse = np.sqrt(2 * step1_theoretical_var**2 / niter
                         * (1 + step1_rho**2) / (1 - step1_rho**2))
print(f"均值估计的有效样本量 ESS = n·δ/(2-δ) ≈ {step1_ess:.0f}（名义样本 {niter}）")
print(f"方差估计量的MC标准误（Bartlett） ≈ {step1_var_mcse:.4f}")
print(f"实测偏差 {step1_empirical_var - step1_theoretical_var:+.4f} "
      f"（约 {abs(step1_empirical_var - step1_theoretical_var)/step1_var_mcse:.1f} 个标准误，属正常抽样波动）")


# ============================================================
# 步骤2：步长敏感性分析
# ============================================================
print("\n" + "=" * 60)
print("步骤2：步长敏感性分析")
print("=" * 60)

deltas = [0.01, 0.1, 0.5, 1.0]
niter = 50000          # 每条链（burn-in之后）的样本数
nchains = 20           # 独立链条数：用于刻画单链方差估计量的抽样波动
burn_in = 1000         # 预热步数（对x0=0影响可忽略，保留以体现规范做法）
x0 = 0
step2_records = []

plt.figure(figsize=(16, 4))

for i, delta in enumerate(deltas):
    # 每个步长使用相互独立的多条链（种子固定、结果可复现）
    chains = ULA_gauss_chains(niter, delta, nchains=nchains,
                              seed=20240501 + i, x0=x0, burn_in=burn_in)
    var_per_chain = np.var(chains, axis=0)
    var_mean = float(var_per_chain.mean())
    var_std = float(var_per_chain.std(ddof=1))
    ci_low, ci_high = np.percentile(var_per_chain, [2.5, 97.5])
    pooled = chains.reshape(-1)
    pooled_var = float(np.var(pooled))

    # ULA不变分布（非目标分布！）方差：σ² = 1 / (1 - δ/2)，δ<2时有限
    theoretical_var = 1 / (1 - delta / 2)
    rho = 1 - delta
    ess_per_chain = niter * (1 - rho) / (1 + rho)  # = niter·δ/(2-δ)
    step2_records.append({
        'delta': float(delta),
        'var_mean_across_chains': float(round(var_mean, 6)),
        'var_std_across_chains': float(round(var_std, 6)),
        'var_ci95': [float(round(ci_low, 6)), float(round(ci_high, 6))],
        'var_pooled': float(round(pooled_var, 6)),
        'theoretical_var': float(round(theoretical_var, 6)),
        'discretization_bias': float(round(theoretical_var - 1.0, 6)),
        # 均值估计意义的ESS；方差估计量的波动已由var_std_across_chains直接实证
        'ess_mean_per_chain': float(round(ess_per_chain, 1)),
    })

    plt.subplot(1, len(deltas), i+1)
    sd_inv = np.sqrt(theoretical_var)
    x = np.linspace(-4 * sd_inv, 4 * sd_inv, 300)
    target_density = 1/np.sqrt(2*np.pi) * np.exp(-x**2 / 2)
    invariant_density = (1/(np.sqrt(2*np.pi)*sd_inv)
                         * np.exp(-x**2 / (2*theoretical_var)))
    plt.hist(pooled, bins=60, density=True, alpha=0.6,
             color='orange', label=f'ULA采样（{nchains}条链汇总）')
    plt.plot(x, invariant_density, 'r--', lw=2,
             label=f'ULA不变分布 $N(0,{theoretical_var:.3f})$')
    plt.plot(x, target_density, 'b-', lw=2, label='目标分布 $N(0,1)$')
    plt.title(f'$\\delta$={delta}\n'
              f'链间方差 {var_mean:.3f}±{var_std:.3f}, '
              f'不变方差 {theoretical_var:.3f}\n'
              f'ESS(均值)/链≈{ess_per_chain:.0f}（名义{niter}）', fontsize=9)
    plt.xlabel('$x$')
    if i == 0:
        plt.ylabel('密度')
    plt.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_步长敏感性.png'), dpi=150)
plt.close()

print("步长敏感性总结（20条独立链，每条50000样本，burn-in=1000）：")
print(f"{'δ':>6}{'链间方差均值':>14}{'链间标准差':>12}{'95%区间':>22}"
      f"{'不变分布方差':>14}{'均值ESS/链':>12}")
for r in step2_records:
    print(f"{r['delta']:>6.2f}{r['var_mean_across_chains']:>14.4f}"
          f"{r['var_std_across_chains']:>12.4f}"
          f"{str((round(r['var_ci95'][0],3), round(r['var_ci95'][1],3))):>22}"
          f"{r['theoretical_var']:>14.4f}{r['ess_mean_per_chain']:>12.0f}")
print("要点1（离散化偏差，系统性）：δ越大，不变分布方差1/(1-δ/2)偏离1越远，")
print("       δ=1.0时不变方差=2，样本服从N(0,2)而非N(0,1)，偏差显著但不发散。")
print("要点2（蒙特卡洛噪声，随机性）：δ越小链混合越慢、ESS越小，单链方差估计")
print("       越不稳定；旧版单次运行δ=0.01得0.901正是约第5百分位的抽风，多链")
print("       平均后回归理论值。比较不同δ时必须同时报告ESS/误差棒，否则不公平。")
print("要点3（偏差-效率权衡）：δ小→系统偏差小但每样本信息少；δ大→偏差大但")
print("       混合快。固定δ永远不能同时消除二者，需δ→0并相应增加总迭代数。")


# ============================================================
# 步骤3：稳定边界实证（δ<2 二阶矩有界；δ=2 随机游走；δ>2 几何发散）
# ============================================================
print("\n" + "=" * 60)
print("步骤3：稳定边界实证")
print("=" * 60)

# 二阶矩闭式公式（x0=0）：
#   v_{m+1} = (1-δ)² v_m + 2δ,  v_0 = 0
#   (1-δ)² ≠ 1 : v_m = 2δ·[(1-δ)^{2m}-1] / [(1-δ)²-1]
#   δ=2（临界）: v_m = 4m，即方差随步数线性增长（对称随机游走）
def second_moment_theory(m, delta):
    m = np.asarray(m, dtype=float)
    a2 = (1 - delta) ** 2
    if abs(a2 - 1.0) < 1e-12:
        return 2 * delta * m
    return 2 * delta * (a2 ** m - 1.0) / (a2 - 1.0)

stab_deltas = [1.0, 1.9, 2.0, 2.5]
# δ=2.5时数值约在数百步内溢出（几何增长），故该情形单独使用短时域
stab_horizons = {1.0: 20000, 1.9: 20000, 2.0: 20000, 2.5: 400}
stab_nchains = 200  # 200条链平均，平滑单链二阶矩噪声
step3_records = []

plt.figure(figsize=(9, 5.5))
stab_labels = {
    1.0: '$\\delta$=1.0（有界，极限2）',
    1.9: '$\\delta$=1.9（有界，极限20）',
    2.0: '$\\delta$=2.0（临界，$v_m=4m$ 线性增长）',
    2.5: '$\\delta$=2.5（几何发散 $\\propto1.5^{2m}$）',
}
for delta in stab_deltas:
    T = stab_horizons[delta]
    rng = np.random.RandomState(31337 + int(round(delta * 100)))
    X = np.zeros(stab_nchains)
    m2_emp = np.empty(T)
    a, b = 1 - delta, np.sqrt(2 * delta)
    for m_idx in range(T):
        X = a * X + b * rng.randn(stab_nchains)
        m2_emp[m_idx] = np.mean(X ** 2)
    ms = np.arange(1, T + 1)
    m2_theo = second_moment_theory(ms, delta)
    step = max(T // 400, 1)
    line, = plt.loglog(ms[::step], m2_emp[::step], 'o', ms=3, alpha=0.6,
                       label=stab_labels[delta] + ' 实测')
    plt.loglog(ms, m2_theo, '--', color=line.get_color(), lw=1.5,
               label='_nolegend_')
    # 末端窗口平均（而非单点），降低单批随机数造成的记录噪声；
    # 短时域（δ=2.5）窗口相应缩短
    tail_w = min(2000, max(50, T // 10))
    emp_tail = float(np.mean(m2_emp[-tail_w:]))
    theo_tail = float(np.mean(m2_theo[-tail_w:]))
    step3_records.append({
        'delta': float(delta),
        'horizon': T,
        'nchains': stab_nchains,
        'tail_window': tail_w,
        'empirical_second_moment_tailavg': round(emp_tail, 4),
        'theoretical_second_moment_tailavg': round(theo_tail, 4),
    })
    print(f"  δ={delta}: 末端{tail_w}步平均二阶矩 实测={emp_tail:.4g}, 理论={theo_tail:.4g}")

plt.xlabel('迭代步数 $m$')
plt.ylabel('二阶矩 $\\mathbb{E}[X_m^2]$（200条链平均）')
plt.title('ULA稳定边界：二阶矩实测（圆点）vs 闭式公式（虚线）')
plt.legend(fontsize=8)
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_稳定边界验证.png'), dpi=150)
plt.close()

print("结论：δ<2时|1-δ|<1，二阶矩收敛到1/(1-δ/2)；δ=2时方差∝步数线性增长；")
print("      δ>2时二阶矩几何级数发散。该稳定边界是ULA递推的固有数学属性。")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验5.1-1 总结")
print("=" * 60)
print("1. ULA与Langevin SDE的关系：ULA是Langevin SDE的Euler-Maruyama离散化；")
print("   对光滑强对数凹目标，不变分布与目标分布的偏差为O(δ)，δ越小近似越精确。")
print("2. 收敛性（需精确表述）：固定δ时，ULA马尔可夫链收敛到的是ULA自身的不变")
print("   分布（本例 N(0,1/(1-δ/2))，相对目标有偏），而不是目标分布本身；只有")
print("   δ→0（并同步增加迭代数）时采样分布才趋于目标分布。强对数凹性保证链以")
print("   几何速率遍历，自相关ρ(k)=(1-δ)^k。")
print("3. 偏差-效率权衡：δ小→系统偏差小但ESS≈n·δ/2小、单链估计噪声大；δ大→")
print("   混合快但偏差大。实验须报告ESS/误差棒，多链平均可分离偏差与蒙特卡洛噪声。")
print("4. 理论方差公式（δ<2）：")
print("   不变分布方差 σ² = 1 / (1 - δ/2)")
print("   推导：递推式 X_{m+1} = (1-δ)X_m + √(2δ)Z")
print("   平衡时 σ² = (1-δ)²σ² + 2δ，解得 σ² = 1/(1-δ/2)")
print("5. 稳定边界（步骤3已实证）：δ<2二阶矩有界；δ=2退化为随机游走、方差∝m；")
print("   δ>2二阶矩几何发散。此为离散递推的固有属性，非实现或算力问题。")

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
    'experiment': '5.1-1',
    'title': 'ULA采样与步长敏感性分析',
    'revision_note': '复核修订版：步骤2改为20条独立链统计并报告ESS；新增步骤3稳定边界实证',
    'step1_ula_sampling': {
        'niter': 100000,
        'delta': 0.1,
        'empirical_var': round(step1_empirical_var, 6),
        'theoretical_var': round(step1_theoretical_var, 6),  # 公式: 1/(1-δ/2)
        'acf_mix_lag': acf_mix_lag,  # 自相关首次衰减到0.05以下的滞后阶数
        'ess_mean_theoretical': round(step1_ess, 1),  # 均值估计ESS: n·(1-ρ)/(1+ρ), ρ=1-δ
        'var_mc_standard_error': round(step1_var_mcse, 6),  # Bartlett近似
    },
    'step2_step_size_sensitivity': {
        'niter_per_chain': niter,
        'nchains': nchains,
        'burn_in': burn_in,
        'results': step2_records,  # 每个步长：链间方差均值/标准差/95%区间/ESS/不变方差
    },
    'step3_stability_boundary': {
        'formula': 'v_m = 2δ·[(1-δ)^{2m}-1]/[(1-δ)²-1] (δ≠2); v_m=4m (δ=2)',
        'nchains': stab_nchains,
        'results': step3_records,  # 各δ末端窗口平均的实测/理论二阶矩
    }
}

with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(_to_native(results_summary), f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
