"""
实验4.3-1 1D高斯ULA采样与平稳方差验证
对应章节：4.3 ULA——Langevin采样的Euler离散
知识点：ULA递推式；AR(1)过程与不变分布；平稳（边际）方差公式 Var_pi = 2/(2-delta)；
        偏差的精确量化；步长delta对偏差的影响；
        术语辨析：平稳方差 2/(2-delta)（衡量分布偏差）
                  vs 遍历均值的渐近（长程）方差 2/delta（衡量混合效率）；
        有效样本量 ESS = n*delta/(2-delta)

修改说明：
  从原参考实验4.1.py迁移，聚焦4.3节ULA核心概念，
  增加AR(1)理论推导与平稳方差公式的数值验证。
  复核修订：
    (1) “渐近方差”改称“平稳（边际）方差”，并补充长程方差 2/delta 与 ESS 的
        术语辨析和数值验证（分批均值法）；
    (2) delta=1 不再称“边界步长”（ULA 的发散边界是 delta=2），改称“大步长
        特例”：phi=1-delta=0，链一步去相关（i.i.d.~N(0,2)），混合最快但仍有偏；
    (3) 输出图与步骤编号对齐：step1 直方图 / step2 方差·偏差·ESS / step3 大步长特例。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    SAVE_DIR = os.path.join(_gdrive, '实验4.3-1')
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

np.random.seed(42)

# ══════════════════════════════════════════════════════════
# 1. ULA_gauss 函数
# ══════════════════════════════════════════════════════════
# 从1D标准高斯 N(0,1) 中用ULA采样
# 势能 U(x)=x^2/2, nabla U(x)=x
# ULA: X_{m+1} = X_m - delta * nabla U(X_m) + sqrt(2*delta) * Z_{m+1}
#                  = (1-delta) * X_m + sqrt(2*delta) * Z_{m+1}
# 这是一个 AR(1) 过程，自回归系数 phi = 1-delta，噪声方差 sigma_eps^2 = 2*delta

def ULA_gauss(niter, delta, x0=0, burn_in=1000):
    """
    ULA 采样器（含 burn-in）

    参数:
        niter: 采样数（不含 burn-in）
        delta: 步长（必须 < 2，否则 AR(1) 发散）
        x0: 初始状态
        burn_in: 预烧期样本数

    返回:
        Y: 采样结果（不含 burn-in）
        var_Y: 经验方差
        mean_Y: 经验均值

    异常:
        ValueError: 当 delta >= 2 时抛出
    """
    if delta >= 2:
        raise ValueError(f"delta={delta} >= 2，ULA 发散，请使用 delta < 2")

    X = x0
    # Burn-in 阶段：不记录样本
    for _ in range(burn_in):
        X = (1 - delta) * X + np.sqrt(2 * delta) * np.random.randn()

    # 采样阶段
    Y = np.zeros(niter)
    for i in range(niter):
        X = (1 - delta) * X + np.sqrt(2 * delta) * np.random.randn()
        Y[i] = X

    return Y, np.var(Y), np.mean(Y)


# ══════════════════════════════════════════════════════════
# 2. 步骤1：ULA直方图 vs 真实密度
# ══════════════════════════════════════════════════════════
x0 = 0
niter = int(5e4)  # 降至 5e4 以减少运行时间
delta = 0.1

Y, _, _ = ULA_gauss(niter, delta, x0)

x = np.linspace(-3, 3, 100)
y = 1 / (np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)

plt.figure(figsize=(8, 5))
plt.plot(x, y, 'b-', linewidth=2, label=r'真实密度 $\mathcal{N}(0,1)$')
plt.hist(Y, bins=100, range=[-3, 3], density=True, alpha=0.6, label=rf'ULA样本 ($\delta={delta}$)')
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'密度', fontsize=12)
plt.title(r'步骤1: ULA采样 vs 真实密度 ($\delta=0.1$)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(SAVE_DIR, 'step1_ULA_histogram.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════
# 3. 步骤2：平稳（边际）方差验证，附长程方差/ESS的术语辨析与量化
# ══════════════════════════════════════════════════════════
# AR(1) 平稳（边际）方差公式（不变分布 pi_delta ~ N(0, 2/(2-delta)) 的方差）:
# 代入 phi = 1-delta, sigma_eps^2 = 2*delta:
#   sigma^2_pi = 2*delta / (1 - (1-delta)^2) = 2*delta / (2*delta - delta^2) = 2 / (2-delta)
#
# 术语辨析（两个“方差”不可混用）:
#   (1) 平稳（边际）方差 sigma^2_pi = 2/(2-delta)：描述不变分布相对目标 N(0,1) 的偏差；
#   (2) 遍历均值的渐近（长程）方差 sigma^2_lr：即 sqrt(n)*样本均值 的 CLT 方差，
#       AR(1) 下 sigma^2_lr = sigma^2_pi * (1+phi)/(1-phi) = 2/delta，
#       delta 越小它越大——偏差小与混合慢是一对权衡；
#   (3) 有效样本量 ESS = n * sigma^2_pi / sigma^2_lr = n*delta/(2-delta)。
# 长程方差用样本 lag-1 自相关拟合 AR(1) 估计：phi_hat 由链自身估计（不假设已知 delta），
# IAC=(1+phi_hat)/(1-phi_hat), sigma^2_lr = var * IAC, ESS = n/IAC。
# （对高斯 AR(1) 这是精确关系；比小批长的分批均值法偏差更小——delta=0.01 时 IAC≈199，
#   批长需数千才能忽略截断偏差。）

def compute_variance_stats(delta, niter=int(5e4)):
    """计算给定 delta 下经验/理论平稳方差、长程方差（phi_hat拟合）与ESS"""
    Y, var_emp, mean_emp = ULA_gauss(niter, delta, x0=0)
    var_theo = 2 / (2 - delta)
    lrv_theo = 2 / delta                       # 遍历均值的渐近（长程）方差
    ess_theo = niter * delta / (2 - delta)     # 理论有效样本量
    Yc = Y - mean_emp
    phi_hat = np.mean(Yc[:-1] * Yc[1:]) / np.mean(Yc**2)  # 样本 lag-1 自相关
    iac_hat = (1 + phi_hat) / (1 - phi_hat)               # 积分自相关时间
    lrv_emp = var_emp * iac_hat
    ess_emp = niter / iac_hat
    return var_emp, var_theo, mean_emp, lrv_emp, lrv_theo, ess_emp, ess_theo

print("=" * 60)
print("实验4.3-1 1D高斯ULA采样与平稳方差验证")
print("=" * 60)
print(f"\n[步骤1] ULA直方图 vs 真实密度 (delta={delta})")
print(f"  ULA采样 {niter} 步，直方图已保存")

print(f"\n[步骤2] 平稳（边际）方差公式验证: " + r"Var_pi = 2/(2-delta)")
print(f"  注1：均值无偏（ULA不变分布均值为0），平稳方差有偏")
print(f"  注2：术语辨析——遍历均值的渐近（长程）方差为 2/delta（delta 越小越大），")
print(f"        有效样本量 ESS = n*delta/(2-delta)，下表 ESS 由样本 phi_hat 拟合估计")
print("")
print(f"  {'delta':>7} {'emp var':>10} {'theo var':>10} {'rel.err':>8} {'mean':>8} {'ESS emp':>9} {'ESS theo':>9}")
print("  " + "-" * 68)

deltas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
emp_vars = []
theo_vars = []
emp_lrvs, theo_lrvs, emp_ess, theo_ess = [], [], [], []
for d in deltas:
    v1, v2, mean_emp, lrv_e, lrv_t, ess_e, ess_t = compute_variance_stats(d)
    emp_vars.append(v1)
    theo_vars.append(v2)
    emp_lrvs.append(lrv_e)
    theo_lrvs.append(lrv_t)
    emp_ess.append(ess_e)
    theo_ess.append(ess_t)
    rel_err = abs(v1 - v2) / v2 * 100
    print(f"  {d:>7.2f} {v1:>10.6f} {v2:>10.6f} {rel_err:>7.2f}% {mean_emp:>8.4f} {ess_e:>9.0f} {ess_t:>9.0f}")

print("")
print("  结论：均值接近0（无偏）；平稳方差随 delta 增大而增大（有偏）；")
print("        ESS 随 delta 减小而骤降（delta=0.01 时仅约 250/50000）——偏差与混合效率的权衡")

# ══════════════════════════════════════════════════════════
# 4. 步骤2（续）：平稳方差、偏差与ESS随步长变化
# ══════════════════════════════════════════════════════════
# 限制 delta_fine 到 1.5，使前两幅图的曲线在纵轴范围内完整可见
delta_fine = np.linspace(0.001, 1.5, 200)
var_theo_fine = 2 / (2 - delta_fine)

fig, axes = plt.subplots(1, 3, figsize=(19, 5))

# 左图：平稳（边际）方差 vs 步长
axes[0].plot(delta_fine, var_theo_fine, 'b-', linewidth=2, label=r'理论值 $\frac{2}{2-\delta}$')
axes[0].axhline(y=1, color='r', linestyle='--', linewidth=1.5, label=r'真实方差 $\sigma^2=1$')
axes[0].scatter(deltas, emp_vars, color='green', s=60, zorder=5, label='经验值')
axes[0].scatter(deltas, theo_vars, color='orange', marker='x', s=60, zorder=5, label='理论值（离散点）')
# 标注发散边界
axes[0].axvline(x=2, color='purple', linestyle=':', linewidth=1.5, label=r'$\delta=2$ 发散边界')
axes[0].set_xlabel(r'步长 $\delta$', fontsize=12)
axes[0].set_ylabel(r'平稳方差 $\sigma^2_{\pi_\delta}=\frac{2}{2-\delta}$（不变分布边际方差）', fontsize=11)
axes[0].set_title(r'步骤2: 平稳方差 vs 步长', fontsize=12)
axes[0].legend(fontsize=9)
axes[0].set_ylim(0.9, 4)
axes[0].set_xlim(0, 2.2)  # 留出空间显示发散边界
axes[0].grid(True, alpha=0.3)

# 右图：偏差 vs 步长
bias = delta_fine / (2 - delta_fine)
axes[1].plot(delta_fine, bias, 'b-', linewidth=2, label=r'偏差 $\frac{\delta}{2-\delta}$')
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='零偏差')
axes[1].scatter(deltas, [v - 1 for v in emp_vars], color='green', s=60, zorder=5, label='经验值')
axes[1].scatter(deltas, [2/(2-d) - 1 for d in deltas], color='orange', marker='x', s=60, zorder=5,
                label='理论值（离散点）')
axes[1].axvline(x=2, color='purple', linestyle=':', linewidth=1.5, label=r'$\delta=2$ 发散边界')
axes[1].set_xlabel(r'步长 $\delta$', fontsize=12)
axes[1].set_ylabel(r'方差偏差 $\sigma^2_{\pi_\delta} - 1$', fontsize=12)
axes[1].set_title(r'步骤2: 偏差 vs 步长', fontsize=12)
axes[1].legend(fontsize=9)
axes[1].set_xlim(0, 2.2)
axes[1].set_ylim(-0.1, 4)  # 与左图显示范围对应
axes[1].grid(True, alpha=0.3)

# 右图之二：有效样本量 ESS vs 步长（量化混合效率；经验值由样本 phi_hat 拟合）
ess_fine = niter * delta_fine / (2 - delta_fine)
axes[2].plot(delta_fine, ess_fine, 'b-', linewidth=2, label=r'理论 ESS $=n\frac{\delta}{2-\delta}$')
axes[2].scatter(deltas, emp_ess, color='green', s=60, zorder=5, label=r'经验值（样本 $\hat{\phi}$ 拟合）')
axes[2].scatter(deltas, theo_ess, color='orange', marker='x', s=60, zorder=5, label='理论值（离散点）')
axes[2].axhline(niter, color='gray', linestyle=':', linewidth=1.2, label=r'独立样本上界 $n$')
axes[2].annotate(r'$\delta=0.01$: ESS$\approx$251' + '\n' + r'（偏差0.5%，但链高度相关）',
                 xy=(0.01, theo_ess[0]), xytext=(0.17, 11000),
                 fontsize=9, color='darkgreen',
                 arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
axes[2].annotate(r'$\delta=1$: ESS$\approx n$' + '\n' + r'（$\phi=0$，一步独立）',
                 xy=(1.0, theo_ess[-1]), xytext=(0.52, 41000),
                 fontsize=9, color='darkgreen',
                 arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
axes[2].set_xlabel(r'步长 $\delta$', fontsize=12)
axes[2].set_ylabel(r'有效样本量 ESS（$n=5\times10^4$）', fontsize=12)
axes[2].set_title(r'步骤2: ESS vs 步长（混合效率）', fontsize=12)
axes[2].legend(fontsize=9, loc='upper left')
axes[2].set_xlim(0, 1.6)
axes[2].set_ylim(0, 60000)
axes[2].grid(True, alpha=0.3)
axes[2].text(0.98, 0.04, r'注: $\delta>1$ 时 $\phi<0$ 负相关，名义 ESS 可超过 $n$，但分布偏差已很大',
             transform=axes[2].transAxes, ha='right', va='bottom', fontsize=8, color='dimgray')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step2_variance_bias_ess.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════
# 5. 步骤3：大步长特例 delta=1（phi=0，一步去相关）
# ══════════════════════════════════════════════════════════
# 术语澄清：delta=1 不是 ULA 的发散边界（发散边界是 delta=2，详见实验4.3-2）。
# 它的特殊性是 phi=1-delta=0：X_{m+1} 与 X_m 独立，样本 i.i.d.~N(0,2)，
# 链混合最快（ESS=n），但不变分布仍是方差翻倍的 N(0,2)——“混合快”不等于“无偏”。
delta_large = 1.0
Y_large, var_large, mean_large = ULA_gauss(int(5e4), delta_large, x0=0)
var_theo_large = 2 / (2 - delta_large)

# 单独定义 x_grid，避免使用步骤1的截断范围
x_fine = np.linspace(-5, 5, 200)
y_true = np.exp(-x_fine**2 / 2) / np.sqrt(2 * np.pi)
y_ula = np.exp(-x_fine**2 / (2 * var_theo_large)) / np.sqrt(2 * np.pi * var_theo_large)

plt.figure(figsize=(8, 5))
plt.plot(x_fine, y_true, 'b-', linewidth=2, label=r'真实密度 $\mathcal{N}(0,1)$')
plt.hist(Y_large, bins=100, range=[-5, 5], density=True, alpha=0.6,
         label=rf'ULA样本 ($\delta=1.0$, 大步长 $\phi=0$)')
plt.plot(x_fine, y_ula, 'g--', linewidth=2,
         label=rf'ULA不变分布 $\mathcal{{N}}(0, {var_theo_large:.0f})$')
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'密度', fontsize=12)
plt.title(rf'步骤3: 大步长特例 ($\phi=0$ 一步独立, 方差={var_theo_large:.1f}$\times$真实值; 发散边界为$\delta=2$)', fontsize=12)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(SAVE_DIR, 'step3_large_step_phi0.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[步骤2] 平稳方差、偏差与ESS随步长变化曲线已保存")
print(f"  注：图中紫色虚线标注 delta=2 发散边界；右图为有效样本量（混合效率）")
print(f"\n[步骤3] 大步长特例 (delta=1.0, phi=0)")
print(f"  理论平稳方差 = 2/(2-1) = {var_theo_large:.1f} (真实方差的2倍)")
print(f"  经验方差 = {var_large:.4f}")
print(f"  经验均值 = {mean_large:.4f} (接近0，均值无偏)")
print(f"  注：phi=0 时样本一步去相关（i.i.d.~N(0,2)）；delta=1 不是发散边界，发散边界为 delta=2")

print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print(r"1. ULA在1D标准高斯下退化为AR(1)过程: X_{m+1} = (1-delta)*X_m + sqrt(2*delta)*Z")
print(r"2. AR(1)平稳（边际）方差: Var_pi = sigma_eps^2 / (1-phi^2) = 2/(2-delta)")
print(r"   注意：这是不变分布的边际方差，不要与遍历均值的渐近(长程)方差混用")
print(r"3. 均值无偏（不变分布均值为0），平稳方差有偏（相对目标 N(0,1)）")
print(r"4. 偏差 = Var_pi - 1 = delta/(2-delta) ~ delta/2 (当delta较小时)")
print(r"5. delta->0: 偏差消失; delta=1: phi=0, 样本一步独立(i.i.d.~N(0,2))、混合最快但方差翻倍;")
print(r"   delta=1 不是发散边界; delta>=2: |1-delta|>=1, ULA发散")
print(r"6. 遍历均值的渐近（长程）方差=2/delta, ESS = n*delta/(2-delta):")
print(r"   delta 越小偏差越小，但 ESS 越低、混合越慢——偏差与效率必须权衡")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")
print(f"\n【运行时间说明】")
print(f"  本实验总迭代约 6×5e4 + 2×5e4 = 4e5 步，运行约 5-10 秒")

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
    'experiment': '4.3-1',
    'title': '1D高斯ULA采样与平稳方差验证',
    'terminology_note': ('2/(2-delta) 为ULA不变分布的平稳（边际）方差（衡量分布偏差）；'
                         '2/delta 为遍历均值的渐近（长程）方差（衡量混合效率）；'
                         'ESS = n*delta/(2-delta)，经验长程方差由样本 lag-1 自相关 phi_hat 拟合估计'),
    'step2_stationary_variance': {
        'niter': int(5e4),
        'deltas': deltas,
        'empirical_vars': [float(round(v, 6)) for v in emp_vars],          # 经验平稳方差
        'theoretical_vars': [float(round(v, 6)) for v in theo_vars],       # 理论 2/(2-delta)
        'empirical_long_run_vars': [float(round(v, 4)) for v in emp_lrvs],  # phi_hat 拟合
        'theoretical_long_run_vars': [float(round(v, 4)) for v in theo_lrvs],  # 理论 2/delta
        'empirical_ess': [float(round(v, 1)) for v in emp_ess],
        'theoretical_ess': [float(round(v, 1)) for v in theo_ess],
    },
    'step3_large_step_phi0': {
        'delta': delta_large,
        'phi': 0.0,
        'note': 'delta=1 为大步长特例（phi=0，样本一步去相关 i.i.d.~N(0,2)），不是发散边界；发散边界为 delta=2',
        'empirical_var': float(round(var_large, 6)),
        'theoretical_var': float(round(var_theo_large, 6)),
        'empirical_mean': float(round(mean_large, 4)),
    }
}

with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(_to_native(results_summary), f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
