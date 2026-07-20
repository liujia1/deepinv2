"""
实验4.3-1 1D高斯ULA采样与渐近方差验证
对应章节：4.3 ULA——Langevin采样的Euler离散
知识点：ULA递推式；AR(1)过程与平稳方差；渐近方差公式 Var_ULA = 2/(2-delta)；
        偏差的精确量化；步长delta对偏差的影响

修改说明：
  从原参考实验4.1.py迁移，聚焦4.3节ULA核心概念，
  增加AR(1)理论推导与渐近方差公式的数值验证。
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
# 3. 步骤2：渐近方差验证
# ══════════════════════════════════════════════════════════
# AR(1) 平稳方差公式: sigma^2_ULA = sigma_eps^2 / (1 - phi^2)
# 代入 phi = 1-delta, sigma_eps^2 = 2*delta:
#   sigma^2_ULA = 2*delta / (1 - (1-delta)^2) = 2*delta / (2*delta - delta^2) = 2 / (2-delta)

def compute_variance_stats(delta, niter=int(5e4)):
    """计算给定 delta 下的经验方差和理论方差"""
    Y, var_emp, mean_emp = ULA_gauss(niter, delta, x0=0)
    var_theo = 2 / (2 - delta)
    return var_emp, var_theo, mean_emp

print("=" * 60)
print("实验4.3-1 1D高斯ULA采样与渐近方差验证")
print("=" * 60)
print(f"\n[步骤1] ULA直方图 vs 真实密度 (delta={delta})")
print(f"  ULA采样 {niter} 步，直方图已保存")

print(f"\n[步骤2] 渐近方差公式验证: " + r"Var_ULA = 2/(2-delta)")
print(f"  注：均值无偏（ULA平稳分布均值为0），方差有偏")
print("")
print(f"  {'delta':>8} {'empirical':>12} {'theoretical':>12} {'rel. error':>11}  {'mean':>10}")
print(f"  {'':>8} {'variance':>12} {'variance':>12} {'(%)':>11}  {'(unbiased)':>10}")
print("  " + "-" * 60)

deltas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
emp_vars = []
theo_vars = []
for d in deltas:
    v1, v2, mean_emp = compute_variance_stats(d)
    emp_vars.append(v1)
    theo_vars.append(v2)
    rel_err = abs(v1 - v2) / v2 * 100
    print(f"  {d:>8.2f} {v1:>12.6f} {v2:>12.6f} {rel_err:>10.2f}%  {mean_emp:>10.4f}")

print("")
print("  结论：均值接近0（无偏），方差随 delta 增大而增大（有偏）")

# ══════════════════════════════════════════════════════════
# 4. 步骤3：渐近方差随步长变化
# ══════════════════════════════════════════════════════════
# 限制 delta_fine 到 1.5，使曲线在 ylim(0.9, 4) 范围内完整可见
delta_fine = np.linspace(0.001, 1.5, 200)
var_theo_fine = 2 / (2 - delta_fine)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：渐近方差 vs 步长
axes[0].plot(delta_fine, var_theo_fine, 'b-', linewidth=2, label=r'理论值 $\frac{2}{2-\delta}$')
axes[0].axhline(y=1, color='r', linestyle='--', linewidth=1.5, label=r'真实方差 $\sigma^2=1$')
axes[0].scatter(deltas, emp_vars, color='green', s=60, zorder=5, label='经验值')
axes[0].scatter(deltas, theo_vars, color='orange', marker='x', s=60, zorder=5, label='理论值（离散点）')
# 标注发散边界
axes[0].axvline(x=2, color='purple', linestyle=':', linewidth=1.5, label=r'$\delta=2$ 发散边界')
axes[0].set_xlabel(r'步长 $\delta$', fontsize=12)
axes[0].set_ylabel(r'渐近方差 $\sigma^2_{\mathrm{ULA}}$', fontsize=12)
axes[0].set_title(r'步骤3: 渐近方差 vs 步长', fontsize=12)
axes[0].legend(fontsize=9)
axes[0].set_ylim(0.9, 4)
axes[0].set_xlim(0, 2.2)  # 留出空间显示发散边界
axes[0].grid(True, alpha=0.3)

# 右图：偏差 vs 步长
bias = delta_fine / (2 - delta_fine)
axes[1].plot(delta_fine, bias, 'b-', linewidth=2, label=r'偏差 $\frac{\delta}{2-\delta}$')
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='零偏差')
axes[1].scatter(deltas, [2/(2-d) - 1 for d in deltas], color='orange', marker='x', s=60, zorder=5,
                label='理论值（离散点）')
axes[1].axvline(x=2, color='purple', linestyle=':', linewidth=1.5, label=r'$\delta=2$ 发散边界')
axes[1].set_xlabel(r'步长 $\delta$', fontsize=12)
axes[1].set_ylabel(r'方差偏差 $\sigma^2_{\mathrm{ULA}} - 1$', fontsize=12)
axes[1].set_title(r'步骤3: 偏差 vs 步长', fontsize=12)
axes[1].legend(fontsize=9)
axes[1].set_xlim(0, 2.2)
axes[1].set_ylim(-0.1, 4)  # 与左图显示范围对应
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step3_variance_and_bias.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════
# 5. 步骤4：边界步长行为 delta=1
# ══════════════════════════════════════════════════════════
delta_boundary = 1.0
Y_bdy, var_bdy, mean_bdy = ULA_gauss(int(5e4), delta_boundary, x0=0)
var_theo_bdy = 2 / (2 - delta_boundary)

# 单独定义 x_grid，避免使用步骤1的截断范围
x_fine = np.linspace(-5, 5, 200)
y_true = np.exp(-x_fine**2 / 2) / np.sqrt(2 * np.pi)
y_ula = np.exp(-x_fine**2 / (2 * var_theo_bdy)) / np.sqrt(2 * np.pi * var_theo_bdy)

plt.figure(figsize=(8, 5))
plt.plot(x_fine, y_true, 'b-', linewidth=2, label=r'真实密度 $\mathcal{N}(0,1)$')
plt.hist(Y_bdy, bins=100, range=[-5, 5], density=True, alpha=0.6,
         label=rf'ULA样本 ($\delta=1.0$, 边界步长)')
plt.plot(x_fine, y_ula, 'g--', linewidth=2,
         label=rf'ULA平稳分布 $\mathcal{{N}}(0, {var_theo_bdy:.0f})$')
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'密度', fontsize=12)
plt.title(rf'步骤4: 边界步长 ($\delta=1.0$, 方差={var_theo_bdy:.1f}$\times$真实值)', fontsize=12)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(SAVE_DIR, 'step4_boundary_step.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[步骤3] 渐近方差与偏差随步长变化曲线已保存")
print(f"  注：图中紫色虚线标注 $\delta=2$ 发散边界")
print(f"\n[步骤4] 边界步长行为 (delta=1.0)")
print(f"  理论方差 = 2/(2-1) = {var_theo_bdy:.1f} (真实方差的2倍)")
print(f"  经验方差 = {var_bdy:.4f}")
print(f"  经验均值 = {mean_bdy:.4f} (接近0，均值无偏)")

print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print(r"1. ULA在1D标准高斯下退化为AR(1)过程: X_{m+1} = (1-delta)*X_m + sqrt(2*delta)*Z")
print(r"2. AR(1)平稳方差公式: Var_ULA = sigma_eps^2 / (1-phi^2) = 2/(2-delta)")
print(r"3. 均值无偏（ULA平稳分布均值为0），方差有偏")
print(r"4. 偏差 = Var_ULA - 1 = delta/(2-delta) ~ delta/2 (当delta较小时)")
print(r"5. delta->0: 偏差消失(无偏); delta=1: 方差翻倍(边界); delta>=2: ULA发散")
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
    'title': '1D高斯ULA采样与渐近方差验证',
    'step2_variance_verification': {
        'deltas': deltas,
        'empirical_vars': [float(round(v, 6)) for v in emp_vars],
        'theoretical_vars': [float(round(v, 6)) for v in theo_vars],  # 理论公式: 2/(2-delta)
    },
    'step4_boundary_step': {
        'delta': delta_boundary,
        'empirical_var': float(round(var_bdy, 6)),
        'theoretical_var': float(round(var_theo_bdy, 6)),
        'empirical_mean': float(round(mean_bdy, 4)),
    }
}

with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(_to_native(results_summary), f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
