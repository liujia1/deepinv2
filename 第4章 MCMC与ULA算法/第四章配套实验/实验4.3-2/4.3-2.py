"""
实验4.3-2 ULA步长delta对收敛的影响
对应章节：4.3 ULA——Langevin采样的Euler离散
知识点：ULA稳定性条件（delta<2）；偏差与收敛的权衡；
        渐近方差公式 Var_ULA = 2/(2-delta) 的步长依赖

修改说明：
  从原参考实验4.2.py迁移，聚焦4.3节步长选择与收敛性。
  核心修正：ULA的发散条件是 delta>=2（AR(1)稳定性 |1-delta|<1），
  而非 delta>1/L（后者是连续时间Langevin的步长上界，对应梯度下降收敛条件）。
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
    SAVE_DIR = os.path.join(_gdrive, '实验4.3-2')
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
# 1. ULA_gauss 函数（含 burn-in 和发散检测）
# ══════════════════════════════════════════════════════════
# 从1D标准高斯 N(0,1) 中用ULA采样
# 势能 U(x)=x^2/2, nabla U(x)=x, Lipschitz常数 L=1
#
# ULA稳定性条件（AR(1)）: |1-delta| < 1，即 0 < delta < 2
# - delta < 1: 正相关游走，方差膨胀
# - 1 < delta < 2: 负相关游走，方差膨胀更严重
# - delta >= 2: 发散

def ULA_gauss(niter, delta, x0=0, burn_in=1000):
    """
    ULA 采样器（含 burn-in 和发散检测）

    参数:
        niter: 采样数（不含 burn-in）
        delta: 步长（必须 < 2，否则 AR(1) 发散）
        x0: 初始状态
        burn_in: 预烧期样本数

    返回:
        Y: 采样结果（不含 burn-in）
        var_Y: 经验方差（发散时返回 inf）
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
        if abs(X) > 1e10:
            # 数值溢出检测（阈值 1e10，避免 δ 接近 2 时误判）
            return np.full(niter, np.nan), float('inf'), float('nan')

    # 采样阶段
    Y = np.zeros(niter)
    for i in range(niter):
        X = (1 - delta) * X + np.sqrt(2 * delta) * np.random.randn()
        if abs(X) > 1e10:
            # 数值溢出检测
            return np.full(niter, np.nan), float('inf'), float('nan')
        Y[i] = X

    return Y, np.var(Y), np.mean(Y)


# ══════════════════════════════════════════════════════════
# 2. 步骤1：不同步长下的ULA行为
# ══════════════════════════════════════════════════════════
n_iter = 50000
L = 1.0
# 展示四档偏差（小、中、大、极端）和接近发散边界的情况
delta_list = [0.01, 0.1, 0.5, 0.8, 1.0, 1.5, 1.9]

# 自适应布局
n = len(delta_list)
ncols = 3
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))

x_grid = np.linspace(-4, 4, 500)
true_pdf = np.exp(-x_grid**2 / 2) / np.sqrt(2 * np.pi)

results = []

for idx, delta in enumerate(delta_list):
    row, col = idx // ncols, idx % ncols
    ax = axes[row][col]

    chain, var_emp, mean_emp = ULA_gauss(n_iter, delta)
    diverged = np.any(np.isnan(chain))

    if not diverged:
        ax.plot(x_grid, true_pdf, 'k-', linewidth=2, label=r'真实密度 $\mathcal{N}(0,1)$')
        ax.hist(chain, bins=80, range=(-4, 4),
                density=True, alpha=0.6, label=r'ULA样本')
        var_theory = 2.0 / (2.0 - delta)
        # 用偏差百分比分档，与教学内容对齐
        # 偏差 = Var_ULA - 1 = delta/(2-delta)
        bias_pct = delta / (2 - delta) * 100  # 偏差百分比
        if bias_pct < 50:
            status = r'小偏差'
        elif bias_pct < 100:
            status = r'中等偏差'
        elif bias_pct < 200:
            status = r'大偏差'
        else:
            status = r'极端偏差'
        ax.set_title(rf'$\delta={delta}$ ({status})' + '\n'
                     + rf'方差={var_emp:.3f} (理论={var_theory:.3f})')
        results.append((delta, var_emp, var_theory, mean_emp, 'converged'))
    else:
        ax.text(0.5, 0.5,
                rf'$\delta={delta} \geq 2$' + '\n发散!',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, color='red', fontweight='bold')
        ax.set_title(rf'$\delta={delta}$: 发散')
        results.append((delta, float('inf'), float('inf'), float('nan'), 'diverged'))

    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'密度', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# 关闭多余的子图
for idx in range(n, nrows * ncols):
    row, col = idx // ncols, idx % ncols
    axes[row][col].axis('off')

fig.suptitle(r'步骤1: ULA行为 vs 步长 ($L=1$, 发散边界 $\delta=2$)',
             fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(SAVE_DIR, 'step1_stepsize_effect.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════
# 3. 步骤2：方差与偏差随步长的变化
# ══════════════════════════════════════════════════════════
# 限制 delta_fine 到 1.5，使曲线在 ylim 范围内完整可见
delta_fine = np.linspace(0.001, 1.5, 300)
var_theo_fine = np.where(delta_fine < 2, 2 / (2 - delta_fine), np.nan)
bias_fine = np.where(delta_fine < 2, delta_fine / (2 - delta_fine), np.nan)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：渐近方差 vs 步长
axes[0].plot(delta_fine, var_theo_fine, 'b-', linewidth=2,
             label=r'理论值 $\frac{2}{2-\delta}$')
axes[0].axhline(y=1, color='r', linestyle='--', linewidth=1.5,
                label=r'真实方差 $\sigma^2=1$')
# 标注发散边界（delta=2）而非"稳定性边界"
axes[0].axvline(x=2, color='purple', linestyle=':', linewidth=1.5,
                label=r'发散边界 $\delta=2$')

# 绘制散点（delta <= 1.5 在 ylim 范围内可见）
for r in results:
    if r[4] == 'converged' and r[0] <= 1.5:
        axes[0].scatter(r[0], r[1], color='green', s=60, zorder=5, label='经验值' if r[0] == 0.01 else '')
        axes[0].scatter(r[0], r[2], color='orange', marker='x', s=60, zorder=5, label='理论值（离散点）' if r[0] == 0.01 else '')

# delta=1.9 的散点超出 ylim，用文字注释标注
# 注意：若修改 delta_list，需同步更新此处的注释内容和坐标
axes[0].annotate(r'$\delta=1.9$: 方差$\approx 20$（超出纵轴范围）',
                 xy=(1.9, 3.9), xytext=(1.0, 3.5),
                 fontsize=9, color='darkgreen',
                 arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

axes[0].set_xlabel(r'步长 $\delta$', fontsize=12)
axes[0].set_ylabel(r'渐近方差 $\sigma^2_{\mathrm{ULA}}$', fontsize=12)
axes[0].set_title(r'步骤2: 方差 vs 步长', fontsize=12)
axes[0].legend(fontsize=9)
axes[0].set_ylim(0.9, 4)
axes[0].set_xlim(0, 2.2)
axes[0].grid(True, alpha=0.3)

# 右图：偏差 vs 步长
axes[1].plot(delta_fine, bias_fine, 'b-', linewidth=2,
             label=r'偏差 $\frac{\delta}{2-\delta}$')
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1.5,
                label='零偏差')
axes[1].axvline(x=2, color='purple', linestyle=':', linewidth=1.5,
                label=r'发散边界 $\delta=2$')

# 绘制经验偏差散点（delta <= 1.5 在 ylim 范围内可见）
for r in results:
    if r[4] == 'converged' and r[0] <= 1.5:
        emp_bias = r[1] - 1  # 经验偏差 = 经验方差 - 1
        theo_bias = r[2] - 1  # 理论偏差 = 理论方差 - 1
        axes[1].scatter(r[0], emp_bias, color='green', s=60, zorder=5, label='经验偏差' if r[0] == 0.01 else '')
        axes[1].scatter(r[0], theo_bias, color='orange', marker='x', s=60, zorder=5, label='理论偏差（离散点）' if r[0] == 0.01 else '')

# delta=1.9 的偏差超出 ylim，用文字注释标注
# 注意：若修改 delta_list，需同步更新此处的注释内容和坐标
axes[1].annotate(r'$\delta=1.9$: 偏差$\approx 19$（超出纵轴范围）',
                 xy=(1.9, 3.9), xytext=(1.0, 3.5),
                 fontsize=9, color='darkgreen',
                 arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

axes[1].set_xlabel(r'步长 $\delta$', fontsize=12)
axes[1].set_ylabel(r'方差偏差 $\sigma^2_{\mathrm{ULA}} - 1$', fontsize=12)
axes[1].set_title(r'步骤2: 偏差 vs 步长', fontsize=12)
axes[1].legend(fontsize=9)
axes[1].set_ylim(-0.1, 4)
axes[1].set_xlim(0, 2.2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step2_variance_bias_vs_delta.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════
# 4. 输出结论
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("实验4.3-2 ULA步长delta对收敛的影响")
print("=" * 60)
print(f"\n[步骤1] 不同步长下的ULA行为 (L={L})")
print(f"  {'delta':>8} {'状态':>10} {'经验方差':>12} {'理论方差':>12} {'经验均值':>10}")
for r in results:
    d, ve, vt, me, status = r
    ve_str = f'{ve:.4f}' if ve != float('inf') else 'inf'
    vt_str = f'{vt:.4f}' if vt != float('inf') else 'inf'
    me_str = f'{me:.4f}' if not np.isnan(me) else 'nan'
    status_cn = '收敛' if status == 'converged' else '发散'
    print(f"  {d:>8.2f} {status_cn:>10} {ve_str:>12} {vt_str:>12} {me_str:>10}")

print(f"\n[步骤2] 方差与偏差随步长变化曲线已保存")

print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print(r"1. ULA稳定性条件: 0 < delta < 2（AR(1) 稳定性 |1-delta| < 1）")
print(r"   - delta < 1: 正相关游走，方差膨胀")
print(r"   - 1 < delta < 2: 负相关游走，方差膨胀更严重")
print(r"   - delta >= 2: 发散，|X_m| -> inf")
print(r"")
print(r"2. 渐近方差公式: Var_ULA = 2/(2-delta)")
print(r"   - delta=0.01: 方差≈1.005（偏差0.5%）")
print(r"   - delta=0.5:  方差≈1.333（偏差33%）")
print(r"   - delta=1.0:  方差=2.0（偏差100%，方差翻倍）")
print(r"   - delta=1.5:  方差=4.0（偏差300%）")
print(r"   - delta=1.9:  方差=20.0（偏差1900%）")
print(r"")
print(r"3. 实践建议:")
print(r"   - 小delta: 偏差小但收敛慢（需要更多迭代）")
print(r"   - 大delta: 收敛快但偏差大")
print(r"   - 通常选择 delta << 1/L 以控制偏差")
print(r"   - delta=1/L 时方差翻倍，但仍收敛")
print(r"")
print(r"4. 注意: delta=1/L=1 是连续时间Langevin的步长上界，")
print(r"   对应梯度下降（零噪声）的收敛条件，不是ULA的发散边界。")
print(r"   ULA的发散边界是 delta=2。")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")
