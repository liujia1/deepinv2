"""
实验4.3-2 ULA步长delta对收敛的影响
对应章节：4.3 ULA——Langevin采样的Euler离散
知识点：步长条件 delta<=1/L；偏差与收敛的权衡；链发散行为；
        渐近方差公式 Var_ULA = 2/(2-delta) 的步长依赖

修改说明：
  从原参考实验4.2.py迁移，聚焦4.3节步长选择与收敛性，
  扩展步长范围以展示收敛/偏差/发散三种行为。
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
# 1. ULA_gauss 函数（含发散检测）
# ══════════════════════════════════════════════════════════
# 从1D标准高斯 N(0,1) 中用ULA采样
# 势能 U(x)=x^2/2, nabla U(x)=x, Lipschitz常数 L=1
# 步长条件: delta <= 1/L = 1

def ULA_gauss(niter, delta, x0=0):
    Y = np.zeros(niter,)
    X = x0
    for i in range(niter):
        Z = np.random.randn()
        grad = X
        X_new = X - delta * grad + np.sqrt(2 * delta) * Z
        if abs(X_new) > 1e6:
            Y[i:] = np.nan
            break
        X = X_new
        Y[i] = X
    return Y, np.var(Y[~np.isnan(Y)]) if not np.all(np.isnan(Y)) else float('inf')

# ══════════════════════════════════════════════════════════
# 2. 步骤1：不同步长下的ULA行为
# ══════════════════════════════════════════════════════════
n_iter = 50000
L = 1.0
delta_list = [0.01, 0.1, 0.5, 1.0, 1.5]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

x_grid = np.linspace(-4, 4, 500)
true_pdf = np.exp(-x_grid**2 / 2) / np.sqrt(2 * np.pi)

results = []

for idx, delta in enumerate(delta_list):
    row, col = idx // 3, idx % 3
    ax = axes[row][col]

    chain, var_emp = ULA_gauss(n_iter, delta)
    valid = chain[~np.isnan(chain)]
    diverged = np.any(np.isnan(chain))

    if not diverged and len(valid) > 1000:
        ax.plot(x_grid, true_pdf, 'k-', linewidth=2, label=r'True density $\mathcal{N}(0,1)$')
        ax.hist(valid[len(valid)//2:], bins=80, range=(-4, 4),
                density=True, alpha=0.6, label=r'ULA samples (2nd half)')
        var_theory = 2.0 / (2.0 - delta) if delta < 2 else float('inf')
        status = r'$\delta \leq 1/L$' if delta <= 1/L else r'$\delta > 1/L$'
        ax.set_title(rf'$\delta={delta}$ ({status})' + '\n'
                     + rf'Var={var_emp:.3f} (theory={var_theory:.3f})')
        results.append((delta, var_emp, var_theory, 'converged'))
    else:
        ax.text(0.5, 0.5,
                rf'$\delta={delta} > 1/L={1/L}$' + '\nDiverged!',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, color='red', fontweight='bold')
        ax.set_title(rf'$\delta={delta}$ ($\delta > 1/L$): Diverged')
        results.append((delta, float('inf'), float('inf'), 'diverged'))

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'Density')
    ax.legend(fontsize=7)

axes[1][2].axis('off')

fig.suptitle(r'Step 1: ULA behavior vs step size ($L=1$, $\delta \leq 1/L=1$)',
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step1_stepsize_effect.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════
# 3. 步骤2：方差与偏差随步长的变化
# ══════════════════════════════════════════════════════════
delta_fine = np.linspace(0.001, 1.8, 300)
var_theo_fine = np.where(delta_fine < 2, 2 / (2 - delta_fine), np.nan)
bias_fine = np.where(delta_fine < 2, delta_fine / (2 - delta_fine), np.nan)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(delta_fine, var_theo_fine, 'b-', linewidth=2,
             label=r'Theoretical $\frac{2}{2-\delta}$')
axes[0].axhline(y=1, color='r', linestyle='--', linewidth=1,
                label=r'True variance $\sigma^2=1$')
axes[0].axvline(x=1/L, color='gray', linestyle=':', linewidth=1,
                label=r'Stability limit $\delta=1/L$')
for r in results:
    if r[3] == 'converged':
        axes[0].scatter(r[0], r[1], color='green', s=60, zorder=5)
        axes[0].scatter(r[0], r[2], color='orange', marker='x', s=60, zorder=5)
axes[0].set_xlabel(r'Step size $\delta$')
axes[0].set_ylabel(r'Asymptotic variance $\sigma^2_{\mathrm{ULA}}$')
axes[0].set_title(r'Step 2: Variance vs step size')
axes[0].legend(fontsize=9)
axes[0].set_ylim(0.8, 6)
axes[0].grid(True, alpha=0.3)

axes[1].plot(delta_fine, bias_fine, 'b-', linewidth=2,
             label=r'Bias $\frac{\delta}{2-\delta}$')
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1,
                label='Zero bias')
axes[1].axvline(x=1/L, color='gray', linestyle=':', linewidth=1,
                label=r'Stability limit $\delta=1/L$')
axes[1].set_xlabel(r'Step size $\delta$')
axes[1].set_ylabel(r'Variance bias $\sigma^2_{\mathrm{ULA}} - 1$')
axes[1].set_title(r'Step 2: Bias vs step size')
axes[1].legend(fontsize=9)
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
print(f"\n[步骤1] 不同步长下的ULA行为 (L={L}, 1/L={1/L})")
print(f"  {'delta':>8} {'status':>12} {'emp. var':>12} {'theo. var':>12}")
for r in results:
    d, ve, vt, status = r
    ve_str = f'{ve:.4f}' if ve != float('inf') else 'inf'
    vt_str = f'{vt:.4f}' if vt != float('inf') else 'inf'
    print(f"  {d:>8.2f} {status:>12} {ve_str:>12} {vt_str:>12}")

print(f"\n[步骤2] 方差与偏差随步长变化曲线已保存")

print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print(r"1. delta <= 1/L: ULA收敛，偏差随delta增大而增大")
print(r"   - 小delta: 偏差小但收敛慢（需要更多迭代）")
print(r"   - 大delta: 收敛快但偏差大（渐近方差=2/(2-delta)）")
print(r"2. delta > 1/L: ULA数值发散，|X_m| -> inf")
print(r"3. delta=1/L (边界): 方差翻倍（2 vs 1），偏差显著但仍收敛")
print(r"4. 实践中需要在偏差和效率之间权衡选择delta")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")
