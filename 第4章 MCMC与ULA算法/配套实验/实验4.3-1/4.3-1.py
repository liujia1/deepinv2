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

def ULA_gauss(niter, delta, x0=0):
    Y = np.zeros(niter,)
    X = x0
    for i in range(niter):
        Z = np.random.randn()
        grad = X
        X = X - delta * grad + np.sqrt(2 * delta) * Z
        Y[i] = X
    return Y, np.var(Y)

# ══════════════════════════════════════════════════════════
# 2. 步骤1：ULA直方图 vs 真实密度
# ══════════════════════════════════════════════════════════
x0 = 0
niter = int(1e5)
delta = 0.1

Y, _ = ULA_gauss(niter, delta, x0)

x = np.linspace(-3, 3, 100)
y = 1 / (np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)

plt.figure(figsize=(8, 5))
plt.plot(x, y, 'b-', linewidth=2, label=r'True density $\mathcal{N}(0,1)$')
plt.hist(Y, bins=100, range=[-3, 3], density=1, alpha=0.6, label=r'ULA samples ($\delta=0.1$)')
plt.xlabel(r'$x$')
plt.ylabel(r'Density')
plt.title(r'Step 1: ULA samples vs true density ($\delta=0.1$)')
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, r'step1_ULA_histogram.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════
# 3. 步骤2：渐近方差验证
# ══════════════════════════════════════════════════════════
# AR(1) 平稳方差公式: sigma^2_ULA = sigma_eps^2 / (1 - phi^2)
# 代入 phi = 1-delta, sigma_eps^2 = 2*delta:
#   sigma^2_ULA = 2*delta / (1 - (1-delta)^2) = 2*delta / (2*delta - delta^2) = 2 / (2-delta)

def var_compare(delta):
    _, var1 = ULA_gauss(int(1e5), delta, x0=0)
    var2 = 2 / (2 - delta)
    print(f"  delta={delta:.2f}: empirical={var1:.6f}, theoretical={var2:.6f}, "
          f"relative error={abs(var1-var2)/var2*100:.2f}%")
    return var1, var2

print("=" * 60)
print("实验4.3-1 1D高斯ULA采样与渐近方差验证")
print("=" * 60)
print(f"\n[步骤1] ULA直方图 vs 真实密度 (delta={delta})")
print(f"  ULA采样 {niter} 步，直方图已保存")

print(f"\n[步骤2] 渐近方差公式验证: " + r"Var_ULA = 2/(2-delta)")
print(f"  {'delta':>8} {'empirical':>14} {'theoretical':>14} {'rel. error':>12}")
deltas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
emp_vars = []
theo_vars = []
for d in deltas:
    v1, v2 = var_compare(d)
    emp_vars.append(v1)
    theo_vars.append(v2)

# ══════════════════════════════════════════════════════════
# 4. 步骤3：渐近方差随步长变化
# ══════════════════════════════════════════════════════════
delta_fine = np.linspace(0.001, 1.5, 200)
var_theo_fine = 2 / (2 - delta_fine)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(delta_fine, var_theo_fine, 'b-', linewidth=2, label=r'Theoretical $\frac{2}{2-\delta}$')
axes[0].axhline(y=1, color='r', linestyle='--', linewidth=1, label=r'True variance $\sigma^2=1$')
axes[0].scatter(deltas, emp_vars, color='green', s=60, zorder=5, label='Empirical')
axes[0].scatter(deltas, theo_vars, color='orange', marker='x', s=60, zorder=5, label='Theoretical (discrete)')
axes[0].set_xlabel(r'Step size $\delta$')
axes[0].set_ylabel(r'Asymptotic variance $\sigma^2_{\mathrm{ULA}}$')
axes[0].set_title(r'Step 3: Asymptotic variance vs step size')
axes[0].legend(fontsize=9)
axes[0].set_ylim(0.8, 5)
axes[0].grid(True, alpha=0.3)

bias = delta_fine / (2 - delta_fine)
axes[1].plot(delta_fine, bias, 'b-', linewidth=2, label=r'Bias $\frac{\delta}{2-\delta}$')
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1, label='Zero bias')
axes[1].scatter(deltas, [2/(2-d) - 1 for d in deltas], color='orange', marker='x', s=60, zorder=5,
                label='Theoretical (discrete)')
axes[1].set_xlabel(r'Step size $\delta$')
axes[1].set_ylabel(r'Variance bias $\sigma^2_{\mathrm{ULA}} - 1$')
axes[1].set_title(r'Step 3: Bias vs step size')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step3_variance_and_bias.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════
# 5. 步骤4：边界步长行为 delta=1
# ══════════════════════════════════════════════════════════
delta_boundary = 1.0
Y_bdy, var_bdy = ULA_gauss(int(1e5), delta_boundary, x0=0)
var_theo_bdy = 2 / (2 - delta_boundary)

plt.figure(figsize=(8, 5))
plt.plot(x, y, 'b-', linewidth=2, label=r'True density $\mathcal{N}(0,1)$')
plt.hist(Y_bdy, bins=100, range=[-5, 5], density=1, alpha=0.6,
         label=rf'ULA samples ($\delta=1.0$, boundary)')
x_bdy = np.linspace(-5, 5, 100)
y_bdy = 1 / np.sqrt(2 * np.pi * var_theo_bdy) * np.exp(-x_bdy**2 / (2 * var_theo_bdy))
plt.plot(x_bdy, y_bdy, 'g--', linewidth=2,
         label=rf'ULA stationary $\mathcal{{N}}(0, {var_theo_bdy:.0f})$')
plt.xlabel(r'$x$')
plt.ylabel(r'Density')
plt.title(rf'Step 4: Boundary step size ($\delta=1.0$, variance={var_theo_bdy:.1f}$\times$true)')
plt.legend(fontsize=9)
plt.savefig(os.path.join(SAVE_DIR, 'step4_boundary_step.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[步骤3] 渐近方差与偏差随步长变化曲线已保存")
print(f"\n[步骤4] 边界步长行为 (delta=1.0)")
print(f"  理论方差 = 2/(2-1) = {var_theo_bdy:.1f} (真实方差的2倍)")
print(f"  经验方差 = {var_bdy:.4f}")

print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print(r"1. ULA在1D标准高斯下退化为AR(1)过程: X_{m+1} = (1-delta)*X_m + sqrt(2*delta)*Z")
print(r"2. AR(1)平稳方差公式: Var_ULA = sigma_eps^2 / (1-phi^2) = 2/(2-delta)")
print(r"3. 偏差 = Var_ULA - 1 = delta/(2-delta) ~ delta/2 (当delta较小时)")
print(r"4. delta->0: 偏差消失(无偏); delta=1: 方差翻倍(边界); delta>2: ULA发散")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")
