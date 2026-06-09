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
    """计算自相关函数"""
    x = x - np.mean(x)
    n = len(x)
    acf_vals = np.zeros(nlags + 1)
    for lag in range(nlags + 1):
        acf_vals[lag] = np.sum(x[:n-lag] * x[lag:]) / (n * np.var(x))
    return acf_vals

acf_values = compute_acf(samples, nlags=50)
plt.subplot(1, 3, 3)
plt.plot(acf_values, 'r-o')
plt.xlabel('滞后阶数')
plt.ylabel('自相关')
plt.title('自相关函数')
plt.axhline(y=0.05, color='k', linestyle='--', label='5%阈值')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_ULA采样验证.png'), dpi=150)
plt.close()

# 计算统计量
print(f"采样方差: {np.var(samples):.4f}")
print(f"理论方差: {2/(2-delta):.4f}")  # 2/(2-δ)是ULA的理论方差
print(f"采样均值: {np.mean(samples):.4f}")
print(f"理论均值: {0:.4f}")


# ============================================================
# 步骤2：步长敏感性分析
# ============================================================
print("\n" + "=" * 60)
print("步骤2：步长敏感性分析")
print("=" * 60)

deltas = [0.01, 0.1, 0.5, 1.0]
niter = 50000

plt.figure(figsize=(15, 4))

for i, delta in enumerate(deltas):
    samples = ULA_gauss(niter, delta, x0=0)
    empirical_var = np.var(samples)
    theoretical_var = 2 / (2 - delta)

    plt.subplot(1, len(deltas), i+1)
    x = np.linspace(-4, 4, 100)
    true_density = 1/np.sqrt(2*np.pi) * np.exp(-x**2 / 2)
    plt.plot(x, true_density, 'b-', lw=2, label='真实 $N(0,1)$')
    plt.hist(samples, bins=50, density=True, alpha=0.7,
             color='orange', label='ULA采样')
    plt.title(f'$\\delta$={delta}, Var_emp={empirical_var:.3f}, Var_theo={theoretical_var:.3f}')
    plt.xlabel('$x$')
    if i == 0:
        plt.ylabel('密度')
    plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_步长敏感性.png'), dpi=150)
plt.close()

print("步长敏感性总结：")
print("  $\\delta$=1.0 时，理论方差为∞（不稳定），实际采样会发散")
print("  小$\\delta$（如0.01）方差更接近1，但收敛慢")
print("  存在偏差-方差权衡：$\\delta$小→偏差小但收敛慢；$\\delta$大→偏差大但收敛快")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验5.1-1 总结")
print("=" * 60)
print("1. ULA与Langevin SDE的关系：ULA是Langevin SDE的Euler-Maruyama离散化")
print("   步长$\\delta$越小，近似越精确")
print("2. 收敛性：在小步长且目标分布强对数凹时，ULA样本分布收敛到真实目标分布")
print("3. 偏差-方差权衡：离散化引入误差，可通过调整$\\delta$平衡近似精度与计算效率")
print("4. 理论方差公式：Var = 2/(2-$\\delta$)，步长越大偏差越大")
