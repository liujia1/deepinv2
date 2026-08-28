# -*- coding: utf-8 -*-
"""
第3章 章末可运行代码：同一模糊逆问题下，不同先验（正则化）如何把"噪声答案"拉回合理
=================================================================================
本脚本用纯 numpy / scipy 复现本章核心现象之一：
  - 一个 1D 信号，经过"模糊算子 A"并加高斯噪声，得到观测 y；
  - 我们用三种 MAP 正则化去重建：
      1) Tikhonov（高斯先验，光滑，闭式解 / 梯度下降）
      2) L1 近端（Laplace 先验，稀疏，ISTA 软阈值）
      3) TV 近端（TV 先验，保边，Chambolle-Pock 原始-对偶）
  - 直观看到：不同先验=不同正则项，把"朴素逆（噪声炸开）"的答案拉回了不同形状的合理解。

只需 numpy + scipy + matplotlib，无私有包依赖。运行：python 3.0-1.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve, circulant
from scipy.fft import fft, ifft


# ---------------------------------------------------------------------------
# 0) 构造一个最朴素的 1D 真解：分段常数（稀疏 + 有边缘，正好对比三种先验）
# ---------------------------------------------------------------------------
rng = np.random.default_rng(20240715)
n = 200
x_true = np.zeros(n)
x_true[40:70] = 1.0      # 第一段平台
x_true[110:150] = -0.8   # 第二段平台（负值）
# 其余区间为 0


# ---------------------------------------------------------------------------
# 1) 模糊算子 A：圆周卷积的高斯模糊核（循环矩阵，可用 FFT 高效作用）
#    定义 A 为一个 circulant 矩阵，第一列是模糊核。
# ---------------------------------------------------------------------------
sigma_blur = 2.0
xx = np.arange(n)
kernel = np.exp(-0.5 * (xx - 0) ** 2 / sigma_blur ** 2)
kernel = np.roll(kernel, -n // 2)          # 把核中心挪到 0 号位置
kernel /= kernel.sum()                      # 归一化，保证 A 近似保能量

A_mat = circulant(kernel)                   # 显式循环矩阵，仅用于本演示（n 不大）


# ---------------------------------------------------------------------------
# 2) 加噪观测 y = A x_true + 噪声
# ---------------------------------------------------------------------------
noise_level = 0.05
y = A_mat @ x_true + noise_level * rng.standard_normal(n)


# ---------------------------------------------------------------------------
# 3) 朴素逆（无正则）：直接最小二乘解 A^{-1} y（用伪逆）
#    这会"炸开"——小奇异值方向把噪声无限放大，是 MAP 要修正的对象。
# ---------------------------------------------------------------------------
x_naive = np.linalg.pinv(A_mat) @ y


# ---------------------------------------------------------------------------
# 4) 方法一：Tikhonov（高斯先验）——闭式解 (A^T A + lam I)^{-1} A^T y
#    直观：把解往 0（先验均值）收缩；对应第3.3节的低通滤波。
# ---------------------------------------------------------------------------
lam_tik = 0.02
x_tik = solve(A_mat.T @ A_mat + lam_tik * np.eye(n), A_mat.T @ y)


# ---------------------------------------------------------------------------
# 5) 方法二：L1 近端（Laplace 先验）——ISTA
#    先梯度步（沿光滑数据项），再做软阈值（近端校正，制造稀疏）。
#    软阈值：sign(v)*max(|v|-mu, 0)
# ---------------------------------------------------------------------------
def soft_threshold(v, mu):
    return np.sign(v) * np.maximum(np.abs(v) - mu, 0.0)

# 估计数据项 Lipschitz 常数 L = ||A^T A|| 的谱半径（这里直接取最大特征值近似）
L = np.linalg.eigvalsh(A_mat.T @ A_mat).max()
tau = 0.99 / L                       # 步长 < 2/L，保证下降（见3.2节）
mu_l1 = 0.05                          # L1 正则权重（对应 lam 在复合形式里）
x_l1 = np.zeros(n)
for _ in range(2000):
    grad = A_mat.T @ (A_mat @ x_l1 - y)        # 数据项梯度
    x_l1 = soft_threshold(x_l1 - tau * grad, tau * mu_l1)


# ---------------------------------------------------------------------------
# 6) 方法三：TV 近端（TV 先验，保边）——Chambolle-Pock 原始-对偶
#    核心：用 Fenchel 对偶把"不可近端"的 TV 变成对偶空间的简单投影。
#    这里用一维各向异性 TV：TV(x) = sum_i |x_{i+1} - x_i|，梯度算子 D 是差分。
#    为与前两种方法公平对比，我们同样带数据保真项 1/2||Ax-y||^2（A=I 的演示里可加 tau*A^T(Ax-y)）。
# ---------------------------------------------------------------------------
D = (np.roll(np.eye(n), -1, axis=0) - np.eye(n))   # 前向差分（循环边界）
Dt = D.T
lam_tv = 0.05
# 算子范数上界：组合算子 K = [A; sqrt(lam)*D] 的范数平方近似 <= ||A||^2 + lam*||D||^2
# 一维差分 ||D||^2 <= 4（见3.5节），这里取保守步长保证 tau*sigma*||K||^2 < 1
K_norm2 = np.linalg.eigvalsh(A_mat.T @ A_mat).max() + lam_tv * 4.0
tau_tv = 1.0 / np.sqrt(K_norm2 + 1e-8)
sigma_tv = tau_tv

x_tv = np.zeros(n)
p_tv = np.zeros(n)        # 对偶变量（梯度空间）
x_bar = np.zeros(n)       # 外推点
for _ in range(3000):
    # 对偶步：投影到无穷范数球（各向异性 TV 的对偶约束 ||p||_inf <= lam）
    p_tv = p_tv + sigma_tv * (D @ x_bar)
    p_tv = np.clip(p_tv, -lam_tv, lam_tv)          # 简单投影 = 软阈值的一种对偶形态
    # 原始步：既要往"数据保真"方向走（减 tau*A^T(Ax-y)），又要往"TV 对偶反馈"方向走（减 tau*Dt*p）
    x_new = x_tv - tau_tv * (A_mat.T @ (A_mat @ x_tv - y) + Dt @ p_tv)
    x_tv = x_new
    x_bar = 2 * x_tv - x_new       # 外推


# ---------------------------------------------------------------------------
# 7) 计算一个"离真解多近"的指标（相对误差），直观对比
# ---------------------------------------------------------------------------
def rel_err(x):
    return np.linalg.norm(x - x_true) / np.linalg.norm(x_true)

print("相对误差（越小越好）：")
print(f"  朴素逆（无正则） : {rel_err(x_naive):.3f}")
print(f"  Tikhonov (L2)   : {rel_err(x_tik):.3f}")
print(f"  L1 近端 (ISTA)   : {rel_err(x_l1):.3f}")
print(f"  TV 近端 (CP)     : {rel_err(x_tv):.3f}")


# ---------------------------------------------------------------------------
# 8) 画图：一眼看到"不同先验如何把噪声答案拉回合理"
# ---------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(x_true, 'k-', lw=2, label='true x')
plt.plot(y, 'gray', alpha=0.5, lw=1, label='observation y (blurred+noise)')
plt.plot(x_naive, 'c--', lw=1.2, label=f'naive inverse (err={rel_err(x_naive):.2f})')
plt.plot(x_tik, 'b-', lw=1.5, label=f'Tikhonov L2 (err={rel_err(x_tik):.2f})')
plt.plot(x_l1, 'g-', lw=1.5, label=f'L1 proximal ISTA (err={rel_err(x_l1):.2f})')
plt.plot(x_tv, 'r-', lw=1.5, label=f'TV proximal CP (err={rel_err(x_tv):.2f})')
plt.title('Same blurred inverse problem: different priors pull the noisy answer back to different plausible shapes')
plt.xlabel('pixel index'); plt.ylabel('signal value')
plt.legend(fontsize=8)
plt.tight_layout()
# 若处于交互式后端（读者本机有窗口）则弹窗；若在无头环境（如 CI/Agg）则存盘，避免警告
if plt.get_backend().lower() not in ('agg', 'pdf', 'svg', 'ps'):
    plt.show()
else:
    plt.savefig('chapter3_demo_reconstructions.png', dpi=120)
    print('saved figure -> chapter3_demo_reconstructions.png')

"""
一句话收尾（zoom-out）：
  正则化不是"调参旋钮"，而是"把我们对 x 的信念写进优化"——
  L2 把解拉向平滑，L1 把解拉向稀疏，TV 把解拉向分段常数保边。
  三种先验、三种正则、三种重建，正是第3章那条因果链最直观的注脚。
"""
