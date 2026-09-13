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
import matplotlib
matplotlib.use('Agg')            # 非交互式后端：直接保存图片文件，忽略环境差异
import matplotlib.pyplot as plt
from scipy.linalg import solve, circulant
from scipy.fft import fft, ifft
from tqdm import tqdm            # 多轮迭代进度条（ISTA / TV）
import os
import sys
import json

# ====== 中文字体配置（兼容本地和 Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验3.0-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)  # 递归创建所有父目录
else:
    _chinese_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.chinese')
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

# 输出路径：图片 + 数值结果 JSON
SAVE_PATH = os.path.join(SAVE_DIR, 'chapter3_demo_reconstructions.png')
JSON_PATH = os.path.join(SAVE_DIR, 'results_summary.json')


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
#    注意：核必须先以 n//2 为中心构造、再 roll 回 0 号位。若峰值直接落在 0 再
#    roll(-n//2)，峰值会跑到 n//2=100，A 变成近乎正交的移位算子（条件数约 6），
#    病态逆问题就不存在了。sigma_blur=1.0 使条件数约 70：既有明显病态（朴素逆会
#    放大噪声），又能保证 ISTA / 原始-对偶在几千次迭代内收敛。
# ---------------------------------------------------------------------------
sigma_blur = 1.0
xx = np.arange(n)
kernel = np.exp(-0.5 * (xx - n // 2) ** 2 / sigma_blur ** 2)  # 以 n//2 为中心构造对称高斯核
kernel = np.roll(kernel, -n // 2)          # 把核中心挪到 0 号位置（circulant 的零延迟位）
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
# 多轮迭代步骤加进度条（ISTA 逐次迭代，同一行持续刷新）
for _ in tqdm(range(10000), desc='ISTA 迭代', ncols=100):
    grad = A_mat.T @ (A_mat @ x_l1 - y)        # 数据项梯度
    x_l1 = soft_threshold(x_l1 - tau * grad, tau * mu_l1)


# ---------------------------------------------------------------------------
# 6) 方法三：TV 近端（TV 先验，保边）——Chambolle-Pock 原始-对偶（Condat-Vu 形式）
#    核心：用 Fenchel 对偶把"不可近端"的 TV 变成对偶空间的简单投影。
#    这里用一维各向异性 TV：TV(x) = sum_i |x_{i+1} - x_i|，梯度算子 D 是差分。
#    数据项用显式梯度步，TV 项通过对偶投影处理：对偶算子只含 D（lam 在对偶投影
#    阈值里，而不是组合进算子 K），即 Condat-Vu 形式。
# ---------------------------------------------------------------------------
D = (np.roll(np.eye(n), -1, axis=0) - np.eye(n))   # 前向差分（循环边界）
Dt = D.T
lam_tv = 0.05
# L 已在第 5 节求出（数据项梯度 Lipschitz 常数 ||A^T A||）
D_norm2 = 4.0                                       # 循环前向差分算子范数平方 ||D||^2 = 4（见3.5节）
# Condat-Vu 收敛条件（tau=sigma=t）：1/t - t*||D||^2 >= L/2，
# 令 t = (-L/2 + sqrt(L^2/4 + 4*||D||^2)) / (2*||D||^2) 即等号成立的最大步长
tau_tv = (-L / 2 + np.sqrt(L ** 2 / 4 + 4 * D_norm2)) / (2 * D_norm2)
tau_tv *= 0.99                                       # 留一点安全余量
sigma_tv = tau_tv

x_tv = np.zeros(n)
p_tv = np.zeros(n)        # 对偶变量（梯度空间）
x_bar = np.zeros(n)       # 外推点
# 多轮迭代步骤加进度条（原始-对偶逐次迭代，同一行持续刷新）
for _ in tqdm(range(3000), desc='TV-CP 迭代', ncols=100):
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

res = {
    '朴素逆（无正则）': float(rel_err(x_naive)),
    'Tikhonov (L2)': float(rel_err(x_tik)),
    'L1 近端 (ISTA)': float(rel_err(x_l1)),
    'TV 近端 (CP)': float(rel_err(x_tv)),
}

print(f"\n相对误差（越小越好）：")
for name, val in res.items():
    print(f"  {name:<18}: {val:.3f}")


# ---------------------------------------------------------------------------
# 8) 画图：一眼看到"不同先验如何把噪声答案拉回合理"
#    数学符号与下标使用 matplotlib 的 LaTeX 数学模式（$...$），保证正确渲染；
#    其余普通文字（中文/英文）用纯文本即可。
# ---------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(x_true, 'k-', lw=2, label=r'真值 $x$')
plt.plot(y, 'gray', alpha=0.5, lw=1, label=r'观测 $y$（模糊+噪声）')
plt.plot(x_naive, 'c--', lw=1.2, label=f"朴素逆（无正则，误差={res['朴素逆（无正则）']:.2f}）")
plt.plot(x_tik, 'b-', lw=1.5, label=f"Tikhonov $L_2$（误差={res['Tikhonov (L2)']:.2f}）")
plt.plot(x_l1, 'g-', lw=1.5, label=f"L1 近端（$L_1$，误差={res['L1 近端 (ISTA)']:.2f}）")
plt.plot(x_tv, 'r-', lw=1.5, label=f"TV 近端（误差={res['TV 近端 (CP)']:.2f}）")
plt.title(r'同一模糊逆问题：不同先验把噪声答案拉回不同的合理解释')
plt.xlabel(r'像素索引'); plt.ylabel(r'信号值')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=120)
plt.close()
print(f"\n图片已保存 -> {SAVE_PATH}")


# ---------------------------------------------------------------------------
# 9) 把数值结果保存为 JSON（含元参数，保证可复现）
# ---------------------------------------------------------------------------
results_summary = {
    'n': n,
    'sigma_blur': sigma_blur,
    'noise_level': noise_level,
    'lam_tik': lam_tik,
    'mu_l1': mu_l1,
    'lam_tv': lam_tv,
    'rel_err_naive': round(res['朴素逆（无正则）'], 4),
    'rel_err_tikhonov': round(res['Tikhonov (L2)'], 4),
    'rel_err_l1': round(res['L1 近端 (ISTA)'], 4),
    'rel_err_tv': round(res['TV 近端 (CP)'], 4),
}

def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    return obj

results_summary = _to_native(results_summary)
with open(JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存 -> {JSON_PATH}")

"""
一句话收尾（zoom-out）：
  正则化不是"调参旋钮"，而是"把我们对 x 的信念写进优化"——
  L2 把解拉向平滑，L1 把解拉向稀疏，TV 把解拉向分段常数保边。
  三种先验、三种正则、三种重建，正是第3章那条因果链最直观的注脚。
"""
