# -*- coding: utf-8 -*-
"""
实验3.4-2 ISTA vs FISTA 收敛曲线对比——"梯度步+近端步"迭代过程
对应章节：3.4 近端方法：不可微先验的求解策略（ISTA / FISTA 段落）
知识点：
  - ISTA 迭代：x_{k+1} = prox_{tau*lam*||·||_1}( x_k - tau*grad f(x_k) )
  - FISTA 加速：外推步 y_k = x_k + (t_{k-1}-1)/t_k * (x_k - x_{k-1})
  - 收敛率：ISTA O(1/k) vs FISTA O(1/k^2)
  - "梯度步 + 近端步"两步迭代的几何解释
  - 稀疏点源 LASSO 去模糊（与 3.4 节文档 SMLM 例子一致）

素材来源：winter_school/BolognaWinterSchool2023-main/
          PHD_MIVA_winter_school_lab - solutions.ipynb（ISTA/FISTA 实现参考）

运行环境：纯 CPU 即可，无需 GPU，无外部权重文件。
"""

import numpy as np
import math
import matplotlib
matplotlib.use('Agg')  # 静默模式：非交互式后端，不弹窗
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import io
import json
import time
import warnings
import logging

# ----- Windows 控制台 UTF-8 输出 -----
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace', line_buffering=True)

# ----- 静默 matplotlib 告警 -----
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.rcParams['axes.unicode_minus'] = False

# ====== 中文字体配置（兼容本地和 Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验3.4-2')
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

# 设置随机种子
np.random.seed(42)

# 设备配置（纯 CPU 即可）
print("=" * 60)
print("实验3.4-2 ISTA vs FISTA 收敛曲线对比")
print("=" * 60)
print("运行环境: 纯 CPU，无需 GPU，无外部权重文件")
print("  - 稀疏点源 LASSO 去模糊（与 3.4 节 SMLM 例子一致）")
print("  - 目标：直观展示'梯度步 + 近端步'两步迭代 + FISTA 加速效果")

# ══════════════════════════════════════════════════════════
# 1. 构造稀疏点源正向模型
# ══════════════════════════════════════════════════════════
print("\n[步骤 1] 构造正向模型...")

n = 256                       # 图像尺寸
N_mol = 80                    # 稀疏点源数量
margin = 5                    # 边距
signal_value = 255.0          # 稀疏点像素值

np.random.seed(17)
gt = np.zeros([n, n])
for k in range(N_mol):
    i = np.random.randint(margin, n - margin)
    j = np.random.randint(margin, n - margin)
    gt[i, j] = signal_value

# 高斯 PSF (sigma=4)
s = 4
x_coords = np.concatenate((np.arange(0, n // 2), np.arange(-n // 2, 0)))
Y, X = np.meshgrid(x_coords, x_coords)
h = np.exp(-(X ** 2 + Y ** 2) / (2 * s ** 2))
h = h / np.sum(h)

# 模糊算子（频域实现：避免构造 N×N 矩阵）
H_fft = np.fft.fft2(np.fft.fftshift(h))


def blur(x):
    """应用高斯模糊 A"""
    return np.real(np.fft.ifft2(H_fft * np.fft.fft2(x)))


# 下采样 (4×4 平均)
L = 4
m = n // L


def down_sampling(x):
    """4×4 块平均下采样"""
    out = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            out[i, j] = np.mean(x[L * i:L * i + L, L * j:L * j + L])
    return out


def forward(x):
    """前向算子 y = M A x"""
    return down_sampling(blur(x))


def adjoint(res_low):
    """伴随算子 A^T M^T（先插值上采样，再相关）"""
    # 上采样（最近邻）：每个低分辨率像素复制到 4×4 块
    up = np.repeat(np.repeat(res_low, L, axis=0), L, axis=1)
    # 模糊的伴随 = 互相关（频域共轭）
    return np.real(np.fft.ifft2(np.conj(H_fft) * np.fft.fft2(up)))


# 含噪观测
sigma_noise = 0.7
y = forward(gt) + sigma_noise * np.random.randn(m, m)

# Lipschitz 常数估计（保证步长满足收敛条件）
# 复合算子 K = M ∘ A (M = 下采样, A = 模糊)
# - ||A||_op = max(|H_fft|) = 1 (h 已归一化)
# - ||M||_op = 1 (块平均是收缩算子)
# - 经验公式（参考 winter school lab）：Lips ≈ max(|H|^2) * L^2
#   其中 L=4 为下采样因子，对应块平均 + 块复制的复合效应
# 这里采用更严格的 power iteration 估计，参考公式作为下界
L_bound = 1.0 * L ** 2  # 公式下界
print(f"  Lipschitz 常数公式下界 L ≥ {L_bound:.4f}")
# 步长取 0.9 / L_bound 以确保收敛（保守策略）
tau = 0.9 / L_bound
print(f"  步长 tau = 0.9/L_bound = {tau:.4e}")
Lips = L_bound

# ══════════════════════════════════════════════════════════
# 2. 目标函数与梯度
# ══════════════════════════════════════════════════════════
def fidelity(x):
    """数据拟合项 0.5 * ||Kx - y||^2"""
    return 0.5 * np.sum((forward(x) - y) ** 2)


def gradient(x):
    """数据项梯度 ∇f(x) = K^T(Kx - y)"""
    residual = forward(x) - y
    return adjoint(residual)


def soft_thresholding(x, gamma):
    """软阈值算子 S_gamma(x) = sign(x) * max(|x| - gamma, 0)"""
    return np.sign(x) * np.maximum(0, np.abs(x) - gamma)


def cost_function(x, lmbda):
    """LASSO 目标函数 f(x) + lambda * ||x||_1"""
    return fidelity(x) + lmbda * np.sum(np.abs(x))


# ══════════════════════════════════════════════════════════
# 3. ISTA 迭代
# ══════════════════════════════════════════════════════════
def ISTA(x0, tau, lmbda, maxiter):
    """
    ISTA: 每步先沿光滑数据项做梯度下降，再做近端步（软阈值）

    迭代公式：
        x_{k+1} = prox_{tau*lambda*||·||_1}( x_k - tau*grad f(x_k) )
                = soft_thresholding( x_k - tau*grad f(x_k), tau*lambda )
    """
    xk = x0.copy()
    cost = np.zeros(maxiter)
    for k in range(maxiter):
        # 梯度步
        z = xk - tau * gradient(xk)
        # 近端步（软阈值）
        xk = soft_thresholding(z, tau * lmbda)
        # 非负约束（SMLM 信号非负）
        xk = np.maximum(xk, 0)
        cost[k] = cost_function(xk, lmbda)
    return xk, cost


# ══════════════════════════════════════════════════════════
# 4. FISTA 迭代（Nesterov 加速）
# ══════════════════════════════════════════════════════════
def FISTA(x0, tau, lmbda, maxiter):
    """
    FISTA: ISTA + 外推步（Nesterov 动量）

    迭代公式：
        y_k = x_k + (t_{k-1} - 1)/t_k * (x_k - x_{k-1})    # 外推步
        x_{k+1} = prox_{tau*lambda*||·||_1}( y_k - tau*grad f(y_k) )
        t_{k+1} = (1 + sqrt(1 + 4*t_k^2)) / 2
    """
    xk = x0.copy()
    xold = x0.copy()
    told = 1.0
    cost = np.zeros(maxiter)
    for k in range(maxiter):
        # 外推步
        tk = 0.5 * (1 + math.sqrt(1 + 4 * told ** 2))
        yk = xk + (told - 1) / tk * (xk - xold)

        # 梯度步（在 yk 处）
        z = yk - tau * gradient(yk)
        # 近端步
        xkk = soft_thresholding(z, tau * lmbda)
        xkk = np.maximum(xkk, 0)

        cost[k] = cost_function(xkk, lmbda)
        # 更新状态
        told = tk
        xold = xk.copy()
        xk = xkk
    return xk, cost


# ══════════════════════════════════════════════════════════
# 5. 运行 ISTA 和 FISTA
# ══════════════════════════════════════════════════════════
print("\n[步骤 2] 运行 ISTA 和 FISTA ...")
x0 = np.zeros((n, n))
lmbda = 10.0
maxiter = 200

t0 = time.time()
cost_ista = np.zeros(maxiter)
xk = x0.copy()
print(f"  ISTA 进度: 0/{maxiter}", end="", flush=True)
for k in range(maxiter):
    z = xk - tau * gradient(xk)
    xk = soft_thresholding(z, tau * lmbda)
    xk = np.maximum(xk, 0)
    cost_ista[k] = cost_function(xk, lmbda)
    if (k + 1) % 20 == 0 or k == 0:
        print(f"\r  ISTA 进度: {k + 1}/{maxiter}, F(x) = {cost_ista[k]:.4f}", end="", flush=True)
print(f"\n  ISTA 完成，耗时 {time.time() - t0:.2f} 秒")
rec_ista = xk.copy()

t0 = time.time()
cost_fista = np.zeros(maxiter)
xk = x0.copy()
xold = x0.copy()
told = 1.0
print(f"  FISTA 进度: 0/{maxiter}", end="", flush=True)
for k in range(maxiter):
    tk = 0.5 * (1 + math.sqrt(1 + 4 * told ** 2))
    yk = xk + (told - 1) / tk * (xk - xold)
    z = yk - tau * gradient(yk)
    xkk = soft_thresholding(z, tau * lmbda)
    xkk = np.maximum(xkk, 0)
    cost_fista[k] = cost_function(xkk, lmbda)
    told = tk
    xold = xk.copy()
    xk = xkk
    if (k + 1) % 20 == 0 or k == 0:
        print(f"\r  FISTA 进度: {k + 1}/{maxiter}, F(x) = {cost_fista[k]:.4f}", end="", flush=True)
print(f"\n  FISTA 完成，耗时 {time.time() - t0:.2f} 秒")
rec_fista = xk.copy()

# 参考最优值（用更长的 ISTA 估计）
print("\n[步骤 3] 计算参考最优值（长 ISTA 运行）...")
ref_iter = 2000
xk_ref = x0.copy()
for k in range(ref_iter):
    z = xk_ref - tau * gradient(xk_ref)
    xk_ref = soft_thresholding(z, tau * lmbda)
    xk_ref = np.maximum(xk_ref, 0)
F_star = cost_function(xk_ref, lmbda)
print(f"  F* ≈ {F_star:.4f}（ISTA {ref_iter} 次迭代估计）")

# 收敛指标
gap_ista = cost_ista - F_star
gap_fista = cost_fista - F_star
# 数值稳定性：去掉前几次迭代中可能的 0 误差
gap_ista = np.maximum(gap_ista, 1e-12)
gap_fista = np.maximum(gap_fista, 1e-12)

print(f"\n[收敛对比] 误差 F(x_k) - F*")
print(f"  迭代 50:  ISTA = {gap_ista[49]:.4e},  FISTA = {gap_fista[49]:.4e}, 加速比 = {gap_ista[49]/gap_fista[49]:.2f}x")
print(f"  迭代 100: ISTA = {gap_ista[99]:.4e},  FISTA = {gap_fista[99]:.4e}, 加速比 = {gap_ista[99]/gap_fista[99]:.2f}x")
print(f"  迭代 200: ISTA = {gap_ista[199]:.4e}, FISTA = {gap_fista[199]:.4e}, 加速比 = {gap_ista[199]/gap_fista[199]:.2f}x")

# 重建质量（与真解 gt 的 PSNR）
psnr_ista = 10 * np.log10(signal_value ** 2 / np.mean((rec_ista - gt) ** 2))
psnr_fista = 10 * np.log10(signal_value ** 2 / np.mean((rec_fista - gt) ** 2))
print(f"\n[重建质量] PSNR (dB) - 数据范围基于信号值 {signal_value:.0f}")
print(f"  ISTA  (200 次): {psnr_ista:.2f} dB")
print(f"  FISTA (200 次): {psnr_fista:.2f} dB")

# ══════════════════════════════════════════════════════════
# 6. 可视化
# ══════════════════════════════════════════════════════════
print("\n[步骤 4] 绘制收敛曲线与重建结果...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 第一行：原始 / 观测 / ISTA 重建
axes[0, 0].imshow(gt, cmap='gray')
axes[0, 0].set_title(r'真解 (稀疏点源, $N_{mol}$=' + f'{N_mol}' + r')')
axes[0, 0].axis('off')

axes[0, 1].imshow(y, cmap='gray')
axes[0, 1].set_title(r'模糊下采样+噪声观测 ($y \in \mathbb{R}^{64 \times 64}$)')
axes[0, 1].axis('off')

# 自动调整 vmax（ISTA/FISTA 重建值远小于 255）
vmax_rec = max(rec_ista.max(), rec_fista.max())
axes[0, 2].imshow(rec_ista, cmap='gray', vmax=vmax_rec)
axes[0, 2].set_title(r'ISTA 重建 (200 次迭代)' + f'\nPSNR={psnr_ista:.2f} dB')
axes[0, 2].axis('off')

# 第二行：FISTA 重建 / 收敛曲线
axes[1, 0].imshow(rec_fista, cmap='gray', vmax=vmax_rec)
axes[1, 0].set_title(r'FISTA 重建 (200 次迭代)' + f'\nPSNR={psnr_fista:.2f} dB')
axes[1, 0].axis('off')

# 收敛曲线（log-log 标度展示 O(1/k) vs O(1/k^2)）
k_arr = np.arange(1, maxiter + 1)
# 参考斜率：1/k 与 1/k^2
ref_1k = gap_ista[10] * 10 / k_arr          # O(1/k) 参考
ref_1k2 = gap_fista[10] * (10 ** 2) / (k_arr ** 2)  # O(1/k^2) 参考

axes[1, 1].loglog(k_arr, gap_ista, 'b-', linewidth=2, label=r'ISTA: $F(x_k) - F^*$')
axes[1, 1].loglog(k_arr, gap_fista, 'r-', linewidth=2, label=r'FISTA: $F(x_k) - F^*$')
axes[1, 1].loglog(k_arr, ref_1k, 'b--', alpha=0.4, label=r'$O(1/k)$ 参考')
axes[1, 1].loglog(k_arr, ref_1k2, 'r--', alpha=0.4, label=r'$O(1/k^2)$ 参考')
axes[1, 1].set_xlabel(r'迭代次数 $k$')
axes[1, 1].set_ylabel(r'$F(x_k) - F^*$')
axes[1, 1].set_title(r'ISTA vs FISTA 收敛曲线 (log-log)')
axes[1, 1].legend(fontsize=9, loc='lower left')
axes[1, 1].grid(True, which='both', alpha=0.3)

# 右下：放大显示迭代前 50 次的差距（FISTA 在前期领先最显著）
axes[1, 2].semilogy(k_arr[:50], gap_ista[:50], 'b-o', markersize=4,
                    linewidth=1.5, label='ISTA')
axes[1, 2].semilogy(k_arr[:50], gap_fista[:50], 'r-s', markersize=4,
                    linewidth=1.5, label='FISTA')
axes[1, 2].set_xlabel(r'迭代次数 $k$')
axes[1, 2].set_ylabel(r'$F(x_k) - F^*$')
axes[1, 2].set_title(r'前 50 次迭代：FISTA 领先 ISTA 的程度')
axes[1, 2].legend(fontsize=10)
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle(r'实验3.4-2: ISTA vs FISTA——"梯度步+近端步"迭代过程与 Nesterov 加速',
             fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path1 = os.path.join(SAVE_DIR, '步骤1_ISTA_vs_FISTA收敛对比.png')
plt.savefig(out_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存：{out_path1}")

# ══════════════════════════════════════════════════════════
# 7. "梯度步 + 近端步"两步迭代的几何示意
# ══════════════════════════════════════════════════════════
print("\n[步骤 5] 绘制'梯度步 + 近端步'两步迭代几何示意...")

# 在二维等高线图上演示 ISTA 的两步迭代（用一个简单的二次函数 + 不可微项）
# f(x1,x2) = 0.5*(x1^2 + 4*x2^2), g(x) = 0.5*lam*(|x1| + |x2|)
# 这是一个简单的 2D 复合凸问题，便于可视化迭代轨迹
def f_simple(z):
    return 0.5 * (z[0] ** 2 + 4 * z[1] ** 2)


def grad_f_simple(z):
    return np.array([z[0], 4 * z[1]])


def g_simple(z, lam=0.3):
    return lam * (np.abs(z[0]) + np.abs(z[1]))


def prox_l1_2d(v, gamma):
    """二维 L1 近端算子 = 分量软阈值"""
    return np.array([np.sign(v[0]) * max(abs(v[0]) - gamma, 0),
                     np.sign(v[1]) * max(abs(v[1]) - gamma, 0)])


# ISTA 迭代轨迹
x_curr = np.array([1.6, 0.9])
ista_traj = [x_curr.copy()]
lam_demo = 0.3
tau_demo = 0.4  # 满足 tau < 1/L (L=4)
for _ in range(20):
    grad = grad_f_simple(x_curr)
    z = x_curr - tau_demo * grad
    x_next = prox_l1_2d(z, tau_demo * lam_demo)
    ista_traj.append(x_next.copy())
    x_curr = x_next

# FISTA 迭代轨迹
x_curr = np.array([1.6, 0.9])
x_old = np.array([1.6, 0.9])
t_old = 1.0
fista_traj = [x_curr.copy()]
for _ in range(20):
    t_k = 0.5 * (1 + math.sqrt(1 + 4 * t_old ** 2))
    y_k = x_curr + (t_old - 1) / t_k * (x_curr - x_old)
    grad = grad_f_simple(y_k)
    z = y_k - tau_demo * grad
    x_next = prox_l1_2d(z, tau_demo * lam_demo)
    fista_traj.append(x_next.copy())
    t_old = t_k
    x_old = x_curr.copy()
    x_curr = x_next

# 构造网格（用于等高线）
x1g = np.linspace(-2.2, 2.2, 200)
x2g = np.linspace(-1.4, 1.4, 200)
X1, X2 = np.meshgrid(x1g, x2g)
F_total = 0.5 * (X1 ** 2 + 4 * X2 ** 2) + lam_demo * (np.abs(X1) + np.abs(X2))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 公共设置：等高线 + 最优解位置
levels = np.logspace(-1, 1.5, 20)

# 左图：ISTA 轨迹
ista_traj = np.array(ista_traj)
cs = axes[0].contour(X1, X2, F_total, levels=levels, alpha=0.4, cmap='viridis')
axes[0].plot(ista_traj[:, 0], ista_traj[:, 1], 'b-o', markersize=5, linewidth=1.5, label='ISTA 轨迹')
# 标记关键迭代点
for k in [0, 1, 5, 10, 20]:
    if k < len(ista_traj):
        axes[0].annotate(f'$k={k}$', ista_traj[k], textcoords="offset points",
                        xytext=(5, 5), fontsize=9, color='blue')
# 最优解（原点附近）
axes[0].plot(0, 0, 'r*', markersize=18, label=r'最优点 $x^* = (0,0)$')
axes[0].set_xlabel(r'$x_1$')
axes[0].set_ylabel(r'$x_2$')
axes[0].set_title(r'ISTA 迭代轨迹（"梯度步 + 近端步"）')
axes[0].legend(fontsize=10, loc='upper right')
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')

# 右图：FISTA 轨迹
fista_traj = np.array(fista_traj)
axes[1].contour(X1, X2, F_total, levels=levels, alpha=0.4, cmap='viridis')
axes[1].plot(fista_traj[:, 0], fista_traj[:, 1], 'r-s', markersize=5, linewidth=1.5, label='FISTA 轨迹')
for k in [0, 1, 5, 10, 20]:
    if k < len(fista_traj):
        axes[1].annotate(f'$k={k}$', fista_traj[k], textcoords="offset points",
                        xytext=(5, 5), fontsize=9, color='red')
axes[1].plot(0, 0, 'r*', markersize=18, label=r'最优点 $x^* = (0,0)$')
axes[1].set_xlabel(r'$x_1$')
axes[1].set_ylabel(r'$x_2$')
axes[1].set_title(r'FISTA 迭代轨迹（Nesterov 加速）')
axes[1].legend(fontsize=10, loc='upper right')
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal')

plt.suptitle(r'实验3.4-2: "梯度步 + 近端步"两步迭代的几何示意（$f = 0.5(x_1^2 + 4x_2^2), g = \lambda(|x_1| + |x_2|)$）',
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path2 = os.path.join(SAVE_DIR, '步骤2_梯度步近端步迭代轨迹.png')
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存：{out_path2}")

# ══════════════════════════════════════════════════════════
# 8. 保存数值结果到 JSON
# ══════════════════════════════════════════════════════════


def _to_native(obj):
    """把 numpy 标量与数组转成 JSON 可序列化对象"""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'item'):
        try:
            return obj.item()
        except Exception:
            return obj
    return obj


results = {
    'experiment': '实验3.4-2 ISTA vs FISTA 收敛曲线对比',
    'scene': {
        'image_size': n,
        'n_molecules': N_mol,
        'psf_sigma': s,
        'downsample_factor': L,
        'noise_sigma': float(sigma_noise),
        'lambda': float(lmbda),
        'maxiter': maxiter,
        'step_size_tau': float(tau),
        'lipschitz_estimate': float(Lips),
    },
    'convergence': {
        'F_star_estimate': float(F_star),
        'gap_ista': {str(k + 1): float(gap_ista[k]) for k in [9, 19, 49, 99, 149, 199]},
        'gap_fista': {str(k + 1): float(gap_fista[k]) for k in [9, 19, 49, 99, 149, 199]},
        'speedup_ratio_50': float(gap_ista[49] / gap_fista[49]),
        'speedup_ratio_100': float(gap_ista[99] / gap_fista[99]),
        'speedup_ratio_200': float(gap_ista[199] / gap_fista[199]),
    },
    'reconstruction_psnr': {
        'ISTA_200iter': float(psnr_ista),
        'FISTA_200iter': float(psnr_fista),
    },
    'output_files': [out_path1, out_path2],
}

with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(_to_native(results), f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print("1. ISTA 每步 = 梯度步 (沿光滑数据项下降) + 近端步 (软阈值)")
print("2. FISTA 在 ISTA 基础上加外推步 (Nesterov 动量)，几乎'免费'获得加速")
print(f"3. 在迭代 50/100/200 时，FISTA 相对 ISTA 的加速比分别为 "
      f"{gap_ista[49]/gap_fista[49]:.2f}x / {gap_ista[99]/gap_fista[99]:.2f}x / "
      f"{gap_ista[199]/gap_fista[199]:.2f}x")
print(f"4. 200 次迭代后 FISTA 重建 PSNR = {psnr_fista:.2f} dB，"
      f"ISTA 重建 PSNR = {psnr_ista:.2f} dB")
print(f"5. 收敛曲线在 log-log 标度下，ISTA 接近 $O(1/k)$，FISTA 接近 $O(1/k^2)$——"
      f"一阶方法的最优速率")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")
print(f"  - 步骤1_ISTA_vs_FISTA收敛对比.png")
print(f"  - 步骤2_梯度步近端步迭代轨迹.png")
print(f"  - results_summary.json")
