# -*- coding: utf-8 -*-
"""
实验3.4-2 ISTA 与 FISTA 的收敛速率对比——1D 信号去卷积
对应章节：3.4 近端方法：不可微先验的求解策略（ISTA / FISTA 段落）
知识点：
  - LASSO 去卷积：min_x 0.5*||Ax-y||^2 + lambda*||x||_1
  - ISTA：x_{k+1} = S_{gamma*lambda}( x_k - gamma*grad f(x_k) )，收敛 O(1/k)
  - FISTA：Nesterov 外推动量，收敛 O(1/k^2)
  - log-log 收敛曲线的斜率 = 收敛阶
  - 步骤2：2D 玩具问题展示"梯度步 + 近端步"两段轨迹

运行环境：纯 CPU，无需 GPU，无外部权重文件。
"""

import math
import os
import sys
import io
import json
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 静默模式：非交互式后端，不弹窗
import matplotlib.pyplot as plt

# ----- Windows 控制台 UTF-8 输出 -----
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace', line_buffering=True)

# ----- 静默 matplotlib 告警 -----
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
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

np.random.seed(42)

print("=" * 60)
print("实验3.4-2 ISTA 与 FISTA 的收敛速率对比（1D 信号去卷积）")
print("=" * 60)
print("运行环境: 纯 CPU，无需 GPU，无外部权重文件")
print("  - LASSO 去卷积: min_x 0.5*||Ax-y||^2 + lambda*||x||_1")
print("  - 目标1: log-log 收敛曲线验证 ISTA O(1/k) 与 FISTA O(1/k^2)")
print("  - 目标2: 2D 玩具问题展示'梯度步 + 近端步'两段轨迹")

# ══════════════════════════════════════════════════════════
# 1. 正向模型：1D 分段常值信号 + 圆周卷积高斯模糊 + 高斯噪声
# ══════════════════════════════════════════════════════════
n = 256                     # 信号长度
sigma_blur = 2.0            # 高斯模糊核宽度
sigma_noise = 0.05          # 观测噪声标准差
lam = 0.1                   # L1 正则权重


def make_piecewise_constant(n, seed=17):
    """分段常值真信号：随机段长 15~40，幅值 ±0.3~1.0，天然含跳变"""
    rng = np.random.RandomState(seed)
    x = np.zeros(n)
    i = 0
    while i < n:
        seg_len = rng.randint(15, 40)
        amp = rng.uniform(0.3, 1.0) * rng.choice([-1.0, 1.0])
        x[i:i + seg_len] = amp
        i += seg_len
    return x


x_true = make_piecewise_constant(n, seed=17)


def gaussian_kernel_fft(n, sigma):
    """周期化一维高斯核的 FFT（居中核，系数和为 1，DC 增益为 1）"""
    t = np.fft.fftfreq(n) * n          # 格点坐标: 0..n/2-1, -n/2..-1
    h = np.exp(-t ** 2 / (2 * sigma ** 2))
    h = h / h.sum()
    return np.fft.fft(h)


H_fft = gaussian_kernel_fft(n, sigma_blur)


def forward(x):
    """正向算子 A：圆周卷积高斯模糊（频域实现，避免构造 n×n 矩阵）"""
    return np.real(np.fft.ifft(H_fft * np.fft.fft(x)))


def adjoint(z):
    """伴随算子 A^T：循环互相关（频域取共轭）"""
    return np.real(np.fft.ifft(np.conj(H_fft) * np.fft.fft(z)))


y = forward(x_true) + sigma_noise * np.random.randn(n)

# Lipschitz 常数 L_f = ||A^T A||_2：循环卷积可频域精确计算
L_exact = float(np.max(np.abs(H_fft)) ** 2)

# 幂迭代验证（通用方法，适用于任意无法对角化的 A）
np.random.seed(0)
v_pi = np.random.randn(n)
v_pi = v_pi / np.linalg.norm(v_pi)
for _ in range(50):
    w = adjoint(forward(v_pi))
    w_norm = np.linalg.norm(w)
    v_pi = w / w_norm
L_power = float(np.sum(v_pi * adjoint(forward(v_pi))))   # Rayleigh 商

gamma = 0.99 / L_exact       # 步长取 0.99/L_f，略小于收敛上限 1/L_f

print("\n[参数设定]")
print(f"  信号长度 n={n}（分段常值，含跳变），模糊核 sigma_blur={sigma_blur}，噪声 sigma={sigma_noise}")
print(f"  L1 权重 lambda={lam}")
print(f"  L_f: 频域精确值 {L_exact:.6f}，幂迭代估计 {L_power:.6f}（二者应一致）")
print(f"  步长 gamma = 0.99/L_f = {gamma:.4f}")

# ══════════════════════════════════════════════════════════
# 2. 目标函数、梯度与近端算子
# ══════════════════════════════════════════════════════════


def fidelity(x):
    """数据项 0.5*||Ax-y||^2"""
    residual = forward(x) - y
    return 0.5 * float(np.sum(residual ** 2))


def grad_f(x):
    """数据项梯度 A^T(Ax-y)"""
    return adjoint(forward(x) - y)


def soft_threshold(v, thr):
    """软阈值 S_thr(v) = sign(v)*max(|v|-thr, 0)——L1 的近端算子"""
    return np.sign(v) * np.maximum(np.abs(v) - thr, 0.0)


def cost(x):
    """LASSO 目标 F(x) = f(x) + lambda*||x||_1"""
    return fidelity(x) + lam * float(np.sum(np.abs(x)))


# ══════════════════════════════════════════════════════════
# 3. ISTA 与 FISTA（记录完整目标值曲线，判停次数事后统计）
# ══════════════════════════════════════════════════════════


def run_ista(max_iter):
    """ISTA：先梯度步，再近端步
        x_{k+1} = prox_{gamma*g}( x_k - gamma*grad f(x_k) )
                = S_{gamma*lambda}( x_k - gamma*A^T(Ax_k - y) )
    近端算子作用于 gamma*g，故软阈值参数是 gamma*lambda 而非 lambda
    """
    xk = np.zeros(n)
    curve = np.zeros(max_iter)
    for k in range(max_iter):
        z = xk - gamma * grad_f(xk)            # 梯度步：沿光滑数据项下降
        xk = soft_threshold(z, gamma * lam)    # 近端步：施加稀疏性
        curve[k] = cost(xk)
    return xk, curve


def run_fista(max_iter):
    """FISTA：ISTA + Nesterov 外推动量
        t_{k+1} = (1+sqrt(1+4*t_k^2))/2
        y_k = x_k + (t_{k-1}-1)/t_k * (x_k - x_{k-1})    # 外推点
        x_{k+1} = S_{gamma*lambda}( y_k - gamma*grad f(y_k) )
    """
    xk = np.zeros(n)
    x_old = np.zeros(n)
    t_old = 1.0
    curve = np.zeros(max_iter)
    for k in range(max_iter):
        t_k = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_old ** 2))
        y_k = xk + (t_old - 1.0) / t_k * (xk - x_old)   # 外推步：借前一步的势头
        z = y_k - gamma * grad_f(y_k)                    # 梯度步（在外推点处）
        x_new = soft_threshold(z, gamma * lam)           # 近端步
        curve[k] = cost(x_new)
        t_old = t_k
        x_old = xk
        xk = x_new
    return xk, curve


max_iter = 2000
print(f"\n[步骤 1] 运行 ISTA 与 FISTA（各 {max_iter} 次）...")
t0 = time.time()
x_ista, curve_ista = run_ista(max_iter)
t_ista = time.time() - t0
print(f"  ISTA 完成，耗时 {t_ista:.2f} 秒")

t0 = time.time()
x_fista, curve_fista = run_fista(max_iter)
t_fista = time.time() - t0
print(f"  FISTA 完成，耗时 {t_fista:.2f} 秒")

# 参考最优值与参考解：用 FISTA 长迭代估计（1D 问题代价低，可跑足量迭代）
ref_iter = 20000
x_ref, curve_ref = run_fista(ref_iter)
F_star = float(curve_ref[-1])

gap_ista = np.maximum(curve_ista - F_star, 1e-15)   # 数值下限保护
gap_fista = np.maximum(curve_fista - F_star, 1e-15)
gap0 = float(curve_ista[0] - F_star)                # 初始偏差基准：第一步后的 F(x_1)-F*（两法共用同一基准）
print(f"  参考最优值 F* ≈ {F_star:.6f}（FISTA {ref_iter} 次估计）")
print(f"  初始偏差 F(x0) - F* = {gap0:.4f}")

# 达容差判停：F(x_k) - F* <= tol_rel * (F(x0) - F*)
# 容差取初始偏差的 2%：既保证 ISTA 在 2000 次内可达（其 O(1/k) 速率较慢），
# 又是足够实际的精度要求；更紧的容差只会放大两者的迭代数差距，不改变结论方向
tol_rel = 2e-2
tol_abs = tol_rel * gap0


def first_reach(gap, tol):
    """第一次满足 gap <= tol 的迭代数；若始终未满足返回 None"""
    hit = np.nonzero(gap <= tol)[0]
    return int(hit[0]) + 1 if len(hit) > 0 else None


k_ista = first_reach(gap_ista, tol_abs)
k_fista = first_reach(gap_fista, tol_abs)
speedup = k_ista / k_fista


def fit_loglog_slope(gap, k_start, k_end):
    """log-log 斜率拟合：在指定窗口 [k_start, k_end] 内拟合
    斜率即收敛指数（理论值：ISTA -1，FISTA -2）"""
    k_arr = np.arange(k_start, k_end + 1)
    slope, _ = np.polyfit(np.log(k_arr), np.log(gap[k_start - 1:k_end]), 1)
    return float(slope), (k_start, k_end)


# 斜率拟合窗口（实验确定性种子下固定）：
# - ISTA 局部斜率直到 k≈1500 才逼近 -1（k=500 处约 -0.6），k=500~2000 为过渡段平均
# - FISTA 在中前段（k=50~300）局部斜率已达 -2.3~-3.1，全窗口均陡于理论 -2；
#   k>500 后活跃集逐步固定、问题在支撑集上呈（弱）强凸性，收敛快于 O(1/k^2)
#   （尾部局部斜率达 -6 量级，见 Beck 10.6 节），拟合窗口取中前段是对 FISTA 最保守的取法
slope_ista, fit_rng_ista = fit_loglog_slope(gap_ista, 500, max_iter)
slope_fista, fit_rng_fista = fit_loglog_slope(gap_fista, 50, 300)

rel_ista = gap_ista[k_ista - 1] / gap0
rel_fista = gap_fista[k_fista - 1] / gap0

print(f"\n[收敛对比]  容差: F(x_k)-F* <= {tol_rel:.0e}*(F(x0)-F*) = {tol_abs:.3e}")
print(f"  ISTA : 达容差 {k_ista} 次 | 停时相对偏差 {rel_ista:.2e} | "
      f"log-log 斜率 {slope_ista:.3f}（k={fit_rng_ista[0]}~{fit_rng_ista[1]}）")
print(f"  FISTA: 达容差 {k_fista} 次 | 停时相对偏差 {rel_fista:.2e} | "
      f"log-log 斜率 {slope_fista:.3f}（k={fit_rng_fista[0]}~{fit_rng_fista[1]}）")
print(f"  加速比 = {speedup:.2f}x")

# 与最优解的偏差（收敛度量：两法解同一 LASSO，最优解相同）
# 注意不用 ||x_k - x_true||/||x_true||：LASSO 解因正则偏差与模糊本身不等于真信号，
# 该距离不反映算法收敛快慢
err_ista = np.linalg.norm(x_ista - x_ref) / np.linalg.norm(x_ref)
err_fista = np.linalg.norm(x_fista - x_ref) / np.linalg.norm(x_ref)
print(f"\n[与最优解的偏差] ||x_k - x_ref||/||x_ref||（x_ref 为 FISTA {ref_iter} 次解）")
print(f"  ISTA  ({max_iter} 次): {err_ista:.4f}")
print(f"  FISTA ({max_iter} 次): {err_fista:.4f}")

# ══════════════════════════════════════════════════════════
# 4. 步骤1 可视化：收敛曲线 + 1D 重建对比
# ══════════════════════════════════════════════════════════
print("\n[步骤 2] 绘制收敛曲线与重建对比...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ---- 左上：log-log 收敛曲线（主图）----
k_arr = np.arange(1, max_iter + 1)
# 参考斜率线：从各自第 100 次迭代锚定，斜率 -1 与 -2
anchor_ista = gap_ista[99]
anchor_fista = gap_fista[99]
ref_1k = anchor_ista * 100 / k_arr            # O(1/k) 参考
ref_1k2 = anchor_fista * 100 ** 2 / k_arr ** 2  # O(1/k^2) 参考

axes[0, 0].loglog(k_arr, gap_ista, 'b-', linewidth=2, label=r'ISTA: $F(x_k)-F^*$')
axes[0, 0].loglog(k_arr, gap_fista, 'r-', linewidth=2, label=r'FISTA: $F(x_k)-F^*$')
axes[0, 0].loglog(k_arr, ref_1k, 'b--', alpha=0.4, label=r'$O(1/k)$ 参考')
axes[0, 0].loglog(k_arr, ref_1k2, 'r--', alpha=0.4, label=r'$O(1/k^2)$ 参考')
axes[0, 0].axhline(tol_abs, color='gray', linestyle=':', linewidth=1,
                   label=f'容差 {tol_rel:.0e}·(F(x0)-F*)')
axes[0, 0].set_xlabel(r'迭代次数 $k$')
axes[0, 0].set_ylabel(r'$F(x_k) - F^*$')
axes[0, 0].set_title(f'收敛曲线 (log-log)：ISTA 实测斜率 {slope_ista:.2f}，'
                     f'FISTA 实测斜率 {slope_fista:.2f}')
axes[0, 0].legend(fontsize=9, loc='lower left')
axes[0, 0].grid(True, which='both', alpha=0.3)

# ---- 右上：前 100 次迭代放大（FISTA 早期领先最显著）----
axes[0, 1].semilogy(k_arr[:100], gap_ista[:100], 'b-o', markersize=3,
                    linewidth=1.5, label='ISTA')
axes[0, 1].semilogy(k_arr[:100], gap_fista[:100], 'r-s', markersize=3,
                    linewidth=1.5, label='FISTA')
axes[0, 1].set_xlabel(r'迭代次数 $k$')
axes[0, 1].set_ylabel(r'$F(x_k) - F^*$')
axes[0, 1].set_title(r'前 100 次迭代：FISTA 早期领先')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# ---- 左下：任务示意——真信号与无噪模糊观测 ----
# 注：LASSO 解因正则偏差与病态反卷积本身含振荡伪影，此处展示任务而非解；
# 算法收敛质量由左上收敛曲线与"与最优解偏差"度量
t_axis = np.arange(n)
y_clean = forward(x_true)
axes[1, 0].plot(t_axis, x_true, 'k-', linewidth=2, label='真信号 $x_{true}$', alpha=0.7)
axes[1, 0].plot(t_axis, y_clean, 'g-', linewidth=1.2, label=r'模糊观测 $Ax_{true}$（无噪）')
axes[1, 0].set_xlabel('样本索引')
axes[1, 0].set_ylabel('幅值')
axes[1, 0].set_title('任务：从模糊观测中恢复分段常值信号（去卷积）')
axes[1, 0].legend(fontsize=9, loc='upper right')
axes[1, 0].grid(True, alpha=0.3)

# ---- 右下：含噪观测 y ----
axes[1, 1].plot(t_axis, y, 'g-', linewidth=0.8)
axes[1, 1].set_xlabel('样本索引')
axes[1, 1].set_ylabel('幅值')
axes[1, 1].set_title(r'含噪模糊观测 $y = Ax_{true} + \eta$'
                     f'（$\\sigma$={sigma_noise}）')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(r'实验3.4-2: ISTA vs FISTA 收敛速率对比'
             r'（$\lambda=0.1$, $\gamma=0.99/L_f$, 最大 2000 次）',
             fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path1 = os.path.join(SAVE_DIR, '步骤1_ISTA_vs_FISTA收敛对比.png')
plt.savefig(out_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存：{out_path1}")

# ══════════════════════════════════════════════════════════
# 5. 步骤2：2D 玩具问题展示"梯度步 + 近端步"两段轨迹
# ══════════════════════════════════════════════════════════
print("\n[步骤 3] 绘制'梯度步 + 近端步'两段轨迹示意...")

# 简单 2D 复合凸问题：f(x1,x2) = 0.5*(x1^2 + 4*x2^2)，g = lam*(|x1| + |x2|)
# 便于在等高线图上同时看清两个子步


def f_simple(z):
    return 0.5 * (z[0] ** 2 + 4 * z[1] ** 2)


def grad_f_simple(z):
    return np.array([z[0], 4 * z[1]])


lam_demo = 0.25
tau_demo = 0.2  # 满足 tau < 1/L（此处 L=4）；阈值 tau*lambda=0.05 与梯度步尺度相配，
                # 使"陡峭梯度步 + 向原点收缩的近端步"的 L 形两段结构清晰可见


def prox_l1_2d(v, thr):
    """二维 L1 近端算子 = 逐分量软阈值（L1 可分，向量近端=分量近端）"""
    return np.array([np.sign(v[0]) * max(abs(v[0]) - thr, 0.0),
                     np.sign(v[1]) * max(abs(v[1]) - thr, 0.0)])


def trace_ista(x0, n_steps):
    """ISTA 轨迹：记录每步 (x_k, 梯度步终点 z_k, 近端步终点 x_{k+1})"""
    x_curr = x0.copy()
    segs = []                       # (x_k, z_k, x_next)
    pts = [x_curr.copy()]           # 近端步终点序列
    for _ in range(n_steps):
        z = x_curr - tau_demo * grad_f_simple(x_curr)
        x_next = prox_l1_2d(z, tau_demo * lam_demo)
        segs.append((x_curr.copy(), z.copy(), x_next.copy()))
        pts.append(x_next.copy())
        x_curr = x_next
    return segs, np.array(pts)


def trace_fista(x0, n_steps):
    """FISTA 轨迹：记录每步 (外推点 y_k, 梯度步终点 z_k, 近端步终点 x_{k+1})"""
    x_curr = x0.copy()
    x_old = x0.copy()
    t_old = 1.0
    segs = []                       # (y_k, z_k, x_next)
    pts = [x_curr.copy()]
    for _ in range(n_steps):
        t_k = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_old ** 2))
        y_k = x_curr + (t_old - 1.0) / t_k * (x_curr - x_old)   # 外推点
        z = y_k - tau_demo * grad_f_simple(y_k)
        x_next = prox_l1_2d(z, tau_demo * lam_demo)
        segs.append((y_k.copy(), z.copy(), x_next.copy()))
        pts.append(x_next.copy())
        t_old = t_k
        x_old = x_curr
        x_curr = x_next
    return segs, np.array(pts)


x0_demo = np.array([1.8, 1.5])
n_demo_steps = 25
ista_segs, ista_pts = trace_ista(x0_demo, n_demo_steps)
fista_segs, fista_pts = trace_fista(x0_demo, n_demo_steps)

# 等高线网格（总目标函数）
x1g = np.linspace(-2.2, 2.2, 200)
x2g = np.linspace(-1.4, 1.4, 200)
X1, X2 = np.meshgrid(x1g, x2g)
F_total = 0.5 * (X1 ** 2 + 4 * X2 ** 2) + lam_demo * (np.abs(X1) + np.abs(X2))
levels = np.logspace(-1, 1.5, 20)


def draw_two_step(ax, segs, pts, color, label_alg, label_grad, label_prox, label_extra):
    """画两段轨迹：x_k→z_k（梯度步）与 z_k→x_{k+1}（近端步）"""
    for i, (p0, p1, p2) in enumerate(segs):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='steelblue',
                lw=1.3, alpha=0.85, label=label_grad if i == 0 else None)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='darkorange',
                lw=1.3, alpha=0.85, label=label_prox if i == 0 else None)
        if label_extra is not None and not np.allclose(p0, p2):
            ax.plot(p0[0], p0[1], marker='o', markersize=4, mfc='none',
                    mec='purple', mew=1.2,
                    label=label_extra if i == 0 else None)
    ax.plot(pts[:, 0], pts[:, 1], color=color, marker='o', markersize=4,
            linewidth=1.0, alpha=0.6, label=label_alg)
    for j, k in enumerate([0, 1, 3, 7, 25]):
        if k < len(pts):
            dy = 6 if j % 2 == 0 else -13   # 上下交替偏移，防后段密集点标注重叠
            ax.annotate(f'$k={k}$', pts[k], textcoords="offset points",
                        xytext=(5, dy), fontsize=9, color=color)


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：ISTA 轨迹
cs = axes[0].contour(X1, X2, F_total, levels=levels, alpha=0.4, cmap='viridis')
draw_two_step(axes[0], ista_segs, ista_pts, 'b', 'ISTA 轨迹点',
              r'梯度步 $x_k \to z_k$', r'近端步 $z_k \to x_{k+1}$', None)
axes[0].plot(0, 0, 'r*', markersize=18, label=r'最优点 $x^* = (0,0)$')
axes[0].set_xlabel(r'$x_1$')
axes[0].set_ylabel(r'$x_2$')
axes[0].set_title(r'ISTA："梯度步 → 近端步"两段结构')
axes[0].legend(fontsize=9, loc='lower left')
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')
axes[0].set_ylim(axes[0].get_ylim()[0], axes[0].get_ylim()[1] + 0.18)  # 顶部留白防 k=0 标注被裁

# 右图：FISTA 轨迹（多一个外推点 y_k）
axes[1].contour(X1, X2, F_total, levels=levels, alpha=0.4, cmap='viridis')
draw_two_step(axes[1], fista_segs, fista_pts, 'r', 'FISTA 轨迹点',
              r'梯度步 $y_k \to z_k$', r'近端步 $z_k \to x_{k+1}$',
              r'外推点 $y_k$')
axes[1].plot(0, 0, 'r*', markersize=18, label=r'最优点 $x^* = (0,0)$')
axes[1].set_xlabel(r'$x_1$')
axes[1].set_ylabel(r'$x_2$')
axes[1].set_title(r'FISTA：Nesterov 外推（空心圈为外推点 $y_k$）')
axes[1].legend(fontsize=9, loc='lower left')
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal')
axes[1].set_ylim(axes[1].get_ylim()[0], axes[1].get_ylim()[1] + 0.18)

plt.suptitle(r'实验3.4-2 步骤2: "梯度步 + 近端步"两段轨迹'
             r'（$f = 0.5(x_1^2 + 4x_2^2)$, $g = \lambda(|x_1| + |x_2|)$）',
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path2 = os.path.join(SAVE_DIR, '步骤2_梯度步近端步迭代轨迹.png')
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存：{out_path2}")

# ══════════════════════════════════════════════════════════
# 6. 保存数值结果到 JSON
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
    'experiment': '实验3.4-2 ISTA 与 FISTA 的收敛速率对比（1D 信号去卷积）',
    'scene': {
        'signal_length': n,
        'signal_type': '分段常值（随机段长15~40，幅值±0.3~1.0）',
        'blur_sigma': sigma_blur,
        'noise_sigma': sigma_noise,
        'lambda': lam,
        'L_exact': L_exact,
        'L_power_iteration': L_power,
        'gamma': gamma,
        'max_iter': max_iter,
        'ref_iter_for_F_star': ref_iter,
        'tol_relative': tol_rel,
        'tol_abs': float(tol_abs),
    },
    'convergence': {
        'gap0': float(gap0),
        'k_ista_to_tol': k_ista,
        'k_fista_to_tol': k_fista,
        'speedup_ratio': float(speedup),
        'relative_gap_at_stop_ista': float(rel_ista),
        'relative_gap_at_stop_fista': float(rel_fista),
        'measured_slope_ista': float(slope_ista),
        'measured_slope_fista': float(slope_fista),
        'slope_fit_range_ista': f'k={fit_rng_ista[0]}~{fit_rng_ista[1]}',
        'slope_fit_range_fista': f'k={fit_rng_fista[0]}~{fit_rng_fista[1]}',
    },
    'reconstruction': {
        'dist_to_opt_ista': float(err_ista),
        'dist_to_opt_fista': float(err_fista),
    },
    'output_files': [out_path1, out_path2],
}

with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(_to_native(results), f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print("1. ISTA 每步 = 梯度步（沿光滑数据项下降）+ 近端步（软阈值施加稀疏性）")
print("2. FISTA 在 ISTA 基础上加 Nesterov 外推动量，每步只多一次向量加法")
print(f"3. 达到容差 F(x_k)-F* <= {tol_rel:.0e}*(F(x0)-F*) 所需迭代："
      f"ISTA {k_ista} 次 vs FISTA {k_fista} 次，加速 {speedup:.2f} 倍")
print(f"4. log-log 斜率：ISTA {slope_ista:.3f} ≈ -1（O(1/k)）；FISTA {slope_fista:.3f}，"
      f"不慢于理论 -2（O(1/k^2) 为最坏情形上界，该问题上中段略陡，"
      f"k>500 后活跃集固定、收敛进一步加快）")
print(f"5. 2000 次迭代后与最优解的偏差：ISTA {err_ista:.4f}，FISTA {err_fista:.4f}"
      f"（解同一问题最优解相同，FISTA 快得多）")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")
print("  - 步骤1_ISTA_vs_FISTA收敛对比.png")
print("  - 步骤2_梯度步近端步迭代轨迹.png")
print("  - results_summary.json")
