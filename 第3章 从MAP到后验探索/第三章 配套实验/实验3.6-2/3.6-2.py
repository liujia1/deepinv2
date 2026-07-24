# -*- coding: utf-8 -*-
"""
实验3.6-2 Landweber 半收敛：U 形误差曲线
对应章节：3.6 收敛性分析与正则化参数选择（核心例子）
        3.3 节 补充说明（Landweber 迭代作为最简单的迭代正则化方法）
知识点：
  - Landweber 迭代：x_{k+1} = x_k - tau * K^T(Kx_k - y)
  - 半收敛现象：误差 ||x_k - x_true|| 随 k 先降后升（U 形曲线）
  - 迭代次数 k 本身即"正则化参数"——与 3.3-1（用 lambda 作参数）形成对比
  - 早停/晚停的反差
  - 离散 Picard 条件

素材来源（理论参考）：
  - Vogel, S.M.  Comput. Phys.  (2002) 第 5 章
  - winter school 讲义
  - 与 3.6 节 L-曲线（Morozov 偏差原理）方法形成姊妹实验

运行环境：纯 CPU 即可，无需 GPU，无外部权重文件。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 静默模式
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
    SAVE_DIR = os.path.join(_gdrive, '实验3.6-2')
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

print("=" * 60)
print("实验3.6-2 Landweber 半收敛：U 形误差曲线")
print("=" * 60)
print("运行环境: 纯 CPU，无需 GPU，无外部权重文件")
print("  - 1D 反演问题（对角算子，与 winter school lab / Vogel 2002 一致）")
print("  - 目标：直观展示'先下降后上升'的半收敛现象 + 迭代次数即正则化参数")

# ══════════════════════════════════════════════════════════
# 1. 构造对角前向算子（IP10 风格）
# ══════════════════════════════════════════════════════════
print("\n[步骤 1] 构造对角前向算子...")

n = 64                              # 信号维度
# 奇异值：sigma_i = exp(-5*i/(n-1))，i=0..n-1
# 随 i 增大快速衰减，模拟"病态"问题
sigma = np.exp(-5 * np.arange(n) / (n - 1))
# 1D 反演：K 是对角算子 diag(sigma)，无正交变换
K_diag = sigma.copy()


def forward(x):
    return sigma * x


def adjoint(y):
    return sigma * y


# 真解：分片常数（含突变 + 平滑区）
x_true = np.zeros(n)
x_true[0:10] = 1.0
x_true[15:30] = 0.5
x_true[40:55] = 0.8
# 频域视角：x_true 的 Fourier 系数 a_i = <x_true, v_i>
# 当奇异值 sigma_i 衰减比 |a_i| 快时，Picard 条件被破坏
a_coef = x_true.copy()

# 含噪观测
# 噪声水平需足够大才能观察到半收敛（否则噪声项始终小于信号项）
# 选择 sigma_n ≈ 0.05: 既能污染小 sigma 分量，又不淹没大 sigma 分量
sigma_noise = 0.05
y = forward(x_true) + sigma_noise * np.random.randn(n)

# Landweber 步长选择
# 收敛条件: 0 < tau < 2 / ||K^T K|| = 2 / max(sigma^2) = 2 / 1 = 2
# 最优步长（最快收敛）: tau_opt = 2 / (max + min) = 2 / (1 + sigma_min^2) ≈ 2
# 但 Vogel 2002 推荐保守步长 tau = 1 / L，L = max(sigma^2) = 1
tau = 1.0
L_K = np.max(K_diag ** 2)
print(f"  信号维度 n = {n}")
print(f"  奇异值范围: [{K_diag.min():.4f}, {K_diag.max():.4f}]")
print(f"  ||K^T K|| = {L_K:.4f}")
print(f"  步长 tau = {tau}（保守选择，满足 0 < tau < 2/L_K = {2 / L_K:.2f}）")

# ══════════════════════════════════════════════════════════
# 2. Landweber 迭代
# ══════════════════════════════════════════════════════════
print("\n[步骤 2] 运行 Landweber 迭代...")

maxiter = 2000
t_start = time.time()
xk = np.zeros(n)                    # 零初值
errors = np.zeros(maxiter + 1)      # ||x_k - x_true||
residual_norms = np.zeros(maxiter + 1)  # ||Kx_k - y||
errors[0] = np.linalg.norm(xk - x_true)
residual_norms[0] = np.linalg.norm(forward(xk) - y)

print(f"  Landweber 进度: 0/{maxiter}", end="", flush=True)
for k in range(1, maxiter + 1):
    # 梯度步
    xk = xk - tau * adjoint(forward(xk) - y)
    errors[k] = np.linalg.norm(xk - x_true)
    residual_norms[k] = np.linalg.norm(forward(xk) - y)
    if k % 20 == 0 or k == maxiter:
        print(f"\r  Landweber 进度: {k}/{maxiter}, 误差 = {errors[k]:.4f}", end="", flush=True)
print(f"\n  Landweber 完成，耗时 {time.time() - t_start:.2f} 秒")

# ══════════════════════════════════════════════════════════
# 3. 分析半收敛现象
# ══════════════════════════════════════════════════════════
print("\n[步骤 3] 分析半收敛现象...")

# 找最优停止点（误差最小）
k_opt = int(np.argmin(errors))
err_min = errors[k_opt]
err_final = errors[-1]
err_0 = errors[0]

print(f"  误差最小点: k_opt = {k_opt}, ||x_k_opt - x_true|| = {err_min:.4f}")
print(f"  初值误差: ||x_0 - x_true|| = {err_0:.4f}")
print(f"  最终误差（k={maxiter}）: ||x_{maxiter} - x_true|| = {err_final:.4f}")
print(f"  半收敛比率: err_{maxiter} / err_{k_opt} = {err_final / err_min:.2f}")
print(f"  -> 误差先从 {err_0:.4f} 降到 {err_min:.4f}，再升到 {err_final:.4f}，呈典型 U 形")

# Picard 条件：x_true 的系数 |a_i| 必须比 1/sigma_i 衰减慢才能被稳定恢复
print(f"\n[Picard 条件分析]")
print(f"  真解系数 a_i = x_true 本身（对角算子下频域即原域）")
print(f"  1/sigma_i = exp(5*i/(n-1))，增长指数级")
# 离散 Picard 条件：稳定恢复要求 |a_i| * sigma_i >> noise_level
# 在有效支撑区近似令 |a_i| ≈ 1，解 sigma_i * |a_i| ≈ noise_level
# 即 exp(-5*i/(n-1)) ≈ sigma_noise → i ≈ -(n-1)/5 * ln(sigma_noise)
# 注意：这是对分段真解 |a_i| 的粗略估计（本例 x_true 在零值段会直接跌破噪声线），
# 具体失效索引应以 ax4 图上 |a_i|*sigma_i 曲线与噪声传递曲线的交点为准。
i_threshold = int(-(n - 1) / 5 * np.log(sigma_noise))
print(f"  粗略估计：当 i > ~{i_threshold} 时 |a_i| * sigma_i ≈ sigma_i < noise_level")
print(f"    （假设 |a_i| 在有效支撑区近似不变；本例 x_true 含零值段，精确失效点见下方 ax4 图）")
print(f"  → 从该索引往后，噪声开始压过被恢复的信号系数，Picard 条件（稳定恢复条件）逐步失效")

# 三个关键时刻的解
x_early = np.zeros(n)               # k=0 几乎为 0
x_opt = np.zeros(n)
xk = np.zeros(n)
for k in range(1, k_opt + 1):
    xk = xk - tau * adjoint(forward(xk) - y)
x_opt = xk.copy()

xk = np.zeros(n)
for k in range(1, maxiter + 1):
    xk = xk - tau * adjoint(forward(xk) - y)
x_late = xk.copy()

# ══════════════════════════════════════════════════════════
# 4. 可视化
# ══════════════════════════════════════════════════════════
print("\n[步骤 4] 绘制 U 形误差曲线 + 三个时刻的解...")

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# 左：U 形误差曲线（核心可视化）
ax1.semilogy(errors, 'b-', linewidth=2, label=r'$\|x_k - x_{true}\|_2$')
ax1.axvline(k_opt, color='r', linestyle='--', alpha=0.7,
            label=r'$k_{opt} = ' + str(k_opt) + '$')
ax1.scatter([k_opt], [err_min], color='red', s=80, zorder=5,
            label=r'最小误差 = ' + f'{err_min:.4f}')
ax1.set_xlabel(r'迭代次数 $k$')
ax1.set_ylabel(r'$\|x_k - x_{true}\|_2$')
ax1.set_title(r'Landweber 半收敛：误差先降后升的 U 形曲线')
ax1.legend(fontsize=10, loc='lower right')
ax1.grid(True, alpha=0.3)
# y 轴下限设为最小误差的 80%，上限设为最终误差的 1.15 倍，让 U 形充分展示
y_bottom = err_min * 0.8
y_top = max(err_0, err_final) * 1.15
ax1.set_ylim(bottom=y_bottom, top=y_top)
# 注释：早停 / 晚停（文字放在图表内部，避免跑出边框）
k_early_pos = max(5, k_opt // 4)
ax1.annotate('早停\n(欠拟合)', xy=(k_early_pos, errors[k_early_pos]),
             xytext=(k_opt * 0.35, err_min * 1.15),
             fontsize=10, color='blue',
             arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6))
k_late_pos = int(maxiter * 0.85)
ax1.annotate('晚停\n(噪声放大)', xy=(k_late_pos, errors[k_late_pos]),
             xytext=(maxiter * 0.6, err_final * 0.85),
             fontsize=10, color='red',
             arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))

# 右：残差范数（用于 Morozov 偏差原理）
ax2.semilogy(residual_norms, 'g-', linewidth=2, label=r'$\|Kx_k - y\|_2$')
ax2.axhline(sigma_noise * np.sqrt(n), color='orange', linestyle='--', alpha=0.7,
            label=r'噪声水平 $\tau \|n\| = ' + f'{sigma_noise * np.sqrt(n):.4f}' + r'$')
ax2.set_xlabel(r'迭代次数 $k$')
ax2.set_ylabel(r'$\|Kx_k - y\|_2$')
ax2.set_title(r'残差范数（Morozov 偏差原理判据）')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3)

plt.suptitle(r'实验3.6-2: Landweber 半收敛——"先下降后上升"的 U 形误差曲线',
             fontsize=14)
plt.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.10, wspace=0.25)
out_path1 = os.path.join(SAVE_DIR, '步骤1a_U形误差曲线.png')
plt.savefig(out_path1, dpi=150)
plt.close()
print(f"  已保存：{out_path1}")

# 第二张图：三个时刻的解 + Picard 条件
fig = plt.figure(figsize=(14, 6))
ax3 = fig.add_subplot(1, 2, 1)
ax4 = fig.add_subplot(1, 2, 2)

# 三个时刻的解
k_early = max(1, k_opt // 4)
xk = np.zeros(n)
for k in range(1, k_early + 1):
    xk = xk - tau * adjoint(forward(xk) - y)
x_early = xk.copy()

x_idx = np.arange(n)
ax3.plot(x_idx, x_true, 'k-', linewidth=2.5, label=r'真解 $x_{true}$')
ax3.plot(x_idx, x_early, 'b--', linewidth=1.5, alpha=0.7,
         label=fr'早停 $k={k_early}$')
ax3.plot(x_idx, x_opt, 'g-', linewidth=1.5,
         label=fr'最优 $k={k_opt}$')
ax3.plot(x_idx, x_late, 'r--', linewidth=1.5, alpha=0.7,
         label=fr'晚停 $k={maxiter}$')
ax3.set_xlabel(r'像素 $i$')
ax3.set_ylabel(r'信号值')
ax3.set_title(r'三个迭代时刻的解对比')
ax3.legend(fontsize=9, loc='upper right')
ax3.grid(True, alpha=0.3)

# Picard 条件可视化
ax4.semilogy(K_diag, 'b-', linewidth=2, label=r'奇异值 $\sigma_i$')
ax4.semilogy(np.abs(x_true), 'k-', linewidth=2, label=r'真解系数 $|a_i| = |x_i^{true}|$')
ax4.semilogy(np.abs(x_true) * K_diag, 'g-', linewidth=2,
             label=r'$|a_i| \sigma_i$（被恢复的部分）')
noise_curve = sigma_noise / K_diag
ax4.semilogy(noise_curve, 'r--', alpha=0.7,
             label=r'噪声传递 $\sigma_{noise} / \sigma_i$')
ax4.set_xlabel(r'索引 $i$')
ax4.set_ylabel(r'幅度')
ax4.set_title(r'Picard 条件：$\sigma_i$ 衰减与真解系数')
ax4.legend(fontsize=9, loc='upper right')
ax4.grid(True, alpha=0.3)
ax4.set_ylim(1e-6, 10)

plt.suptitle(r'实验3.6-2: 半收敛的解与 Picard 条件分析',
             fontsize=14)
plt.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.10, wspace=0.25)
out_path1b = os.path.join(SAVE_DIR, '步骤1b_解对比与Picard条件.png')
plt.savefig(out_path1b, dpi=150)
plt.close()
print(f"  已保存：{out_path1b}")

# ══════════════════════════════════════════════════════════
# 5. 不同噪声水平下的半收敛（多个 U 形曲线叠加）
# ══════════════════════════════════════════════════════════
print("\n[步骤 5] 不同噪声水平下的半收敛对比...")

sigma_noise_levels = [1e-2, 2e-2, 5e-2, 1e-1, 2e-1]
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

colors = plt.cm.viridis(np.linspace(0, 0.9, len(sigma_noise_levels)))

for idx, noise_level in enumerate(sigma_noise_levels):
    np.random.seed(42)
    y_noise = forward(x_true) + noise_level * np.random.randn(n)
    xk = np.zeros(n)
    errs = np.zeros(maxiter + 1)
    errs[0] = np.linalg.norm(x_true)  # 显式初始化，避免 argmin 错误返回 0
    for k in range(1, maxiter + 1):
        xk = xk - tau * adjoint(forward(xk) - y_noise)
        errs[k] = np.linalg.norm(xk - x_true)
    k_opt_n = int(np.argmin(errs))
    axes[0].semilogy(errs, color=colors[idx], linewidth=1.5,
                     label=fr'$\sigma_n = {noise_level}$, $k_{{opt}} = {k_opt_n}$')
    axes[0].scatter([k_opt_n], [errs[k_opt_n]], color=colors[idx], s=30, zorder=5)

axes[0].set_xlabel(r'迭代次数 $k$')
axes[0].set_ylabel(r'$\|x_k - x_{true}\|_2$')
axes[0].set_title(r'不同噪声水平下的半收敛曲线')
axes[0].legend(fontsize=9, loc='lower right')
axes[0].grid(True, alpha=0.3)

# 右图：最优迭代次数 k_opt 随噪声水平的变化
k_opts = []
for noise_level in sigma_noise_levels:
    np.random.seed(42)
    y_noise = forward(x_true) + noise_level * np.random.randn(n)
    xk = np.zeros(n)
    errs = np.zeros(maxiter + 1)
    errs[0] = np.linalg.norm(x_true)
    for k in range(1, maxiter + 1):
        xk = xk - tau * adjoint(forward(xk) - y_noise)
        errs[k] = np.linalg.norm(xk - x_true)
    k_opts.append(int(np.argmin(errs)))

axes[1].semilogx(sigma_noise_levels, k_opts, 'bo-', markersize=8, linewidth=2)
axes[1].set_xlabel(r'噪声水平 $\sigma_n$')
axes[1].set_ylabel(r'最优迭代次数 $k_{opt}$')
axes[1].set_title(r'噪声越大，最优停止越早（Morozov 偏差原理的体现）')
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(sigma_noise_levels)
axes[1].set_xticklabels([f'{s:.0e}' for s in sigma_noise_levels], rotation=30)

plt.suptitle(r'实验3.6-2: 半收敛曲线族——迭代次数 $k$ 即"正则化参数"',
             fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path2 = os.path.join(SAVE_DIR, '步骤2_不同噪声下半收敛曲线族.png')
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存：{out_path2}")

# ══════════════════════════════════════════════════════════
# 6. 保存数值结果到 JSON
# ══════════════════════════════════════════════════════════


def _to_native(obj):
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
    'experiment': '实验3.6-2 Landweber 半收敛：U 形误差曲线',
    'scene': {
        'signal_dim': n,
        'sigma_diag_min': float(K_diag.min()),
        'sigma_diag_max': float(K_diag.max()),
        'noise_sigma': float(sigma_noise),
        'step_size_tau': float(tau),
        'maxiter': maxiter,
    },
    'semiconvergence': {
        'k_opt': int(k_opt),
        'err_at_k0': float(err_0),
        'err_at_k_opt': float(err_min),
        'err_at_k_max': float(err_final),
        'semiconvergence_ratio': float(err_final / err_min),
    },
    'k_opt_vs_noise': {
        str(noise_level): int(k_opt_n)
        for noise_level, k_opt_n in zip(sigma_noise_levels, k_opts)
    },
    'output_files': [out_path1, out_path1b, out_path2],
}

with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(_to_native(results), f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print(f"1. Landweber 迭代的误差 ||x_k - x_true|| 呈典型 U 形：")
print(f"   初值 ({err_0:.4f}) → 最优点 k={k_opt} ({err_min:.4f}) → 最终 ({err_final:.4f})")
print(f"2. '半收敛'——早停保留信号但欠拟合，晚停过度拟合噪声")
print(f"3. 迭代次数 k 本身即'正则化参数'（与 3.3-1 中用 lambda 作参数形成对比）")
print(f"4. 噪声水平越高，最优停止越早（Morozov 偏差原理的体现）")
print(f"5. 本实验场景在 3.6 节收敛性分析中具核心地位")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")
print(f"  - 步骤1a_U形误差曲线.png")
print(f"  - 步骤1b_解对比与Picard条件.png")
print(f"  - 步骤2_不同噪声下半收敛曲线族.png")
print(f"  - results_summary.json")
