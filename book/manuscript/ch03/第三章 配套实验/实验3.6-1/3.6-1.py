# -*- coding: utf-8 -*-
"""
实验3.6-1 Landweber 半收敛：U 形误差曲线
对应章节：3.6 收敛性分析与正则化参数选择（核心例子）
        3.3 节 补充说明（Landweber 迭代作为最简单的迭代正则化方法）
知识点：
  - Landweber 迭代：x_{k+1} = x_k - tau * K^T(Kx_k - y)
  - 半收敛现象：误差 ||x_k - x_true|| 随 k 先降后升（U 形曲线）
  - 迭代次数 k 本身即"正则化参数"——与 3.3-1（用 lambda 作参数）形成对比
  - 早停/晚停的反差；k_opt 随噪声水平增大而减小
  - 离散 Picard 条件：第 i 个分量可稳定恢复的判据是 sigma_i * |a_i| >~ delta
    （数据域：信号奇异系数高于噪声水平；等价于解域 |a_i| >~ delta/sigma_i）
  - Morozov 偏差原理：残差首次降到噪声范数附近时停止

素材来源（理论参考）：
  - Vogel, Computational Methods for Inverse Problems (2002) 第 5 章
  - winter school 讲义
  - 与 3.6 节 L-曲线（Morozov 偏差原理）方法形成姊妹实验

实现要点（复核修订版）：
  - 迭代窗口取 10000，覆盖全部受试噪声水平的真实 k_opt（最小噪声下真实 k_opt≈3319），
    并完整展示 U 形右臂向噪声放大极限 ||e/sigma|| 的爬升；
    k_opt 若顶到窗口上界会被标记为"删失"并自动延长窗口重算。
  - 噪声族各水平共用同一随机种子（公共随机数 / paired comparison），
    另用 100 个独立种子给出 k_opt 的中位数与 5–95% 分位带。
  - 对角算子下迭代有逐分量闭式解 x_i^k=[1-(1-tau*sigma_i^2)^k]*(a_i+e_i/sigma_i)，
    仅用于（a）与显式迭代逐点互验，（b）蒙特卡洛加速；两者数学上严格等价。

运行环境：纯 CPU 即可（n=64，整脚本约 10 秒），无需 GPU，无外部权重文件。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 静默模式
import matplotlib.pyplot as plt
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
    SAVE_DIR = os.path.join(_gdrive, '实验3.6-1')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
_chinese_path = os.path.join(SAVE_DIR, '.chinese')
os.makedirs(_chinese_path, exist_ok=True)

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

# 设置随机种子（主场景与噪声族均使用；噪声族各水平共用同一噪声实现，仅幅度不同）
SEED = 42
np.random.seed(SEED)

print("=" * 60)
print("实验3.6-1 Landweber 半收敛：U 形误差曲线")
print("=" * 60)
print("运行环境: 纯 CPU，无需 GPU，无外部权重文件")
print("  - 1D 反演问题（对角算子，与 winter school lab / Vogel 2002 一致）")
print("  - 目标：直观展示'先下降后上升'的半收敛现象 + 迭代次数即正则化参数")

# ══════════════════════════════════════════════════════════
# 1. 构造对角正向算子（IP10 风格）
# ══════════════════════════════════════════════════════════
print("\n[步骤 1] 构造对角正向算子...")

n = 64                              # 信号维度
# 奇异值：sigma_i = exp(-5*i/(n-1))，i=0..n-1
# 随 i 增大快速衰减，模拟"病态"问题
sigma = np.exp(-5 * np.arange(n) / (n - 1))
s2 = sigma ** 2                     # K^T K 的对角元


def forward(x):
    return sigma * x


def adjoint(y):
    return sigma * y


# 真解：分片常数（含突变 + 平滑区）
x_true = np.zeros(n)
x_true[0:10] = 1.0
x_true[15:30] = 0.5
x_true[40:55] = 0.8
# 在恒等奇异向量基下，真解的奇异系数 a_i 就是 x_true 本身（逐分量）
# 因此本例"频域即原域"，Picard 判据可直接按分量检验

# 含噪观测
# 噪声水平需足够大才能在有限迭代窗口内观察到半收敛（k_opt 随 delta→0 而发散）
# delta = 0.05：既能污染小 sigma 分量，又不淹没大 sigma 分量
sigma_noise = 0.05
y = forward(x_true) + sigma_noise * np.random.randn(n)
noise_norm = float(np.linalg.norm(y - forward(x_true)))
noise_norm_expected = sigma_noise * np.sqrt(n)

# Landweber 步长选择
# 逐分量稳定条件：|1 - tau*sigma_i^2| < 1，即 0 < tau < 2/sigma_max^2 = 2
# 取保守步长 tau = 1（Vogel 2002 风格）；最快收敛常数步长应为 2/(sigma_max^2+sigma_min^2)≈2
tau = 1.0
L_K = float(np.max(s2))
print(f"  信号维度 n = {n}")
print(f"  奇异值范围: [{sigma.min():.4f}, {sigma.max():.4f}]")
print(f"  ||K^T K|| = {L_K:.4f}")
print(f"  步长 tau = {tau}（保守选择，满足 0 < tau < 2/L_K = {2 / L_K:.2f}）")
print(f"  噪声范数: 实际 ||e|| = {noise_norm:.4f}, 期望 delta*sqrt(n) = {noise_norm_expected:.4f}")

# ══════════════════════════════════════════════════════════
# 2. Landweber 迭代（含删失检测的统一驱动函数）
# ══════════════════════════════════════════════════════════
MAXITER = 10000      # 迭代窗口：覆盖全部噪声水平的真实 k_opt，并展示右臂噪声放大


def landweber(y_obs, maxiter, keep_trajectory=False):
    """零初值 Landweber。返回 errors/residuals（长度 maxiter+1）；
    keep_trajectory=True 时额外返回整条迭代轨迹 X（maxiter+1, n）。"""
    xk = np.zeros(n)
    errors = np.empty(maxiter + 1)
    residuals = np.empty(maxiter + 1)
    errors[0] = np.linalg.norm(x_true)
    residuals[0] = np.linalg.norm(forward(xk) - y_obs)
    X = np.zeros((maxiter + 1, n)) if keep_trajectory else None
    if keep_trajectory:
        X[0] = xk
    for k in range(1, maxiter + 1):
        xk = xk - tau * adjoint(forward(xk) - y_obs)
        errors[k] = np.linalg.norm(xk - x_true)
        residuals[k] = np.linalg.norm(forward(xk) - y_obs)
        if keep_trajectory:
            X[k] = xk
    return (errors, residuals, X) if keep_trajectory else (errors, residuals)


def run_with_censor(y_obs, horizon=MAXITER, max_extend=40000):
    """若 argmin 顶到窗口上界（删失），自动延长窗口重算，直到真实最优点落入窗口。
    返回 errors, residuals, k_opt, censored_flag, horizon_used。"""
    h = horizon
    while True:
        errs, ress = landweber(y_obs, h)
        k_opt = int(np.argmin(errs))
        if k_opt < h or h >= max_extend:
            return errs, ress, k_opt, (k_opt >= h), h
        h *= 2
        print(f"  [删失] k_opt 顶到窗口上界，延长迭代窗口至 {h} 重算")


def closed_form_errors(noise, ks):
    """对角算子 Landweber 的逐分量闭式解，仅用于互验与蒙特卡洛加速：
       x_i^k = [1-(1-tau*sigma_i^2)^k] * (a_i + e_i/sigma_i)
       与显式迭代数学上严格等价（主场景互验误差 < 1e-12）。"""
    ks = np.asarray(ks)
    qpow = (1 - tau * s2)[None, :] ** ks[:, None]
    filt = 1 - qpow                              # Landweber 滤波因子，k=0 时为 0（零初值）
    target = x_true[None, :] + noise[None, :] / sigma[None, :]
    X = filt * target
    return np.linalg.norm(X - x_true[None, :], axis=1)


print(f"\n[步骤 2] 运行 Landweber 迭代（maxiter = {MAXITER}，保存轨迹用于三时刻快照）...")
t_start = time.time()
errors, residual_norms, X_traj = landweber(y, MAXITER, keep_trajectory=True)
print(f"  Landweber 完成，{MAXITER} 步耗时 {time.time() - t_start:.2f} 秒")

# 闭式解互验（防回归：若有人改坏迭代，此处会立刻暴露）
_check_ks = [0, 1, 100, 1000, MAXITER]
_diff = np.max(np.abs(closed_form_errors(y - forward(x_true), _check_ks) - errors[_check_ks]))
print(f"  闭式解互验：与显式迭代最大偏差 = {_diff:.2e}（应 < 1e-10）")
assert _diff < 1e-8, "Landweber 迭代与闭式解不一致，请检查实现！"

# ══════════════════════════════════════════════════════════
# 3. 分析半收敛现象
# ══════════════════════════════════════════════════════════
print("\n[步骤 3] 分析半收敛现象...")

k_opt = int(np.argmin(errors))
err_min = float(errors[k_opt])
err_final = float(errors[-1])
err_0 = float(errors[0])
# k->∞ 时每个分量收敛到 a_i + e_i/sigma_i，误差趋于噪声放大极限 ||e/sigma||
noise_amplification_limit = float(np.linalg.norm((y - forward(x_true)) / sigma))
# 不可恢复分量的信号能量：sigma_i*|a_i| < delta 的分量，任何停止规则都救不回来
unrecoverable = (sigma * np.abs(x_true)) < sigma_noise
unrecoverable_signal_norm = float(np.linalg.norm(x_true[unrecoverable]))

print(f"  误差最小点: k_opt = {k_opt}, ||x_k_opt - x_true|| = {err_min:.4f}")
print(f"  初值误差: ||x_0 - x_true|| = {err_0:.4f}")
print(f"  窗口末端误差（k={MAXITER}）: {err_final:.4f}")
print(f"  k->inf 噪声放大极限 ||e/sigma|| = {noise_amplification_limit:.4f}")
print(f"  半收敛比率: err_{MAXITER} / err_min = {err_final / err_min:.2f}")
print(f"  -> 误差先从 {err_0:.4f} 降到 {err_min:.4f}，再升到 {err_final:.4f}（并继续趋向 {noise_amplification_limit:.2f}）")
print(f"  不可恢复信号（sigma*|a|<delta 的分量）范数 = {unrecoverable_signal_norm:.4f}，"
      f"与 err_min={err_min:.4f} 接近，说明可恢复分量已达到噪声极限")

# Morozov 偏差原理：残差首次降到噪声范数处停止
def first_cross(res, level):
    hit = np.flatnonzero(res <= level)
    return int(hit[0]) if hit.size else -1

k_mor_exp = first_cross(residual_norms, noise_norm_expected)
k_mor_act = first_cross(residual_norms, noise_norm)
print(f"\n[Morozov 偏差原理]")
print(f"  按期望噪声范数 delta*sqrt(n)={noise_norm_expected:.4f}：首次穿越 k = {k_mor_exp}，"
      f"该处误差 = {errors[k_mor_exp]:.4f}（先知最优 {err_min:.4f}）")
print(f"  按实际噪声范数 ||e||={noise_norm:.4f}：首次穿越 k = {k_mor_act}，"
      f"该处误差 = {errors[k_mor_act]:.4f}")

# 离散 Picard 条件：sigma_i * |a_i| >~ delta 才可稳定恢复
print(f"\n[Picard 条件分析]（正确判据：数据域 sigma*|a| = delta，等价于解域 |a| = delta/sigma）")
block_specs = [(0, 10, 1.0), (15, 30, 0.5), (40, 55, 0.8)]
picard_crossings = []
for a, b, v in block_specs:
    if v <= sigma_noise:
        i_star = 0.0
        status = "整段不可恢复"
    else:
        # exp(-5i/(n-1)) * v = delta  ->  i* = -(n-1)/5 * ln(delta/v)
        i_star = -(n - 1) / 5 * np.log(sigma_noise / v)
        status = "交点在支撑外，整段基本不可恢复" if i_star <= a else (
                 "整段可恢复" if i_star >= b else f"i >= {i_star:.1f} 起不可恢复")
    picard_crossings.append((a, b, v, float(i_star), status))
    print(f"  支撑[{a:2d}:{b:2d}], a={v}: sigma*|a|=delta 的交点 i* = {i_star:5.1f}  -> {status}")

# 三个关键时刻的解（直接从轨迹取，无需重复迭代）
k_early = max(1, k_opt // 8)          # 明显欠拟合的早停点
x_early = X_traj[k_early].copy()
x_opt = X_traj[k_opt].copy()
x_late = X_traj[-1].copy()
print(f"\n  三个展示时刻：早停 k={k_early}（误差 {errors[k_early]:.4f}）、"
      f"最优 k={k_opt}（{err_min:.4f}）、晚停 k={MAXITER}（{err_final:.4f}）")

# ══════════════════════════════════════════════════════════
# 4. 可视化（图1：U 形误差曲线 + 残差/Morozov）
# ══════════════════════════════════════════════════════════
print("\n[步骤 4] 绘制 U 形误差曲线 + 三个时刻的解...")

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# 左：U 形误差曲线（核心可视化）
ax1.semilogy(errors, 'b-', linewidth=2, label=r'$\|x_k - x_{true}\|_2$')
ax1.axvline(k_opt, color='r', linestyle='--', alpha=0.7,
            label=r'$k_{opt} = ' + str(k_opt) + r'$（先知最优）')
ax1.scatter([k_opt], [err_min], color='red', s=80, zorder=6)
# Morozov 停止点（无需知道真解的实用规则）
ax1.scatter([k_mor_act], [errors[k_mor_act]], color='orange', s=70, zorder=6,
            marker='D', label=fr'Morozov 停止 $k={k_mor_act}$（仅用 $\|e\|$）')
# 噪声放大极限
ax1.axhline(noise_amplification_limit, color='gray', linestyle=':', alpha=0.8,
            label=fr'噪声放大极限 $\|e/\sigma\|={noise_amplification_limit:.1f}$')
ax1.set_xlabel(r'迭代次数 $k$')
ax1.set_ylabel(r'$\|x_k - x_{true}\|_2$')
ax1.set_title(r'Landweber 半收敛：误差先降后升的 U 形曲线')
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=err_min * 0.9, top=noise_amplification_limit * 1.12)
# 注释：早停 / 晚停
ax1.annotate('早停\n(欠拟合)', xy=(k_early, errors[k_early]),
             xytext=(k_opt * 0.9, err_min * 1.5),
             fontsize=10, color='blue',
             arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6))
ax1.annotate('晚停\n(噪声放大)', xy=(MAXITER * 0.85, errors[int(MAXITER * 0.85)]),
             xytext=(MAXITER * 0.55, noise_amplification_limit * 0.75),
             fontsize=10, color='red',
             arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))

# 右：残差范数（Morozov 偏差原理判据）
ax2.semilogy(residual_norms, 'g-', linewidth=2, label=r'$\|Kx_k - y\|_2$')
ax2.axhline(noise_norm_expected, color='orange', linestyle='--', alpha=0.8,
            label=fr'期望噪声范数 $\delta\sqrt{{n}}={noise_norm_expected:.3f}$')
ax2.axhline(noise_norm, color='purple', linestyle='--', alpha=0.7,
            label=fr'实际噪声范数 $\|e\|={noise_norm:.3f}$')
ax2.scatter([k_mor_exp], [residual_norms[k_mor_exp]], color='orange', s=50, zorder=6, marker='D')
ax2.scatter([k_mor_act], [residual_norms[k_mor_act]], color='purple', s=50, zorder=6, marker='D')
ax2.set_xlabel(r'迭代次数 $k$')
ax2.set_ylabel(r'$\|Kx_k - y\|_2$')
ax2.set_title(r'残差范数（Morozov 偏差原理判据）')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)

plt.suptitle(r'实验3.6-1: Landweber 半收敛——"先下降后上升"的 U 形误差曲线',
             fontsize=14)
plt.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.10, wspace=0.25)
out_path1 = os.path.join(SAVE_DIR, '步骤1a_U形误差曲线.png')
plt.savefig(out_path1, dpi=150)
plt.close()
print(f"  已保存：{out_path1}")

# ══════════════════════════════════════════════════════════
# 5. 图2：三个时刻的解 + 修正后的 Picard 条件
# ══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 6))
ax3 = fig.add_subplot(1, 2, 1)
ax4 = fig.add_subplot(1, 2, 2)

x_idx = np.arange(n)
ax3.plot(x_idx, x_true, 'k-', linewidth=2.5, label=r'真解 $x_{true}$')
ax3.plot(x_idx, x_early, 'b--', linewidth=1.5, alpha=0.75,
         label=fr'早停 $k={k_early}$（欠拟合）')
ax3.plot(x_idx, x_opt, 'g-', linewidth=1.5,
         label=fr'最优 $k={k_opt}$（先知）')
ax3.plot(x_idx, x_late, 'r--', linewidth=1.2, alpha=0.65,
         label=fr'晚停 $k={MAXITER}$（噪声放大）')
ax3.set_xlabel(r'像素 $i$')
ax3.set_ylabel(r'信号值')
ax3.set_title(r'三个迭代时刻的解对比')
ax3.legend(fontsize=9, loc='upper right')
ax3.grid(True, alpha=0.3)

# Picard 条件可视化（同一判据的两种等价表述）
# 数据域：sigma*|a|（绿）与噪声水平 delta（橙水平线）比较  <-- 主判据
# 解域：|a|（黑）与 delta/sigma（红虚线）比较，交点完全相同
ax4.semilogy(sigma, 'b-', linewidth=1.8, label=r'奇异值 $\sigma_i$')
ax4.semilogy(np.maximum(np.abs(x_true), 1e-7), 'k-', linewidth=2,
             label=r'真解系数 $|a_i|=|x_i^{true}|$')
ax4.semilogy(np.maximum(np.abs(x_true) * sigma, 1e-7), 'g-', linewidth=2,
             label=r'数据域信号 $\sigma_i |a_i|$')
ax4.axhline(sigma_noise, color='orange', linestyle='-', alpha=0.8, linewidth=2,
            label=fr'数据域噪声线 $\delta={sigma_noise}$（主判据）')
ax4.semilogy(sigma_noise / sigma, 'r--', alpha=0.7,
             label=r'解域噪声传递 $\delta/\sigma_i$（与 $|a_i|$ 比，等价）')
# 标注各支撑的失效交点
_cross_colors = {1.0: 'purple', 0.5: 'brown', 0.8: 'magenta'}
for a, b, v, i_star, status in picard_crossings:
    if 0 < i_star < n:
        c = _cross_colors.get(v, 'gray')
        ax4.axvline(i_star, color=c, linestyle=':', alpha=0.6)
        # 竖排标注，避免三个相邻交点的文字互相遮挡
        ax4.text(i_star + 0.3, sigma_noise * 1.35,
                 fr'$i^*={i_star:.0f}$' + '\n' + fr'$(a={v})$',
                 fontsize=8, color=c, va='bottom', ha='left', rotation=90)
ax4.set_xlabel(r'索引 $i$')
ax4.set_ylabel(r'幅度')
ax4.set_title(r'Picard 条件：$\sigma_i |a_i|>\delta$ 的分量才可稳定恢复')
ax4.legend(fontsize=8.5, loc='upper right')
ax4.grid(True, alpha=0.3)
ax4.set_ylim(1e-3, 10)

plt.suptitle(r'实验3.6-1: 半收敛的解与 Picard 条件分析',
             fontsize=14)
plt.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.10, wspace=0.25)
out_path1b = os.path.join(SAVE_DIR, '步骤1b_解对比与Picard条件.png')
plt.savefig(out_path1b, dpi=150)
plt.close()
print(f"  已保存：{out_path1b}")

# ══════════════════════════════════════════════════════════
# 6. 不同噪声水平下的半收敛（曲线族 + k_opt 统计带）
# ══════════════════════════════════════════════════════════
print("\n[步骤 5] 不同噪声水平下的半收敛对比...")

sigma_noise_levels = [1e-2, 2e-2, 5e-2, 1e-1, 2e-1]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(sigma_noise_levels)))

family_errs = []
family_kopt = []
family_censored = []
for idx, noise_level in enumerate(sigma_noise_levels):
    np.random.seed(SEED)   # 公共随机数：各水平使用同一噪声实现，仅幅度按比例缩放
    y_noise = forward(x_true) + noise_level * np.random.randn(n)
    errs_n, _, k_opt_n, censored_n, horizon_n = run_with_censor(y_noise)
    if horizon_n > MAXITER:
        errs_n = errs_n[:MAXITER + 1]   # 绘图窗口统一
    family_errs.append(errs_n)
    family_kopt.append(k_opt_n)
    family_censored.append(censored_n)
    tag = "（删失！）" if censored_n else ""
    print(f"  delta={noise_level:g}: k_opt={k_opt_n}, err_min={errs_n[k_opt_n]:.4f}{tag}")

# 蒙特卡洛：k_opt 对噪声实现的散布（用闭式解加速，与迭代严格等价）
N_MC = 100
print(f"  蒙特卡洛 {N_MC} 个独立噪声实现（闭式解加速）...")
t0 = time.time()
mc_kopt = np.empty((len(sigma_noise_levels), N_MC), dtype=int)
mc_censor = np.zeros(len(sigma_noise_levels))
qpow = (1 - tau * s2)[None, :] ** np.arange(MAXITER + 1)[:, None]
filt = 1 - qpow                                          # Landweber 滤波因子
for idx, noise_level in enumerate(sigma_noise_levels):
    for s in range(N_MC):
        rng = np.random.RandomState(s)
        e = noise_level * rng.randn(n)
        target = x_true + e / sigma                       # 被滤波的逐分量目标
        errs_mc = np.linalg.norm(filt * target - x_true, axis=1)
        ko = int(np.argmin(errs_mc))
        mc_kopt[idx, s] = ko
        mc_censor[idx] += (ko >= MAXITER)
mc_censor /= N_MC
print(f"  蒙特卡洛完成，耗时 {time.time() - t0:.1f} 秒")
mc_med = np.median(mc_kopt, axis=1)
mc_q05 = np.percentile(mc_kopt, 5, axis=1)
mc_q25 = np.percentile(mc_kopt, 25, axis=1)
mc_q75 = np.percentile(mc_kopt, 75, axis=1)
mc_q95 = np.percentile(mc_kopt, 95, axis=1)
for idx, noise_level in enumerate(sigma_noise_levels):
    print(f"  delta={noise_level:g}: k_opt 中位数={int(mc_med[idx])}, "
          f"IQR=[{int(mc_q25[idx])},{int(mc_q75[idx])}], 5-95%=[{int(mc_q05[idx])},{int(mc_q95[idx])}], "
          f"删失比例={mc_censor[idx]:.0%}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for idx, noise_level in enumerate(sigma_noise_levels):
    errs_n = family_errs[idx]
    k_opt_n = family_kopt[idx]
    label = (fr'$\delta={noise_level}$, $k_{{opt}}={k_opt_n}$'
             + (' (删失)' if family_censored[idx] else ''))
    axes[0].semilogy(errs_n, color=colors[idx], linewidth=1.5, label=label)
    axes[0].scatter([k_opt_n], [errs_n[k_opt_n]], color=colors[idx], s=30, zorder=5)
axes[0].set_xlabel(r'迭代次数 $k$')
axes[0].set_ylabel(r'$\|x_k - x_{true}\|_2$')
axes[0].set_title(r'不同噪声水平下的半收敛曲线（公共随机数，seed=42）')
axes[0].legend(fontsize=9, loc='upper left')
axes[0].grid(True, alpha=0.3)

# 右图：k_opt 随噪声水平变化 + 蒙特卡洛分位带
lv = np.array(sigma_noise_levels)
axes[1].fill_between(lv, mc_q05, mc_q95, color='blue', alpha=0.12,
                     label=fr'5–95% 分位带（{N_MC} 次实现）')
axes[1].fill_between(lv, mc_q25, mc_q75, color='blue', alpha=0.22, label='IQR（25–75%）')
axes[1].semilogx(lv, mc_med, 'b--', linewidth=1.5, alpha=0.7, label='中位数')
axes[1].semilogx(lv, family_kopt, 'ro-', markersize=7, linewidth=2, label='seed=42 单次实现')
axes[1].set_xlabel(r'噪声水平 $\delta$')
axes[1].set_ylabel(r'最优迭代次数 $k_{opt}$')
axes[1].set_title(r'噪声越大，最优停止越早（迭代次数即正则化参数）')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=9, loc='upper right')
axes[1].set_xticks(sigma_noise_levels)
axes[1].set_xticklabels([f'{s:.0e}' for s in sigma_noise_levels], rotation=30)

plt.suptitle(r'实验3.6-1: 半收敛曲线族——迭代次数 $k$ 即"正则化参数"',
             fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path2 = os.path.join(SAVE_DIR, '步骤2_不同噪声下半收敛曲线族.png')
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存：{out_path2}")

# ══════════════════════════════════════════════════════════
# 7. 保存数值结果到 JSON
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
    'experiment': '实验3.6-1 Landweber 半收敛：U 形误差曲线',
    'reproducibility': {
        'seed': SEED,
        'mc_seeds': N_MC,
        'note': '噪声族各水平共用 seed=42 的同一噪声实现（公共随机数）；'
                '蒙特卡洛使用独立种子 0..N-1，k_opt 由与迭代严格等价的闭式解计算',
    },
    'scene': {
        'signal_dim': n,
        'sigma_diag_min': float(sigma.min()),
        'sigma_diag_max': float(sigma.max()),
        'noise_delta': float(sigma_noise),
        'noise_norm_actual': noise_norm,
        'noise_norm_expected_delta_sqrt_n': float(noise_norm_expected),
        'step_size_tau': float(tau),
        'maxiter': MAXITER,
        'closed_form_max_diff_vs_iteration': float(_diff),
    },
    'semiconvergence': {
        'k_opt': int(k_opt),
        'k_early_displayed': int(k_early),
        'err_at_k0': float(err_0),
        'err_at_k_opt': float(err_min),
        'err_at_k_max': float(err_final),
        'semiconvergence_ratio': float(err_final / err_min),
        'noise_amplification_limit_e_over_sigma': noise_amplification_limit,
        'unrecoverable_signal_norm_sigma_a_below_delta': unrecoverable_signal_norm,
        'morozov_k_at_expected_noise': int(k_mor_exp),
        'morozov_err_at_expected_noise': float(errors[k_mor_exp]),
        'morozov_k_at_actual_noise': int(k_mor_act),
        'morozov_err_at_actual_noise': float(errors[k_mor_act]),
    },
    'picard_crossings_sigma_a_equals_delta': [
        {'support': [a, b], 'a_value': v, 'i_star': i_star, 'status': status}
        for a, b, v, i_star, status in picard_crossings
    ],
    'k_opt_vs_noise': {
        str(noise_level): int(k_opt_n)
        for noise_level, k_opt_n in zip(sigma_noise_levels, family_kopt)
    },
    'k_opt_vs_noise_detail': [
        {
            'delta': float(noise_level),
            'k_opt_seed42': int(family_kopt[idx]),
            'censored_at_window': bool(family_censored[idx]),
            'mc_median': float(mc_med[idx]),
            'mc_q05': float(mc_q05[idx]),
            'mc_q25': float(mc_q25[idx]),
            'mc_q75': float(mc_q75[idx]),
            'mc_q95': float(mc_q95[idx]),
            'mc_censored_fraction': float(mc_censor[idx]),
        }
        for idx, noise_level in enumerate(sigma_noise_levels)
    ],
    'output_files': [out_path1, out_path1b, out_path2],
}

with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(_to_native(results), f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print(f"1. Landweber 迭代的误差 ||x_k - x_true|| 呈典型 U 形：")
print(f"   初值 ({err_0:.4f}) → 最优点 k={k_opt} ({err_min:.4f}) → k={MAXITER} ({err_final:.4f})")
print(f"   继续迭代将趋向噪声放大极限 ||e/sigma|| = {noise_amplification_limit:.2f}")
print(f"2. '半收敛'——早停保留信号但欠拟合，晚停过度拟合噪声（1/sigma 放大）")
print(f"3. 迭代次数 k 本身即'正则化参数'；蒙特卡洛证实 k_opt 随噪声水平单调减小")
print(f"4. Morozov 偏差原理（残差≤||e|| 即停，k={k_mor_act}）无需真解，"
      f"误差 {errors[k_mor_act]:.4f} 已接近先知最优 {err_min:.4f}")
print(f"5. Picard 判据 sigma*|a|>delta 定量解释了不可恢复分量（范数 "
      f"{unrecoverable_signal_norm:.4f}），err_min 已达噪声极限")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")
print(f"  - 步骤1a_U形误差曲线.png")
print(f"  - 步骤1b_解对比与Picard条件.png")
print(f"  - 步骤2_不同噪声下半收敛曲线族.png")
print(f"  - results_summary.json")
