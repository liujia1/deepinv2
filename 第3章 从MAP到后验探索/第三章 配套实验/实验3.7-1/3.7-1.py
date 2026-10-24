# -*- coding: utf-8 -*-
"""
实验3.7-1 高维高斯的众数≠典型 + 两条路径分叉
对应章节：3.7 从 MAP 到后验：点估计的局限与分叉
知识点：
  - 高维高斯分布的众数（mode）在原点 x=0
  - 典型样本（typical sample）位于半径 sqrt(n) 的壳层（范数集中现象）
  - 众数≠典型：点估计的局限性
  - 两条路径分叉：后验 → 采样（Part II 主线） / 近似（Part III 副线） → 扩散模型

素材来源（理论参考）：
  - Vershynin, "High-Dimensional Probability" (2018) 第 3 章
  - 与 3.7 节文本中"众数≠典型"和"两条路径"两个核心论点对应
  - 两张配图均依据 3.7 节正文（高维高斯反例 + 两条路径分叉）自行设计实现，
    并非任何既有教材插图的复刻或替代

运行环境：纯 CPU 即可，无需 GPU，无外部权重文件。
"""

import numpy as np
import math
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
from scipy import stats
from matplotlib.ticker import ScalarFormatter, NullLocator

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
    SAVE_DIR = os.path.join(_gdrive, '实验3.7-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)
else:
    # 注意：字体目录必须锚定脚本所在目录（SAVE_DIR），不能用相对路径 '.chinese'，
    # 否则从其他工作目录启动时会找不到 chinese_font 模块、中文显示为方框。
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

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
print("实验3.7-1 高维高斯的众数≠典型 + 两条路径分叉")
print("=" * 60)
print("运行环境: 纯 CPU，无需 GPU，无外部权重文件")
print("  - 场景A：高维高斯蒙特卡洛采样（展示众数≠典型）")
print("  - 场景B：两条路径分叉流程图（呼应3.7节核心论点）")

# ══════════════════════════════════════════════════════════
# 场景A：高维高斯众数≠典型
# ══════════════════════════════════════════════════════════
print("\n[场景 A] 高维高斯众数≠典型...")

dimensions = [2, 10, 100, 1000]
n_samples = 500
samples = {}

print(f"  在维度 n ∈ {dimensions} 下各采样 {n_samples} 个样本", end="", flush=True)
t_start = time.time()
for d in dimensions:
    # 一次性打印进度（多维度一起）
    samples[d] = np.random.randn(n_samples, d)
    print(".", end="", flush=True)
print(f" 完成，耗时 {time.time() - t_start:.2f} 秒")

# 范数统计
print(f"\n  维度   众数范数*  典型范数(中位)  经验均值  理论典型 sqrt(n)")
print(f"  {'-' * 60}")
# 众数范数是【解析值】而非采样统计量：联合密度 p(x)=(2π)^{-n/2}e^{-‖x‖²/2} 的
# 众数在 x=0，范数恒为 0，与 n、样本数无关；因此不能由 500 个样本估计出来
# （有限样本的"经验众数"只是最近邻样本点，与理论众数无关）。
mode_norm = {}  # 解析值：联合密度众数范数 ≡ 0
typical_norm = {}  # 典型范数（用中位数）
mean_norm = {}  # 经验平均范数
for d in dimensions:
    norms = np.linalg.norm(samples[d], axis=1)
    mode_norm[d] = 0.0
    typical_norm[d] = np.median(norms)
    mean_norm[d] = np.mean(norms)
    print(f"  n={d:<5} {mode_norm[d]:<10.4f} {typical_norm[d]:<16.4f} {mean_norm[d]:<10.4f} {np.sqrt(d):.4f}")
print(f"  * 众数范数为解析值（联合密度众数在 x=0），非采样统计量；"
      f"其余三列均来自 seed=42 的 500 样本实测。")

# 众数≠典型的核心观察
print(f"\n  核心观察:")
print(f"  - 众数（mode）在 x=0（密度最大点），但该点的概率体积为零")
print(f"  - 典型样本（typical sample）位于半径 ≈ sqrt(n) 的薄壳层")
print(f"  - 当 n=1000 时，典型范数 ≈ {typical_norm[1000]:.1f}（接近√n≈{np.sqrt(1000):.1f}），而众数处范数恒为 0")
print(f"  - MAP 估计（点估计）追求众数=0，但该点附近无任何样本")

# ── 精确理论矩（R = ||x|| ~ chi(n)）与多种子稳健性复核 ──
# 说明：500 样本的单种子（seed=42）结果存在蒙特卡洛涨落，需用精确理论矩
# 与多种子区间对照，才能区分"现象本身"与"单次抽样抖动"。
# 注意：以下使用独立的新 RNG（default_rng），不消耗 seed=42 的随机流，
# 因此上节所有 seed=42 数值保持逐位不变。
M_SEEDS = 200
MS_RNG_SEED = 20261014
ms_rng = np.random.default_rng(MS_RNG_SEED)
exact_median, exact_mean, exact_std = {}, {}, {}
ms_med_lo, ms_med_hi = {}, {}
ms_mean_lo, ms_mean_hi = {}, {}
print(f"\n  精确理论矩（chi(n)）与多种子稳健性（{M_SEEDS} 独立种子 × "
      f"{n_samples} 样本，default_rng({MS_RNG_SEED})）")
print(f"  n       精确中位     精确均值     精确标准差   "
      f"median 2.5-97.5%区间        mean 2.5-97.5%区间")
for d in dimensions:
    chi_d = stats.chi(d)
    exact_median[d] = float(chi_d.ppf(0.5))
    exact_mean[d] = float(chi_d.mean())
    exact_std[d] = float(chi_d.std())
    med_list, mean_list = [], []
    for _ in range(M_SEEDS):
        r_ms = np.linalg.norm(ms_rng.standard_normal((n_samples, d)), axis=1)
        med_list.append(np.median(r_ms))
        mean_list.append(np.mean(r_ms))
    ms_med_lo[d], ms_med_hi[d] = [float(v) for v in np.percentile(med_list, [2.5, 97.5])]
    ms_mean_lo[d], ms_mean_hi[d] = [float(v) for v in np.percentile(mean_list, [2.5, 97.5])]
    print(f"  n={d:<5} {exact_median[d]:<12.4f} {exact_mean[d]:<12.4f} {exact_std[d]:<12.4f}"
          f"[{ms_med_lo[d]:.4f}, {ms_med_hi[d]:.4f}]      "
          f"[{ms_mean_lo[d]:.4f}, {ms_mean_hi[d]:.4f}]")
print(f"  概念区分：联合密度 p(x) 的众数在 r=0；而径向（壳层）密度 chi(n) 的众数在 "
      f"sqrt(n-1)，")
print(f"            两者是不同对象——下面 n=10 子图同时标出这两条参考线。")

# ══════════════════════════════════════════════════════════
# 场景A 可视化
# ══════════════════════════════════════════════════════════
print("\n[场景 A] 绘制高维高斯采样可视化...")

fig = plt.figure(figsize=(19.5, 6))

# 左图：2D 投影散点（n=2）
ax1 = fig.add_subplot(1, 4, 1)
sample_2d = samples[2]
ax1.scatter(sample_2d[:, 0], sample_2d[:, 1], alpha=0.4, s=15, c='steelblue', label='样本')
# 画典型圆（RMS 半径 sqrt(n)；注意 n=2 时尚无薄壳集中）
theta = np.linspace(0, 2 * np.pi, 200)
ax1.plot(np.sqrt(2) * np.cos(theta), np.sqrt(2) * np.sin(theta),
         'r--', linewidth=2, label=r'RMS 半径圆 $\|x\|=\sqrt{n}=\sqrt{2}$')
# 众数
ax1.scatter([0], [0], s=200, c='red', marker='*', zorder=5, label='众数 $x^*=0$')
ax1.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
ax1.axvline(0, color='gray', linewidth=0.5, alpha=0.3)
# n=2 时 R^2~chi^2(2)，圆内质量 = 1-e^-1 ≈ 63.2%——二维没有薄壳，这是高维现象
ax1.text(0.03, 0.03, r'圆内质量 $1-e^{-1}\approx63.2\%$；$n=2$ 尚无薄壳',
         transform=ax1.transAxes, fontsize=8, color='#444444',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CCCCCC', alpha=0.85))
ax1.set_xlabel(r'$x_1$')
ax1.set_ylabel(r'$x_2$')
ax1.set_title(r'$n=2$ 维：众数=原点，样本围绕 RMS 半径圆')
ax1.legend(fontsize=8.5, loc='upper right')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-3.5, 3.5)
ax1.set_ylim(-3.5, 3.5)

# 中图：n=10 的样本范数经验分布 vs chi(n) 理论
ax2 = fig.add_subplot(1, 4, 2)
sample_10 = samples[10]
norms_10 = np.linalg.norm(sample_10, axis=1)
# 理论：||x||^2 ~ chi^2(n)，所以 ||x|| 的分布 = sqrt(chi^2(n)) = chi(n)
# chi(n) 的 PDF: f(r) = r^(n-1) * exp(-r^2/2) / (2^(n/2-1) * Gamma(n/2))
n_dim_10 = 10
x_theory = np.linspace(0, 8, 200)
chi_pdf = (x_theory ** (n_dim_10 - 1)) * np.exp(-x_theory ** 2 / 2) / \
          (2 ** (n_dim_10 / 2 - 1) * math.gamma(n_dim_10 / 2))
ax2.hist(norms_10, bins=40, density=True, alpha=0.6, color='steelblue',
         label='经验分布（500 样本，seed=42）')
ax2.plot(x_theory, chi_pdf, 'r-', linewidth=2.5,
         label=fr'$\chi(n)$ 理论 (n=10)')
ax2.axvline(np.sqrt(10), color='orange', linestyle='--', linewidth=2,
            label=r'$\sqrt{n}=' + f'{np.sqrt(10):.2f}' + r'$：集中半径')
# chi(n) 径向（壳层）密度的众数 = sqrt(n-1)；勿与联合密度 p(x) 的众数 r=0 混淆
ax2.axvline(np.sqrt(n_dim_10 - 1), color='#7B1FA2', linestyle='-.', linewidth=2,
            label=fr'壳层密度众数 $\sqrt{{n-1}}={np.sqrt(n_dim_10 - 1):.2f}$')
ax2.axvline(0, color='red', linestyle=':', linewidth=2,
            label=r'联合密度众数 $r=0$（壳层密度为 0）')
ax2.set_xlabel(r'$\|x\|_2$')
ax2.set_ylabel(r'径向概率密度')
ax2.set_title(r'$n=10$ 维：两种众数 vs 集中半径')
ax2.legend(fontsize=8, loc='upper right', framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.3, 8)  # 让 x=0 的众数标注线可见，不与 y 轴重合

# 右图：n=1000 范数分布的 CLT 标准化
ax3 = fig.add_subplot(1, 4, 3)
sample_1000 = samples[1000]
norms_1000 = np.linalg.norm(sample_1000, axis=1)
# 标准化到 (||x|| - sqrt(n)) / 1/sqrt(2)，当 n 大时 ~ N(0,1)（delta method，方差 1/2）
z = (norms_1000 - np.sqrt(1000)) / (1.0 / np.sqrt(2))
ax3.hist(z, bins=50, density=True, alpha=0.6, color='steelblue',
         label=r'经验分布 $(\|x\|-\sqrt{n})/(1/\sqrt{2})$')
xx = np.linspace(-5, 5, 200)
ax3.plot(xx, stats.norm.pdf(xx), 'r-', linewidth=2.5, label=r'$\mathcal{N}(0,1)$ 理论')
ax3.set_xlabel(r'标准化范数 $(\|x\| - \sqrt{n}) / (1/\sqrt{2})$')
ax3.set_ylabel(r'概率密度')
ax3.set_title(r'$n=1000$ 维：典型集现象（CLT）')
ax3.legend(fontsize=9, loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-5, 5)  # 与 z 值范围一致

# 第四图：典型范数随维度变化 + 多种子稳健性
ax4 = fig.add_subplot(1, 4, 4)
_ds = np.array(dimensions, dtype=float)
# 区间只在实际采样的 4 个维度处画误差棒（不跨维度插值，避免在对数轴上误导）
_band_lo = np.array([ms_med_lo[d] for d in dimensions])
_band_hi = np.array([ms_med_hi[d] for d in dimensions])
_band_mid = (_band_lo + _band_hi) / 2
ax4.errorbar(_ds, _band_mid,
             yerr=[_band_mid - _band_lo, _band_hi - _band_mid],
             fmt='s', markersize=5, markerfacecolor='orange', markeredgecolor='#B8860B',
             ecolor='orange', elinewidth=1.6, capsize=5, capthick=1.6,
             label=fr'median 的 2.5–97.5%（{M_SEEDS} 种子）', zorder=4)
ax4.plot(_ds, [np.sqrt(d) for d in dimensions], 'k-', lw=1.8,
         label=r'$\sqrt{n}$：集中半径', zorder=3)
ax4.plot(_ds, [exact_mean[d] for d in dimensions], 'g--', lw=1.8,
         label=r'精确均值 $E\|x\|$（$\chi(n)$）', zorder=3)
ax4.plot(_ds, [exact_median[d] for d in dimensions], color='blue',
         linestyle=':', lw=2.0, label=r'精确中位（$\chi(n)$）', zorder=3)
ax4.scatter(_ds, [typical_norm[d] for d in dimensions],
            marker='*', s=150, c='red', zorder=6, label='seed=42 实测 median')
ax4.axhline(0, color='red', linestyle=':', lw=1.3, label='众数范数 = 0')
ax4.text(2.6, 5.2, '众数恒为 0：典型范数\n从 1.1（n=2）拉大到\n31.6（n=1000）',
         fontsize=8, color='#8B0000',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#E0B0B0', alpha=0.85))
ax4.set_xscale('log')
ax4.set_xticks(_ds)
ax4.get_xaxis().set_major_formatter(ScalarFormatter())
ax4.xaxis.set_minor_locator(NullLocator())
ax4.set_xlabel(r'维度 $n$（对数轴）')
ax4.set_ylabel(r'典型范数 $\|x\|_2$')
ax4.set_title(fr'典型范数 vs 维度：{M_SEEDS} 种子稳健性')
ax4.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax4.grid(True, alpha=0.3, which='both')
ax4.set_xlim(1.6, 1300)
ax4.set_ylim(-1.5, 36)

plt.suptitle(r'实验3.7-1: 高维高斯的众数$\neq$典型——众数在原点，典型样本在 $\sqrt{n}$ 壳层',
             fontsize=13)
plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.10, wspace=0.28)
out_path1 = os.path.join(SAVE_DIR, '步骤1_高维高斯众数不等于典型.png')
plt.savefig(out_path1, dpi=150)
plt.close()
print(f"  已保存：{out_path1}")

# ══════════════════════════════════════════════════════════
# 场景B：两条路径分叉（概念图）
# ══════════════════════════════════════════════════════════
print("\n[场景 B] 绘制'从 MAP 到后验'的两条路径分叉概念图...")
print("  后验推断的两条路径（众数 ≠ 典型驱动从点估计走向分布级推理）：")
print("  - 采样主线（Part II 第4–7章）：MCMC/ULA(4) → 朗之万动力学与得分(5) → 得分匹配(6) → 扩散SDE(7)")
print("  - 近似副线（Part III 第8–11章）：变分推断(8) → ELBO/VAE(9) → 层级VAE(10) → 扩散VLB(11)")
print("  - 殊途同归：两路在第12章汇合于扩散模型（Score ≡ ELBO），第13章闭环于逆问题")


def draw_path_fork(save_dir):
    """绘制'从 MAP 到后验'两条路径分叉概念图，保存为 步骤2_两条路径分叉.png。"""
    fig, ax = plt.subplots(figsize=(13, 9.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10.3)
    ax.axis('off')

    def box(x, y, w, h, text, fc, ec, fs=10):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc,
                                   edgecolor=ec, linewidth=1.6, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fs, zorder=4, linespacing=1.4)

    def arrow(x1, y1, x2, y2, color='#555555'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.0), zorder=2)

    # 顶部：高维后验
    box(4.3, 8.45, 3.4, 1.0, r"后验 $p(x|y)$" + "\n（高维后验，算不出）",
        fc='#FFF3CD', ec='#B8860B', fs=11)
    ax.text(6, 8.0, "分叉点：MAP 只给一个点，够吗？", ha='center', va='center',
            fontsize=9, color='#8B0000')

    # 左侧：采样主线（Part II 第4–7章），顺序与 3.7 节正文演进链一致
    box(0.3, 6.7, 5.3, 0.95, "采样主线 · 逆问题驱动\n（Part II 第4–7章）",
        fc='#E8F5E9', ec='#2E7D32', fs=10)
    box(0.3, 5.3, 5.3, 0.85, "MCMC / ULA（第4章）", fc='#F1F8E9', ec='#2E7D32')
    box(0.3, 3.9, 5.3, 0.85, "朗之万动力学与得分函数（第5章）", fc='#F1F8E9', ec='#2E7D32')
    box(0.3, 2.5, 5.3, 0.85, "得分匹配：去噪中学得分（第6章）", fc='#F1F8E9', ec='#2E7D32')

    # 右侧：近似副线（Part III 第8–11章）
    box(6.4, 6.7, 5.3, 0.95, "近似副线 · 生成建模驱动\n（Part III 第8–11章）",
        fc='#FFF8E1', ec='#EF6C00', fs=10)
    box(6.4, 5.3, 5.3, 0.85, "变分推断（第8章）", fc='#FFFDE7', ec='#EF6C00')
    box(6.4, 3.9, 5.3, 0.85, "ELBO / VAE（第8–9章）", fc='#FFFDE7', ec='#EF6C00')
    box(6.4, 2.5, 5.3, 0.85, "层级模型 · 层级 VAE（第10章）", fc='#FFFDE7', ec='#EF6C00')

    # 底部：汇合点
    box(4.15, 1.15, 3.7, 0.95, "扩散模型\n（第12章 殊途同归：Score ≡ ELBO）", fc='#F3E5F5', ec='#6A1B9A', fs=10.5)
    ax.text(6, 0.62, "条件生成 = 逆问题求解（第13章 · 全书闭环）", ha='center', va='center',
            fontsize=9.5, color='#6A1B9A', fontweight='bold')

    # 箭头：后验 → 两条路径
    arrow(5.9, 8.45, 2.95, 7.65, '#2E7D32')
    arrow(6.1, 8.45, 9.05, 7.65, '#EF6C00')
    # 左列纵向
    arrow(2.95, 6.7, 2.95, 6.15, '#2E7D32')
    arrow(2.95, 5.3, 2.95, 4.75, '#2E7D32')
    arrow(2.95, 3.9, 2.95, 3.35, '#2E7D32')
    # 右列纵向
    arrow(9.05, 6.7, 9.05, 6.15, '#EF6C00')
    arrow(9.05, 5.3, 9.05, 4.75, '#EF6C00')
    arrow(9.05, 3.9, 9.05, 3.35, '#EF6C00')
    # 两路汇聚到扩散模型（第7章 SDE 视角 / 第11章 VLB 视角在汇聚箭上标出）
    arrow(3.2, 2.5, 5.4, 2.1, '#2E7D32')
    arrow(8.8, 2.5, 6.6, 2.1, '#EF6C00')
    ax.text(4.15, 2.36, "扩散 SDE（第7章）", ha='center', va='center',
            fontsize=8.5, color='#2E7D32', zorder=5,
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.85))
    ax.text(7.85, 2.36, "VLB 变分视角（第11章）", ha='center', va='center',
            fontsize=8.5, color='#EF6C00', zorder=5,
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.85))
    # 扩散 → 闭环
    arrow(6, 1.15, 6, 0.72, '#6A1B9A')

    ax.text(6, 9.95, "从 MAP 点估计到后验探索：两条路径，殊途同归",
            ha='center', va='center', fontsize=13, fontweight='bold')

    out_path2 = os.path.join(save_dir, '步骤2_两条路径分叉.png')
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path2


out_path2 = draw_path_fork(SAVE_DIR)
print(f"  已保存：{out_path2}")

# ══════════════════════════════════════════════════════════
# 保存数值结果
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
    'experiment': '实验3.7-1 高维高斯众数≠典型 + 两条路径分叉',
    'scene_a_typical_set': {
        'dimensions': dimensions,
        'n_samples': n_samples,
        # 解析值：联合密度 p(x)=(2π)^(-n/2)exp(-‖x‖²/2) 的众数在 x=0，范数恒为 0。
        # 这不是采样统计量，无法由 500 个样本估计，故单独标注以免与下方实测列混淆。
        'mode_norm_analytic': {str(d): float(mode_norm[d]) for d in dimensions},
        'mode_norm_note': ('解析值：联合密度众数 x*=0 的范数，与 n 和样本数无关；'
                           '非由采样估计得到（有限样本的经验众数只是最近邻样本点）'),
        # 以下两项为 seed=42、每维 500 样本的实测统计量
        'typical_norm_median': {str(d): float(typical_norm[d]) for d in dimensions},
        'mean_norm': {str(d): float(mean_norm[d]) for d in dimensions},
        'theoretical_typical_sqrt_n': {str(d): float(np.sqrt(d)) for d in dimensions},
    },
    'theoretical_exact_chi': {
        # R=||x|| ~ chi(n) 的精确矩（scipy.stats.chi），用于判定单种子偏差是否在 MC 误差内
        str(d): {'median': exact_median[d], 'mean': exact_mean[d], 'std': exact_std[d]}
        for d in dimensions
    },
    'multiseed_robustness': {
        'rng': 'numpy.random.default_rng',
        'rng_seed': MS_RNG_SEED,
        'n_independent_seeds': M_SEEDS,
        'samples_per_seed': n_samples,
        'percentiles': '2.5-97.5',
        'median_band': {str(d): [ms_med_lo[d], ms_med_hi[d]] for d in dimensions},
        'mean_band': {str(d): [ms_mean_lo[d], ms_mean_hi[d]] for d in dimensions},
        'note': '该多种子流与 seed=42 的展示抽样相互独立，不改变其逐位可复现性',
    },
    'key_observation': (
        f'高维高斯 N(0, I_n) 的众数在 x=0（范数=0），但典型样本位于半径 sqrt(n) 的薄壳层。'
        f'当 n=1000 时，典型范数 ≈ {typical_norm[1000]:.1f}（接近√n≈{np.sqrt(1000):.1f}），而众数处范数恒为 0。'
    ),
    'output_files': [out_path1, out_path2],
}

with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(_to_native(results), f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print("1. 高维高斯 N(0, I_n) 的众数 = 0，但典型样本范数 ≈ sqrt(n)")
print(f"2. n={dimensions[-1]} 时众数=0，典型范数={typical_norm[1000]:.1f}（接近√n≈{np.sqrt(1000):.1f}）")
print("3. MAP 估计（追求众数）在高维时与典型样本偏差巨大——点估计的根本局限")
print("4. 驱动推断从'点估计'走向'分布级推理'（采样主线 / 变分近似副线）")
print("5. 路径归属：采样主线=Part II 第4–7章（MCMC/ULA→朗之万与得分→得分匹配→扩散SDE）；")
print("   近似副线=Part III 第8–11章（变分推断→ELBO/VAE→层级VAE→扩散VLB）；第12章汇合")
print(f"6. 多种子复核（{M_SEEDS} 种子×{n_samples} 样本）：各维 median/mean 的跨种子区间"
      f"稳定覆盖 chi(n) 精确理论值，")
print("   seed=42 与理论的偏差均属蒙特卡洛涨落，不随种子选择改变结论")
print("7. 概念辨析：联合密度 p(x) 的众数在 r=0；径向壳层密度 chi(n) 的众数在 sqrt(n-1)；")
print("   sqrt(n) 是集中半径（典型集尺度），三者不可混为一谈")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")
print(f"  - 步骤1_高维高斯众数不等于典型.png")
print(f"  - 步骤2_两条路径分叉.png")
print(f"  - results_summary.json")
