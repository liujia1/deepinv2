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
  - 图3-7（缺失材料）的替代实现

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
print(f"\n  维度   众数范数   典型范数(中位)  经验均值  理论典型 sqrt(n)")
print(f"  {'-' * 60}")
mode_norm = {}  # 众数 = 0
typical_norm = {}  # 典型范数（用中位数）
mean_norm = {}  # 经验平均范数
for d in dimensions:
    norms = np.linalg.norm(samples[d], axis=1)
    mode_norm[d] = 0.0
    typical_norm[d] = np.median(norms)
    mean_norm[d] = np.mean(norms)
    print(f"  n={d:<5} {mode_norm[d]:<10.4f} {typical_norm[d]:<16.4f} {mean_norm[d]:<10.4f} {np.sqrt(d):.4f}")

# 众数≠典型的核心观察
print(f"\n  核心观察:")
print(f"  - 众数（mode）在 x=0（密度最大点），但该点的概率体积为零")
print(f"  - 典型样本（typical sample）位于半径 ≈ sqrt(n) 的薄壳层")
print(f"  - 当 n=1000 时，典型范数 ≈ {typical_norm[1000]:.1f}（接近√n≈{np.sqrt(1000):.1f}），而众数处范数恒为 0")
print(f"  - MAP 估计（点估计）追求众数=0，但该点附近无任何样本")

# ══════════════════════════════════════════════════════════
# 场景A 可视化
# ══════════════════════════════════════════════════════════
print("\n[场景 A] 绘制高维高斯采样可视化...")

fig = plt.figure(figsize=(15, 6))

# 左图：2D 投影散点（n=2）
ax1 = fig.add_subplot(1, 3, 1)
sample_2d = samples[2]
ax1.scatter(sample_2d[:, 0], sample_2d[:, 1], alpha=0.4, s=15, c='steelblue', label='样本')
# 画典型圆
theta = np.linspace(0, 2 * np.pi, 200)
ax1.plot(np.sqrt(2) * np.cos(theta), np.sqrt(2) * np.sin(theta),
         'r--', linewidth=2, label=r'典型圆 $\|x\|=\sqrt{n}=\sqrt{2}$')
# 众数
ax1.scatter([0], [0], s=200, c='red', marker='*', zorder=5, label='众数 $x^*=0$')
ax1.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
ax1.axvline(0, color='gray', linewidth=0.5, alpha=0.3)
ax1.set_xlabel(r'$x_1$')
ax1.set_ylabel(r'$x_2$')
ax1.set_title(r'$n=2$ 维：众数=原点，样本围绕典型圆')
ax1.legend(fontsize=9, loc='upper right')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-3.5, 3.5)
ax1.set_ylim(-3.5, 3.5)

# 中图：n=10 的样本范数经验分布 vs chi(n) 理论
ax2 = fig.add_subplot(1, 3, 2)
sample_10 = samples[10]
norms_10 = np.linalg.norm(sample_10, axis=1)
# 理论：||x||^2 ~ chi^2(n)，所以 ||x|| 的分布 = sqrt(chi^2(n)) = chi(n)
# chi(n) 的 PDF: f(r) = r^(n-1) * exp(-r^2/2) / (2^(n/2-1) * Gamma(n/2))
n_dim_10 = 10
x_theory = np.linspace(0, 8, 200)
chi_pdf = (x_theory ** (n_dim_10 - 1)) * np.exp(-x_theory ** 2 / 2) / \
          (2 ** (n_dim_10 / 2 - 1) * math.gamma(n_dim_10 / 2))
ax2.hist(norms_10, bins=40, density=True, alpha=0.6, color='steelblue',
         label='经验分布（500 样本）')
ax2.plot(x_theory, chi_pdf, 'r-', linewidth=2.5,
         label=fr'$\chi(n)$ 理论 (n=10)')
ax2.axvline(np.sqrt(10), color='orange', linestyle='--', linewidth=2,
            label=r'$\sqrt{n} = \sqrt{10} \approx ' + f'{np.sqrt(10):.2f}' + r'$')
ax2.axvline(0, color='red', linestyle=':', linewidth=2, label='众数位置 = 0')
ax2.set_xlabel(r'$\|x\|_2$')
ax2.set_ylabel(r'概率密度')
ax2.set_title(r'$n=10$ 维：范数集中在 $\sqrt{n}$ 附近')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.3, 8)  # 让 x=0 的众数标注线可见，不与 y 轴重合

# 右图：n=1000 范数分布 + 众数位置对比
ax3 = fig.add_subplot(1, 3, 3)
sample_1000 = samples[1000]
norms_1000 = np.linalg.norm(sample_1000, axis=1)
# 标准化到 (||x|| - sqrt(n)) / 1/sqrt(2)，当 n 大时 ~ N(0,1)
z = (norms_1000 - np.sqrt(1000)) / (1.0 / np.sqrt(2))
ax3.hist(z, bins=50, density=True, alpha=0.6, color='steelblue',
         label=r'经验分布 $(\|x\|-\sqrt{n})/(1/\sqrt{2})$')
xx = np.linspace(-5, 5, 200)
ax3.plot(xx, stats.norm.pdf(xx), 'r-', linewidth=2.5, label=r'$\mathcal{N}(0,1)$ 理论')
ax3.set_xlabel(r'标准化范数 $(\|x\| - \sqrt{n}) / (1/\sqrt{2})$')
ax3.set_ylabel(r'概率密度')
ax3.set_title(r'$n=1000$ 维：典型集现象')
ax3.legend(fontsize=10, loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-5, 5)  # 与 z 值范围一致

plt.suptitle(r'实验3.7-1: 高维高斯的众数$\neq$典型——众数在原点，典型样本在 $\sqrt{n}$ 壳层',
             fontsize=13)
plt.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.10, wspace=0.30)
out_path1 = os.path.join(SAVE_DIR, '步骤1_高维高斯众数不等于典型.png')
plt.savefig(out_path1, dpi=150)
plt.close()
print(f"  已保存：{out_path1}")

# ══════════════════════════════════════════════════════════
# 场景B：两条路径分叉（文字说明）
# ══════════════════════════════════════════════════════════
print("\n[场景 B] 两条路径分叉——从 MAP 点估计到分布级推理")
print("  后验推断的两条路径（众数 ≠ 典型驱动从点估计走向分布级推理）：")
print("  ")
print("  后验 p(x|y)")
print("    ├─ 点估计 (MAP): x_MAP = argmax p(x|y)  →  众数 → 典型?")
print("    └─ 分叉点")
print("         ├─ 路径 1: 采样 (Part II)")
print("         │    MCMC / 变分推断  →  扩散模型采样: x_T ~ p_T → x_0")
print("         └─ 路径 2: 近似 (Part III)")
print("              直接近似 / 端到端学习  →  条件生成器 G_θ(y) 端到端反演网络")
print("  ")
print("  高维高斯：众数在原点 (范数=0)，典型样本在 sqrt(n) 壳层（范数集中）")

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
        'mode_norm': {str(d): float(mode_norm[d]) for d in dimensions},
        'typical_norm_median': {str(d): float(typical_norm[d]) for d in dimensions},
        'mean_norm': {str(d): float(mean_norm[d]) for d in dimensions},
        'theoretical_typical_sqrt_n': {str(d): float(np.sqrt(d)) for d in dimensions},
    },
    'key_observation': (
        f'高维高斯 N(0, I_n) 的众数在 x=0（范数=0），但典型样本位于半径 sqrt(n) 的薄壳层。'
        f'当 n=1000 时，典型范数 ≈ {typical_norm[1000]:.1f}（接近√n≈{np.sqrt(1000):.1f}），而众数处范数恒为 0。'
    ),
    'output_files': [out_path1],
}

with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(_to_native(results), f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print("1. 高维高斯 N(0, I_n) 的众数 = 0，但典型样本范数 ≈ sqrt(n)")
print(f"2. n={dimensions[-1]} 时众数=0，典型范数={typical_norm[1000]:.1f}（接近√n≈{np.sqrt(1000):.1f}）")
print("3. MAP 估计（追求众数）在高维时与典型样本偏差巨大——点估计的根本局限")
print("4. 驱动推断从'点估计'走向'分布级推理'（采样 / 端到端近似）")
print("5. 两条路径分叉：MCMC/变分推断（Part II 主线）vs 端到端学习（Part III 副线）")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")
print(f"  - 步骤1_高维高斯众数不等于典型.png")
print(f"  - results_summary.json")
