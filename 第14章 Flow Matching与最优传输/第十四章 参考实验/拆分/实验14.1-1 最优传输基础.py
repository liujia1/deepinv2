# -*- coding: utf-8 -*-
"""
实验14.1-1 最优传输基础：路径形态对比
对应知识点：
  - 14.1节 最优传输基础（Monge/Kantorovich问题、Wasserstein距离）
  - 14.3.5节 独立耦合 vs OT耦合的路径差异

本实验不需要GPU，通过2D点云的可视化直观理解最优传输的核心概念。

素材来源：
  - book_plan.md的实验14.1计划
  - 14.1节的理论内容
  - ★ 原创设计：独立耦合 vs OT耦合的路径可视化

实验内容：
  最优传输 vs 独立耦合——路径形态对比
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import warnings

# ====== 解决中文乱码的核心代码 ======
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
import platform
from matplotlib.font_manager import FontManager

def _find_chinese_font():
    candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong'] if platform.system() == 'Windows' else ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei']
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available:
            return font
    import os as _os, re
    for f in fm.ttflist:
        for pat in ['cjk', 'wqy', 'noto.*cjk', 'simhei']:
            if re.search(pat, f.name.lower()):
                return f.name
    return None

_cn_font = _find_chinese_font()
if _cn_font:
    plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['font.family'] = 'sans-serif'
    print(f"[Font] 已检测到中文字体: {_cn_font}")
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
# ========================================================

np.random.seed(42)
import torch
torch.manual_seed(42)

SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()


# ============================================================
# 2D点云分布定义
# ============================================================
def sample_source(n):
    """源分布：两个高斯的混合（左上+右下）"""
    mode = np.random.rand(n) < 0.5
    x = np.where(mode, -2, 2) + 0.3 * np.random.randn(n)
    y = np.where(mode, 2, -2) + 0.3 * np.random.randn(n)
    return np.stack([x, y], axis=1)

def sample_target(n):
    """目标分布：两个高斯的混合（右上+左下），与源交叉"""
    mode = np.random.rand(n) < 0.5
    x = np.where(mode, 2, -2) + 0.3 * np.random.randn(n)
    y = np.where(mode, 2, -2) + 0.3 * np.random.randn(n)
    return np.stack([x, y], axis=1)


# ============================================================
# 最优传输耦合（小规模：使用匈牙利算法）
# ============================================================
def ot_coupling(source, target):
    """计算最优传输配对（最小化总距离的配对）

    对于2D点云，使用匈牙利算法（O(n^3)）
    这对应14.1节Kantorovich问题的离散解
    """
    from scipy.optimize import linear_sum_assignment
    # 代价矩阵: C[i,j] = ||source[i] - target[j||^2
    diff = source[:, None, :] - target[None, :, :]  # (n, n, 2)
    cost = np.sum(diff**2, axis=-1)  # (n, n)
    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind  # 返回source[i]应该配对的target索引

def independent_coupling(n):
    """独立耦合：随机配对（14.3.5节的默认CFM）"""
    return np.random.permutation(n)


# ============================================================
# 路径插值函数
# ============================================================
def linear_interp(z, x0, t):
    """线性插值路径（14.3.4节 OT路径/14.4.1节直线插值）
    x_t = (1-t)z + t*x_0,  v_t = x_0 - z
    """
    return (1 - t) * z + t * x0


# ============================================================
# 最优传输 vs 独立耦合——路径形态对比（14.1节/14.3.5节）
# ============================================================
print("=" * 60)
print("实验14.1-1：最优传输 vs 独立耦合——路径形态对比（14.1节/14.3.5节）")
print("=" * 60)

print("""
14.1节核心：Monge问题寻找确定性传输映射T，使得T_#p_0 = p_1
14.3.5节核心：独立耦合→弯曲路径，OT耦合→直线路径

McCann插值（14.1.4节）：x_t = (1-t)x_0 + t*T^*(x_0)
  - OT映射T^*下的路径是直线（Wasserstein空间的测地线）
  - 独立配对下的路径是弯曲的（轨迹交叉导致向量场需要"折中"）
""")

n_points = 50
source = sample_source(n_points)
target = sample_target(n_points)

# OT耦合
np.random.seed(42)
ot_idx = ot_coupling(source, target)
target_ot = target[ot_idx]

# 独立耦合
np.random.seed(123)
ind_idx = independent_coupling(n_points)
target_ind = target[ind_idx]

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# (a) 源和目标分布
ax = axes[0]
ax.scatter(source[:, 0], source[:, 1], c='blue', alpha=0.6, s=30, label='源 p_0')
ax.scatter(target[:, 0], target[:, 1], c='red', alpha=0.6, s=30, label='目标 p_1')
ax.set_title('(a) 源分布 vs 目标分布', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(alpha=0.3)
ax.set_aspect('equal')

# (b) 独立耦合路径
ax = axes[1]
t_vals = np.linspace(0, 1, 20)
for i in range(n_points):
    path = np.array([linear_interp(source[i], target_ind[i], t) for t in t_vals])
    ax.plot(path[:, 0], path[:, 1], 'purple', alpha=0.2, lw=1)
ax.scatter(source[:, 0], source[:, 1], c='blue', alpha=0.6, s=20)
ax.scatter(target_ind[:, 0], target_ind[:, 1], c='red', alpha=0.6, s=20)
ax.set_title('(b) 独立耦合：路径弯曲交叉', fontsize=13)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(alpha=0.3)
ax.set_aspect('equal')

# (c) OT耦合路径
ax = axes[2]
for i in range(n_points):
    path = np.array([linear_interp(source[i], target_ot[i], t) for t in t_vals])
    ax.plot(path[:, 0], path[:, 1], 'green', alpha=0.2, lw=1)
ax.scatter(source[:, 0], source[:, 1], c='blue', alpha=0.6, s=20)
ax.scatter(target_ot[:, 0], target_ot[:, 1], c='red', alpha=0.6, s=20)
ax.set_title('(c) OT耦合：路径短且直', fontsize=13)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(alpha=0.3)
ax.set_aspect('equal')

plt.suptitle('实验14.1-1：独立耦合 vs OT耦合（14.1节/14.3.5节）', fontsize=14, y=1.01)
plt.tight_layout()
fig_path1 = os.path.join(SAVE_DIR, '步骤1_耦合对比.png')
plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"图1已保存: {fig_path1}")

# 计算传输代价
cost_indep = np.mean(np.sum((source - target_ind)**2, axis=1))
cost_ot = np.mean(np.sum((source - target_ot)**2, axis=1))
print(f"  独立耦合平均传输代价: {cost_indep:.4f}")
print(f"  OT耦合平均传输代价:   {cost_ot:.4f}")
print(f"  OT代价 / 独立代价:    {cost_ot/cost_indep:.4f}")
print(f"  → OT耦合传输代价更低（Wasserstein距离更短）")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print("实验14.1-1 完成!")
print("=" * 60)
print("""
关键结论:
1. 最优传输基础（14.1节）
   - Monge问题：寻找确定性传输映射T，最小化传输代价
   - Kantorovich问题：离散最优传输的线性规划形式
   - Wasserstein距离：最优传输代价定义的距离度量

2. 路径形态对比（14.3.5节）
   - 独立耦合：路径弯曲交叉，传输代价高
   - OT耦合：路径短且直，传输代价低（Wasserstein距离）
   - OT映射下的McCann插值是Wasserstein空间的测地线

3. 实践意义
   - OT耦合能够获得更直的传输路径
   - 直线路径有助于减少ODE求解步数
   - 为Flow Matching的效率优化奠定基础
""")