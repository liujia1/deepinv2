"""
图2-1：L1 vs L2 几何解释——为什么L1促稀疏
对应章节：2.2 经典先验族 - Laplace先验
知识点：L1约束（菱形）与L2约束（圆形）的几何差异；角点处相切→稀疏解
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
import sys

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
_chinese_path = os.path.join(SAVE_DIR, '.chinese')
if os.path.isdir(_chinese_path):
    sys.path.insert(0, _chinese_path)
    try:
        from chinese_font import setup_chinese_font
        setup_chinese_font(save_dir=_chinese_path)
    except ImportError:
        print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
else:
    print("警告: .chinese 文件夹未找到，中文字体可能无法正常显示")

# ─── 参数设置 ───
# 确保中文字体在所有绘图操作之前加载
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] + plt.rcParams.get('font.sans-serif', [])
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

theta = np.linspace(0, 2*np.pi, 400)

# L2 正则项的等高线：圆形
# R(w) = w1^2 + w2^2 = C → 半径 r = sqrt(C)
r_l2_1 = 1.0
r_l2_2 = 1.5

# L1 正则项的等高线：菱形（旋转45度的正方形）
# R(w) = |w1| + |w2| = C → 参数方程
# 第一象限: w2 = C - w1, w1 ∈ [0, C]
def l1_contour(C, n=100):
    """生成L1等高线（菱形）的点"""
    pts = []
    # 第一象限: (t, C-t), t∈[0,C]
    t = np.linspace(0, C, n)
    pts.extend([(x, C-x) for x in t])
    # 第二象限: (-t, C-t), t∈[0,C]
    pts.extend([(-x, C-x) for x in t[1:]])
    # 第三象限: (-t, -(C-t)), t∈[0,C]
    pts.extend([(-x, -(C-x)) for x in t[1:]])
    # 第四象限: (t, -(C-t)), t∈[0,C]
    pts.extend([(x, -(C-x)) for x in t[1:]])
    return np.array(pts)

# 损失函数等高线（数据项）：椭圆
# L(w) = (w1 - w1_0)^2 + (w2 - w2_0)^2 = C
def ellipse_contour(cx, cy, a, b, n=200):
    """椭圆等高线"""
    t = np.linspace(0, 2*np.pi, n)
    return np.column_stack([cx + a*np.cos(t), cy + b*np.sin(t)])

# 约束集（用于LASSO）：|w1| + |w2| <= t
def l1_ball_boundary(t, n=100):
    """L1球的边界（菱形）"""
    pts = []
    t_arr = np.linspace(0, t, n)
    pts.extend([(x, t-x) for x in t_arr])          # 第一象限
    pts.extend([(-x, t-x) for x in t_arr[1:]])     # 第二象限
    pts.extend([(-x, -(t-x)) for x in t_arr[1:]])  # 第三象限
    pts.extend([(x, -(t-x)) for x in t_arr[1:]])   # 第四象限
    return np.array(pts)

# ─── 绘图 ───
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── 左子图：L2正则化（圆形）──
ax = axes[0]
# L2正则项等高线（同心圆）
for r, alpha in [(r_l2_1, 0.3), (r_l2_2, 0.15)]:
    ax.fill(np.cos(theta)*r, np.sin(theta)*r, alpha=alpha, color='steelblue')
    ax.plot(np.cos(theta)*r, np.sin(theta)*r, 'b-', linewidth=1.5)
# 损失函数等高线（椭圆，中心偏离原点）
ell = ellipse_contour(1.2, 0.8, 1.5, 1.0)
ax.plot(ell[:, 0], ell[:, 1], 'r-', linewidth=2, label='损失函数等高线')
ell2 = ellipse_contour(1.2, 0.8, 2.0, 1.3)
ax.plot(ell2[:, 0], ell2[:, 1], 'r-', linewidth=1.5, alpha=0.5)
# 最优解（圆与椭圆切点）
# 切点方向与中心到原点方向一致
w_opt_l2 = np.array([0.85, 0.57])  # 近似
ax.plot(w_opt_l2[0], w_opt_l2[1], 'ko', markersize=12, zorder=5)
ax.annotate('最优解\n(非稀疏)', xy=w_opt_l2, xytext=(w_opt_l2[0]+0.5, w_opt_l2[1]+0.3),
            fontsize=11, ha='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
# 坐标轴
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.set_xlabel('$w_1$', fontsize=13)
ax.set_ylabel('$w_2$', fontsize=13)
ax.set_title('L2正则化（高斯先验）\n约束集：圆形', fontsize=13, fontweight='bold')
# 添加图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='steelblue', alpha=0.3, edgecolor='blue', label='L2正则项等高线'),
                   plt.Line2D([0], [0], color='red', linewidth=2, label='损失函数等高线')]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

# ── 右子图：L1正则化（菱形）──
ax = axes[1]
# L1正则项等高线（菱形）
l1_c1 = l1_contour(1.2, n=80)
l1_c2 = l1_contour(1.8, n=80)
ax.fill(l1_c1[:, 0], l1_c1[:, 1], alpha=0.3, color='forestgreen')
ax.plot(l1_c1[:, 0], l1_c1[:, 1], 'g-', linewidth=1.5)
ax.fill(l1_c2[:, 0], l1_c2[:, 1], alpha=0.15, color='forestgreen')
ax.plot(l1_c2[:, 0], l1_c2[:, 1], 'g-', linewidth=1.5, alpha=0.7)
# 损失函数等高线（椭圆，中心偏离原点）
ell = ellipse_contour(1.0, 0.6, 1.2, 0.8)
ax.plot(ell[:, 0], ell[:, 1], 'r-', linewidth=2, label='损失函数等高线')
ell2 = ellipse_contour(1.0, 0.6, 1.6, 1.1)
ax.plot(ell2[:, 0], ell2[:, 1], 'r-', linewidth=1.5, alpha=0.5)
# 最优解（菱形角点与椭圆切点）→ 在坐标轴上
w_opt_l1 = np.array([1.2, 0.0])  # 在w1轴上（w2=0）
ax.plot(w_opt_l1[0], w_opt_l1[1], 'ko', markersize=12, zorder=5)
ax.annotate('最优解\n(稀疏：$w_2=0$)', xy=w_opt_l1, xytext=(w_opt_l1[0]+0.6, w_opt_l1[1]+0.5),
            fontsize=11, ha='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
# 标注角点
ax.plot(0, 1.2, 'g^', markersize=8, alpha=0.7)
ax.plot(0, -1.2, 'gv', markersize=8, alpha=0.7)
ax.plot(1.2, 0, 'g>', markersize=8, alpha=0.7)
ax.plot(-1.2, 0, 'g<', markersize=8, alpha=0.7)
ax.text(1.4, 0.15, '角点', fontsize=9, color='green')
# 坐标轴
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.set_xlabel('$w_1$', fontsize=13)
ax.set_ylabel('$w_2$', fontsize=13)
ax.set_title('L1正则化（Laplace先验）\n约束集：菱形', fontsize=13, fontweight='bold')
legend_elements = [Patch(facecolor='forestgreen', alpha=0.3, edgecolor='green', label='L1正则项等高线'),
                   plt.Line2D([0], [0], color='red', linewidth=2, label='损失函数等高线')]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

#  总标题和注释 ─
fig.suptitle('L1 vs L2 几何解释\n为什么L1促稀疏', fontsize=15, fontweight='bold', y=0.96, linespacing=1.5)

# 底部注释（使用fig.text在整个图底部居中）
fig.text(0.5, 0.04, 
         '核心直觉：L2的圆形约束集没有"角点"，最优解通常不在坐标轴上；'
         'L1的菱形约束集有4个角点，椭圆更容易在角点处与之相切，从而得到稀疏解（某些分量为0）',
         ha='center', va='bottom', fontsize=9, linespacing=1.5,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3), wrap=True)

plt.subplots_adjust(wspace=0.12, hspace=0.3, bottom=0.15, top=0.85)
plt.savefig(os.path.join(SAVE_DIR, '图2-1_L1_vs_L2几何解释.png'), dpi=150, bbox_inches='tight')
plt.close()

print("图2-1已生成：", os.path.join(SAVE_DIR, '图2-1_L1_vs_L2几何解释.png'))
