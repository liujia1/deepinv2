"""
图2-3：偏斜分布下 MMSE vs MAP 分歧示意
对应章节：2.3 先验的质量：MMSE vs MAP估计器 - "分歧的情形"
知识点：偏斜后验下，MAP偏向峰值（零点），MMSE被尾部拉远
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
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

# ─── 构造偏斜后验分布 ───
# Laplace先验 + 高斯似然 → 偏斜后验
# 先验: p(x) ∝ exp(-|x|/b)  (Laplace, b=0.5)
# 似然: p(y|x) ∝ exp(-(y-x)^2/(2σ^2))  (σ=0.3)
# 后验: p(x|y) ∝ exp(-|x|/b - (y-x)^2/(2σ^2))
# 这个后验是偏斜的：峰值偏向零点，尾部向远离零点方向延伸

b = 0.5      # Laplace先验尺度
sigma = 0.3  # 噪声标准差
y_obs = 0.8  # 观测值

# 计算后验（非归一化）
def log_posterior(x, y, b, sigma):
    """对数后验: log p(x|y) = -|x|/b - (y-x)^2/(2σ^2) + const"""
    return -np.abs(x)/b - (y - x)**2 / (2 * sigma**2)

x_range = np.linspace(-2, 3, 1000)
log_p = log_posterior(x_range, y_obs, b, sigma)
p = np.exp(log_p - log_p.max())  # 归一化到峰值=1

# 数值归一化（使积分为1）
dx = x_range[1] - x_range[0]
p = p / (p.sum() * dx)

# ─── 计算 MAP 和 MMSE ───
# MAP：后验峰值位置
map_idx = np.argmax(p)
x_map = x_range[map_idx]

# MMSE：后验均值
x_mmse = np.trapezoid(x_range * p, x_range)

print(f"观测值 y = {y_obs}")
print(f"MAP  = {x_map:.4f} (后验峰值)")
print(f"MMSE = {x_mmse:.4f} (后验均值)")
print(f"分歧: MMSE - MAP = {x_mmse - x_map:.4f}")

# ─── 绘图 ───
fig, ax = plt.subplots(figsize=(11, 6))

# 后验分布曲线
ax.fill_between(x_range, p, alpha=0.3, color='steelblue')
ax.plot(x_range, p, 'b-', linewidth=2, label='后验分布 $p(x|y)$')

# MAP 标注
ax.axvline(x=x_map, color='red', linestyle='--', linewidth=2, alpha=0.8)
ax.plot(x_map, p[map_idx], 'ro', markersize=12, zorder=5)
ax.annotate(f'MAP = {x_map:.2f}\n(峰值，偏向零点)', 
            xy=(x_map, p[map_idx]), xytext=(x_map - 0.8, p[map_idx] + 0.15),
            fontsize=11, ha='center', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# MMSE 标注
ax.axvline(x=x_mmse, color='green', linestyle='--', linewidth=2, alpha=0.8)
# 找到MMSE位置对应的概率密度
mmse_p_idx = np.argmin(np.abs(x_range - x_mmse))
ax.plot(x_mmse, p[mmse_p_idx], 'go', markersize=12, zorder=5)
ax.annotate(f'MMSE = {x_mmse:.2f}\n(均值，被尾部拉远)', 
            xy=(x_mmse, p[mmse_p_idx]), xytext=(x_mmse + 0.8, p[mmse_p_idx] + 0.15),
            fontsize=11, ha='center', color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

# 标注"尾部拉动"效应
# 找到尾部区域（x > 1.5）
tail_mask = x_range > 1.5
ax.fill_between(x_range[tail_mask], p[tail_mask], alpha=0.2, color='orange')
ax.annotate('尾部质量\n拉动MMSE向右', xy=(2.0, 0.02), xytext=(2.3, 0.08),
            fontsize=10, ha='center', color='darkorange',
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))

# 零点标注
ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.text(0.05, ax.get_ylim()[1]*0.9, '零点', fontsize=9, color='gray')

# 观测值标注
ax.axvline(x=y_obs, color='purple', linestyle=':', linewidth=1.5, alpha=0.6)
ax.text(y_obs + 0.05, ax.get_ylim()[1]*0.8, f'观测 $y={y_obs}$', fontsize=9, color='purple')

ax.set_xlabel('$x$', fontsize=13)
ax.set_ylabel('后验概率密度 $p(x|y)$', fontsize=13)
ax.set_title('图2-3：偏斜后验下 MAP vs MMSE 分歧\nLaplace先验 + 高斯似然 → 偏斜后验', 
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# 底部说明
fig.text(0.5, 0.02,
         '核心观察：MAP选择后验峰值（最可能值），偏向零点；'
         'MMSE计算后验均值，被右侧尾部的质量拉远。两者在非对称分布下产生分歧。',
         ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(os.path.join(SAVE_DIR, '图2-3_偏斜分布MMSE_vs_MAP.png'), dpi=150, bbox_inches='tight')
plt.close()

print("图2-3已生成：", os.path.join(SAVE_DIR, '图2-3_偏斜分布MMSE_vs_MAP.png'))
