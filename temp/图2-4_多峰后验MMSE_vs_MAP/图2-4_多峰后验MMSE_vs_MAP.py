"""
图2-4：多峰后验下 MMSE vs MAP 分歧示意
对应章节：2.3 先验的质量：MMSE vs MAP估计器 - "分歧的情形"
知识点：双峰后验下，MAP选最高的峰，MMSE在两峰之间取加权平均
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# ── 构造双峰后验分布 ───
# 确保中文字体在所有绘图操作之前加载
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] + plt.rcParams.get('font.sans-serif', [])
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 模拟二值图像场景：像素值应为0或1，但观测被噪声污染
# 先验：两个峰在0和1（二值先验）
# 似然：高斯噪声
# 后验：两个高斯峰的混合

# 双峰后验：两个高斯成分的混合
# p(x|y) = 0.6 * N(x; μ1=0.0, σ=0.15) + 0.4 * N(x; μ2=1.0, σ=0.15)
# 第一个峰更高（概率0.6），但第二个峰也在

mu1, sigma1, w1 = 0.0, 0.15, 0.6
mu2, sigma2, w2 = 1.0, 0.15, 0.4

def gaussian(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

x_range = np.linspace(-0.5, 1.5, 1000)
p = w1 * gaussian(x_range, mu1, sigma1) + w2 * gaussian(x_range, mu2, sigma2)

# ─── 计算 MAP 和 MMSE ───
# MAP：最高的峰
map_idx = np.argmax(p)
x_map = x_range[map_idx]

# MMSE：后验均值
x_mmse = np.trapezoid(x_range * p, x_range)

# 验证：理论均值 = w1*mu1 + w2*mu2
x_mmse_theory = w1 * mu1 + w2 * mu2

print(f"双峰后验: {w1}*N({mu1}, {sigma1}^2) + {w2}*N({mu2}, {sigma2}^2)")
print(f"MAP  = {x_map:.4f} (最高峰值)")
print(f"MMSE = {x_mmse:.4f} (数值积分)")
print(f"MMSE(理论) = {x_mmse_theory:.4f} (加权平均)")
print(f"分歧: MMSE - MAP = {x_mmse - x_map:.4f}")

# ─── 绘图 ───
fig, ax = plt.subplots(figsize=(11, 6))

# 后验分布曲线
ax.fill_between(x_range, p, alpha=0.3, color='steelblue')
ax.plot(x_range, p, 'b-', linewidth=2, label='后验分布 $p(x|y)$')

# 两个峰的分量（虚线）
p1 = w1 * gaussian(x_range, mu1, sigma1)
p2 = w2 * gaussian(x_range, mu2, sigma2)
ax.plot(x_range, p1, 'b--', linewidth=1, alpha=0.4, label=f'分量1: $w_1={w1}$, $\mu_1={mu1}$')
ax.plot(x_range, p2, 'b--', linewidth=1, alpha=0.4, label=f'分量2: $w_2={w2}$, $\mu_2={mu2}$')

# MAP 标注（最高峰值）
ax.axvline(x=x_map, color='red', linestyle='--', linewidth=2, alpha=0.8)
ax.plot(x_map, p[map_idx], 'ro', markersize=12, zorder=5)
ax.annotate(f'MAP = {x_map:.2f}\n(选最高的峰)', 
            xy=(x_map, p[map_idx]), xytext=(x_map - 0.35, p[map_idx] + 0.15),
            fontsize=11, ha='center', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# MMSE 标注（两峰之间）
ax.axvline(x=x_mmse, color='green', linestyle='--', linewidth=2, alpha=0.8)
mmse_p_idx = np.argmin(np.abs(x_range - x_mmse))
ax.plot(x_mmse, p[mmse_p_idx], 'go', markersize=12, zorder=5)
ax.annotate(f'MMSE = {x_mmse:.2f}\n(加权平均)', 
            xy=(x_mmse, p[mmse_p_idx]), xytext=(x_mmse + 0.35, p[mmse_p_idx] + 0.3),
            fontsize=11, ha='center', color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

# 标注"两峰之间"
ax.annotate('', xy=(x_mmse, 0.05), xytext=(mu1, 0.05),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
ax.annotate('', xy=(mu2, 0.05), xytext=(x_mmse, 0.05),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
ax.text(x_mmse, 0.08, 'MMSE在两峰之间', fontsize=9, ha='center', color='gray')

# 标注二值图像含义
ax.text(mu1, -0.15, '像素值=0\n(背景)', fontsize=9, ha='center', color='blue')
ax.text(mu2, -0.15, '像素值=1\n(前景)', fontsize=9, ha='center', color='blue')

ax.set_xlabel('$x$ (像素值)', fontsize=13)
ax.set_ylabel('后验概率密度 $p(x|y)$', fontsize=13)
ax.set_title('图2-4：多峰后验下 MAP vs MMSE 分歧\n二值图像场景：像素应为0或1，但噪声使后验呈双峰', 
             fontsize=15, fontweight='bold', pad=12)
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(bottom=-0.2)
ax.grid(True, alpha=0.3)

# 底部说明（使用ax.text + transAxes，继承全局字体设置）
ax.text(0.5, -0.15,
        '核心观察：MAP选择概率最高的峰（此处为0），但MMSE取两峰的加权平均（0.4）。'
        '在二值图像去噪中，MAP给出"背景"判断，而MMSE给出模糊的0.4——两者代表不同的决策哲学。',
        ha='center', va='top', fontsize=9, transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3), wrap=True)

plt.subplots_adjust(bottom=0.20)
plt.savefig(os.path.join(SAVE_DIR, '图2-4_多峰后验MMSE_vs_MAP.png'), dpi=150, bbox_inches='tight')
plt.close()

print("图2-4已生成：", os.path.join(SAVE_DIR, '图2-4_多峰后验MMSE_vs_MAP.png'))
