# -*- coding: utf-8 -*-
"""(D) 图的像素级校准核对; (E) md 中"换种子可能均值先冲过0.7再回落"的可实现性"""
import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.image as mpimg
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
img = mpimg.imread(os.path.join(HERE, 'rerun', '实验1_5_1_硬币贝叶斯更新.png')).astype(np.float32)
print('img shape', img.shape)

# 找蓝色(matplotlib tab:blue = (0.122,0.466,0.706)) 与紫色(tab:purple=(0.494,0.184,0.556)) 像素
def mask_of(c, tol=0.06):
    return (np.abs(img[..., 0] - c[0] < tol) & np.abs(img[..., 1] - c[1] < tol) & np.abs(img[..., 2] - c[2] < tol))
blue = mask_of((0.122, 0.466, 0.706))
purple = mask_of((0.494, 0.184, 0.556))
ys, xs = np.where(blue)
print('blue 像素数', len(xs), 'x 范围', xs.min(), xs.max())
# 图例区会包含蓝/紫短线，先按 y 直方图看分布
import collections
# 取图内（排除图例框）：图例在右上，x 较大且 y 较小
# 用列剖面：对若干 x 位置，列出 blue 的 y
for xv in [0.05, 0.3, 0.5, 0.7, 0.95]:
    sub = (ys[(xs > int(img.shape[1]*xv)) & (xs < int(img.shape[1]*xv) + 4)])
    print(f'  x={xv}: blue y 像素 {sorted(set(sub.tolist()))[:6]}')

# 轴校准：找黑色坐标轴刻度位置比较复杂，改用两条参考：
# 1) 真实概率竖线 (黑色虚线) 应在 x=0.7; 2) 蓝线应在 y=1.0
# 先定位绘图区 bbox：通过 grid 线 (浅灰 alpha0.3 -> 约 (0.85..)) 太麻烦，改用轴脊(黑色实线)
dark = (img[..., 0] < 0.25) & (img[..., 1] < 0.25) & (img[..., 2] < 0.25)
col_counts = dark.sum(0); row_counts = dark.sum(1)
H, W = dark.shape
spine_x = np.where(col_counts > 0.5 * H)[0]
spine_y = np.where(row_counts > 0.5 * W)[0]
print('竖直轴脊 x 像素', spine_x, '水平轴脊 y 像素', spine_y)

th = 0.7
rng = np.random.default_rng(7)
M = 40000
allf = (rng.random((M, 500)) < th)
h = {n: allf[:, :n].sum(1) for n in (2, 10, 50, 500)}
m = {n: (1 + h[n]) / (2 + n) for n in h}
print('\n=== E. 均值"冲过 0.7 再回落" ===')
print('P(m10 > 0.7) =', np.mean(m[10] > 0.7))
print('P(m10 > 0.7 且 m50 < m10) =', np.mean((m[10] > 0.7) & (m[50] < m[10])))
print('P(m50 > 0.7 且 m500 < m50) =', np.mean((m[50] > 0.7) & (m[500] < m[50])))
print('P(存在某检查点 > 0.7 且之后回落) =', np.mean(np.any(np.stack([m[10] > 0.7, m[50] > 0.7, m[500] > 0.7]), 0) & np.any(np.diff(np.stack([m[2], m[10], m[50], m[500]]).T) < 0, 1)))
# 具体给几个种子演示
import matplotlib.pyplot as plt
for seed in [42, 0, 1, 2, 3, 7, 11, 123]:
    f = (np.random.default_rng(seed).random(500) < th).astype(int)
    vals = [(1 + f[:n].sum()) / (2 + n) for n in (0, 2, 10, 50, 500)]
    print(f'  seed={seed:>4}: ' + ' '.join(f'{v:.4f}' for v in vals))
