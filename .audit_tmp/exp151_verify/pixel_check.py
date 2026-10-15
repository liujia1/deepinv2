# -*- coding: utf-8 -*-
"""图的轴校准核对：蓝线是否 = 1.0，紫峰是否 = beta.pdf(mode)"""
import numpy as np, os
import matplotlib.image as mpimg
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
img = mpimg.imread(os.path.join(HERE, 'rerun', '实验1_5_1_硬币贝叶斯更新.png')).astype(np.float32)[..., :3]
H, W, _ = img.shape
blue = (np.abs(img[..., 0] - 0.122) < .04) & (np.abs(img[..., 1] - 0.466) < .04) & (np.abs(img[..., 2] - 0.706) < .04)
purple = (np.abs(img[..., 0] - 0.494) < .04) & (np.abs(img[..., 1] - 0.184) < .04) & (np.abs(img[..., 2] - 0.556) < .04)
gray = (np.abs(img[..., 0] - img[..., 1]) < .02) & (np.abs(img[..., 1] - img[..., 2]) < .02) & (img[..., 0] > .72) & (img[..., 0] < .93)

col = 700  # 避开图例
print('col', col, 'blue rows', np.where(blue[:, col])[0])
print('col', col, 'grid rows', np.where(gray[:, col])[0])
rows = np.where(gray[:, col])[0]
# 连续段合并
groups = np.split(rows, np.where(np.diff(rows) > 2)[0] + 1)
gc = [g.mean() for g in groups if len(g)]
print('gridline 中心 y 像素:', gc)
# 从下往上应是 0.0,2.5,5.0,...,20.0
if len(gc) >= 3:
    p0, p1 = gc[-1], gc[-2]          # y=0.0 与 y=2.5
    scale = 2.5 / (p0 - p1)          # 数据单位 / 像素
    brow = np.where(blue[:, col])[0].mean()
    print(f'蓝线数据值 = {(p0 - brow) * scale:.4f}  (应为 1.0)')
    prow = np.where(purple[:, :])[0].min()
    # 紫峰顶（图内最高紫色像素）
    print(f'紫峰顶数据值 ≈ {(p0 - prow) * scale:.3f}  解析 mode 密度 = {beta.pdf((356-1)/(502-2), 356, 146):.3f}')
