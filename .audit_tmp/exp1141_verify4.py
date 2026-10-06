# -*- coding: utf-8 -*-
"""复核脚本4：图1a曲线与checkpoint history一致性（宽松颜色掩码 + 左右端y标定）"""
import sys, os
import numpy as np
import torch
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
EXP = r"D:\works\deepinv\第11章 扩散模型：变分视角\第十一章 配套实验\实验11.4-1"

arr = np.array(Image.open(os.path.join(EXP, '步骤1a_训练曲线.png')))[..., :3].astype(int)
R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
blue = (B > R + 40) & (B > G + 30) & (B > 100)
red = (R > B + 40) & (R > G + 40) & (R > 100)
print("blue px:", blue.sum(), " red px:", red.sum())

# 轴刻度标定：找y轴刻度位置较麻烦，改用相对形状验证：
# 检查每条曲线 左端y像素(高值) 与 右端y像素(低值) 的相对顺序
def curve_ys(mask):
    xs_out = {}
    W = mask.shape[1]
    for x in range(W):
        ys = np.where(mask[:, x])[0]
        if len(ys):
            # 取最底部像素簇(曲线在图例下方), 避免右上图例污染
            top = ys.max()
            cluster = ys[ys >= top - 8]
            xs_out[x] = cluster.mean()
    return xs_out

cb, cr = curve_ys(blue), curve_ys(red)
bx = sorted(cb); rx = sorted(cr)
# 排除图例区域（图例在右上）：图例附近行范围约? 直接看左端与右端
print("blue x范围:", bx[0], bx[-1])
print("red   x范围:", rx[0], rx[-1])
# 左端（曲线起点）
bl = [cb[x] for x in bx[:5]]; br_ = [cb[x] for x in bx[-5:]]
rl = [cr[x] for x in rx[:5]]; rr_ = [cr[x] for x in rx[-5:]]
print("blue 左端y像素均值=%.1f 右端=%.1f (y像素越小值越大)" % (np.mean(bl), np.mean(br_)))
print("red  左端y像素均值=%.1f 右端=%.1f" % (np.mean(rl), np.mean(rr_)))

# y轴数据标定: 用两条曲线左端比值验证。history: eps[0]=0.1963, x0[0]=0.1003, 比值1.957
# 轴范围近似 [0.04?, 0.20?] 由图可见ticks 0.04..0.20。取像素: y(0.20)≈? 我们用两端线性解:
# 设 y_pix = a - b*value, 由左端两曲线: (a-b*0.1963)/(a-b*0.1003)
yb_l, yr_l = np.mean(bl), np.mean(rl)
yb_r, yr_r = np.mean(br_), np.mean(rr_)
# 解 a,b: 0.1963 -> yb_l, 0.1003 -> yr_l
b = (yr_l - yb_l) / (0.1963 - 0.1003); a = yb_l + b * 0.1963
print("用左端两点标定: value(y) = (a - y)/b, 预测右端: eps=%.4f x0=%.4f" %
      ((a - yb_r) / b, (a - yr_r) / b))
print("checkpoint 右端真值: eps=0.04755 x0=0.03342")
