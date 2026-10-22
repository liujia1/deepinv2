# -*- coding: utf-8 -*-
"""为 17.6.2/17.6.4 补的最小机制验证：
   p4 群卷积（4 个旋转角）对 90° 旋转严格等变——输入旋转 = group 维度循环移位；
   标准卷积（同一随机核）对 90° 旋转不等变。
   循环边界，避免边界伪影（见 17.6-1 Step4 的教训）。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn.functional as F

torch.manual_seed(0)
H = W = 16
w = torch.randn(1, 1, 3, 3)
x = torch.randn(1, 1, H, W)

def rot(x, k):
    return torch.rot90(x, k, dims=[2, 3])

def conv_c(x, w):
    """循环边界 3x3 卷积"""
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode='circular'), w, padding=0)

def gconv(x, w):
    """p4 群卷积（单通道最小版）：输出 group 维度 4 个旋转副本"""
    return torch.stack([conv_c(rot(x, r), w)[0, 0] for r in range(4)])

# 检验：GConv(R90 x)[r] 应等于 GConv(x)[(r+1) % 4]
Gx = gconv(x, w)
GRx = gconv(rot(x, 1), w)
err_shift = (GRx - torch.stack([Gx[(r + 1) % 4] for r in range(4)])).abs().max().item()

# 对照：标准卷积 conv(R90 x) 与 R90 conv(x)
yc = conv_c(x, w)
ycr = conv_c(rot(x, 1), w)
err_std = (ycr - rot(yc, 1)).abs().max().item()

print(f'[p4 群卷积] |GConv(R90 x) - shift_group(GConv(x))|_max = {err_shift:.2e}  (应为数值零)')
print(f'[标准卷积]  |conv(R90 x) - R90 conv(x)|_max          = {err_std:.2e}  (显著>0)')
print(f'相对量级: p4 误差 / 标准卷积误差 = {err_shift / err_std:.1e}')
