# -*- coding: utf-8 -*-
"""独立复算实验14.1-1的数值，不覆盖原产物"""
import numpy as np
from scipy.optimize import linear_sum_assignment

np.random.seed(42)

def sample_source(n):
    mode = np.random.rand(n) < 0.5
    x = np.where(mode, -2, 2) + 0.3 * np.random.randn(n)
    y = np.where(mode, 2, -2) + 0.3 * np.random.randn(n)
    return np.stack([x, y], axis=1)

def sample_target(n):
    mode = np.random.rand(n) < 0.5
    x = np.where(mode, 2, -2) + 0.3 * np.random.randn(n)
    y = np.where(mode, 2, -2) + 0.3 * np.random.randn(n)
    return np.stack([x, y], axis=1)

n_points = 50
source = sample_source(n_points)
target = sample_target(n_points)

diff = source[:, None, :] - target[None, :, :]
cost = np.sum(diff**2, axis=-1)
row, col = linear_sum_assignment(cost)
target_ot = target[col]

np.random.seed(123)
ind_idx = np.random.permutation(n_points)
target_ind = target[ind_idx]

cost_indep = np.mean(np.sum((source - target_ind)**2, axis=1))
cost_ot = np.mean(np.sum((source - target_ot)**2, axis=1))
print("cost_indep =", cost_indep)
print("cost_ot    =", cost_ot)
print("ratio      =", cost_ot / cost_indep)
print("sum_ot     =", cost_ot * n_points, " sum_indep =", cost_indep * n_points)

# 簇内/簇间配对计数
src_mode = source[:, 0] < 0   # True=左上(-2,2)
tgt_mode = target_ot[:, 0] < 0  # True=左下(-2,-2)
print("同簇(垂直)配对数:", int(np.sum(src_mode == tgt_mode)), "/", n_points)
print("源簇计数:", int(src_mode.sum()), "目标簇计数:", int((target[:,0] < 0).sum()))
