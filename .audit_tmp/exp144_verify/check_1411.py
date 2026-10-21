# -*- coding: utf-8 -*-
"""核查 14.1-1：对称交叉几何下 OT vs 独立耦合的代价差异来自哪里（路径长度 vs 噪声对齐）"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def sample_source(n, rng=None):
    # rng=None 时用全局 legacy RNG（与实验脚本完全一致）
    if rng is None:
        mode = np.random.rand(n) < 0.5
        x = np.where(mode, -2, 2) + 0.3 * np.random.randn(n)
        y = np.where(mode, 2, -2) + 0.3 * np.random.randn(n)
    else:
        mode = rng.random(n) < 0.5
        x = np.where(mode, -2, 2) + 0.3 * rng.standard_normal(n)
        y = np.where(mode, 2, -2) + 0.3 * rng.standard_normal(n)
    return np.stack([x, y], axis=1)


def sample_target(n, rng=None):
    if rng is None:
        mode = np.random.rand(n) < 0.5
        x = np.where(mode, 2, -2) + 0.3 * np.random.randn(n)
        y = np.where(mode, 2, -2) + 0.3 * np.random.randn(n)
    else:
        mode = rng.random(n) < 0.5
        x = np.where(mode, 2, -2) + 0.3 * rng.standard_normal(n)
        y = np.where(mode, 2, -2) + 0.3 * rng.standard_normal(n)
    return np.stack([x, y], axis=1)


np.random.seed(42)
src = sample_source(50)
tgt = sample_target(50)

# 复刻脚本：OT + seed(123) 的 permutation
diff = src[:, None, :] - tgt[None, :, :]
C = (diff ** 2).sum(-1)
_, ot_idx = linear_sum_assignment(C)

np.random.seed(123)
ind_idx = np.random.permutation(50)

d_ot = np.linalg.norm(src - tgt[ot_idx], axis=1)
d_ind = np.linalg.norm(src - tgt[ind_idx], axis=1)
print("=== 复刻交付数字 ===")
print(f"OT   : 平均平方代价={np.mean(d_ot**2):.4f}  平均欧氏长度={np.mean(d_ot):.4f}")
print(f"独立 : 平均平方代价={np.mean(d_ind**2):.4f}  平均欧氏长度={np.mean(d_ind):.4f}")
print(f"比值 : {np.mean(d_ot**2)/np.mean(d_ind**2):.4f}")

# 独立耦合多次重复（真正 iid 采样目标，允许重复）
print("\n=== 独立耦合 200 次重复（iid 抽样目标，非 permutation） ===")
c2, l2 = [], []
for k in range(200):
    r = np.random.default_rng(1000 + k)
    t = sample_target(50, r)
    d = np.linalg.norm(src - t, axis=1)
    c2.append(np.mean(d ** 2)); l2.append(np.mean(d))
print(f"平均平方代价 {np.mean(c2):.4f} ± {np.std(c2):.4f}   平均欧氏长度 {np.mean(l2):.4f}")

# 独立耦合 permutation 200 次
print("\n=== 独立耦合 200 次重复（permutation，同交付口径） ===")
c3 = []
for k in range(200):
    p = np.random.default_rng(2000 + k).permutation(50)
    c3.append(np.mean(np.sum((src - tgt[p]) ** 2, axis=1)))
print(f"平均平方代价 {np.mean(c3):.4f} ± {np.std(c3):.4f}  （交付值 15.5402 的 z={(15.5402-np.mean(c3))/np.std(c3):+.2f}σ）")

# 分解：OT 里有多少是"模式中心"、多少是"噪声"
cent = np.array([[2.0, 2.0], [-2.0, -2.0]])
def nearest(p):
    return cent[np.argmin(((p[:, None, :] - cent[None]) ** 2).sum(-1), 1)]
# 源中心
csrc = np.array([[-2.0, 2.0], [2.0, -2.0]])
def nearest_src(p):
    return csrc[np.argmin(((p[:, None, :] - csrc[None]) ** 2).sum(-1), 1)]

# 只用模式中心（去掉噪声）计算，检验对称简并
src_c = nearest_src(src)
tgt_c_ot = nearest(tgt[ot_idx])
tgt_c_ind = nearest(tgt[ind_idx])
print("\n=== 中心项（去噪声）的平方代价 ===")
print(f"OT  中心项={np.mean(((src_c-tgt_c_ot)**2).sum(1)):.3f}   独立 中心项={np.mean(((src_c-tgt_c_ind)**2).sum(1)):.3f}")
print("  （若两者都=16，说明几何上路径长度完全相同，代价差异只能来自噪声对齐）")

# 路径长度：欧氏
print(f"\nOT  平均路径长度={np.mean(d_ot):.4f}；独立 平均路径长度={np.mean(d_ind):.4f}（差 {100*(np.mean(d_ind)/np.mean(d_ot)-1):+.1f}%）")
