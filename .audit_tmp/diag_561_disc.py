# -*- coding: utf-8 -*-
# 检验"37样本导致统计波动"免责是否成立：子采样看 std 估计与相关系数的波动
import numpy as np, os
base = r"d:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验"
d = np.load(os.path.join(base, "实验5.5-1", "sampling_results.npz"))
mc = d["mc_samples"]; pm = d["post_mean"]; xt = d["x_true"]
err = np.abs(xt - pm)
M = len(mc)
rng = np.random.default_rng(0)

def corr_from_subset(idx):
    sub = mc[idx]
    std = np.sqrt(np.var(sub, axis=0, ddof=1))
    return np.corrcoef(std.flatten(), err.flatten())[0,1], std

full_corr, full_std = corr_from_subset(np.arange(M))
print("full M=%d corr=%.4f" % (M, full_corr))

# leave-one-out / bootstrap 波动
for k in [10, 15, 20, 25, 30, 37]:
    cs = []
    for _ in range(200):
        idx = rng.choice(M, size=min(k, M), replace=(k>M))
        c,_ = corr_from_subset(idx)
        cs.append(c)
    cs = np.array(cs)
    print("k=%2d  corr mean=%.4f std=%.4f  [%.3f,%.3f](2.5-97.5)" % (
        k, cs.mean(), cs.std(), np.percentile(cs,2.5), np.percentile(cs,97.5)))

# 每个像素 std 估计的相对误差（37样本，ddof=1）理论 ~ sqrt(2/(M-1)) for var
print("per-pixel var rel-SE (indep approx) sqrt(2/(M-1)) = %.3f" % np.sqrt(2/(M-1)))
# 自相关：MC链是否高度相关（有效样本 << 37）
flat = mc.reshape(M, -1)
c0 = flat - flat.mean(axis=1, keepdims=True)
# lag-1 correlation across consecutive samples (pixel-averaged)
num = np.mean(np.sum(c0[:-1]*c0[1:], axis=1))
den = np.mean(np.sum(c0*c0, axis=1))
print("lag-1 autocorr across samples ~ %.3f" % (num/den))
