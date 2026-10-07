# -*- coding: utf-8 -*-
"""比对归档缓存 vs 当前代码 fresh 重跑结果"""
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import numpy as np

old = np.load(r"d:\works\deepinv\第9章 VAE与重参数化\第九章 配套实验\实验9.2-1\numerical_results_cache.npz", allow_pickle=True)
new = np.load(r"d:\works\deepinv\.audit_tmp\exp921_verify\numerical_results_cache.npz", allow_pickle=True)

for k in old.files:
    a, b = old[k], new[k]
    if a.shape != b.shape:
        print(f"{k}: 形状不同 old={a.shape} new={b.shape}")
        continue
    exact = np.array_equal(a, b)
    rel = np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
    print(f"{k}: 逐位一致={exact}  max相对偏差={rel:.3e}  | old: mean={a.mean():.4f} std={a.std():.4f} | new: mean={b.mean():.4f} std={b.std():.4f}")
