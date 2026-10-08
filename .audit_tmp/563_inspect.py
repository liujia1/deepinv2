# -*- coding: utf-8 -*-
import numpy as np, os, json

npz = r"d:/works/deepinv/第5章 朗之万动力学与得分函数/第五章配套实验/实验5.5-1/sampling_results.npz"
data = np.load(npz)
print("keys:", list(data.keys()))
for k in data.keys():
    try:
        a = data[k]
        print(k, getattr(a, 'shape', None), getattr(a, 'dtype', None))
    except Exception as e:
        print(k, "ERR", e)

post_mean = data['post_mean']
post_var0 = data['post_var']
mc_samples = data['mc_samples']
x_true = data['x_true']
y_obs = data['y_obs']

# 与 5.6-3 相同的重估
post_var = np.var(mc_samples, axis=0, ddof=1)

def compute_psnr(x, y):
    mse = np.mean((x - y) ** 2)
    if mse == 0: return float('inf')
    max_val = max(x.max(), 1.0)
    return 10 * np.log10(max_val ** 2 / mse)

def compute_global_similarity(x, y):
    C1 = 0.01 ** 2; C2 = 0.03 ** 2
    mu_x = np.mean(x); mu_y = np.mean(y)
    sigma_x = np.var(x); sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    return ((2*mu_x*mu_y+C1)*(2*sigma_xy+C2))/((mu_x**2+mu_y**2+C1)*(sigma_x+sigma_y+C2))

def compute_nrmse(x, y):
    std_x = np.std(x)
    if std_x == 0: return float('inf')
    return np.sqrt(np.mean((x - y) ** 2)) / std_x

x_mmse = post_mean
x_map = y_obs

print("\nx_true range:", x_true.min(), x_true.max(), " y_obs range:", y_obs.min(), y_obs.max())
print("post_mean shape", post_mean.shape, " mc_samples", mc_samples.shape)

print("\n=== 步骤2 指标（5.6-3 定义）===")
print("MMSE  PSNR=%.4f  sim=%.6f  NRMSE=%.6f" % (compute_psnr(x_true,x_mmse), compute_global_similarity(x_true,x_mmse), compute_nrmse(x_true,x_mmse)))
print("y_obs PSNR=%.4f  sim=%.6f  NRMSE=%.6f" % (compute_psnr(x_true,x_map),  compute_global_similarity(x_true,x_map),  compute_nrmse(x_true,x_map)))

post_std = np.sqrt(post_var)
print("\n=== 步骤5 不确定性 ===")
print("post_std mean=%.6f max=%.6f" % (np.mean(post_std), np.max(post_std)))
unc = (post_std - post_std.min())/(post_std.max()-post_std.min()+1e-10)
thr = np.percentile(unc, 90)
mask = unc > thr
print("high_unc_ratio_pct=%.4f" % (np.mean(mask)*100))
print("base_var(mean post_var)=%.8f" % np.mean(post_var))
