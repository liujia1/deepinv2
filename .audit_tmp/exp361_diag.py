# -*- coding: utf-8 -*-
"""独立复核 3.6-1：复现 JSON + 讲义三表 + 若干定量论断"""
import numpy as np
np.random.seed(42)
n = 64
sigma = np.exp(-5*np.arange(n)/(n-1))
fwd = lambda x: sigma*x
adj = lambda y: sigma*y
x_true = np.zeros(n); x_true[0:10]=1.0; x_true[15:30]=0.5; x_true[40:55]=0.8
sigma_noise=0.05
y = fwd(x_true) + sigma_noise*np.random.randn(n)
tau=1.0; maxiter=2000

xk=np.zeros(n); errors=np.zeros(maxiter+1); resid=np.zeros(maxiter+1)
errors[0]=np.linalg.norm(xk-x_true); resid[0]=np.linalg.norm(fwd(xk)-y)
for k in range(1,maxiter+1):
    xk=xk-tau*adj(fwd(xk)-y); errors[k]=np.linalg.norm(xk-x_true); resid[k]=np.linalg.norm(fwd(xk)-y)
k_opt=int(np.argmin(errors))
print("=== 步骤1 JSON 复现 ===")
print(f"k_opt={k_opt}(J64) err0={errors[0]:.3f}(J4.832) errmin={errors[k_opt]:.3f}(J3.278) errmax={errors[-1]:.3f}(J5.815) ratio={errors[-1]/errors[k_opt]:.2f}(J1.77)")

# 步骤2: k=16
xk=np.zeros(n)
for k in range(1,17): xk=xk-tau*adj(fwd(xk)-y)
print(f"\n=== 步骤2 ===\n  k=16 误差={np.linalg.norm(xk-x_true):.3f}(讲义3.391)")

# 论断: ||K^{-1}noise|| 最小二乘误差 "约16"
noise_vec = y - fwd(x_true)  # = sigma_noise*randn (同一随机流已消耗, 重取)
# 直接: x_LS = y/sigma, 误差 = ||(y/sigma)-x_true|| = ||noise/sigma||
x_ls = y/sigma
print(f"  最小二乘解误差 ||x_LS-x_true|| = ||noise/sigma|| = {np.linalg.norm(x_ls-x_true):.2f} (讲义'约16')")
print(f"  其中 noise/sigma 最大分量={np.max(np.abs((y-fwd(x_true))/sigma)):.2f}")

# Morozov: k*处残差 vs 噪声水平
print(f"\n=== Morozov 裁决 ===")
print(f"  残差@k*={resid[k_opt]:.3f}  噪声水平 σ_n√n={sigma_noise*np.sqrt(n):.3f}  (讲义0.246 vs 0.4)")
below=np.nonzero(resid<sigma_noise*np.sqrt(n))[0]
print(f"  残差首次跌破0.4的k={int(below[0]) if len(below) else None}(讲义≈6)")

# 步骤3: 噪声扫描
print(f"\n=== 步骤3 噪声扫描 (k*, 最小误差) ===")
for nl in [1e-2,2e-2,5e-2,1e-1,2e-1]:
    np.random.seed(42); yn=fwd(x_true)+nl*np.random.randn(n); xk=np.zeros(n)
    errs=np.zeros(maxiter+1); errs[0]=np.linalg.norm(x_true)
    for k in range(1,maxiter+1):
        xk=xk-tau*adj(fwd(xk)-yn); errs[k]=np.linalg.norm(xk-x_true)
    ko=int(np.argmin(errs)); print(f"  σ_n={nl}: k*={ko}(J{ {0.01:2000,0.02:1254,0.05:64,0.1:11,0.2:3}[nl] }) minerr={errs[ko]:.3f}")

# σ_min²
print(f"\n  σ_min²={sigma.min()**2:.2e} (讲义4.5e-5)")

# Picard: 真解非零平台段 |a_i|σ_i>σ_n 的前占比
nz=x_true>0; frac=(np.abs(x_true)*sigma>sigma_noise)
recoverable=nz&frac
print(f"  非零平台段总数={int(nz.sum())}  其中可恢复(|a|σ>σ_n)={int(recoverable.sum())}  占全域={int(recoverable.sum())}/{n}={recoverable.sum()/n*100:.0f}% (讲义前39%)")
idxs=np.flatnonzero(recoverable); print(f"  可恢复索引范围: {idxs.min()}..{idxs.max()}")
