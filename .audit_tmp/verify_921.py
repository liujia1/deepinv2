# -*- coding: utf-8 -*-
"""实验9.2-1 复核脚本：缓存数值 vs 理论精确值 vs 讲义叙述"""
import sys, io, os
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import numpy as np

D = r"d:\works\deepinv\第9章 VAE与重参数化\第九章 配套实验\实验9.2-1"
d = np.load(os.path.join(D, "numerical_results_cache.npz"), allow_pickle=True)
print("keys:", d.files)

MU, L = 2.0, 50
TRUE = 4.0
rg = d['reinforce_estimates']; rp = d['reparam_estimates']
rs = d['reinforce_small']; rps = d['reparam_small']
print(f"n_trials step2: {len(rg)} {len(rp)} {len(rs)} {len(rps)}")

def reinforce_exact_var(mu, sigma, L):
    # 单样本 x=z^2*(z-mu)/sigma^2: E[x^2]=15+12mu^2/s^2+mu^4/s^4, E[x]=2mu
    # Var = 2 sigma^4 + 4 mu^2 sigma^2 (解析结果), 除L
    s2 = sigma**2
    per = 2*s2**2 + 4*mu**2*s2
    return per / L

def reparam_exact_var(mu, sigma, L):
    return 4*sigma**2 / L

print("\n== 步骤2 σ=1.0 ==")
print(f"REINFORCE  mean={rg.mean():.4f} (true 4.0)  var={rg.var():.4f}  exact_theory={reinforce_exact_var(MU,1.0,L):.4f}")
print(f"重参数化   mean={rp.mean():.4f}  var={rp.var():.4f}  exact_theory={reparam_exact_var(MU,1.0,L):.4f}")
print(f"方差比 实测={rg.var()/rp.var():.2f}  理论比={(reinforce_exact_var(MU,1.0,L)/reparam_exact_var(MU,1.0,L)):.2f}")
# 讲义代码里的"theory"形状线
print(f"代码theory线 (σ²+μ²/σ²)/L @σ=1: {(1+4)/50:.4f}  ; 4σ²/L: {4/50:.4f}")

print("\n== 步骤2 σ=0.1 ==")
print(f"REINFORCE  mean={rs.mean():.4f}  var={rs.var():.4f}  exact_theory={reinforce_exact_var(MU,0.1,L):.4f}")
print(f"重参数化   mean={rps.mean():.4f}  var={rps.var():.6f}  exact_theory={reparam_exact_var(MU,0.1,L):.6f}")
print(f"方差比 实测={rs.var()/rps.var():.1f}  理论比={(reinforce_exact_var(MU,0.1,L)/reparam_exact_var(MU,0.1,L)):.1f}")
print(f"标准差 实测 REINFORCE={rs.std():.4f} 重参数化={rps.std():.4f}")
print(f"讲义图(b)注释: REINFORCE std≈5.2 峰值≈0.077 ; 重参数化 std≈0.032 峰值≈12.5")

sig = d['sigma_list']; rv = d['reinforce_vars']; pv = d['reparam_vars']
print("\n== 步骤3 σ扫描 ==")
print(f"sigma范围: {sig[0]:.4f} .. {sig[-1]:.4f}, 共{len(sig)}")
exact_r = np.array([reinforce_exact_var(2.0, s, 50) for s in sig])
exact_p = np.array([reparam_exact_var(2.0, s, 50) for s in sig])
code_th_r = (sig**2 + 4/sig**2)/50
code_th_p = 4*sig**2/50
for i in [0, 6, 12, 18, 24]:
    print(f"σ={sig[i]:.4f}: R_var={rv[i]:.4f} exact={exact_r[i]:.4f} code_th={code_th_r[i]:.4f}({rv[i]/code_th_r[i]:.2f}x) | "
          f"P_var={pv[i]:.6f} exact={exact_p[i]:.6f} | ratio={rv[i]/pv[i]:.1f}")
# log-log 相关：实测 vs 精确理论
lr = np.log(rv); lp = np.log(pv)
print(f"corr(log R_var, log exact_R)={np.corrcoef(lr, np.log(exact_r))[0,1]:.4f}")
print(f"corr(log P_var, log exact_P)={np.corrcoef(lp, np.log(exact_p))[0,1]:.4f}")
# 斜率检验：重参数化在logσ-logVar上的斜率应为2
A = np.polyfit(np.log(sig), lp, 1); print(f"重参数化 log-log 斜率={A[0]:.2f} (理论≈2 在小σ端)")
# REINFORCE 小σ端斜率应为 -2
small = sig < 0.3
A2 = np.polyfit(np.log(sig[small]), lr[small], 1); print(f"REINFORCE 小σ端 log-log 斜率={A2[0]:.2f} (理论≈-2)")
large = sig > 2
A3 = np.polyfit(np.log(sig[large]), lr[large], 1); print(f"REINFORCE 大σ端 log-log 斜率={A3[0]:.2f} (理论≈+2)")
print(f"方差比范围: min={ (rv/pv).min():.1f} @σ={sig[(rv/pv).argmin()]:.3f}, max={(rv/pv).max():.1f} @σ={sig[(rv/pv).argmax()]:.3f}")

Ls = d['L_list']; rmse = d['reinforce_mse']; pmse = d['reparam_mse']
print("\n== 步骤4 样本效率 ==")
print(f"L_list={Ls}")
for L_, r_, p_ in zip(Ls, rmse, pmse):
    # 理论 MSE = Var of single estimator scale: mse*L 应近似常数
    print(f"L={int(L_):5d}: R_MSE={r_:.4f} (MSE*L={r_*L_:8.2f}) P_MSE={p_:.4f} (MSE*L={p_*L_:7.2f}) ratio={r_/p_:.1f}x")
# 精确理论 MSE@L=50: R = var(single trial scaled) -> mse(L)= per_sample_var*? 直接由公式:
# REINFORCE MSE ≈ Var_per_trial = reinforce_exact_var(L=50)*(50/L)
pred_r50 = reinforce_exact_var(2.0, 0.5, 50); pred_p50 = reparam_exact_var(2.0, 0.5, 50)
print(f"理论@L=50: R_MSE≈{pred_r50:.4f} 实测{rmse[1]:.4f} ; P_MSE≈{pred_p50:.4f} 实测{pmse[1]:.4f}")
# 插值所需样本数（复刻图2b逻辑）
from scipy.interpolate import interp1d
fr = interp1d(np.log10(rmse), np.log10(Ls), fill_value='extrapolate')
fp = interp1d(np.log10(pmse), np.log10(Ls), fill_value='extrapolate')
Lr = 10**fr(np.log10(0.1)); Lp = 10**fp(np.log10(0.1))
print(f"达到MSE=0.1所需L: REINFORCE≈{Lr:.0f} 重参数化≈{Lp:.0f} 比值={Lr/Lp:.1f}x")
