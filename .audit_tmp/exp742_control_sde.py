# -*- coding: utf-8 -*-
"""对照实验：验证步骤3"逆向SDE插值不平滑"是固有属性还是人为放大。
- 情形A(实验所用): 一条RNG流贯穿全部α(每个α独立噪声) -> 预期不平滑
- 情形B(共随机数): 每个α共享同一套噪声(每次重置同一seed) -> 若变平滑,
  说明"不平滑"来自独立噪声(即SDE默认采样), 属固有属性; 实验对照公平。
"""
import numpy as np

beta_min, beta_max = 0.1, 20.0
def vp_beta(t): return beta_min + t * (beta_max - beta_min)
def vp_marginal_params(t):
    lmc = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    mc = np.exp(lmc); return mc, 1 - mc**2
def vp_gmm_1d_score(x, t, weights=[0.3,0.7], means=[-2,2], stds=[1,1]):
    mc, vc = vp_marginal_params(t); st = np.sqrt(vc)
    pdf = np.zeros_like(x, float); dpdf = np.zeros_like(x, float)
    for w, m, s in zip(weights, means, stds):
        nm = mc*m; ns = np.sqrt(mc**2*s**2 + st**2)
        g = np.exp(-0.5*((x-nm)/ns)**2)/(ns*np.sqrt(2*np.pi))
        pdf += w*g; dpdf += w*(-(x-nm)/ns**2)*g
    return dpdf/(pdf+1e-30)

def reverse_sde(x_init, shared_noise, N_steps=500, T=1.0):
    dt = T/N_steps; x = x_init.copy(); traj=[x.copy()]
    for i in range(N_steps):
        t = T - i*dt; bt = vp_beta(t); sc = vp_gmm_1d_score(x, t)
        x = x + bt*dt*(0.5*x + sc) + np.sqrt(bt*dt)*shared_noise[i]
        traj.append(x.copy())
    return np.array(traj)

z_A, z_B = -3.0, 3.0
alphas = np.linspace(0,1,11)

# 情形A: 一条流贯穿(独立噪声)
rngA = np.random.RandomState(42)
resA = []
for a in alphas:
    z = a*z_A + (1-a)*z_B
    noise = rngA.randn(500)  # 每个α消耗新的独立噪声
    resA.append(reverse_sde(np.array([z]), noise)[-1][0])

# 情形B: 共随机数(所有α共享同一套噪声)
rngB = np.random.RandomState(42)
shared = rngB.randn(500)  # 固定一套噪声
resB = []
for a in alphas:
    z = a*z_A + (1-a)*z_B
    resB.append(reverse_sde(np.array([z]), shared)[-1][0])

def smoothness(v):
    v = np.asarray(v); return np.mean(np.abs(np.diff(v)))  # 相邻点平均跳幅

print("情形A(独立噪声, 实验所用):")
print("  ", np.round(resA,3))
print("  平均相邻跳幅=%.3f, 最大=%.3f" % (smoothness(resA), np.max(np.abs(np.diff(resA)))))
print("情形B(共随机数, 人为耦合):")
print("  ", np.round(resB,3))
print("  平均相邻跳幅=%.3f, 最大=%.3f" % (smoothness(resB), np.max(np.abs(np.diff(resB)))))
print("单调性检查: A单调=%s, B单调=%s" % (
    bool(np.all(np.diff(resA)<0) or np.all(np.diff(resA)>0)),
    bool(np.all(np.diff(resB)<0) or np.all(np.diff(resB)>0))))
