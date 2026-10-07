# -*- coding: utf-8 -*-
"""计时：确定步骤4可用的 N 与 seeds 配置；并验证修正后的 VE/SMLD 与解析边际的一致性"""
import time
import numpy as np
from scipy.special import erf

beta_min, beta_max = 0.1, 20.0
T = 1.0
W, M, S = [0.3, 0.7], [-2, 1], [1, 1]

def vp_marginal(t):
    lm = -0.25*t**2*(beta_max-beta_min) - 0.5*t*beta_min
    return np.exp(lm), np.sqrt(1-np.exp(2*lm))

def vp_score(x, t):
    mt, st = vp_marginal(t)
    pdf = np.zeros_like(x); dpdf = np.zeros_like(x)
    for w, m, s in zip(W, M, S):
        nm, ns = mt*m, np.sqrt(mt**2*s**2+st**2)
        pdf += w*np.exp(-0.5*((x-nm)/ns)**2)/(ns*np.sqrt(2*np.pi))
        dpdf += w*(-(x-nm)/ns**2)*np.exp(-0.5*((x-nm)/ns)**2)/(ns*np.sqrt(2*np.pi))
    return dpdf/(pdf+1e-30)

def ddpm(N, n_steps, seed):
    np.random.seed(seed)
    h = T/n_steps
    x = np.random.randn(N)
    for i in range(n_steps):
        t = T - i*h
        b = beta_min + t*(beta_max-beta_min)
        x = x + b*h*(0.5*x + vp_score(x, t)) + np.sqrt(b*h)*np.random.randn(N)
    return x

xg = np.linspace(-12, 12, 40001)
Fg = np.zeros_like(xg)
for w, m, s in zip(W, M, S):
    Fg += w*0.5*(1+erf((xg-m)/(s*np.sqrt(2))))

def ks_ana(f):
    xs = np.sort(f)
    Fe = np.arange(1, len(xs)+1)/len(xs)
    return np.max(np.abs(Fe - np.interp(xs, xg, Fg)))

t0 = time.time()
for N, seeds in [(100000, 3)]:
    vals = []
    for n_steps in [10, 25, 50, 100, 200, 500, 1000]:
        ks = [ks_ana(ddpm(N, n_steps, 42+7*k)) for k in range(seeds)]
        vals.append((n_steps, np.mean(ks), np.std(ks)))
        print(f"  N={N} steps={n_steps:>5}: KS_ana={np.mean(ks):.5f}±{np.std(ks):.5f}")
print(f"elapsed(N=100000,3seeds) = {time.time()-t0:.1f}s")
