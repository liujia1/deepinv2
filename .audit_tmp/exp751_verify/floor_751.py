# -*- coding: utf-8 -*-
"""7.5-1 步骤2/4 指标噪声地板量化：
1) 两样本KS的噪声地板（同一分布的两个有限样本）
2) var_diff 的采样噪声（对比 x0 样本方差 vs 总体方差 2.89）
3) 改用单样本KS(对解析CDF) + 大N 后是否单调
"""
import numpy as np
from scipy.special import erf

beta_min, beta_max = 0.1, 20.0
T = 1.0
W, M, S = [0.3, 0.7], [-2, 1], [1, 1]

def vp_marginal(t):
    lm = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    return np.exp(lm), np.sqrt(1 - np.exp(2 * lm))

def vp_score(x, t):
    mt, st = vp_marginal(t)
    pdf = np.zeros_like(x); dpdf = np.zeros_like(x)
    for w, m, s in zip(W, M, S):
        nm, ns = mt * m, np.sqrt(mt**2 * s**2 + st**2)
        pdf += w * np.exp(-0.5*((x-nm)/ns)**2)/(ns*np.sqrt(2*np.pi))
        dpdf += w * (-(x-nm)/ns**2)*np.exp(-0.5*((x-nm)/ns)**2)/(ns*np.sqrt(2*np.pi))
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

# 解析目标 CDF
xg = np.linspace(-12, 12, 40001)
Fg = np.zeros_like(xg)
for w, m, s in zip(W, M, S):
    Fg += w*0.5*(1+erf((xg-m)/(s*np.sqrt(2))))

def ks_ana(f):
    xs = np.sort(f)
    Fe = np.arange(1, len(xs)+1)/len(xs)
    return np.max(np.abs(Fe - np.interp(xs, xg, Fg)))

# 总体矩
mu_pop = sum(w*m for w, m in zip(W, M))
e2 = sum(w*(m**2+s**2) for w, m, s in zip(W, M, S))
var_pop = e2 - mu_pop**2
mu4 = sum(w*(m**4+6*m**2*s**2+3*s**4) for w, m, s in zip(W, M, S))
print(f"总体: mu={mu_pop:.4f} var={var_pop:.4f}  E[x^4]={mu4:.4f}")

# 1) 两样本KS地板：从解析p_0抽两组样本
np.random.seed(7)
floor = []
for _ in range(20):
    a = np.where(np.random.rand(20000) < 0.3, np.random.randn(20000)-2, np.random.randn(20000)+1)
    b = np.where(np.random.rand(10000) < 0.3, np.random.randn(10000)-2, np.random.randn(10000)+1)
    from scipy import stats as sp
    floor.append(sp.ks_2samp(a, b).statistic)
print(f"[1] 两样本KS噪声地板 (20000 vs 10000, 同分布): mean={np.mean(floor):.4f} std={np.std(floor):.4f}")

vd = []
for _ in range(20):
    a = np.where(np.random.rand(5000) < 0.3, np.random.randn(5000)-2, np.random.randn(5000)+1)
    b = np.where(np.random.rand(10000) < 0.3, np.random.randn(10000)-2, np.random.randn(10000)+1)
    vd.append(abs(np.var(a)-np.var(b)))
print(f"[2] var_diff噪声地板 (5000 vs 10000, 同分布): mean={np.mean(vd):.4f} std={np.std(vd):.4f}")
print(f"    理论 std(sample var, n=5000) = sqrt((mu4-var^2)/n) = {np.sqrt((mu4-var_pop**2)/5000):.4f}")

print("\n[3] 单样本KS(对解析CDF), N=200000, 5 seeds:")
for n_steps in [10, 25, 50, 100, 200, 500, 1000]:
    ks = [ks_ana(ddpm(200000, n_steps, 42+7*k)) for k in range(5)]
    print(f"  steps={n_steps:>5}: KS_ana={np.mean(ks):.5f}±{np.std(ks):.5f}")

print("\n[4] 单样本KS, N=20000（原粒子数）, 5 seeds:")
for n_steps in [10, 25, 50, 100, 200, 500, 1000]:
    ks = [ks_ana(ddpm(20000, n_steps, 42+7*k)) for k in range(5)]
    print(f"  steps={n_steps:>5}: KS_ana={np.mean(ks):.5f}±{np.std(ks):.5f}")

print("\n[5] var_diff 对总体方差(2.890), N=200000:")
for n_steps in [50, 200, 500, 1000]:
    f = ddpm(200000, n_steps, 42)
    print(f"  steps={n_steps:>5}: mu_diff={abs(np.mean(f)-mu_pop):.5f} var_diff_vs_pop={abs(np.var(f)-var_pop):.5f}")
