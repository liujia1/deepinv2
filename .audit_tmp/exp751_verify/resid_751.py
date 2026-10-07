# -*- coding: utf-8 -*-
"""步骤4 残差定位：KS 的非单调来自哪里？
- 全区间 sup vs 限制在分位数窗口 [q_lo,q_hi] 内的 sup
- Wasserstein-2 (对解析分位数) 作为平滑度量
"""
import numpy as np
from scipy.special import erf
from scipy.stats import norm

beta_min, beta_max = 0.1, 20.0
T = 1.0
W, M, S = [0.3, 0.7], [-2, 1], [1, 1]
mu_pop = sum(w*m for w, m in zip(W, M))
var_pop = sum(w*(m**2+s**2) for w, m, s in zip(W, M, S)) - mu_pop**2

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

# 解析 p_0 分位数函数（混合分布，用密集网格插值）
xg = np.linspace(-15, 15, 200001)
Fg = np.zeros_like(xg)
for w, m, s in zip(W, M, S):
    Fg += w*0.5*(1+erf((xg-m)/(s*np.sqrt(2))))

def F_of(x):
    return np.interp(x, xg, Fg)

def Q_of(u):
    return np.interp(u, Fg, xg)

def ks_window(f, qlo=0.0, qhi=1.0):
    xs = np.sort(f)
    n = len(xs)
    Fe = np.arange(1, n+1)/n
    Fth = F_of(xs)
    d = np.abs(Fe-Fth)
    lo = int(qlo*n); hi = min(n, int(np.ceil(qhi*n)))
    k_all = d.max()
    k_win = d[lo:hi].max() if hi > lo else 0.0
    argx = xs[np.argmax(d)]
    return k_all, k_win, argx

def w2(f, nq=2000):
    xs = np.sort(f)
    n = len(xs)
    # 限制在 [0.002,0.998] 分位数窗口，避免解析CDF饱和区导致插值失真
    u = np.linspace(0.002, 0.998, nq)
    idx = np.clip((u*n).astype(int), 0, n-1)
    return np.sqrt(np.mean((xs[idx]-Q_of(u))**2))

print(f"{'steps':>6} {'KS_all':>9} {'KS[1%,99%]':>11} {'argmax_x':>9} {'W2':>8} {'var':>8} {'max|x|':>8}")
for n_steps in [10, 25, 50, 100, 200, 500, 1000]:
    rows = []
    for k in range(3):
        f = ddpm(100000, n_steps, 42+7*k)
        ka, kw, ax = ks_window(f, 0.01, 0.99)
        rows.append((ka, kw, ax, w2(f), np.var(f), np.abs(f).max()))
    r = np.mean(rows, axis=0)
    s = np.std(rows, axis=0)
    print(f"{n_steps:>6} {r[0]:>9.5f} {r[1]:>11.5f} {r[2]:>9.3f} {r[3]:>8.5f} {r[4]:>8.4f} {r[5]:>8.2f}"
          f"   (std: KS={s[0]:.5f} W2={s[3]:.5f})")
print(f"\n解析目标: var={var_pop:.4f}")

# 纯 MC 地板：直接从解析 p_0 抽样
np.random.seed(5)
comp = np.random.choice(2, 100000, p=[0.3, 0.7])
samp = np.where(comp == 0, np.random.randn(100000)-2, np.random.randn(100000)+1)
ka, kw, ax = ks_window(samp, 0.01, 0.99)
print(f"MC地板(N=100000, 精确p_0样本): KS_all={ka:.5f} KS_win={kw:.5f} W2={w2(samp):.5f} var={np.var(samp):.4f}")
