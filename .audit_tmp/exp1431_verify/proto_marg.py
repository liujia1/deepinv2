# -*- coding: utf-8 -*-
"""原型: 边际误差判据的log域Sinkhorn, 校准迭代数/耗时, 为定稿脚本提供参数依据"""
import sys, io, time
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment

def sample_gm(n, std=0.2, seed=None):
    rng = np.random.RandomState(42) if seed is None else np.random.RandomState(seed)
    angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
    centers = np.stack([2.0*np.cos(angles), 2.0*np.sin(angles)], axis=1)
    return centers[rng.randint(0, 4, n)] + std*rng.randn(n, 2)

def sinkhorn_marg(C, reg, max_iter=20000, tol=1e-6):
    n, m = C.shape
    log_K = -C / reg
    la, lb = -np.log(n)*np.ones(n), -np.log(m)*np.ones(m)
    lu, lv = np.zeros(n), np.zeros(m)
    a = np.full(n, 1.0/n); b = np.full(m, 1.0/m)
    for it in range(1, max_iter+1):
        lu = la - logsumexp(log_K + lv[None, :], axis=1)
        lv = lb - logsumexp(log_K + lu[:, None], axis=0)
        lp = lu[:, None] + lv[None, :] + log_K
        P = np.exp(lp - lp.max()); P /= P.sum()
        err = max(np.max(np.abs(P.sum(axis=1)-a)), np.max(np.abs(P.sum(axis=0)-b)))
        if err < tol:
            return P, it, float(np.sum(C*P)), True
    return P, max_iter, float(np.sum(C*P)), False

print("n | eps | iters | conv | cost_err% | wall_s")
for n in [50, 100, 200, 300, 500]:
    src = sample_gm(n, seed=42); tgt = sample_gm(n, seed=123)
    C = np.sum((src[:,None,:]-tgt[None,:,:])**2, axis=-1)
    r, c = linear_sum_assignment(C); ch = C[r, c].mean()
    for reg in [0.01, 0.1, 1.0]:
        t0 = time.perf_counter()
        P, it, cst, conv = sinkhorn_marg(C, reg)
        dt = time.perf_counter()-t0
        print(f"{n:>4} | {reg:<5} | {it:>6} | {str(conv):>5} | {abs(cst-ch)/ch*100:6.2f} | {dt:7.2f}")

print()
print("敏感性极值 (n=100, 预算20000):")
src = sample_gm(100, seed=42); tgt = sample_gm(100, seed=123)
C = np.sum((src[:,None,:]-tgt[None,:,:])**2, axis=-1)
r, c = linear_sum_assignment(C); ch = C[r, c].mean()
for reg in [0.001, 0.0017, 0.0028, 0.0046, 0.0076, 0.0126]:
    t0 = time.perf_counter()
    P, it, cst, conv = sinkhorn_marg(C, reg, max_iter=20000)
    dt = time.perf_counter()-t0
    print(f"  eps={reg:<7}: it={it:>6} {'收敛' if conv else '封顶'} err={abs(cst-ch)/ch*100:6.2f}% t={dt:.1f}s")

print()
print("匈牙利大n标定 (scipy, 仅LSA时间):")
for n in [500, 1000, 2000]:
    rng = np.random.RandomState(0)
    src = rng.randn(n,2); tgt = rng.randn(n,2)
    C = np.sum((src[:,None,:]-tgt[None,:,:])**2, axis=-1)
    t0 = time.perf_counter(); linear_sum_assignment(C); print(f"  n={n}: {time.perf_counter()-t0:.3f}s")
