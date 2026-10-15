# -*- coding: utf-8 -*-
import numpy as np

def make_cs_problem(n, k, m, snr, seed=0):
    rng = np.random.default_rng(seed)
    alpha = np.zeros(n)
    idx = rng.choice(n, size=k, replace=False)
    alpha[idx] = rng.uniform(-3, 3, size=k)
    A = rng.standard_normal((m, n))
    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    sigma = np.linalg.norm(A @ alpha) / (10 ** (snr / 20))
    noise = rng.standard_normal(m) * sigma
    return A, A @ alpha + noise, alpha, idx

def IHT(A, y, K, max_iter=200):
    n = A.shape[1]; step = 1.0/np.linalg.norm(A,2)**2; a = np.zeros(n)
    for _ in range(max_iter):
        g = A.T@(y-A@a); an = a+step*g
        th = np.sort(np.abs(an))[-K]; an[np.abs(an)<th]=0.0
        if np.linalg.norm(an-a)<1e-8: a=an;break
        a=an
    return a

def RW(A, y, n_outer=12, eps=1e-2, lam=0.1, max_inner=300):
    n=A.shape[1]; step=1.0/np.linalg.norm(A,2)**2; a=np.zeros(n); w=np.ones(n)
    for _ in range(n_outer):
        for _ in range(max_inner):
            g=A.T@(y-A@a); s=np.sign(a+step*g)*np.maximum(np.abs(a+step*g)-step*lam*w,0)
            if np.linalg.norm(s-a)<1e-8: a=s;break
            a=s
        w=1/(np.abs(a)+eps)
    return a

def OMP(A,y,K):
    n=A.shape[1]; r=y.copy(); idx=[]; a=np.zeros(n)
    for _ in range(K):
        c=A.T@r; ni=int(np.argmax(np.abs(c)))
        if ni in idx:break
        idx.append(ni); As=A[:,idx]; coef,*_=np.linalg.lstsq(As,y,rcond=None); r=y-As@coef
    a[idx]=coef; return a

def rel_err(a,b): return np.linalg.norm(a-b)/np.linalg.norm(b)
def nnz(a): return int(np.sum(np.abs(a)>1e-6))
def hit(a,sup): return len(set(np.where(np.abs(a)>1e-6)[0])&set(sup.tolist()))

print("=== 多seed鲁棒性: IHT / Reweighted / OMP (rel_err | nnz | 命中真支撑/10) ===")
for seed in [7,1,2,3,11,42]:
    A,y,at,sup = make_cs_problem(256,10,80,30,seed=seed)
    ri,rw,ro = IHT(A,y,10),RW(A,y),OMP(A,y,10)
    print(f"seed{seed:3d}: IHT {rel_err(ri,at):.3f}|{nnz(ri)}|{hit(ri,sup)}   "
          f"RW {rel_err(rw,at):.3f}|{nnz(rw)}|{hit(rw,sup)}   OMP {rel_err(ro,at):.3f}|{nnz(ro)}|{hit(ro,sup)}")

print("\n=== IHT/OMP nnz=K 是否为算法构造保证(重言式)? 观察IHT每步都强制保留top-K, OMP恰好选K个原子 -> 必然nnz=K ===")

print("\n=== 固定lam下 Reweighted对eps/lam敏感性(seed=7) ===")
A,y,at,sup = make_cs_problem(256,10,80,30,seed=7)
for lam in [0.05,0.1,0.2]:
    for eps in [1e-3,1e-2,1e-1]:
        rw=RW(A,y,lam=lam,eps=eps)
        print(f"  lam={lam} eps={eps:g}: rel_err={rel_err(rw,at):.4f} nnz={nnz(rw)} hit={hit(rw,sup)}")
