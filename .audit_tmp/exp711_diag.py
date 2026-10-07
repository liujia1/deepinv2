# -*- coding: utf-8 -*-
import numpy as np

def gm1d_pdf(x):
    return 0.3*np.exp(-0.5*(x+2)**2)/np.sqrt(2*np.pi) + \
           0.7*np.exp(-0.5*(x-1)**2)/np.sqrt(2*np.pi)

def gm1d_score_sigma(x, sigma):
    v = 1 + sigma**2
    p1 = np.exp(-0.5*(x+2)**2/v)/np.sqrt(2*np.pi*v)
    p2 = np.exp(-0.5*(x-1)**2/v)/np.sqrt(2*np.pi*v)
    p = 0.3*p1 + 0.7*p2
    return (0.3*-(x+2)/v*p1 + 0.7*-(x-1)/v*p2) / p

def langevin_1d(n, n_iter, delta, sigma_score, rng):
    x = rng.randn(n)*3
    for _ in range(n_iter):
        score = gm1d_score_sigma(x, sigma_score)
        x = x + delta*score + np.sqrt(2*delta)*rng.randn(n)
    return x

def annealed(n, sigmas, T, eps, rng):
    x = rng.randn(n)*sigmas[0]
    for sigma in sigmas:
        alpha = eps*(sigma/sigmas[-1])**2
        for _ in range(T):
            score = gm1d_score_sigma(x, sigma)
            x = x + alpha/2*score + np.sqrt(alpha)*rng.randn(n)
    return x

# theoretical proportions under p and p_sigma=1 for threshold -0.78
from scipy.stats import norm
def prop_below_p(mu_list, w_list, var, thr):
    tot=0
    for w,m in zip(w_list,mu_list):
        tot += w*norm.cdf((thr-m)/np.sqrt(var))
    return tot
print("理论 p (var=1) 左比例@-0.78:", round(prop_below_p([-2,1],[0.3,0.7],1,-0.78),4))
print("理论 p_sigma=1 (var=2) 左比例@-0.78:", round(prop_below_p([-2,1],[0.3,0.7],2,-0.78),4))
print("理论 p std:", round(np.sqrt(0.3*((-2)**2+1)+0.7*(1+1)-(0.1)**2),4))
print("理论 p_sigma=1 std:", round(np.sqrt(0.3*(4+2)+0.7*(1+2)-(0.1)**2),4))

N=20000
sigmas = np.exp(np.linspace(np.log(5.0), np.log(0.05), 10))
print("\nseed | 真实左 单L左 退火左 | 真实std 单L std 退火std | 单L峰值 退火峰值")
for seed in [42,0,1,7,2024]:
    np.random.seed(seed)
    rng=np.random
    xt=np.concatenate([np.random.randn(int(N*0.3))-2, np.random.randn(int(N*0.7))+1])
    xl=langevin_1d(N,2000,0.05,1.0,rng)
    xa=annealed(N,sigmas,200,2e-3,rng)
    thr=-0.78
    pl=lambda s: np.mean(s<thr)
    # peak density estimate via histogram max
    def peak(s):
        h,e=np.histogram(s,bins=60,density=True); return h.max()
    print(f"{seed:5d} | {pl(xt):.3f} {pl(xl):.3f} {pl(xa):.3f} | {xt.std():.3f} {xl.std():.3f} {xa.std():.3f} | {peak(xl):.3f} {peak(xa):.3f} (true peak {peak(xt):.3f})")
