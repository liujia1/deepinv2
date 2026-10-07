# -*- coding: utf-8 -*-
"""8A-1 步骤2改造探针：寻找一个"后验真正双峰"的配置，
量化受限单高斯族的变分间隙（解析 + MC-CRN 两条路径）。"""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from numpy.polynomial.hermite_e import hermegauss

nodes, weights = hermegauss(160)

def make_model(weights_, means, stds, sigma_obs, x_obs):
    def log_prior(z):
        lp = np.full_like(z, -1e30)
        for w, mu, tau in zip(weights_, means, stds):
            lp = np.logaddexp(lp, np.log(w) - 0.5*np.log(2*np.pi) - np.log(tau) - 0.5*((z-mu)/tau)**2)
        return lp
    def log_joint(z):
        return (-0.5*np.log(2*np.pi) - np.log(sigma_obs) - 0.5*((x_obs-z)/sigma_obs)**2) + log_prior(z)
    def log_px():
        v, e = quad(lambda z: np.exp(log_joint(np.array([z]))[0]), min(means)-15, max(means)+15, limit=800)
        return np.log(v)
    return log_joint, log_px

def elbo_exact(log_joint, mu, sigma):
    zs = mu + sigma*nodes
    fj = np.array([log_joint(np.array([z]))[0] for z in zs])
    return np.sum(weights*fj)/np.sum(weights) + 0.5*np.log(2*np.pi*np.e*sigma**2)

def sup_single_gauss(log_joint, log_px, mu_lo, mu_hi):
    best = (-1e30, None)
    for mu in np.linspace(mu_lo, mu_hi, 81):
        for s in np.linspace(0.1, 3.0, 59):
            v = elbo_exact(log_joint, mu, s)
            if v > best[0]:
                best = (v, (mu, s))
    r = minimize(lambda p: -elbo_exact(log_joint, p[0], np.exp(p[1])), [best[1][0], np.log(best[1][1])],
                 method='Nelder-Mead', options={'xatol':1e-10,'fatol':1e-10,'maxiter':8000})
    return -r.fun, r.x[0], np.exp(r.x[1]), log_px - (-r.fun)

def post_shape(log_joint, lo, hi, n=6001):
    zs = np.linspace(lo, hi, n)
    p = np.exp(np.array([log_joint(np.array([z]))[0] for z in zs]))
    p = p/p.sum()
    loc = [zs[i] for i in range(1, n-1) if p[i] > p[i-1] and p[i] >= p[i+1] and p[i] > 0.01*p.max()]
    return loc

configs = {
    'A 原设定': ([0.3,0.7], [-2.0,1.0], [1.0,1.0], 0.5, 0.5, -8, 6),
    'B1 w=.5/.5 mean±3 tau=.6 s=.5': ([0.5,0.5], [-3.0,3.0], [0.6,0.6], 0.5, 0.0, -9, 9),
    'B2 w=.5/.5 mean±2 tau=.5 s=.5': ([0.5,0.5], [-2.0,2.0], [0.5,0.5], 0.5, 0.0, -8, 8),
    'B3 w=.3/.7 mean±2.5 tau=.5 s=.4': ([0.3,0.7], [-2.5,2.5], [0.5,0.5], 0.4, 0.0, -8, 8),
    'B4 w=.5/.5 mean±3 tau=.7 s=.6': ([0.5,0.5], [-3.0,3.0], [0.7,0.7], 0.6, 0.0, -9, 9),
}
for name, (w, means, stds, so, xo, lo, hi) in configs.items():
    lj, lpxf = make_model(w, means, stds, so, xo)
    log_px = lpxf()
    sup, mu_s, sig_s, gap = sup_single_gauss(lj, log_px, lo, hi)
    modes = post_shape(lj, lo, hi)
    print(f"{name}\n  log p(x)={log_px:.6f}  sup ELBO={sup:.6f}  q*=N({mu_s:.4f},{sig_s:.4f}^2)  gap={gap:.6f}  后验峰位={['%.2f'%m for m in modes]}")

    # MC-CRN 路径（与实验代码同法）
    np.random.seed(42)
    eps = np.random.randn(200000)
    def neg_elbo(p):
        mu_q, ls = p; s = np.exp(ls)
        z = mu_q + s*eps
        lpxz = -0.5*np.log(2*np.pi) - np.log(so) - 0.5*((xo-z)/so)**2
        lpz = np.full_like(z, -1e30)
        for ww, mu, tau in zip(w, means, stds):
            lpz = np.logaddexp(lpz, np.log(ww) - 0.5*np.log(2*np.pi) - np.log(tau) - 0.5*((z-mu)/tau)**2)
        lqz = -0.5*np.log(2*np.pi) - ls - 0.5*((z-mu_q)/s)**2
        return -np.mean(lpxz + lpz - lqz)
    bg, bp = np.inf, [0.0, 0.0]
    for m in np.linspace(lo/2, hi/2, 25):
        for s in np.linspace(0.2, 3.0, 15):
            v = neg_elbo([m, np.log(s)])
            if v < bg: bg, bp = v, [m, np.log(s)]
    r = minimize(neg_elbo, bp, method='Nelder-Mead', options={'maxiter':8000,'xatol':1e-9,'fatol':1e-9})
    mu_mc, sig_mc = r.x[0], np.exp(r.x[1]); elbo_mc = -r.fun
    print(f"  MC-CRN: q*=N({mu_mc:.4f},{sig_mc:.4f}^2) ELBO={elbo_mc:.6f} gap={log_px-elbo_mc:.6f} | 该q*解析ELBO={elbo_exact(lj,mu_mc,sig_mc):.6f}")
