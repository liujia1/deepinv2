# -*- coding: utf-8 -*-
"""独立核验：用高精度数值积分(非MC)计算单高斯q=N(mu,sig^2)的精确ELBO，
确认对称双峰后验下的最优 sigma* 与最优 ELBO。"""
import numpy as np
from scipy.integrate import quad

prior_w = [0.5, 0.5]
prior_mu = [-2.0, 2.0]
prior_tau = [1.0, 1.0]
sigma_obs = 1.0
x_obs = 0.0

def log_pz(z):
    t = 0.0
    for w, mu, tau in zip(prior_w, prior_mu, prior_tau):
        t += w * np.exp(-0.5*((z-mu)/tau)**2)/(tau*np.sqrt(2*np.pi))
    return np.log(t)

def exact_elbo(mu_q, sigma_q):
    # E_q[log p(x|z)]
    def integrand_logpxz(z):
        q = np.exp(-0.5*((z-mu_q)/sigma_q)**2)/(sigma_q*np.sqrt(2*np.pi))
        lp = -0.5*np.log(2*np.pi) - np.log(sigma_obs) - 0.5*((x_obs-z)/sigma_obs)**2
        return q*lp
    # E_q[log p(z)]
    def integrand_logpz(z):
        q = np.exp(-0.5*((z-mu_q)/sigma_q)**2)/(sigma_q*np.sqrt(2*np.pi))
        return q*log_pz(z)
    # E_q[log q(z)]
    def integrand_logq(z):
        q = np.exp(-0.5*((z-mu_q)/sigma_q)**2)/(sigma_q*np.sqrt(2*np.pi))
        return q*np.log(q+1e-300)
    e_pxz,_ = quad(integrand_logpxz, -30, 30, limit=400)
    e_pz,_   = quad(integrand_logpz, -30, 30, limit=400)
    e_q,_    = quad(integrand_logq, -30, 30, limit=400)
    return e_pxz + e_pz - e_q

# 扫描 sigma（mu=0 由对称性是最优）
print("mu=0 时扫描 sigma:")
best=(None,-1e9)
for s in np.arange(0.8, 1.8, 0.02):
    e = exact_elbo(0.0, s)
    if e>best[1]: best=(s,e)
    if abs(s-round(s,2))<1e-9 and (abs(s-1.0)<1e-9 or abs(s-1.1)<1e-9 or abs(s-1.16)<1e-9 or abs(s-1.18)<1e-9 or abs(s-1.2)<1e-9 or abs(s-1.4)<1e-9):
        print(f"  sigma={s:.2f} ELBO={e:.5f}")
print(f"  最优 sigma*={best[0]:.4f}, ELBO*={best[1]:.5f}")

# 也扫描 mu 确认 0 是最优
print("sigma=1.17 时扫描 mu:")
for m in [0.0,0.3,0.6,1.0]:
    print(f"  mu={m:.2f} ELBO={exact_elbo(m,1.17):.5f}")

# log p(x)
def log_marginal(x):
    terms=[]
    for w,mu,tau in zip(prior_w,prior_mu,prior_tau):
        ms=np.sqrt(sigma_obs**2+tau**2)
        terms.append(np.log(w)-0.5*np.log(2*np.pi)-np.log(ms)-0.5*((x-mu)/ms)**2)
    mx=max(terms); return mx+np.log(sum(np.exp(t-mx) for t in terms))
lpx=log_marginal(0.0)
print(f"\nlog p(x)={lpx:.5f}")
print(f"精确最优 ELBO={best[1]:.5f}, 变分间隙={lpx-best[1]:.5f}")
print(f"讲义声称: sigma*=1.41, ELBO=-2.35, log p(x)=-2.19, gap=0.16")

# 关键点密度
def post_pdf(z):
    p=0.0
    # 真实后验 0.5 N(-1,0.5)+0.5 N(1,0.5)
    for w,m,s2 in [(0.5,-1.0,0.5),(0.5,1.0,0.5)]:
        p+=w*np.exp(-0.5*((z-m)**2)/s2)/(np.sqrt(s2)*np.sqrt(2*np.pi))
    return p
for z in [0.0,-1.0,1.0]:
    q=np.exp(-0.5*((z-0.0)/best[0])**2)/(best[0]*np.sqrt(2*np.pi))
    print(f"  z={z:+.1f}: p={post_pdf(z):.4f}, q*={q:.4f}, diff={q-post_pdf(z):+.4f}")
