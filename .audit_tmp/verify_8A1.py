# -*- coding: utf-8 -*-
"""实验8A-1复核脚本：
1) 解析（数值积分）计算单高斯族的精确 sup ELBO，与代码MC+Nelder-Mead结果对比
2) 检验代码打印的间隙 0.0005 是否被CRN有限样本过拟合压低
3) 检验"单高斯无法拟合双峰后验"的说法：后验在 x=0.5 处是否真的双峰
4) 验证 |x|, y=1.5 时 x_range 扩大数值解单调增长（免责声称）
5) 复现性：重跑 neg_elbo 优化看结果是否稳定
"""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

np.random.seed(42)

prior_weights = [0.3, 0.7]
prior_means = [-2.0, 1.0]
prior_stds = [1.0, 1.0]
sigma_obs = 0.5
x_obs = 0.5

def log_prior_z(z):
    lp = np.full_like(np.atleast_1d(z), -1e30)
    for w, mu, tau in zip(prior_weights, prior_means, prior_stds):
        lp = np.logaddexp(lp, np.log(w) - 0.5*np.log(2*np.pi) - np.log(tau) - 0.5*((z-mu)/tau)**2)
    return lp

def log_joint_scalar(z):
    lp = log_prior_z(np.array([z]))[0]
    return -0.5*np.log(2*np.pi) - np.log(sigma_obs) - 0.5*((x_obs - z)/sigma_obs)**2 + lp

def log_marginal_exact():
    val, err = quad(lambda z: np.exp(log_joint_scalar(z)), -20, 20, limit=400)
    return np.log(val), err

log_px, qerr = log_marginal_exact()
print(f"[精确 log p(x)] = {log_px:.8f}  (quad误差~{qerr:.1e})")

# ---- 解析 ELBO(q=N(mu,s2))：用数值积分 E_q[log p(x,z)] + H(q) ----
def elbo_exact(mu, sigma):
    # E_q[f] via quad with change of variable z = mu + sigma*t
    from numpy.polynomial.hermite_e import hermegauss
    # Gauss-Hermite (probabilists') for N(0,1): sum w_i/sqrt(2pi) * f(mu+sigma*x_i)
    nodes, weights = hermegauss(120)
    zs = mu + sigma * nodes
    fj = np.array([log_joint_scalar(z) for z in zs])
    # weights sum to sqrt(2pi); normalize
    ej = np.sum(weights * fj) / np.sum(weights)
    H = 0.5*np.log(2*np.pi*np.e*sigma**2)
    return ej + H

# 网格+精修求单高斯族 sup ELBO
best = (-1e30, None)
for mu in np.linspace(-3, 3, 61):
    for s in np.linspace(0.15, 2.5, 49):
        v = elbo_exact(mu, s)
        if v > best[0]:
            best = (v, (mu, s))
res = minimize(lambda p: -elbo_exact(p[0], np.exp(p[1])), [best[1][0], np.log(best[1][1])],
               method='Nelder-Mead', options={'xatol':1e-10,'fatol':1e-10,'maxiter':5000})
mu_ex, sig_ex = res.x[0], np.exp(res.x[1])
elbo_sup = -res.fun
print(f"[解析 sup ELBO(单高斯)] = {elbo_sup:.8f}, q* = N({mu_ex:.4f}, {sig_ex:.4f}^2)")
print(f"[解析变分间隙] = {log_px - elbo_sup:.6f}")

# ---- 代码同配置的MC路径复现（与8A-1.py相同seed消耗顺序） ----
n_opt_samples = 200000
opt_eps = np.random.randn(n_opt_samples)
def neg_elbo_gaussian(params):
    mu_q, log_sigma_q = params
    sigma_q = np.exp(log_sigma_q)
    z = mu_q + sigma_q * opt_eps
    log_pxz = -0.5*np.log(2*np.pi) - np.log(sigma_obs) - 0.5*((x_obs - z)/sigma_obs)**2
    log_pz = np.full_like(z, -1e30)
    for w, mu, tau in zip(prior_weights, prior_means, prior_stds):
        log_comp = np.log(w) - 0.5*np.log(2*np.pi) - np.log(tau) - 0.5*((z-mu)/tau)**2
        log_pz = np.logaddexp(log_pz, log_comp)
    log_joint = log_pxz + log_pz
    log_qz = -0.5*np.log(2*np.pi) - np.log(sigma_q) - 0.5*((z-mu_q)/sigma_q)**2
    return -np.mean(log_joint - log_qz)

mu_c = np.linspace(-3, 3, 15); s_c = np.linspace(0.2, 2.0, 10)
bg, bp = np.inf, [0.0, 0.0]
for m in mu_c:
    for s in s_c:
        v = neg_elbo_gaussian([m, np.log(s)])
        if v < bg: bg, bp = v, [m, np.log(s)]
r2 = minimize(neg_elbo_gaussian, bp, method='Nelder-Mead', options={'maxiter':5000,'xatol':1e-8,'fatol':1e-8})
mu_mc, sig_mc = r2.x[0], np.exp(r2.x[1]); elbo_mc = -r2.fun
print(f"[MC-CRN 路径] q* = N({mu_mc:.4f}, {sig_mc:.4f}^2), MC ELBO = {elbo_mc:.6f}")
print(f"   该 q* 的解析 ELBO = {elbo_exact(mu_mc, sig_mc):.6f}, 解析间隙 = {log_px - elbo_exact(mu_mc, sig_mc):.6f}")
print(f"   MC打印间隙 = {log_px - elbo_mc:.6f}  → 有限样本过拟合使MC间隙偏小: {abs(log_px-elbo_mc) < abs(log_px - elbo_exact(mu_mc, sig_mc))}")

# ---- 后验形态：是否真的双峰？ ----
zs = np.linspace(-6, 5, 4001)
post = np.exp(np.array([log_joint_scalar(z) for z in zs]))
post /= post.sum()
# 找局部极大
loc = [(zs[i]) for i in range(1, len(zs)-1) if post[i] > post[i-1] and post[i] >= post[i+1]]
print(f"[后验局部极大点] {['%.3f' % m for m in loc]}")
for m in loc:
    i = np.argmin(np.abs(zs-m)); print(f"   z={m:.3f}: 密度={post[i]:.5f}")
# 左峰占比（z<-0.5区域质量）
mass_left = post[zs < -0.5].sum()
print(f"[后验 z<-0.5 质量] = {mass_left:.5f}")

# ---- |x|, y=1.5 扩大搜索区间：数值解应单调增长 ----
def fenchel_abs(y, hi, n=10000):
    x = np.linspace(-hi, hi, n)
    return np.max(x*y - np.abs(x))
print("[免责验证] f=|x|, y=1.5, x_range上界扩大:")
for hi in [10, 100, 1000, 10000]:
    print(f"   hi={hi:6d}: f*_num = {fenchel_abs(1.5, hi):.4f}")

# ---- x^2/2 中 |x| y=0.5 的 -0.0005 是否格点问题 ----
x = np.linspace(-10, 10, 10000)
print(f"[格点诊断] linspace(-10,10,10000) 不包含 0.0，离0最近点: {x[np.argmin(np.abs(x))]:.6f}")
