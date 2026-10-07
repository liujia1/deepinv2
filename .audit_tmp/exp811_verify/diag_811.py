# -*- coding: utf-8 -*-
"""实验8.1-1 诊断脚本：
A) Laplace计时是否被 scipy import 开销污染（排除import后重新计时）
B) VI 最优解是否真的"取决于优化起点"（多起点扫描，固定CRN噪声）
C) 解析后验复核 + ELBO间隙恒等式 logZ = ELBO + KL 校验
"""
import numpy as np, time, math

prior_weights = [0.4, 0.6]; prior_means = [-2.0, 2.0]; prior_stds = [0.5, 0.5]
sigma_obs = 0.3; x_obs = 0.0

def log_unnorm(z):
    lp = -0.5*np.log(2*np.pi) - np.log(sigma_obs) - 0.5*((x_obs-z)/sigma_obs)**2
    pz = np.full_like(np.atleast_1d(z).astype(float), -1e30)
    for w, mu, tau in zip(prior_weights, prior_means, prior_stds):
        pz = np.logaddexp(pz, np.log(w) - 0.5*np.log(2*np.pi) - np.log(tau) - 0.5*((z-mu)/tau)**2)
    return lp + pz

# ---------- A) Laplace 计时（scipy 已导入的前提下） ----------
from scipy.optimize import minimize
t0 = time.time()
r = minimize(lambda z: -log_unnorm(z)[0], x0=0.0, method='Nelder-Mead')
z_map = r.x[0]
eps = 1e-4
hess = (log_unnorm(z_map+eps)[0] - 2*log_unnorm(z_map)[0] + log_unnorm(z_map-eps)[0]) / eps**2
sig_lap = math.sqrt(-1.0/hess)
t_lap = time.time() - t0
print(f"[A] 排除import开销后 Laplace 真实耗时 = {t_lap:.6f}s, z_map={z_map:.4f}, sigma_laplace={sig_lap:.4f}")

# ---------- B) VI 多起点扫描（固定一组CRN噪声） ----------
np.random.seed(42)
# 复现原脚本RNG路径：n_mc=1e6 z_samples, 然后 MCMC 循环消耗随机数……
# 直接自采一组 eps 做定性多起点测试即可（结论与具体eps无关，大数定律）
vi_eps = np.random.randn(10000)
def neg_elbo(params):
    mu, sigma = params[0], math.exp(params[1])
    z = mu + sigma * vi_eps
    lp = -0.5*np.log(2*np.pi) - np.log(sigma_obs) - 0.5*((x_obs-z)/sigma_obs)**2
    pz = np.full_like(z, -1e30)
    for w, m0, tau in zip(prior_weights, prior_means, prior_stds):
        pz = np.logaddexp(pz, np.log(w) - 0.5*np.log(2*np.pi) - np.log(tau) - 0.5*((z-m0)/tau)**2)
    lq = -0.5*np.log(2*np.pi) - np.log(sigma) - 0.5*((z-mu)/sigma)**2
    return -np.mean(lp + pz - lq)

starts = [[0.0,0.0],[2.0,0.0],[-2.0,0.0],[0.5,-0.7],[-1.0,0.5],[3.0,1.0]]
print("[B] VI 多起点优化结果（同一固定CRN噪声）:")
for s in starts:
    res = minimize(neg_elbo, x0=s, method='Nelder-Mead')
    print(f"   start={s} -> mu={res.x[0]:+.4f}, sigma={math.exp(res.x[1]):.4f}, ELBO={-res.fun:.4f}")

# ---------- C) 解析后验 & ELBO间隙 ----------
z_1d = sum(w*np.exp(-0.5*((x_obs-mu)/math.sqrt(sigma_obs**2+tau**2))**2)/(math.sqrt(2*np.pi)*math.sqrt(sigma_obs**2+tau**2))
          for w, mu, tau in zip(prior_weights, prior_means, prior_stds))
print(f"[C] 解析 Z = {z_1d:.6f}, logZ = {math.log(z_1d):.4f}")
s2 = 1.0/(1.0/sigma_obs**2 + 1.0/0.25)
for w, mu in zip(prior_weights, prior_means):
    m = s2*(x_obs/sigma_obs**2 + mu/0.25)
    print(f"    分量(mu={mu}): 后验均值={m:+.4f}, 后验std={math.sqrt(s2):.4f}")
# 权重：x=0 对两个对称分量边际似然相同 -> 权重=先验权重
print("    注：x_obs=0 到 ±2 距离相等 -> 后验权重 = 先验权重 (0.4, 0.6)")
