# -*- coding: utf-8 -*-
"""探针2：验证13.3-3中"DPS重建误差"是否就是‖x_t - x2d‖（扩散噪声本身），
即DPS的得分修正项几乎不起作用，误差随t单调增是实现方式导致，而非delta近似固有属性。
"""
import numpy as np

T = 200
betas = np.linspace(1e-4, 0.02, T)
alpha_bars = np.cumprod(1.0 - betas)
GM_WEIGHTS = [0.3, 0.7]; GM_MEANS = [-2.0, 1.0]; GM_STDS = [1.0, 1.0]
A = np.array([[2.0, 0.1], [1.5, 0.5]])
sigma_obs = 0.1

def tweedie_1d(x_t_scalar, abar):
    sqrt_abar = np.sqrt(abar); sigma_t2 = 1 - abar
    lw = np.log(GM_WEIGHTS)
    ll = np.array([-0.5*((x_t_scalar - sqrt_abar*mu)**2)/(abar*sg**2+sigma_t2)
                   - 0.5*np.log(abar*sg**2+sigma_t2) for mu, sg in zip(GM_MEANS, GM_STDS)])
    lj = lw + ll; lj -= lj.max(); pi = np.exp(lj); pi /= pi.sum()
    pm = np.array([(sqrt_abar*sg**2*x_t_scalar + sigma_t2*mu)/(abar*sg**2+sigma_t2)
                   for mu, sg in zip(GM_MEANS, GM_STDS)])
    return (pi*pm).sum()

def tweedie(x_t, t_idx):
    abar = alpha_bars[t_idx]
    return np.array([tweedie_1d(v, abar) for v in x_t])

np.random.seed(42)
for t_idx in [0, 50, 100, 150, 199]:
    dps_errs, dist_xt, corr_mag = [], [], []
    for _ in range(500):
        x0 = np.random.normal() * 0 + (np.random.choice(2, p=GM_WEIGHTS) and 1 or 0)  # 简化：与主脚本一致用混合采样
        comp = np.random.choice(2, p=GM_WEIGHTS)
        x0 = np.random.normal(GM_MEANS[comp], GM_STDS[comp])
        x2d = np.array([x0, x0*0.5])
        y = A @ x2d + np.random.randn(2)*sigma_obs
        ab = alpha_bars[t_idx]
        x_t = np.sqrt(ab)*x2d + np.sqrt(1-ab)*np.random.randn(2)
        x_hat0 = tweedie(x_t, t_idx)
        score = A.T @ (y - A @ x_hat0) / sigma_obs**2
        step = 0.1 * 0.01 * score          # zeta=0.01, 步长0.1
        x_dps = x_t + step
        dps_errs.append(np.linalg.norm(x_dps - x2d))
        dist_xt.append(np.linalg.norm(x_t - x2d))
        corr_mag.append(np.linalg.norm(step))
    print(f"t={t_idx:3d}: DPS误差={np.mean(dps_errs):.4f} | ‖x_t-x2d‖={np.mean(dist_xt):.4f} | 修正项范数={np.mean(corr_mag):.4f}")
