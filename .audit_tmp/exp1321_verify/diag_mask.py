# -*- coding: utf-8 -*-
"""诊断：验证 |x_t|<3 掩码是否为固有数值属性（而非规避策略）
对比有/无掩码时的分解误差，并检查无掩码误差峰值的空间位置。
"""
import numpy as np
import importlib.util, sys, types, os

# 从原脚本导入核心函数（原脚本顶层会执行绘图，这里只复制函数定义所需的常量与函数）
GM_WEIGHTS = [0.3, 0.7]
GM_MEANS = [-2.0, 1.0]
GM_STDS = [1.0, 1.0]
BETA_MIN, BETA_MAX = 0.1, 20.0

def gm1d_pdf(x):
    pdf = np.zeros_like(x)
    for w, m, s in zip(GM_WEIGHTS, GM_MEANS, GM_STDS):
        pdf += w * np.exp(-0.5 * ((x - m) / s)**2) / (s * np.sqrt(2 * np.pi))
    return pdf

def vp_marginal(t):
    log_mean = -0.25 * t**2 * (BETA_MAX - BETA_MIN) - 0.5 * t * BETA_MIN
    return np.exp(log_mean), np.sqrt(1 - np.exp(2 * log_mean))

def vp_score_analytic(x, t):
    mean_t, std_t = vp_marginal(t)
    pdf = np.zeros_like(x); dpdf = np.zeros_like(x)
    for w, m, s in zip(GM_WEIGHTS, GM_MEANS, GM_STDS):
        nm = mean_t * m; ns = np.sqrt(mean_t**2 * s**2 + std_t**2)
        pdf += w * np.exp(-0.5 * ((x - nm) / ns)**2) / (ns * np.sqrt(2 * np.pi))
        dpdf += w * (-(x - nm) / ns**2) * np.exp(-0.5 * ((x - nm) / ns)**2) / (ns * np.sqrt(2 * np.pi))
    return dpdf / (pdf + 1e-30)

def compute_marginal_pxt(x_t, t):
    mean_t, std_t = vp_marginal(t)
    pdf = np.zeros_like(np.atleast_1d(x_t), dtype=float)
    for w, m, s in zip(GM_WEIGHTS, GM_MEANS, GM_STDS):
        nm = mean_t * m; ns = np.sqrt(mean_t**2 * s**2 + std_t**2)
        pdf += w * np.exp(-0.5 * ((x_t - nm) / ns)**2) / (ns * np.sqrt(2 * np.pi))
    return pdf

def compute_likelihood(x_t, t, y_obs, A, sigma_y):
    mean_t, std_t = vp_marginal(t)
    x0_grid = np.linspace(-8, 8, 2000); dx0 = x0_grid[1] - x0_grid[0]
    p_xt_given_x0 = np.exp(-0.5 * ((x_t - mean_t * x0_grid) / std_t)**2) / (std_t * np.sqrt(2 * np.pi))
    p_x0 = gm1d_pdf(x0_grid)
    p_xt = compute_marginal_pxt(x_t, t)
    p_x0_given_xt = p_xt_given_x0 * p_x0 / (p_xt + 1e-30)
    p_y_given_x0 = np.exp(-0.5 * ((y_obs - A * x0_grid) / sigma_y)**2) / (sigma_y * np.sqrt(2 * np.pi))
    return np.sum(p_y_given_x0 * p_x0_given_xt) * dx0

def compute_joint_pxt_y(x_t, t, y_obs, A, sigma_y):
    return compute_marginal_pxt(x_t, t) * compute_likelihood(x_t, t, y_obs, A, sigma_y)

A_val, sigma_y, y_obs = 1.0, 0.5, 0.5
x_grid = np.linspace(-5, 5, 200)
mask = np.abs(x_grid) < 3
eps = 1e-4

print(f"{'t':>5s}  {'max误差(全域)':>16s}  {'argmax位置x_t':>12s}  {'max误差(|x|<3)':>16s}  {'p(x_t)@argmax':>14s}")
for t_val in [0.1, 0.3, 0.5, 0.8]:
    ps = vp_score_analytic(x_grid, t_val)
    ls = np.zeros_like(x_grid); post = np.zeros_like(x_grid)
    for i, xi in enumerate(x_grid):
        pl = compute_likelihood(xi + eps, t_val, y_obs, A_val, sigma_y)
        ml = compute_likelihood(xi - eps, t_val, y_obs, A_val, sigma_y)
        cl = compute_likelihood(xi, t_val, y_obs, A_val, sigma_y)
        ls[i] = (pl - ml) / (2 * eps * cl + 1e-30)
        pj_p = compute_joint_pxt_y(xi + eps, t_val, y_obs, A_val, sigma_y)
        pj_m = compute_joint_pxt_y(xi - eps, t_val, y_obs, A_val, sigma_y)
        pj_c = compute_joint_pxt_y(xi, t_val, y_obs, A_val, sigma_y)
        post[i] = (pj_p - pj_m) / (2 * eps * pj_c + 1e-30)
    err = np.abs(post - (ps + ls))
    imax = np.argmax(err)
    pxt_at = compute_marginal_pxt(x_grid[imax], t_val)[0]
    print(f"{t_val:5.1f}  {err.max():16.3e}  {x_grid[imax]:12.2f}  {err[mask].max():16.3e}  {pxt_at:14.3e}")
