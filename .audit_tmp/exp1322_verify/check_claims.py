# -*- coding: utf-8 -*-
"""诊断13.2-2：检验讲义三条定性论断在全部200个测试点上是否成立"""
import sys, os, io
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 复用原脚本函数定义（不执行画图部分，直接内嵌相同公式）
GM_WEIGHTS = [0.3, 0.7]; GM_MEANS = [-2.0, 1.0]; GM_STDS = [1.0, 1.0]
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
        g = np.exp(-0.5 * ((x - nm) / ns)**2) / (ns * np.sqrt(2 * np.pi))
        pdf += w * g; dpdf += w * (-(x - nm) / ns**2) * g
    return dpdf / (pdf + 1e-30)

def tweedie_estimate(x_t, t):
    mean_t, std_t = vp_marginal(t)
    return (x_t + std_t**2 * vp_score_analytic(x_t, t)) / (mean_t + 1e-10)

def compute_marginal_pxt(x_t, t):
    mean_t, std_t = vp_marginal(t)
    pdf = np.zeros_like(np.atleast_1d(x_t), dtype=float)
    for w, m, s in zip(GM_WEIGHTS, GM_MEANS, GM_STDS):
        nm = mean_t * m; ns = np.sqrt(mean_t**2 * s**2 + std_t**2)
        pdf += w * np.exp(-0.5 * ((x_t - nm) / ns)**2) / (ns * np.sqrt(2 * np.pi))
    return pdf

def _compute_likelihood(x_t, t, y_obs, A, sigma_y):
    mean_t, std_t = vp_marginal(t)
    x0_grid = np.linspace(-8, 8, 2000); dx0 = x0_grid[1] - x0_grid[0]
    p_xt_given_x0 = np.exp(-0.5 * ((x_t - mean_t * x0_grid) / std_t)**2) / (std_t * np.sqrt(2 * np.pi))
    p_x0_given_xt = p_xt_given_x0 * gm1d_pdf(x0_grid) / (compute_marginal_pxt(x_t, t) + 1e-30)
    p_y_given_x0 = np.exp(-0.5 * ((y_obs - A * x0_grid) / sigma_y)**2) / (sigma_y * np.sqrt(2 * np.pi))
    return np.sum(p_y_given_x0 * p_x0_given_xt) * dx0

A_val, sigma_y, y_obs, t_val = 1.0, 0.5, 0.5, 0.3
x_test = np.linspace(-4, 4, 200)
exact = np.array([
    (_compute_likelihood(xi + 1e-4, t_val, y_obs, A_val, sigma_y)
     - _compute_likelihood(xi - 1e-4, t_val, y_obs, A_val, sigma_y))
    / (2e-4 * _compute_likelihood(xi, t_val, y_obs, A_val, sigma_y) + 1e-30)
    for xi in x_test])
x0_hat = tweedie_estimate(x_test, t_val)
resid = (y_obs - A_val * x0_hat) / sigma_y**2
eps_j = 1e-4
J = (tweedie_estimate(x_test + eps_j, t_val) - tweedie_estimate(x_test - eps_j, t_val)) / (2 * eps_j)
full = J * resid
simple = resid

# 1) 方向正确性：sign(exact) vs sign(full/simple)，|exact|>1e-3 的点
mask_sig = np.abs(exact) > 1e-3
bad_full = np.sum(np.sign(exact[mask_sig]) != np.sign(full[mask_sig]))
bad_simple = np.sum(np.sign(exact[mask_sig]) != np.sign(simple[mask_sig]))
print(f"[论断1 方向] |exact|>1e-3 共{mask_sig.sum()}点: 含J变号{bad_full}个, 忽略J变号{bad_simple}个")
for nm, v in [('full', full), ('simple', simple)]:
    m = mask_sig & (np.sign(exact) != np.sign(v))
    if m.any():
        print(f"  {nm} 变号位置 x_t:", np.round(x_test[m], 3))

# 2) 幅度偏大
m2 = mask_sig & (np.abs(exact) > 0.05)
print(f"[论断2 幅度] |full|>|exact| 占比 {(np.abs(full[m2])>np.abs(exact[m2])).mean():.2%}, "
      f"|simple|>|exact| 占比 {(np.abs(simple[m2])>np.abs(exact[m2])).mean():.2%}")

# 3) 相对误差：谁更小（对应讲义“含Jacobian在部分区域精度更好(如x_t≈0)”）
relf = np.abs(full - exact) / (np.abs(exact) + 1e-10)
rels = np.abs(simple - exact) / (np.abs(exact) + 1e-10)
better_full = relf < rels
regions = [(-4, -2.5), (-2.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 4)]
print("[论断3 含J vs 忽略J 分区对比] (含J更优比例)")
for a, b in regions:
    m = (x_test >= a) & (x_test < b)
    print(f"  x_t∈[{a},{b}): 含J更优 {better_full[m].mean():.0%} | 平均误差 含J={relf[m].mean():7.2f} 略J={rels[m].mean():7.2f}")

# 4) 误差随t单调性（t扫描, x_t=0.5）
t_range = np.linspace(0.05, 0.95, 20)
ef, es = [], []
for t_r in t_range:
    ex = (_compute_likelihood(0.5 + 1e-4, t_r, y_obs, A_val, sigma_y)
          - _compute_likelihood(0.5 - 1e-4, t_r, y_obs, A_val, sigma_y)) / \
         (2e-4 * _compute_likelihood(0.5, t_r, y_obs, A_val, sigma_y) + 1e-30)
    xh = tweedie_estimate(np.array([0.5]), t_r)[0]
    xhp = tweedie_estimate(np.array([0.5 + eps_j]), t_r)[0]
    xhm = tweedie_estimate(np.array([0.5 - eps_j]), t_r)[0]
    gx = (xhp - xhm) / (2 * eps_j); r = (y_obs - xh) / sigma_y**2
    ef.append(abs(gx * r - ex) / (abs(ex) + 1e-10)); es.append(abs(r - ex) / (abs(ex) + 1e-10))
    if abs(t_r - t_range[8]) < 1e-9:
        print(f"[尖峰诊断] t={t_r:.3f}: exact={ex:.5f} (接近0导致相对误差尖峰)")
mono_f = np.all(np.diff(ef) > 0); mono_s = np.all(np.diff(es) > 0)
print(f"[论断4 随t单调增] 含J单调={mono_f}, 忽略J单调={mono_s}")
print("  t>=0.6 后 含J更优?", all(eff < ess for eff, ess in zip(ef[12:], es[12:])))
print("  全程含J误差<忽略J?", all(eff < ess for eff, ess in zip(ef, es)))
