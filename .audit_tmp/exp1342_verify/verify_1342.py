# -*- coding: utf-8 -*-
"""
实验13.4-2 复核脚本：
  A) 原样重跑实验算法（修复JSON块顺序后的逻辑），对照讲义表格数值
  B) 验证解析后验数学（手算复核）
  C) 验证docstring中 dx0_hat/dx_t 有限差分数值（0.049 vs mean_t 0.017 vs 1/mean_t 58.8）
  D) 免责裁决对照实验：
     - 精确后验得分采样（ζ=1）应复现真实后验 → 检验SDE代码本身是否正确
     - 精确似然得分引导下的ζ扫描 → 检验"均值偏差非单调"是否源于mean_t近似
     - 朴素 1/mean_t 缩放 → 检验docstring关于其不合理的说法
"""
import sys, io, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(42)

GM_WEIGHTS = [0.3, 0.7]
GM_MEANS = [-2.0, 1.0]
GM_STDS = [1.0, 1.0]
BETA_MIN, BETA_MAX = 0.1, 20.0

def vp_marginal(t):
    log_mean = -0.25 * t**2 * (BETA_MAX - BETA_MIN) - 0.5 * t * BETA_MIN
    mean_t = np.exp(log_mean)
    std_t = np.sqrt(1 - np.exp(2 * log_mean))
    return mean_t, std_t

def vp_beta(t):
    return BETA_MIN + t * (BETA_MAX - BETA_MIN)

def gm_score_mixture(x, mus, sigs, weights):
    """一般高斯混合的得分（softmax责任加权，数值稳定）"""
    mus = np.asarray(mus); sigs = np.asarray(sigs); weights = np.asarray(weights)
    z = (x[:, None] - mus[None, :]) / sigs[None, :]
    logp = -0.5 * z**2 - np.log(sigs)[None, :] + np.log(weights)[None, :]
    logp -= logp.max(axis=1, keepdims=True)
    resp = np.exp(logp); resp /= resp.sum(axis=1, keepdims=True)
    comp_score = -(x[:, None] - mus[None, :]) / sigs[None, :]**2
    return (resp * comp_score).sum(axis=1)

def vp_score_analytic(x, t):
    mean_t, std_t = vp_marginal(t)
    mus = [mean_t * m for m in GM_MEANS]
    sigs = [np.sqrt(mean_t**2 * s**2 + std_t**2) for s in GM_STDS]
    return gm_score_mixture(np.atleast_1d(x), mus, sigs, GM_WEIGHTS)

def analytic_posterior_gm_gaussian(y_obs, A, sigma_y, gm_weights, gm_means, gm_stds):
    post_weights, post_means, post_stds = [], [], []
    log_weights_unnorm = []
    for w, m, s in zip(gm_weights, gm_means, gm_stds):
        post_mean = (sigma_y**2 * m + s**2 * y_obs) / (sigma_y**2 + s**2)
        post_std = sigma_y * s / np.sqrt(sigma_y**2 + s**2)
        post_means.append(post_mean); post_stds.append(post_std)
        marginal_var = sigma_y**2 + s**2
        log_marginal = -0.5 * ((y_obs - m)**2 / marginal_var) - 0.5 * np.log(2 * np.pi * marginal_var)
        log_weights_unnorm.append(np.log(w) + log_marginal)
    max_log = max(log_weights_unnorm)
    wu = [np.exp(lw - max_log) for lw in log_weights_unnorm]
    total = sum(wu)
    post_weights = [a / total for a in wu]
    posterior_mean = sum(w * m for w, m in zip(post_weights, post_means))
    posterior_var = sum(w * (s**2 + m**2) for w, s, m in zip(post_weights, post_stds, post_means)) - posterior_mean**2
    return posterior_mean, np.sqrt(posterior_var), post_weights, post_means, post_stds

def exact_posterior_score(x, t, post_w, post_m, post_s):
    """p(x_t|y) 的精确得分：p(x_t|y)=∫p(x_t|x0)p(x0|y)dx0，先验侧同样为GM"""
    mean_t, std_t = vp_marginal(t)
    pm = [mean_t * m for m in post_m]
    ps = [np.sqrt(mean_t**2 * s**2 + std_t**2) for s in post_s]
    # 注意后验分量权重需重新归一化：p(x_t|y) 的分量权重 ∝ w_k|y * N(x_t; mean_t*m_k|y, ...)
    return gm_score_mixture(x, pm, ps, post_w)

def run_dps(y_obs, A, sigma_y, zeta, mode='meant', N=5000, N_steps=300, T=1.0, seed=42,
            post_params=None):
    """mode: 'meant' 原代码; 'inv_meant' 朴素链式; 'exact' 精确似然得分"""
    np.random.seed(seed)
    h = T / N_steps
    x = np.random.randn(N)
    for i in range(N_steps):
        t = T - i * h
        beta_t = vp_beta(t)
        mean_t, std_t = vp_marginal(t)
        prior_score = vp_score_analytic(x, t)
        mean_t_safe = max(mean_t, 1e-6)
        x0_hat = (x + std_t**2 * prior_score) / mean_t_safe
        if mode == 'meant':
            lik = mean_t * (y_obs - A * x0_hat) / sigma_y**2
        elif mode == 'inv_meant':
            lik = (y_obs - A * x0_hat) / (mean_t_safe * sigma_y**2)
        else:  # exact: ∇log p(y|x_t) = s_post - s_prior
            pw, pm, ps = post_params
            lik = exact_posterior_score(x, t, pw, pm, ps) - prior_score
        score = prior_score + zeta * lik
        x = x + beta_t * h * (0.5 * x + score) + np.sqrt(beta_t * h) * np.random.randn(N)
    return x

# ============================================================
print("=" * 70)
print("A) 原实验重跑（修复变量顺序后）")
A_val, sigma_y, y_obs = 1.0, 0.5, 0.5
pm_true, ps_true, pw, pcm, pcs = analytic_posterior_gm_gaussian(
    y_obs, A_val, sigma_y, GM_WEIGHTS, GM_MEANS, GM_STDS)
print(f"B) 解析后验: mean={pm_true:.4f} std={ps_true:.4f}  分量w={pw} μ={pcm} σ={pcs}")
xg = np.linspace(-10, 10, 100000)
def post_pdf(xv):
    xv = np.asarray(xv, dtype=float)
    out = np.zeros_like(xv)
    for w, m, s in zip(pw, pcm, pcs):
        out += w * np.exp(-0.5 * ((xv - m) / s)**2) / (s * np.sqrt(2 * np.pi))
    return out
true_cons = np.trapezoid(np.abs(xg - y_obs) * post_pdf(xg), xg)
print(f"   真实后验 E|X-y| = {true_cons:.4f}   |E[X]-y| = {abs(pm_true - y_obs):.4f}")

lecture_table = {0.0: (0.051, 1.716), 0.3: (0.525, 0.732), 0.7: (0.540, 0.444),
                 1.0: (0.524, 0.366), 2.0: (0.507, 0.256), 5.0: (0.503, 0.164)}
zeta_values = [0.0, 0.3, 0.7, 1.0, 2.0, 5.0]
print(f"\n{'zeta':>5s} | {'重跑均值':>8s} | {'讲义均值':>8s} | {'重跑std':>7s} | {'讲义std':>7s} | 一致?")
cons_list, div_list = [], []
dps_samples = {}
for z in zeta_values:
    s = run_dps(y_obs, A_val, sigma_y, z, mode='meant')
    dps_samples[z] = s
    m_, sd_ = float(np.mean(s)), float(np.std(s))
    lm, ls = lecture_table[z]
    ok = abs(m_ - lm) < 0.002 and abs(sd_ - ls) < 0.002
    cons_list.append(np.mean(np.abs(s - y_obs))); div_list.append(np.std(s))
    print(f"{z:5.1f} | {m_:8.3f} | {lm:8.3f} | {sd_:7.3f} | {ls:7.3f} | {'OK' if ok else 'MISMATCH'}")

# ============================================================
print("\n" + "=" * 70)
print("C) docstring有限差分验证：t=0.9 处 dx0_hat/dx_t")
t0 = 0.9
mean_t, std_t = vp_marginal(t0)
def x0hat_of(xx):
    sc = vp_score_analytic(np.array([xx]), t0)[0]
    return (xx + std_t**2 * sc) / mean_t
eps_fd = 1e-4
for xt in [0.5, 0.0, -1.0]:
    d = (x0hat_of(xt + eps_fd) - x0hat_of(xt - eps_fd)) / (2 * eps_fd)
    print(f"   x_t={xt:>5.1f}: 有限差分 dx0/dxt = {d:.4f}; mean_t={mean_t:.4f}; 1/mean_t={1/mean_t:.2f}; 比值={d/mean_t:.2f}")
# 在N(0,1)粒子系综上的平均
xt_ens = np.random.RandomState(0).randn(20000)
sc1 = vp_score_analytic(xt_ens + eps_fd, t0)
sc0 = vp_score_analytic(xt_ens - eps_fd, t0)
d_ens = ((xt_ens + eps_fd + std_t**2 * sc1) - (xt_ens - eps_fd + std_t**2 * sc0)) / (2 * eps_fd * mean_t)
print(f"   系综平均(20000个x_t~N(0,1)): dx0/dxt ≈ {np.mean(d_ens):.4f} (median {np.median(d_ens):.4f}), mean_t={mean_t:.4f}")

# ============================================================
print("\n" + "=" * 70)
print("D1) 精确后验得分采样 (ζ=1, exact) —— 检验SDE代码正确性")
s_exact = run_dps(y_obs, A_val, sigma_y, 1.0, mode='exact', post_params=(pw, pcm, pcs))
print(f"   exact ζ=1: mean={np.mean(s_exact):.4f} (真值{pm_true:.4f}), std={np.std(s_exact):.4f} (真值{ps_true:.4f})")

print("\nD2) 精确似然得分下的ζ扫描 —— 免责裁决：非单调是否源于mean_t近似?")
print(f"{'zeta':>5s} | {'均值':>8s} | {'均值偏差':>8s} | {'std':>7s} | {'std偏差':>7s}")
for z in zeta_values:
    s = run_dps(y_obs, A_val, sigma_y, z, mode='exact', post_params=(pw, pcm, pcs))
    mb, sb = np.mean(s) - pm_true, np.std(s) - ps_true
    print(f"{z:5.1f} | {np.mean(s):8.4f} | {mb:+8.4f} | {np.std(s):7.4f} | {sb:+7.4f}")

print("\nD3) 朴素 1/mean_t 缩放（docstring声称不合理）ζ=0.3/0.7/1.0:")
for z in [0.3, 0.7, 1.0]:
    try:
        s = run_dps(y_obs, A_val, sigma_y, z, mode='inv_meant')
        print(f"   inv_meant ζ={z}: mean={np.mean(s):.4f} std={np.std(s):.4f}")
    except FloatingPointError as e:
        print(f"   inv_meant ζ={z}: 数值溢出 {e}")

print("\nD4) mean_t模式下的ζ=1（标准DPS）与more seeds敏感性（甜蜜点检验）")
for z in [0.7, 1.0]:
    for sd in [42, 7, 2024]:
        s = run_dps(y_obs, A_val, sigma_y, z, mode='meant', seed=sd)
        print(f"   meant ζ={z} seed={sd}: mean={np.mean(s):.4f} bias={np.mean(s)-pm_true:+.4f} std={np.std(s):.4f}")

# ============================================================
print("\n" + "=" * 70)
print("生图核对（与实验目录PNG对比）")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
x_hist = np.linspace(-6, 6, 500)
for idx, z in enumerate(zeta_values):
    ax = axes[idx // 3, idx % 3]
    ax.hist(dps_samples[z], bins=60, density=True, alpha=0.5, color='steelblue', range=(-6, 6))
    ax.plot(x_hist, post_pdf(x_hist), 'g-', lw=2)
    ax.set_title(f'zeta={z}')
fig.savefig(os.path.join(SAVE_DIR, 'rerun_引导权重权衡.png'), dpi=100, bbox_inches='tight')
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(div_list, cons_list, 'ro-')
for i, z in enumerate(zeta_values):
    ax2.annotate(f'{z}', (div_list[i], cons_list[i]), xytext=(5, 5), textcoords='offset points')
ax2.scatter([ps_true], [true_cons], marker='*', s=150, c='green')
ax2.set_xlabel('diversity'); ax2.set_ylabel('consistency')
fig2.savefig(os.path.join(SAVE_DIR, 'rerun_权衡曲线.png'), dpi=100, bbox_inches='tight')
print(f"重跑图已保存: {SAVE_DIR}")
print(f"重跑 consistency_list = {[round(c,4) for c in cons_list]}")
print(f"重跑 diversity_list    = {[round(d,4) for d in div_list]}")
