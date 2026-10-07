# -*- coding: utf-8 -*-
"""实验7.5-2 步骤4对照诊断：
1) 复现实验单种子结果（KS vs x0_ref[:3000]）
2) 多种子(10 seeds)平均 KS vs 解析目标CDF（一样本KS，消除参考样本噪声）
3) 多种子平均 KS vs 同一参考样本（复现口径）
裁决"DDIM在少步数(10-50)时KS显著低于DDPM"是否成立/是否噪声。
"""
import sys, io, os
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from scipy import stats as sp_stats
from scipy.special import erf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- 与 7.5-2.py 完全相同的采样器（直接复制核心函数） ----
beta_min, beta_max = 0.1, 20.0

def vp_marginal(t):
    log_mean = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    mean_t = np.exp(log_mean)
    std_t = np.sqrt(1 - np.exp(2 * log_mean))
    return mean_t, std_t

def vp_beta(t):
    return beta_min + t * (beta_max - beta_min)

def vp_score_analytic(x, t):
    mean_t, std_t = vp_marginal(t)
    pdf = np.zeros_like(x); dpdf = np.zeros_like(x)
    for w, m, s in zip([0.3, 0.7], [-2, 1], [1, 1]):
        new_mean = mean_t * m
        new_std = np.sqrt(mean_t**2 * s**2 + std_t**2)
        pdf += w * np.exp(-0.5 * ((x - new_mean) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
        dpdf += w * (-(x - new_mean) / new_std**2) * np.exp(-0.5 * ((x - new_mean) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
    return dpdf / (pdf + 1e-30)

def ddim_sample_vp(score_fn, N_particles, N_steps, T=1.0, eta=0.0):
    dt = T / N_steps
    x = np.random.randn(N_particles)
    traj = [x.copy()]
    for i in range(N_steps):
        t = T - i * dt
        t_prev = max(t - dt, 0)
        mean_t, std_t = vp_marginal(t); alpha_bar_t = mean_t**2
        mean_prev, std_prev = vp_marginal(t_prev); alpha_bar_prev = mean_prev**2
        score = score_fn(x, t)
        eps_theta = -std_t * score
        x0_hat = (x - std_t * eps_theta) / (mean_t + 1e-10)
        if eta > 0 and alpha_bar_prev > 0 and alpha_bar_t > 0:
            sigma_eta = eta * np.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * \
                       np.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        else:
            sigma_eta = 0.0
        dir_xt = np.sqrt(max(1 - alpha_bar_prev - sigma_eta**2, 0)) * eps_theta
        noise = np.random.randn(N_particles) * sigma_eta if sigma_eta > 0 else 0
        x = np.sqrt(alpha_bar_prev) * x0_hat + dir_xt + noise
        traj.append(x.copy())
    return np.array(traj)

# 解析目标 CDF（0.3 N(-2,1) + 0.7 N(1,1)）
def target_cdf(x):
    z1 = 0.5 * (1 + erf((x + 2) / np.sqrt(2)))
    z2 = 0.5 * (1 + erf((x - 1) / np.sqrt(2)))
    return 0.3 * z1 + 0.7 * z2

step_counts = [10, 20, 50, 100, 200, 500]
N_PART = 3000
n_seeds = 10

# 参考样本（与实验相同口径：seed42 先生成 5000 取前 3000）
np.random.seed(42)
x0_ref = np.where(np.random.rand(5000) < 0.3,
                  np.random.randn(5000) - 2,
                  np.random.randn(5000) + 1)[:3000]

print("== A. 单种子(seed42)复现，与实验 print 对照 (KS vs x0_ref[:3000]) ==")
for n in step_counts:
    row = {}
    for eta, name in [(1.0, 'DDPM'), (0.0, 'DDIM')]:
        np.random.seed(42)
        final = ddim_sample_vp(vp_score_analytic, N_PART, n, eta=eta)[-1]
        row[name] = sp_stats.ks_2samp(final, x0_ref).statistic
    print(f"N={n:4d}: DDPM KS={row['DDPM']:.4f}, DDIM KS={row['DDIM']:.4f}, DDIM-DDPM={row['DDIM']-row['DDPM']:+.4f}")

print("\n== B. 10种子平均 KS vs 解析目标CDF（一样本KS，无参考样本噪声） ==")
res = {'DDPM': {n: [] for n in step_counts}, 'DDIM': {n: [] for n in step_counts}}
for s in range(n_seeds):
    for n in step_counts:
        for eta, name in [(1.0, 'DDPM'), (0.0, 'DDIM')]:
            np.random.seed(s * 1000 + n)
            final = ddim_sample_vp(vp_score_analytic, N_PART, n, eta=eta)[-1]
            res[name][n].append(sp_stats.kstest(final, target_cdf).statistic)
print(f"{'N':>5} {'DDPM(解析)':>12} {'DDIM(解析)':>12} {'差(DDIM-DDPM)':>14} {'DDPM_se':>9} {'DDIM_se':>9}")
for n in step_counts:
    dm = np.mean(res['DDPM'][n]); dd = np.mean(res['DDIM'][n])
    sem_d = np.std(res['DDPM'][n], ddof=1)/np.sqrt(n_seeds)
    sem_i = np.std(res['DDIM'][n], ddof=1)/np.sqrt(n_seeds)
    print(f"{n:5d} {dm:12.4f} {dd:12.4f} {dd-dm:+14.4f} {sem_d:9.4f} {sem_i:9.4f}")

print("\n== C. 10种子平均 两样本KS vs 固定参考 x0_ref[:3000]（复现口径的多种子版） ==")
res2 = {'DDPM': {n: [] for n in step_counts}, 'DDIM': {n: [] for n in step_counts}}
for s in range(n_seeds):
    for n in step_counts:
        for eta, name in [(1.0, 'DDPM'), (0.0, 'DDIM')]:
            np.random.seed(s * 1000 + n)
            final = ddim_sample_vp(vp_score_analytic, N_PART, n, eta=eta)[-1]
            res2[name][n].append(sp_stats.ks_2samp(final, x0_ref).statistic)
for n in step_counts:
    dm = np.mean(res2['DDPM'][n]); dd = np.mean(res2['DDIM'][n])
    print(f"N={n:4d}: DDPM={dm:.4f}, DDIM={dd:.4f}, 差={dd-dm:+.4f}")

print("\n== D. 一样本KS理论噪声底：完美采样器(seed42) vs x0_ref[:3000] ==")
np.random.seed(7)
perfect = np.where(np.random.rand(3000) < 0.3, np.random.randn(3000)-2, np.random.randn(3000)+1)
print(f"真实样本 vs 参考: ks_2samp = {sp_stats.ks_2samp(perfect, x0_ref).statistic:.4f}")
print(f"真实样本 vs 解析CDF: kstest = {sp_stats.kstest(perfect, target_cdf).statistic:.4f}")
