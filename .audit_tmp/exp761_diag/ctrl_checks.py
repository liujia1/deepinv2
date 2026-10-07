# -*- coding: utf-8 -*-
"""实验7.6-1 复核对照脚本：
1) KS基线噪声水平（两份独立完美样本）
2) DPM-Solver clip r 的必要性：20-seed 有/无clip 对照（5/10步）
3) SNR变化率std的稳健性：剔除t≈0裁剪悬崖后线性vs余弦排序是否保持
4) DDIM(η=0.1) vs DDIM(η=0)：20-seed 是否所有步数都改善
"""
import numpy as np
from scipy import stats as sp_stats
from scipy.stats import mannwhitneyu, ks_2samp
import importlib.util, sys, os, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 复用 7.6-1 的函数：手动复制核心定义（避免执行整个脚本）
def gm1d_sample(n, weights=[0.3, 0.7], means=[-2, 1], stds=[1, 1]):
    k = np.random.choice(len(weights), p=weights, size=n)
    return np.random.randn(n) * np.array(stds)[k] + np.array(means)[k]

def vp_marginal_linear(t, beta_min=0.1, beta_max=20.0):
    log_mean = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    return np.exp(log_mean), np.sqrt(1 - np.exp(2 * log_mean))

def vp_marginal_cosine(t, s=0.008):
    f = lambda t_: np.cos(np.pi * (t_ + s) / (2 * (1 + s)))**2
    ab = f(t) / f(0)
    return np.sqrt(ab), np.sqrt(1 - ab)

def vp_score_analytic(x, t, marginal_fn):
    mean_t, std_t = marginal_fn(t)
    pdf = np.zeros_like(x); dpdf = np.zeros_like(x)
    for w, m, s in zip([0.3, 0.7], [-2, 1], [1, 1]):
        nm = mean_t * m; ns = np.sqrt(mean_t**2 * s**2 + std_t**2)
        g = np.exp(-0.5 * ((x - nm) / ns)**2) / (ns * np.sqrt(2 * np.pi))
        pdf += w * g; dpdf += w * (-(x - nm) / ns**2) * g
    return dpdf / (pdf + 1e-30)

def evaluate_quality(samples, reference):
    return ks_2samp(samples, reference).statistic

def ddim_sample(score_fn, marginal_fn, N, N_steps, T=1.0, eta=0.0):
    dt = T / N_steps
    x = np.random.randn(N)
    for i in range(N_steps):
        t = T - i * dt; t_prev = max(t - dt, 0)
        mean_t, std_t = marginal_fn(t); mean_prev, std_prev = marginal_fn(t_prev)
        ab_t = mean_t**2; ab_p = mean_prev**2
        eps = -std_t * score_fn(x, t, marginal_fn)
        x0_hat = (x - std_t * eps) / (mean_t + 1e-10)
        if eta > 0 and ab_p > 0 and ab_p > ab_t:
            se = eta * np.sqrt((1 - ab_p) / (1 - ab_t)) * np.sqrt(1 - ab_t / ab_p)
        else:
            se = 0.0
        dir_xt = np.sqrt(max(1 - ab_p - se**2, 0)) * eps
        noise = np.random.randn(N) * se if se > 0 else 0
        x = np.sqrt(ab_p) * x0_hat + dir_xt + noise
    return x

def dpm_solver2_sample(score_fn, marginal_fn, N, N_steps, T=1.0, use_clip=True):
    dt = T / N_steps
    x = np.random.randn(N); prev_eps = None; prev_h = None
    for i in range(N_steps):
        t = T - i * dt; t_prev = max(t - dt, 1e-10)
        mean_t, std_t = marginal_fn(t); mean_prev, _ = marginal_fn(t_prev)
        ab_t = mean_t**2; ab_p = mean_prev**2
        eps = -std_t * score_fn(x, t, marginal_fn)
        lam_t = 0.5 * np.log(ab_t / (1 - ab_t + 1e-30) + 1e-30)
        lam_p = 0.5 * np.log(ab_p / (1 - ab_p + 1e-30) + 1e-30)
        h = lam_p - lam_t
        if prev_eps is not None and abs(h) > 1e-8:
            r = h / (prev_h + 1e-30)
            if use_clip:
                r = np.clip(r, -2.0, 2.0)
            eps_c = eps + 0.5 * r * (eps - prev_eps)
        else:
            eps_c = eps
        x0_hat = (x - std_t * eps_c) / (mean_t + 1e-10)
        x = np.sqrt(ab_p) * x0_hat + np.sqrt(1 - ab_p) * eps_c
        prev_eps = eps.copy(); prev_h = h
    return x

# ---------- 1) KS基线 ----------
base = []
for j in range(20):
    np.random.seed(2000 + j)
    base.append(evaluate_quality(gm1d_sample(3000), gm1d_sample(3000)))
print(f"[1] KS基线(20次, n=3000): mean={np.mean(base):.4f} std={np.std(base):.4f} range=[{min(base):.4f},{max(base):.4f}]")

# ---------- 2) clip r 必要性 20-seed ----------
np.random.seed(42)
x0_ref = gm1d_sample(5000)
ref = x0_ref[:3000]
for ns in (5, 10):
    ks_c, ks_nc = [], []
    for seed in range(42, 62):
        np.random.seed(seed)
        ks_c.append(evaluate_quality(dpm_solver2_sample(vp_score_analytic, vp_marginal_cosine, 3000, ns, use_clip=True), ref))
        np.random.seed(seed)
        ks_nc.append(evaluate_quality(dpm_solver2_sample(vp_score_analytic, vp_marginal_cosine, 3000, ns, use_clip=False), ref))
    p = mannwhitneyu(ks_nc, ks_c, alternative='greater')[1]
    print(f"[2] DPM@{ns}步 20seed: clip KS={np.mean(ks_c):.4f}  无clip KS={np.mean(ks_nc):.4f}  (H1:无clip更差) p={p:.4g}")

# ---------- 3) SNR变化率std稳健性 ----------
for lo, hi, tag in [(0.0, 0.99, '全区间(含t→0裁剪悬崖)'), (0.05, 0.98, '剔除两端')]:
    tg = np.linspace(lo, hi, 500)
    ml, _ = vp_marginal_linear(tg); mc, _ = vp_marginal_cosine(tg)
    abl = np.where(ml**2 < 1, ml**2, 1 - 1e-12)
    def snrdb(ab): return 10*np.log10(np.clip(ab/(1-ab), 1e-10, 1e10))
    rl = np.abs(np.gradient(snrdb(abl), tg)); rc = np.abs(np.gradient(snrdb(mc**2), tg))
    print(f"[3] SNR率std {tag}: 线性={np.std(rl):.1f} 余弦={np.std(rc):.1f} -> {'余弦更均匀' if np.std(rc)<np.std(rl) else '线性更均匀'}")

# ---------- 4) DDIM η=0.1 vs η=0 多seed ----------
for ns in (5, 10, 20, 100, 500):
    d0, d1 = [], []
    for seed in range(42, 62):
        np.random.seed(seed)
        d0.append(evaluate_quality(ddim_sample(vp_score_analytic, vp_marginal_cosine, 3000, ns, eta=0.0), ref))
        np.random.seed(seed)
        d1.append(evaluate_quality(ddim_sample(vp_score_analytic, vp_marginal_cosine, 3000, ns, eta=0.1), ref))
    win = np.sum(np.array(d1) < np.array(d0))
    p = mannwhitneyu(d0, d1, alternative='greater')[1]
    print(f"[4] @{ns}步 20seed: η=0 KS={np.mean(d0):.4f} η=0.1 KS={np.mean(d1):.4f} η=0.1改善{win}/20次 p={p:.4g}")
