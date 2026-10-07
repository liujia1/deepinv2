# -*- coding: utf-8 -*-
"""实验7.5-1 复核诊断：
A) SMLD步骤3失败裁决：代码经验步长 vs 讲义EM公式步长(α=σ_i²−σ_{i−1}²) vs 退火Langevin(2α)
B) 步骤4 KS非单调裁决：用解析CDF的KS（无两样本噪声地板），验证是否固有采样噪声
"""
import numpy as np
from scipy import stats as sp_stats

np.set_printoptions(suppress=True)

beta_min, beta_max = 0.1, 20.0
T = 1.0
weights, means, stds = [0.3, 0.7], [-2, 1], [1, 1]

def vp_marginal(t):
    log_mean = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    return np.exp(log_mean), np.sqrt(1 - np.exp(2 * log_mean))

def vp_score_analytic(x, t, beta_min_val=None, beta_max_val=None):
    mean_t, std_t = vp_marginal(t)
    pdf = np.zeros_like(x); dpdf = np.zeros_like(x)
    for w, m, s in zip(weights, means, stds):
        nm = mean_t * m; ns = np.sqrt(mean_t**2 * s**2 + std_t**2)
        pdf += w * np.exp(-0.5 * ((x - nm) / ns)**2) / (ns * np.sqrt(2 * np.pi))
        dpdf += w * (-(x - nm) / ns**2) * np.exp(-0.5 * ((x - nm) / ns)**2) / (ns * np.sqrt(2 * np.pi))
    return dpdf / (pdf + 1e-30)

def ddpm_sample_vp(N, n_steps, seed=42):
    np.random.seed(seed)
    h = T / n_steps
    x = np.random.randn(N)
    for i in range(n_steps):
        t = T - i * h
        beta_t = beta_min + t * (beta_max - beta_min)
        s = vp_score_analytic(x, t)
        x = x + beta_t * h * (0.5 * x + s) + np.sqrt(beta_t * h) * np.random.randn(N)
    return x

# ---------- B) KS vs 解析CDF（DDPM） ----------
def gm1d_cdf(x):
    c = np.zeros_like(x)
    for w, m, s in zip(weights, means, stds):
        c += w * 0.5 * (1 + np.vectorize(sp_stats.norm.cdf)((x - m) / s)) if False else 0
    from scipy.special import erf
    for w, m, s in zip(weights, means, stds):
        c += w * 0.5 * (1 + erf((x - m) / (s * np.sqrt(2))))
    return c

xg = np.linspace(-10, 10, 20001)
Fcdf = gm1d_cdf(xg)

def ks_vs_analytic(final):
    xs = np.sort(final)
    Fe = np.arange(1, len(xs) + 1) / len(xs)
    Fth = np.interp(xs, xg, Fcdf)
    return np.max(np.abs(Fe - Fth))

print("== B) DDPM: KS vs 解析CDF（无两样本噪声地板）与两样本KS 对比 ==")
np.random.seed(42)
x0ref = np.where(np.random.rand(10000) < 0.3, np.random.randn(10000) - 2, np.random.randn(10000) + 1)
print(f"{'steps':>6} {'KS_analytic(mean over 5 seeds)':>32} {'KS_2samp(原代码)':>18}")
for n_steps in [10, 25, 50, 100, 200, 500, 1000]:
    ka, k2 = [], []
    for so in range(5):
        f = ddpm_sample_vp(20000, n_steps, seed=42 + so * 7)
        ka.append(ks_vs_analytic(f))
        k2.append(sp_stats.ks_2samp(f, x0ref).statistic)
    print(f"{n_steps:>6} {np.mean(ka):>20.5f}±{np.std(ka):<10.5f} {np.mean(k2):>12.5f}±{np.std(k2):<6.5f}")

# ---------- A) SMLD 失败裁决 ----------
sigma_min, sigma_max = 0.01, 10.0
def ve_sigma_orig(t):
    # 原代码：t=0 -> 10 (大噪声), t=1 -> 0.01 (小噪声)  与VP的t约定相反
    return sigma_max * (sigma_min / sigma_max) ** t

def ve_sigma_fix(t):
    # 修正：与VP一致，t=0 -> 小噪声, t=1 -> 大噪声
    return sigma_min * (sigma_max / sigma_min) ** t

def make_ve_score(sig_fn):
    def ve_score_analytic(x, t):
        sig = sig_fn(t)
        pdf = np.zeros_like(x); dpdf = np.zeros_like(x)
        ns = np.sqrt(1 + sig**2)
        for w, m in zip(weights, means):
            pdf += w * np.exp(-0.5 * ((x - m) / ns)**2) / (ns * np.sqrt(2 * np.pi))
            dpdf += w * (-(x - m) / ns**2) * np.exp(-0.5 * ((x - m) / ns)**2) / (ns * np.sqrt(2 * np.pi))
        return dpdf / (pdf + 1e-30)
    return ve_score_analytic

def smld(N, n_steps, mode, sig_fn, seed=42):
    np.random.seed(seed)
    dt = T / n_steps
    score_fn = make_ve_score(sig_fn)
    x = np.random.randn(N) * sig_fn(T)  # 从最大噪声水平开始
    for i in range(n_steps):
        t = T - i * dt
        sig_t = sig_fn(t)
        sig_prev = sig_fn(max(t - dt, 0.0))
        s = score_fn(x, t)
        alpha = abs(sig_t**2 - sig_prev**2)
        if mode == 'code':       # 原代码经验步长
            step = 0.1 * (sig_t / sigma_max)**2 + 1e-5
            x = x + step * s + np.sqrt(2 * step) * np.random.randn(N)
        elif mode == 'em':       # 讲义EM: α = σ_i² − σ_{i−1}², 噪声 √α
            x = x + alpha * s + np.sqrt(alpha) * np.random.randn(N)
        elif mode == 'al':       # 退火Langevin: 噪声 √(2α)
            x = x + alpha * s + np.sqrt(2 * alpha) * np.random.randn(N)
    return x

print("\n== A1) 原代码σ(t)约定（噪声随采样递增）==")
for mode in ['code', 'em', 'al']:
    for n_steps in [50, 200, 500]:
        f = smld(5000, n_steps, mode, ve_sigma_orig)
        print(f"  mode={mode:>4} N={n_steps:>4}: mu={np.mean(f):+.4f} var={np.var(f):8.4f} "
              f"KS_ana={ks_vs_analytic(f):.4f} NaN={np.isnan(f).any()}")

print("\n== A2) 修正σ(t)约定（与VP一致，噪声退火递减）==")
for mode in ['code', 'em', 'al']:
    for n_steps in [50, 200, 500, 1000]:
        f = smld(5000, n_steps, mode, ve_sigma_fix)
        print(f"  mode={mode:>4} N={n_steps:>4}: mu={np.mean(f):+.4f} var={np.var(f):8.4f} "
              f"KS_ana={ks_vs_analytic(f):.4f} NaN={np.isnan(f).any()}")
