# -*- coding: utf-8 -*-
"""
审计对照实验（exp1351_verify）：
裁决"DOC优于DPS / DPS少步退化是固有属性"的讲义说法。

对照维度：
  A) 原版DPS（u = 0.01*clamp(200*A*res, ±100)）—— 复现存档行为
  B) 标准DPS（得分修正形式 u = β_t * ζ * A*res/σ²），ζ ∈ {0.1, 0.5, 1.0}
  C) 标准DPS + 确定性读出（用最终 x̂_{0|0} 而非带噪 x_0）
  D) 贝叶斯理论基线：后验均值(MMSE)与后验标准差（误差下限参考）
  E) DOC 原版（同 13.5-1.py）
  F) 大T检验：T=200 时标准DPS vs DOC（讲义称 T>1000 DPS接近DOC）

所有方法共用同一批试验 (x0*, y, x_T)，seed=42。
"""
import numpy as np
import torch
import torch.optim as optim
import json

torch.set_num_threads(2)
device = torch.device('cpu')

GM_WEIGHTS = [0.5, 0.5]
GM_MEANS = [-2.0, 2.0]
GM_STDS = [0.5, 0.5]

def build_schedule(T):
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alpha_bars

def make_score(alpha_bars):
    def gaussian_mixture_score(x_t, t_idx):
        alpha_bar_t = alpha_bars[t_idx]
        sigma_t_sq = 1.0 - alpha_bar_t
        log_probs = []
        for i in range(len(GM_WEIGHTS)):
            mu_i = torch.tensor(GM_MEANS[i], dtype=x_t.dtype)
            sigma_i_sq = torch.tensor(GM_STDS[i] ** 2, dtype=x_t.dtype)
            total_sigma_sq = sigma_i_sq + sigma_t_sq
            log_prob = torch.log(torch.tensor(GM_WEIGHTS[i], dtype=x_t.dtype)) \
                       - 0.5 * torch.log(2 * np.pi * total_sigma_sq) \
                       - 0.5 * (x_t - mu_i) ** 2 / total_sigma_sq
            log_probs.append(log_prob)
        log_probs_stack = torch.stack(log_probs)
        max_log_prob = torch.max(log_probs_stack)
        exp_probs = torch.exp(log_probs_stack - max_log_prob)
        sum_exp = torch.sum(exp_probs)
        score = torch.tensor(0.0, dtype=x_t.dtype)
        for i in range(len(GM_WEIGHTS)):
            mu_i = torch.tensor(GM_MEANS[i], dtype=x_t.dtype)
            sigma_i_sq = torch.tensor(GM_STDS[i] ** 2, dtype=x_t.dtype)
            total_sigma_sq = sigma_i_sq + sigma_t_sq
            weight = exp_probs[i] / sum_exp
            score = score + weight * (-(x_t - mu_i) / total_sigma_sq)
        return score
    return gaussian_mixture_score

def mmse_baseline(x0_star, y_obs, A=2.0, sigma_obs=0.1):
    """解析后验均值与后验std（对高斯混合先验+线性高斯似然）"""
    mus, sig2s, ws = np.array(GM_MEANS), np.array([s**2 for s in GM_STDS]), np.array(GM_WEIGHTS)
    var_eff = A**2 * sig2s + sigma_obs**2
    logw = np.log(ws) - 0.5 * np.log(2*np.pi*var_eff) - 0.5 * (y_obs - A*mus)**2 / var_eff
    W = np.exp(logw - logw.max()); W /= W.sum()
    post_var = 1.0 / (1.0/sig2s + A**2/sigma_obs**2)
    post_mean = post_var * (mus/sig2s + A*y_obs/sigma_obs**2)
    xhat = (W * post_mean).sum()
    # 总后验方差（含模态间方差）
    total_var = (W * (post_var + (post_mean - xhat)**2)).sum()
    return xhat, np.sqrt(total_var)

def run_dps(trial, betas, alpha_bars, score_fn, mode, zeta, gen, T):
    """mode: 'orig' 原版(0.01阻尼+clamp) | 'beta' 标准得分修正 | 'beta_det_readout' 同beta但读出x̂0"""
    y_obs, A_obs, sigma_obs, x_T = trial
    x_t = x_T.clone()
    for t_idx in range(T-1, -1, -1):
        beta_t = betas[t_idx]
        ab = alpha_bars[t_idx]
        ab_safe = torch.clamp(ab, min=0.01)
        s = score_fn(x_t, t_idx)
        x_hat_0 = (x_t + (1.0 - ab_safe) * s) / torch.sqrt(ab_safe)
        res = y_obs - A_obs * x_hat_0
        g = A_obs * res / sigma_obs**2   # ∂(-log p(y|x̂0))/x̂0
        if mode == 'orig':
            u = 0.01 * torch.clamp(zeta * g, min=-100, max=100)
        else:  # 'beta' / 'beta_det_readout'
            u = beta_t * zeta * g
        x_t = x_t + 0.5 * beta_t * x_t + beta_t * s + u + torch.sqrt(beta_t) * torch.randn(1, generator=gen)
    if mode == 'beta_det_readout':
        ab = torch.clamp(alpha_bars[0], min=0.01)
        s0 = score_fn(x_t, 0)
        x0 = (x_t + (1.0 - ab) * s0) / torch.sqrt(ab)
    else:
        x0 = x_t
    return x0.item()

def run_doc_original(trial, betas, alpha_bars, score_fn, gen, T, n_iters=100, lr=0.01, reg_lambda=0.1):
    """与 13.5-1.py SimpleDOC 相同的优化（单一Adam实例），但噪声用独立generator"""
    y_obs, A_obs, sigma_obs, x_T = trial
    def terminal_cost(x0):
        return 0.5 * torch.sum((y_obs - A_obs * x0)**2) / sigma_obs**2
    def fwd(control, add_noise):
        x_t = x_T.clone()
        for t_idx in range(T-1, -1, -1):
            beta_t = betas[t_idx]
            s = score_fn(x_t, t_idx)
            drift = 0.5 * beta_t * x_t + beta_t * s
            u = control[t_idx]
            diff = torch.sqrt(beta_t) * torch.randn(1, generator=gen) if add_noise else 0.0
            x_t = x_t + drift + u + diff
        return x_t
    control = torch.zeros(T, requires_grad=True)
    opt = optim.Adam([control], lr=lr)
    for _ in range(n_iters):
        opt.zero_grad()
        x0 = fwd(control, add_noise=True)
        loss = terminal_cost(x0) + reg_lambda * torch.sum(control**2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        x0 = fwd(control.detach(), add_noise=False)
    return x0.item()

def make_trials(n, T, seed=42):
    rng = np.random.RandomState(seed)
    gen_t = torch.Generator().manual_seed(seed)
    trials = []
    for _ in range(n):
        comp = rng.choice(2, p=GM_WEIGHTS)
        x0_star = rng.normal(GM_MEANS[comp], GM_STDS[comp])
        A, sig = 2.0, 0.1
        y = A * x0_star + rng.randn() * sig
        x_T = torch.randn(1, generator=gen_t) * 3.0
        trials.append((torch.tensor(y), torch.tensor(A), sig, x_T))
    return trials

def summarize(name, errs):
    e = np.array(errs)
    return {"method": name, "mean_abs_err": round(float(e.mean()), 4),
            "std_abs_err": round(float(e.std()), 4), "n": len(e)}

if __name__ == "__main__":
    results = {}

    # ---------- 主对照：T=20, n=50 ----------
    T = 20
    betas, alpha_bars = build_schedule(T)
    score_fn = make_score(alpha_bars)
    trials = make_trials(50, T)

    # 理论基线（重放试验生成以获得 x0_star）
    rng = np.random.RandomState(42)
    x0_stars = []
    for _ in range(50):
        comp = rng.choice(2, p=GM_WEIGHTS)
        x0s = rng.normal(GM_MEANS[comp], GM_STDS[comp])
        y = 2.0 * x0s + rng.randn() * 0.1
        x0_stars.append((x0s, y))
    mmse_errs, psds = [], []
    for (x0s, y) in x0_stars:
        xhat, psd = mmse_baseline(x0s, y)
        mmse_errs.append(abs(xhat - x0s))
        psds.append(psd)

    variants = {
        'DPS_orig(0.01阻尼+clamp, ζ=1)': ('orig', 1.0),
        'DPS_beta ζ=0.1': ('beta', 0.1),
        'DPS_beta ζ=0.5': ('beta', 0.5),
        'DPS_beta ζ=0.2': ('beta', 0.2),
        'DPS_beta ζ=0.3': ('beta', 0.3),
        'DPS_beta ζ=0.7': ('beta', 0.7),
        'DPS_beta ζ=1.0': ('beta', 1.0),
        'DPS_beta ζ=1.0 确定性读出': ('beta_det_readout', 1.0),
    }
    for name, (mode, zeta) in variants.items():
        errs = []
        for k, trial in enumerate(trials):
            gen = torch.Generator().manual_seed(1000 + k)
            x0 = run_dps(trial, betas, alpha_bars, score_fn, mode, zeta, gen, T)
            errs.append(abs(x0 - x0_stars[k][0]))
        results[name] = summarize(name, errs)
        print(name, results[name])

    results['MMSE后验均值(理论最优)'] = summarize('MMSE', mmse_errs)
    results['MMSE后验均值(理论最优)']['后验std均值(参考下限)'] = round(float(np.mean(psds)), 4)
    print('MMSE', results['MMSE后验均值(理论最优)'])

    # DOC 原版（每试验独立generator）
    errs = []
    for k, trial in enumerate(trials):
        gen = torch.Generator().manual_seed(2000 + k)
        x0 = run_doc_original(trial, betas, alpha_bars, score_fn, gen, T)
        errs.append(abs(x0 - x0_stars[k][0]))
    results['DOC原版'] = summarize('DOC', errs)
    print('DOC原版', results['DOC原版'])

    # ---------- 大T检验：T=200, n=20 ----------
    T2 = 200
    betas2, alpha_bars2 = build_schedule(T2)
    score_fn2 = make_score(alpha_bars2)
    trials2 = make_trials(20, T2)
    rng = np.random.RandomState(42)
    x0_stars2 = []
    for _ in range(20):
        comp = rng.choice(2, p=GM_WEIGHTS)
        x0s = rng.normal(GM_MEANS[comp], GM_STDS[comp])
        y = 2.0 * x0s + rng.randn() * 0.1
        x0_stars2.append((x0s, y))

    errs = []
    for k, trial in enumerate(trials2):
        gen = torch.Generator().manual_seed(3000 + k)
        x0 = run_dps(trial, betas2, alpha_bars2, score_fn2, 'beta', 1.0, gen, T2)
        errs.append(abs(x0 - x0_stars2[k][0]))
    name = 'DPS_beta ζ=1 (T=200)'
    results[name] = summarize(name, errs)
    print(name, results[name])

    errs = []
    for k, trial in enumerate(trials2):
        gen = torch.Generator().manual_seed(4000 + k)
        x0 = run_doc_original(trial, betas2, alpha_bars2, score_fn2, gen, T2)
        errs.append(abs(x0 - x0_stars2[k][0]))
    name = 'DOC (T=200)'
    results[name] = summarize(name, errs)
    print(name, results[name])

    with open('sweep_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("done")
