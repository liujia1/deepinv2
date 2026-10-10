"""
诊断脚本：复现 4.2-1 步骤2 的细致平衡验证，检查 max_error 是否真的来自低访问区间。
严格按 4.2-1.py 的 RNG 消耗顺序：seed(42) -> 步骤1 mh_sampler -> 步骤2 verify。
"""
import numpy as np


def target_density(x):
    x_sq = np.clip(x ** 2, None, 1400)
    return np.exp(-0.5 * x_sq)


def log_target_density(x):
    return -0.5 * np.clip(x ** 2, None, 1400)


def mh_sampler(target, log_target=None, n_samples=50000, sigma=1.0, x0=0.0, burn_in=5000):
    x = x0
    samples_full = []
    n_accept_full = 0
    n_accept_post = 0
    for i in range(n_samples):
        proposal = x + sigma * np.random.randn()
        if log_target is not None:
            log_alpha = min(0.0, log_target(proposal) - log_target(x))
        else:
            log_alpha = min(0.0, np.log(target(proposal)) - np.log(target(x)))
        alpha = np.exp(log_alpha)
        if np.random.rand() < alpha:
            x = proposal
            n_accept_full += 1
            if i >= burn_in:
                n_accept_post += 1
        samples_full.append(x)
    samples_full = np.array(samples_full)
    samples_post = samples_full[burn_in:]
    accept_rate_full = n_accept_full / n_samples
    accept_rate_post = n_accept_post / (n_samples - burn_in) if n_samples > burn_in else 0.0
    return samples_full, samples_post, accept_rate_post, accept_rate_full


def verify_detailed_balance_empirical(target, log_target=None, n_samples=100000, sigma=1.0,
                                      n_bins=20, x_range=(-3, 3), burn_in=5000):
    bins = np.linspace(x_range[0], x_range[1], n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    transition_counts = np.zeros((n_bins, n_bins))
    state_counts = np.zeros(n_bins)
    x = 0.0
    for _ in range(burn_in):
        proposal = x + sigma * np.random.randn()
        if log_target is not None:
            log_alpha = min(0.0, log_target(proposal) - log_target(x))
        else:
            log_alpha = min(0.0, np.log(target(proposal)) - np.log(target(x)))
        alpha = np.exp(log_alpha)
        if np.random.rand() < alpha:
            x = proposal
    for _ in range(n_samples - burn_in):
        proposal = x + sigma * np.random.randn()
        if log_target is not None:
            log_alpha = min(0.0, log_target(proposal) - log_target(x))
        else:
            log_alpha = min(0.0, np.log(target(proposal)) - np.log(target(x)))
        alpha = np.exp(log_alpha)
        if np.random.rand() < alpha:
            new_x = proposal
        else:
            new_x = x
        x_bin = np.digitize(x, bins) - 1
        new_x_bin = np.digitize(new_x, bins) - 1
        if 0 <= x_bin < n_bins and 0 <= new_x_bin < n_bins:
            transition_counts[x_bin, new_x_bin] += 1
            state_counts[x_bin] += 1
        x = new_x
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    empirical_transition = transition_counts / row_sums
    total_visits = state_counts.sum()
    empirical_probs = state_counts / total_visits if total_visits > 0 else np.ones(n_bins) / n_bins
    balance_errors = []
    balance_pairs = []
    for i in range(n_bins):
        for j in range(i + 1, n_bins):
            if empirical_transition[i, j] > 0 and empirical_transition[j, i] > 0:
                lhs = empirical_probs[i] * empirical_transition[i, j]
                rhs = empirical_probs[j] * empirical_transition[j, i]
                error = np.abs(lhs - rhs) / (lhs + rhs + 1e-10)
                balance_errors.append(error)
                balance_pairs.append({
                    'i': i, 'j': j,
                    'ci': bin_centers[i], 'cj': bin_centers[j],
                    'p_i': empirical_probs[i], 'p_j': empirical_probs[j],
                    'N_ij': transition_counts[i, j], 'N_ji': transition_counts[j, i],
                    'visits_i': state_counts[i], 'visits_j': state_counts[j],
                    'lhs': lhs, 'rhs': rhs, 'error': error
                })
    return {
        'transition_matrix': empirical_transition,
        'empirical_probs': empirical_probs,
        'balance_errors': np.array(balance_errors),
        'balance_pairs': balance_pairs,
        'bin_centers': bin_centers,
        'mean_error': np.mean(balance_errors) if balance_errors else 0.0,
        'max_error': np.max(balance_errors) if balance_errors else 0.0,
    }


# ---- 严格复现 RNG 顺序 ----
np.random.seed(42)
# 步骤1（消耗 RNG）
_, sp1, ar1p, ar1f = mh_sampler(target_density, log_target_density, n_samples=50000, sigma=1.0, x0=5.0, burn_in=5000)
print(f"[步骤1] mean={np.mean(sp1):.6f} var={np.var(sp1):.6f} accept_post={ar1p:.4f}")

# 步骤2
db = verify_detailed_balance_empirical(target_density, log_target_density, n_samples=200000,
                                       sigma=1.0, n_bins=20, x_range=(-3, 3), burn_in=5000)
print(f"[步骤2] mean_error={db['mean_error']:.4f} max_error={db['max_error']:.4f} n_pairs={len(db['balance_pairs'])}")

pairs = sorted(db['balance_pairs'], key=lambda p: p['error'], reverse=True)
print("\n误差最大的前 8 个区间对（按 error 降序）：")
print(f"{'rank':>4} {'bin_i':>5} {'bin_j':>5} {'c_i':>6} {'c_j':>6} {'N_ij':>6} {'N_ji':>6} {'visits_i':>9} {'visits_j':>9} {'error':>7}")
for r, p in enumerate(pairs[:8]):
    print(f"{r:>4} {p['i']:>5} {p['j']:>5} {p['ci']:>6.2f} {p['cj']:>6.2f} "
          f"{p['N_ij']:>6.0f} {p['N_ji']:>6.0f} {p['visits_i']:>9.0f} {p['visits_j']:>9.0f} {p['error']:>7.4f}")

# 全体误差与访问量的相关性
errs = np.array([p['error'] for p in db['balance_pairs']])
minvis = np.array([min(p['visits_i'], p['visits_j']) for p in db['balance_pairs']])
print(f"\n误差 vs 该对最小访问次数 的 Pearson 相关: {np.corrcoef(errs, minvis)[0,1]:.4f}")
print(f"访问次数中位数: {np.median(np.array([p['visits_i'] for p in db['balance_pairs']])):.0f}")
top5_minvis = np.median(minvis[:5])
all_minvis = np.median(minvis)
print(f"Top5 高误差对的最小访问次数中位数: {top5_minvis:.0f}  vs  全体最小访问次数中位数: {all_minvis:.0f}")
