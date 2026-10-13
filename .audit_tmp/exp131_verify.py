import numpy as np, os

np.random.seed(42)
n = 100
x_true = np.random.randn(n)
kappas = [10.0**k for k in [0, 1, 2, 3, 4, 6, 7, 8, 10, 12]]
noise_level = 1e-8
sv_cache = {k: np.logspace(0, -np.log10(k), n) for k in kappas}

print("=== 主循环 (seed=42) ===")
print(f"{'kappa':>10} {'rel_err':>14} {'theory':>14} {'err/theory':>11} {'digits_lost':>12}")
for kappa in kappas:
    sv = sv_cache[kappa]
    y = sv * x_true
    y_norm = np.linalg.norm(y)
    y_noisy = y + noise_level * np.random.randn(n)
    x_recon = y_noisy / sv
    rel = np.linalg.norm(x_recon - x_true) / np.linalg.norm(x_true)
    dy = np.linalg.norm(y_noisy - y)
    tb = kappa * dy / y_norm
    dl = None if rel >= 1 else -np.log10(rel)
    print(f"{kappa:10.0e} {rel:14.6e} {tb:14.6e} {rel/tb:11.3e} "
          f"{('完全失真' if rel>=1 else f'remaining={dl:.2f},lost={8-round(dl)}'):>12}")
    print(f"           y_norm={y_norm:.4e}  rel_noise_data={dy/y_norm:.4e}  max_comp_err={np.max(np.abs(x_recon-x_true)):.4e}")

print("\n=== 附图 (seed=123, 共用噪声) ===")
np.random.seed(123)
t_demo = np.linspace(0, 1, n)
x_demo = (np.sin(2*np.pi*6*t_demo) + 0.5*np.cos(2*np.pi*16*t_demo)
          + 0.35*np.sin(2*np.pi*28*t_demo))
e_shared = noise_level * np.random.randn(n)
for kappa in [1e7, 1e10]:
    sv = sv_cache[kappa]
    y_noisy = sv * x_demo + e_shared
    xr = y_noisy / sv
    rel = np.linalg.norm(xr - x_demo) / np.linalg.norm(x_demo)
    err = xr - x_demo
    idx_first_over1 = np.argmax(np.abs(err) > 1.0)
    print(f"kappa={kappa:.0e} rel={rel:.6e} max|err|={np.max(np.abs(err)):.4e} "
          f"first idx where |err|>1: {idx_first_over1} (t={t_demo[idx_first_over1]:.3f}, "
          f"sigma={sv[idx_first_over1]:.3e})")
    print(f"   |xr| range = [{xr.min():.3e}, {xr.max():.3e}]  超出ylim(±1.6)点数={np.sum(np.abs(xr)>1.6)}")

print("\n=== 理论：误差开始超过1的索引（解析） ===")
for kappa in [1e7, 1e10]:
    sv = sv_cache[kappa]
    # 分量误差 ~ noise_level*(randn)/sv, 典型幅值 noise_level/sv
    ratio = noise_level / sv
    idx = np.argmax(ratio > 1.0)
    print(f"kappa={kappa:.0e}: noise_level/sigma_i>1 从 idx={idx} (t={idx/(n-1):.3f}) 开始, sigma={sv[idx]:.3e}")

print("\n=== 检查 baseline_digits 注释 vs 实际 ===")
print("round(-log10(noise_level)) =", round(-np.log10(noise_level)))

print("\n=== 浮点下溢/精度检查：kappa=1e12 时最小奇异值与除法 ===")
sv = sv_cache[1e12]
print("sigma_min =", sv[-1], " 1/sigma_min =", 1/sv[-1], " float64 eps =", np.finfo(float).eps)
print("x_recon 最大分量量级(kappa=1e12):", np.max(np.abs((sv*np.random.RandomState(0).randn(n))/sv)))
