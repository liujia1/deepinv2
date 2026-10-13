import numpy as np

n = 100
noise_level = 1e-8
np.random.seed(42)
x_true = np.random.randn(n)

def rel_err(kappa, rng):
    sv = np.logspace(0, -np.log10(kappa), n)
    y = sv * x_true
    e = noise_level * rng.standard_normal(n)
    xr = (y + e) / sv
    return np.linalg.norm(xr - x_true) / np.linalg.norm(x_true), sv, y, e

print('=== 1) 上界松弛因子的解析预测 vs 实测 ===')
rng = np.random.default_rng(0)
for kappa in [1e3, 1e6, 1e8, 1e10]:
    sv = np.logspace(0, -np.log10(kappa), n)
    # 预测 ratio = E[实际]/上界 = (σ_min·sqrt(Σ1/σ_i²)/sqrt(n)) · (‖y/(σ_minx‖))·... 直接推导：
    # 实际 ≈ noise·sqrt(Σ (x_i²/‖x²)·(1/σ_i²) ... ) 用 MC 估计
    vals, ratios = [], []
    for _ in range(400):
        e = noise_level * np.random.randn(n)
        y = sv * x_true
        re = np.linalg.norm(e / sv) / np.linalg.norm(x_true)
        tb = kappa * np.linalg.norm(e) / np.linalg.norm(y)
        vals.append(re); ratios.append(re / tb)
    print(f'  kappa={kappa:8.0e}  实测ratio(单次seed42流)  MC均值ratio={np.mean(ratios):.4f} '
          f'解析 σ_min·sqrt(sum 1/sig^2)/sqrt(n)={sv[-1]*np.sqrt(np.sum(1/sv**2))/np.sqrt(n):.4f}')

print('\n=== 2) 相对误差=1 的临界 κ 扫描（每个 κ 独立种子，公平比较）===')
for kappa in [1e7, 3e7, 1e8, 3e8, 5e8, 7e8, 1e9, 3e9, 1e10]:
    r, *_ = rel_err(kappa, np.random.default_rng(7))
    print(f'  kappa={kappa:9.1e}  rel_err={r:.3e}')

print('\n=== 3) 双对数斜率（实测误差 vs κ）===')
kappas = [10.0**k for k in [0, 1, 2, 3, 4, 6, 7, 8, 10, 12]]
errs = []
np.random.seed(42)
_ = np.random.randn(n)
for kappa in kappas:
    sv = np.logspace(0, -np.log10(kappa), n)
    y = sv * x_true
    y_noisy = y + noise_level * np.random.randn(n)
    errs.append(np.linalg.norm(y_noisy / sv - x_true) / np.linalg.norm(x_true))
errs = np.array(errs)
for a, b in zip(range(len(kappas) - 1), range(1, len(kappas))):
    sl = np.log10(errs[b] / errs[a]) / np.log10(kappas[b] / kappas[a])
    print(f'  {kappas[a]:.0e}->{kappas[b]:.0e}: 斜率={sl:.2f}')

print('\n=== 4) 信号是否被"抹平"？（朴素逆重建对信号本身零失真的解析验证）===')
np.random.seed(123)
t = np.linspace(0, 1, n)
x_demo = np.sin(2*np.pi*6*t) + 0.5*np.cos(2*np.pi*16*t) + 0.35*np.sin(2*np.pi*28*t)
e_shared = noise_level * np.random.randn(n)
for kappa in [1e7, 1e10]:
    sv = np.logspace(0, -np.log10(kappa), n)
    xr = (sv * x_demo + e_shared) / sv
    err = xr - x_demo
    print(f'  kappa={kappa:.0e}: max|误差|={np.max(np.abs(err)):.3e}  误差/噪声放大预测max(|e|/sigma)={np.max(np.abs(e_shared/sv)):.3e}'
          f'  信号衰减项 x_demo*(1-1) 恒为0 → 无"抹平"，偏差全部来自噪声')
    idx = np.argmax(np.abs(err) > 1.0)
    print(f'      |误差|首次>1 的索引={idx} (t={t[idx]:.3f})  σ={sv[idx]:.2e}')

print('\n=== 5) 数据相对误差是否恒为 1e-8？（跨 κ 的公平性）===')
np.random.seed(42)
_ = np.random.randn(n)
for kappa in kappas:
    sv = np.logspace(0, -np.log10(kappa), n)
    y = sv * x_true
    e = noise_level * np.random.randn(n)
    print(f'  kappa={kappa:9.0e}  ‖y‖={np.linalg.norm(y):.3f}  ‖e‖/‖y‖={np.linalg.norm(e)/np.linalg.norm(y):.3e}')
