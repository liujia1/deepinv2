# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(42)
N, K, LAMBDA, SNR_DB = 50, 5, 0.6, 20

def make_sparse_signal(n, k, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n); idx = rng.choice(n, size=k, replace=False)
    x[idx] = rng.uniform(-2.5, 2.5, size=k); return x, idx

x_true, support = make_sparse_signal(N, K, seed=7)
sig = np.linalg.norm(x_true) / (10 ** (SNR_DB / 20))
noise = np.random.normal(0, sig, size=N)
y = x_true + noise

print("噪声std sig =", round(sig,4))
print("true support:", sorted(support.tolist()), "values:", np.round(x_true[support],4))
print("killed idx48: true x=%.4f obs y=%.4f |y|=%.4f vs lambda=%.2f" % (x_true[48], y[48], abs(y[48]), LAMBDA))

# per-coordinate SNR of each true coefficient
print("\n各真支撑坐标 |x|/noise_std:")
for i in sorted(support):
    print(f"  idx{i}: |x|={abs(x_true[i]):.4f}  |y|={abs(y[i]):.4f}  |x|/sig={abs(x_true[i])/sig:.2f}")

# noise-only entries: how many have |y| bigger than the killed coefficient's |y|?
noise_only = [i for i in range(N) if abs(x_true[i])<1e-9]
abs_y_killed = abs(y[48])
above = [i for i in noise_only if abs(y[i]) > abs_y_killed]
print(f"\n45个纯噪声(真0)分量中, |y| 大于被杀系数|y[48]|={abs_y_killed:.4f} 的有 {len(above)}/45 个")
print("=> 被杀系数在观测量级上排在噪声群里的位置: 比它大的噪声占 %.0f%%" % (100*len(above)/45))

# lambda sweep: does LASSO recall ever reach 100% while keeping sparsity?
def soft(y, lam): return np.sign(y)*np.maximum(np.abs(y)-lam,0)
print("\nλ扫描: (recall=真支撑非零比例, sparsity=置零数/45噪声)")
for lam in [0.02,0.05,0.1,0.2,0.3,0.4,0.6,0.8]:
    xl = soft(y, lam)
    rec = np.mean(np.abs(xl[support])>1e-9)
    nz_noise = sum(1 for i in noise_only if abs(xl[i])>1e-9)
    print(f"  λ={lam:.2f}: LASSO recall={rec:.0%}  噪声误报(非零)={nz_noise}/45  置零数={int(np.sum(np.abs(xl)<1e-9))}")

# Ridge degeneracy check
x_ridge = y/(1+LAMBDA)
print("\nRidge 非零个数 =", int(np.sum(np.abs(x_ridge)>1e-9)), "/", N, " => recall恒=100%%因它从不清零(误报45/45)")
