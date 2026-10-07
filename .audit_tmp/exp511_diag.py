import numpy as np
np.random.seed(42)

def ULA_gauss(niter, delta, x0=0):
    Y = np.zeros(niter,)
    X = x0
    for i in range(niter):
        Z = np.random.randn()
        grad = X
        X = X - delta * grad + np.sqrt(2 * delta) * Z
        Y[i] = X
    return Y

def compute_acf(x, nlags=50):
    x = x - np.mean(x)
    n = len(x)
    acf_vals = np.zeros(nlags + 1)
    for lag in range(nlags + 1):
        acf_vals[lag] = np.sum(x[:n-lag] * x[lag:]) / (n * np.var(x))
    return acf_vals

# Step1
niter = 100000
delta = 0.1
samples = ULA_gauss(niter, delta, 0)
print("=== Step1 ===")
print("empirical var:", np.var(samples))
print("theoretical var 1/(1-d/2):", 1/(1-delta/2))
acf = compute_acf(samples, nlags=60)
# find lag where acf first < 0.05
first_below = None
for k in range(1, len(acf)):
    if acf[k] < 0.05:
        first_below = k
        break
print("lag ACF first <0.05:", first_below)
print("acf at lag 10:", acf[10], " theoretical 0.9^10=", 0.9**10)
print("acf at lag 29:", acf[29], " theoretical 0.9^29=", 0.9**29)
print("acf first 15:", np.round(acf[:15],3))

# Step2
print("\n=== Step2 ===")
for d in [0.01, 0.1, 0.5, 1.0]:
    np.random.seed(42)
    s = ULA_gauss(50000, d, 0)
    print(f"delta={d}: emp_var={np.var(s):.4f} theo_var={1/(1-d/2):.4f}")
