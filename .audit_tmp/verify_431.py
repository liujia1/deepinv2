import numpy as np

def ULA_gauss(niter, delta, x0=0, burn_in=1000):
    if delta >= 2:
        raise ValueError(f"delta={delta} >= 2")
    X = x0
    for _ in range(burn_in):
        X = (1 - delta) * X + np.sqrt(2 * delta) * np.random.randn()
    Y = np.zeros(niter)
    for i in range(niter):
        X = (1 - delta) * X + np.sqrt(2 * delta) * np.random.randn()
        Y[i] = X
    return Y, np.var(Y), np.mean(Y)

np.random.seed(42)
niter = int(5e4)

print("=== 步骤1 delta=0.1 ===")
Y, v, m = ULA_gauss(niter, 0.1, 0)
print(f"var={v:.6f} mean={m:.6f}")

print("=== 步骤2 表 (seed=42 连续调用顺序) ===")
deltas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
for d in deltas:
    v1, m1 = None, None
    Y, var_emp, mean_emp = ULA_gauss(niter, d, x0=0)
    var_theo = 2 / (2 - d)
    rel = abs(var_emp - var_theo)/var_theo*100
    print(f"delta={d:>5} emp={var_emp:.6f} theo={var_theo:.6f} rel={rel:.2f}% mean={mean_emp:.4f}")

print("=== 步骤4 delta=1.0 ===")
Y_b, vb, mb = ULA_gauss(int(5e4), 1.0, x0=0)
print(f"emp_var={vb:.6f} theo_var={2/(2-1):.6f} mean={mb:.4f}")
