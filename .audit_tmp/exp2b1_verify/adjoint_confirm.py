import numpy as np
rng = np.random.default_rng(3)
N = 6
X = [rng.standard_normal((N, N)) for _ in range(8)]
P = [rng.standard_normal((N, N)) for _ in range(8)]
Q = [rng.standard_normal((N, N)) for _ in range(8)]

def D_fwd(x):
    dh = np.zeros_like(x); dv = np.zeros_like(x)
    dh[:, :-1] = x[:, 1:] - x[:, :-1]
    dv[:-1, :] = x[1:, :] - x[:-1, :]
    return dh, dv

# 代码版 div
def div_code(p, q):
    div = np.zeros(p.shape)
    div[:, 1:] += p[:, :-1]
    div -= p
    div[1:, :] += q[:-1, :]
    div -= q
    return div

# 候选修正版（镜像索引）
def div_fix(p, q):
    div = np.zeros(p.shape)
    div[:, :-1] -= p[:, 1:]
    div += p
    div[:-1, :] -= q[1:, :]
    div += q
    return div

for name, f in [("code", div_code), ("fix", div_fix)]:
    err = max(abs(np.sum(D_fwd(x)[0]*p + D_fwd(x)[1]*q) - np.sum(x*(-f(p, q))))
              for x, p, q in zip(X, P, Q))
    print(f"[{name}] max |<Dx,p> - <x,-div(p)>| = {err:.3e}")
