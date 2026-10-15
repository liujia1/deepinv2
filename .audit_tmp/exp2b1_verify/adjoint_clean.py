import numpy as np
np.set_printoptions(precision=3, suppress=True)

N = 4
size = N * N

def grad_fwd(x):
    dh = np.zeros_like(x); dv = np.zeros_like(x)
    dh[:, :-1] = x[:, 1:] - x[:, :-1]
    dv[:-1, :] = x[1:, :] - x[:-1, :]
    return dh, dv

# 用矩阵表示 grad_fwd（逐列施加基向量），避免手推索引出错
Dcols_h = np.zeros((size, size)); Dcols_v = np.zeros((size, size))
for idx in range(size):
    e = np.zeros((N, N)).reshape(-1); e[idx] = 1.0
    dh, dv = grad_fwd(e.reshape(N, N))
    Dcols_h[:, idx] = dh.reshape(-1); Dcols_v[:, idx] = dv.reshape(-1)
D = np.vstack([Dcols_h, Dcols_v])  # (2size, size)

def div_code(p, q):
    div = np.zeros(p.shape)
    div[:, 1:] += p[:, :-1]
    div -= p
    div[1:, :] += q[:-1, :]
    div -= q
    return div

rng = np.random.default_rng(5)
maxerr = 0.0
for t in range(20):
    x = rng.standard_normal((N, N))
    p = rng.standard_normal((N, N))
    q = rng.standard_normal((N, N))
    lhs = np.sum(grad_fwd(x)[0]*p) + np.sum(grad_fwd(x)[1]*q)
    rhs = -np.sum(x * div_code(p, q))
    maxerr = max(maxerr, abs(lhs - rhs))
print(f"[代码 divergence_backward] 伴随性最大误差 = {maxerr:.3e}")

# 真 -D^T 逐元素（通过矩阵）
x = rng.standard_normal((N,N)); p = rng.standard_normal((N,N)); q = rng.standard_normal((N,N))
pv = np.concatenate([p.reshape(-1), q.reshape(-1)])
minusDt = -(D.T @ pv).reshape(N, N)
print("真 -D^T p 矩阵:"); print(minusDt)
print("代码 div(p,q):"); print(div_code(p, q))

# 手写候选：正确应为 div_true[i,j] = p[i,j+1]-p[i,j] + q[i+1,j]-q[i,j]
cand = np.zeros((N,N))
cand[:, :-1] += p[:, 1:] - p[:, :-1]
cand[:, -1] -= p[:, -1]
cand[:-1, :] += q[1:, :] - q[:-1, :]
cand[-1, :] -= q[-1, :]
print("候选(p[j+1]-p[j]型):"); print(cand)
print("候选与 -D^T 误差:", np.abs(cand - minusDt).max())
print("代码div与 -D^T 误差:", np.abs(div_code(p,q) - minusDt).max())
