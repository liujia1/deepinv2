import numpy as np
np.set_printoptions(precision=3, suppress=True)

N = 4
size = N * N

def grad_fwd(x):
    dh = np.zeros_like(x); dv = np.zeros_like(x)
    dh[:, :-1] = x[:, 1:] - x[:, :-1]
    dv[:-1, :] = x[1:, :] - x[:-1, :]
    return dh, dv

def div_code(p, q):
    div = np.zeros(p.shape)
    div[:, 1:] += p[:, :-1]
    div -= p
    div[1:, :] += q[:-1, :]
    div -= q
    return div

# 显式双重循环构造 D 矩阵（行=输出分量展平，列=输入像素）
D = np.zeros((2 * size, size))
for ii in range(N):
    for jj in range(N):
        r = ii * N + jj            # dh 分量 (ii,jj) 的行号
        if jj < N - 1:
            D[r, ii*N + jj+1] += 1.0
            D[r, ii*N + jj] -= 1.0
        r2 = size + ii * N + jj    # dv 分量
        if ii < N - 1:
            D[r2, (ii+1)*N + jj] += 1.0
            D[r2, ii*N + jj] -= 1.0

rng = np.random.default_rng(5)
ok = True
for t in range(50):
    x = rng.standard_normal((N, N)); p = rng.standard_normal((N, N)); q = rng.standard_normal((N, N))
    lhs = np.sum(grad_fwd(x)[0]*p) + np.sum(grad_fwd(x)[1]*q)
    rhs = -np.sum(x * div_code(p, q))
    if abs(lhs - rhs) > 1e-9:
        ok = False
print(f"divergence_backward 是 -D^T ? {ok}")

# 直接验证 D 矩阵与函数一致
x = rng.standard_normal((N,N)); p = rng.standard_normal((N,N)); q = rng.standard_normal((N,N))
xv = x.reshape(-1); pv = np.concatenate([p.reshape(-1), q.reshape(-1)])
Dx = D @ xv
print("D矩阵 vs grad_fwd:", np.abs(Dx[:size].reshape(N,N) - grad_fwd(x)[0]).max(),
      np.abs(Dx[size:].reshape(N,N) - grad_fwd(x)[1]).max())

# 真 -D^T 与代码 div 逐元素
true_minusDt = -(D.T @ pv).reshape(N, N)
codediv_pos = div_code(p, q)
print("-D^T 与 代码div 最大差:", np.abs(true_minusDt - codediv_pos).max())
print("-D^T:"); print(true_minusDt)
print("代码 div:"); print(codediv_pos)
