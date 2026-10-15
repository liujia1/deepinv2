import numpy as np
np.set_printoptions(precision=3, suppress=True)
N = 4; size = N*N

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

# 用线性性逐基向量构造 D 和 DIV 的矩阵
Dmat = np.zeros((2*size, size))
DVmat = np.zeros((size, 2*size))
for idx in range(size):
    e = np.zeros((N,N)); e.reshape(-1)[idx] = 1.0   # 注意: reshape(-1)返回视图, 可写
    dh, dv = grad_fwd(e)
    Dmat[:size, idx] = dh.reshape(-1); Dmat[size:, idx] = dv.reshape(-1)
for r in range(size):
    for c in range(2*size):
        pv = np.zeros(2*size); pv[c] = 1.0
        p = pv[:size].reshape(N,N); q = pv[size:].reshape(N,N)
        DVmat[r, c] = div_code(p, q).reshape(-1)[r]

print("D 矩阵与函数一致(抽查):", np.abs(Dmat[:, 5]).max())
e = np.zeros((N,N)); e[1,2] = 1.0
print("构造检查:", np.abs(grad_fwd(e)[0]).sum(), np.abs(grad_fwd(e)[1]).sum())

rel1 = np.abs(Dmat.T + DVmat).max()   # div = -D^T ?
print(f"||代码div矩阵 + D^T||_max = {rel1:.3e}   (0 => 代码div == -D^T, 伴随正确)")

# 随机双检验
rng = np.random.default_rng(9)
errs = []
for _ in range(30):
    x = rng.standard_normal((N,N)); p = rng.standard_normal((N,N)); q = rng.standard_normal((N,N))
    dh, dv = grad_fwd(x)
    lhs = np.sum(dh*p) + np.sum(dv*q)
    rhs = np.sum(x * (-div_code(p,q)))
    errs.append(abs(lhs - rhs))
print(f"随机检验 max|<Dx,p> - <x,-div p>| = {max(errs):.3e}")
