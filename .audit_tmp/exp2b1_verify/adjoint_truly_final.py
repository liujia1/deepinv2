import numpy as np
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

# 构造 D 矩阵：逐基向量，显式赋值，不依赖 reshape 视图
Dmat = np.zeros((2*size, size))
for ii in range(N):
    for jj in range(N):
        idx = ii*N + jj
        e = np.zeros((N, N)); e[ii, jj] = 1.0
        dh, dv = grad_fwd(e)
        Dmat[:size, idx] = dh.reshape(-1)
        Dmat[size:, idx] = dv.reshape(-1)

# 构造 DIV 矩阵：输入 [p; q]，输出 div
DVmat = np.zeros((size, 2*size))
for a in range(N):
    for b in range(N):
        idx = a*N + b
        ep = np.zeros((N, N)); ep[a, b] = 1.0
        DVa = div_code(ep, np.zeros((N, N)))
        DVmat[:size, idx] = DVa.reshape(-1)
        eq = np.zeros((N, N)); eq[a, b] = 1.0
        DVb = div_code(np.zeros((N, N)), eq)
        DVmat[:size, size + idx] = DVb.reshape(-1)

print("||代码div矩阵 - (-D^T)||_max =", np.abs(DVmat + Dmat.T).max())
print("||代码div矩阵 - (+D^T)||_max =", np.abs(DVmat - Dmat.T).max())

# 打印一个小算例对照
p = np.array([[1.,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
q = np.zeros((N,N))
pv = np.concatenate([p.reshape(-1), q.reshape(-1)])
print("代码 div(p,0):"); print(div_code(p, q))
print("-D^T p:"); print(-(Dmat.T @ pv).reshape(N,N))
print("+D^T p:"); print((Dmat.T @ pv).reshape(N,N))
