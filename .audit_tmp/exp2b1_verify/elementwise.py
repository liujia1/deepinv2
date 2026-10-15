import numpy as np

def gradient_forward(x):
    h, w = x.shape
    dx_h = np.zeros_like(x); dx_v = np.zeros_like(x)
    dx_h[:, :-1] = x[:, 1:] - x[:, :-1]
    dx_v[:-1, :] = x[1:, :] - x[:-1, :]
    return dx_h, dx_v

def divergence_backward(p_h, p_v):
    h, w = p_h.shape
    div = np.zeros((h, w))
    div[:, 1:] += p_h[:, :-1]
    div -= p_h
    div[1:, :] += p_v[:-1, :]
    div -= p_v
    return div

N = 5
x = np.arange(25).reshape(N, N).astype(float)
p = np.ones((N, N)); q = np.zeros((N, N))

# 手工按公式构 D^T p
Dtp = np.zeros((N, N))
# (D^T p)[i,j] = -p[i,j] + p[i,j+1] (若j+1<N) 垂直部分: -q[i,j] + q[i+1,j]
for i in range(N):
    for j in range(N):
        v = -p[i, j] + q[i, j]
        if j + 1 < N: v += p[i, j+1]
        if i + 1 < N: v -= q[i+1, j]
        Dtp[i, j] = v
print("手工 D^T p:"); print(Dtp)
div = divergence_backward(p, q)
print("函数 divergence_backward(p,q):"); print(div)
print("两者和 D^T p + div =", np.abs(Dtp + div).max())

# 伴随性终极检验
Dx = gradient_forward(x)
lhs = np.sum(Dx[0]*p) + np.sum(Dx[1]*q)
print(f"<Dx,p>={lhs}, <x,D^T p>={np.sum(x*Dtp)}, -<x,div>={-np.sum(x*div)}")
