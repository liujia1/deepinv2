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

rng = np.random.default_rng(7)
N = 5
for trial in range(3):
    x = rng.standard_normal((N, N))
    p = rng.standard_normal((N, N)); q = rng.standard_normal((N, N))
    Dx = gradient_forward(x)
    lhs = np.sum(Dx[0]*p) + np.sum(Dx[1]*q)
    dvp = divergence_backward(p, q)
    rhs = -np.sum(x * dvp)
    print(f"trial{trial}: <Dx,p>={lhs:.8f}  -<x,div p>={rhs:.8f}  diff={lhs-rhs:.3e}")

# 逐元素核对 div 公式
x = np.arange(25).reshape(5,5).astype(float)
p = np.ones((5,5)); q = np.zeros((5,5))
div = divergence_backward(p, q)
print("div with p=ones,q=zeros:"); print(div)
# 期望: div[i,j] = p[i,j-1] - p[i,j] = 0 除 j=0 处 = -p[i,0] = -1
print("gradient_forward on arange:")
Dh, Dv_ = gradient_forward(x)
print(Dh); print(Dv_)
