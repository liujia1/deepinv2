import numpy as np
N = 4; S = N*N

def grad_fwd(x):
    dh = np.zeros_like(x); dv = np.zeros_like(x)
    dh[:, :-1] = x[:, 1:] - x[:, :-1]
    dv[:-1, :] = x[1:, :] - x[:-1, :]
    return dh, dv

def sym_grad_fwd(w1, w2):
    E1 = np.zeros_like(w1); E2 = np.zeros_like(w2)
    E1[:, :-1] = w1[:, 1:] - w1[:, :-1]
    E2[:-1, :] = w2[1:, :] - w2[:-1, :]
    d_wv_h = np.zeros_like(w2); d_wh_v = np.zeros_like(w1)
    d_wv_h[:, :-1] = w2[:, 1:] - w2[:, :-1]
    d_wh_v[:-1, :] = w1[1:, :] - w1[:-1, :]
    E3 = 0.5*(d_wv_h + d_wh_v)
    return E1, E2, E3

def basis(k):
    e = np.zeros(S); e[k] = 1.0
    return e.reshape(N, N)

Dm = np.zeros((2*S, S)); Em = np.zeros((3*S, 2*S))
for k in range(S):
    dh, dv = grad_fwd(basis(k))
    Dm[:S, k] = dh.reshape(-1); Dm[S:, k] = dv.reshape(-1)
    r = sym_grad_fwd(basis(k), np.zeros((N,N)))
    Em[:S,k], Em[S:2*S,k], Em[2*S:,k] = r[0].reshape(-1), r[1].reshape(-1), r[2].reshape(-1)
    r = sym_grad_fwd(np.zeros((N,N)), basis(k))
    Em[:S,S+k], Em[S:2*S,S+k], Em[2*S:,S+k] = r[0].reshape(-1), r[1].reshape(-1), r[2].reshape(-1)

# 打印 D 的行/列结构片段，观察转置
np.set_printoptions(precision=1, suppress=True, linewidth=150)
print("D 的水平分量矩阵 D_h (行=(i,j)输出, 列=输入):")
print(Dm[:S, :S])
print("其转置 D_h^T:")
print(Dm[:S, :S].T)

# 从矩阵直接读出 -D^T 的切片公式
# 水平: (-D_h^T p)[j] = p[j] - p[j+1] (j<m-1), = p[m-1] (j=m-1)
#   => out[:, :-1] = p[:, :-1] - p[:, 1:] ; out[:, -1] = p[:, -1]
def div_new(p, q):
    out = np.zeros(p.shape)
    out[:, :-1] = p[:, :-1] - p[:, 1:]
    out[:, -1] = p[:, -1]
    out[:-1, :] += q[:-1, :] - q[1:, :]
    out[-1, :] += q[-1, :]
    return out

def sym_div_new(q1, q2, q3):
    # h: D_h^T q1 的负 + ...; 用 -E^T = 组合 div_new 的单分量版
    def mDT1(p):  # = -D_h^T p
        out = np.zeros(p.shape)
        out[:, :-1] = p[:, :-1] - p[:, 1:]
        out[:, -1] = p[:, -1]
        return out
    def mDT2(p):  # = -D_v^T p
        out = np.zeros(p.shape)
        out[:-1, :] = p[:-1, :] - p[1:, :]
        out[-1, :] = p[-1, :]
        return out
    return mDT1(q1) + 0.5*mDT2(q3), mDT2(q2) + 0.5*mDT1(q3)

# 矩阵级验证
DVn = np.zeros((S, 2*S)); SDn = np.zeros((2*S, 3*S))
for k in range(S):
    DVn[:S, k] = div_new(basis(k), np.zeros((N,N))).reshape(-1)
    DVn[:S, S+k] = div_new(np.zeros((N,N)), basis(k)).reshape(-1)
for c in range(3):
    for k in range(S):
        qs = [np.zeros((N,N)) for _ in range(3)]; qs[c] = basis(k)
        r = sym_div_new(*qs)
        SDn[:S, c*S+k] = r[0].reshape(-1); SDn[S:, c*S+k] = r[1].reshape(-1)
print("新div: ||DV_new + D^T|| =", np.abs(DVn + Dm.T).max())
print("新symdiv: ||SD_new + E^T|| =", np.abs(SDn + Em.T).max())

# 随机内积检验
rng = np.random.default_rng(1)
e1 = e2 = 0.0
for _ in range(20):
    x = rng.standard_normal((N,N)); p = rng.standard_normal((N,N)); q = rng.standard_normal((N,N))
    dh, dv = grad_fwd(x)
    e1 = max(e1, abs(np.sum(dh*p)+np.sum(dv*q) - np.sum(x*(-div_new(p,q)))))
    w1 = rng.standard_normal((N,N)); w2 = rng.standard_normal((N,N))
    q1 = rng.standard_normal((N,N)); q2 = rng.standard_normal((N,N)); q3 = rng.standard_normal((N,N))
    E = sym_grad_fwd(w1, w2); Sq = sym_div_new(q1, q2, q3)
    e2 = max(e2, abs(np.sum(E[0]*q1)+np.sum(E[1]*q2)+np.sum(E[2]*q3) - (-(np.sum(w1*Sq[0])+np.sum(w2*Sq[1])))))
print(f"随机检验: D-div err={e1:.2e}, E-symdiv err={e2:.2e}")
