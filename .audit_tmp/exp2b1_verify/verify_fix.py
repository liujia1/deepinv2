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
    a = np.zeros_like(w2); b = np.zeros_like(w1)
    a[:, :-1] = w2[:, 1:] - w2[:, :-1]
    b[:-1, :] = w1[1:, :] - w1[:-1, :]
    E3 = 0.5*(a + b)
    return E1, E2, E3

def basis(k):
    e = np.zeros(S); e[k] = 1.0; return e.reshape(N, N)

# 真矩阵
Dm = np.zeros((2*S, S)); Em = np.zeros((3*S, 2*S))
for k in range(S):
    dh, dv = grad_fwd(basis(k)); Dm[:S,k]=dh.reshape(-1); Dm[S:,k]=dv.reshape(-1)
    r = sym_grad_fwd(basis(k), np.zeros((N,N))); Em[:S,k]=r[0].reshape(-1); Em[S:2*S,k]=r[1].reshape(-1); Em[2*S:,k]=r[2].reshape(-1)
    r = sym_grad_fwd(np.zeros((N,N)), basis(k)); Em[:S,S+k]=r[0].reshape(-1); Em[S:2*S,S+k]=r[1].reshape(-1); Em[2*S:,S+k]=r[2].reshape(-1)

# 候选 div1: 代码原版
def div_code(p, q):
    d = np.zeros(p.shape)
    d[:, 1:] += p[:, :-1]; d -= p
    d[1:, :] += q[:-1, :]; d -= q
    return d
# 候选 div2: 我推导的正确 -D^T
def div_fix(p, q):
    d = p.copy(); d[:, 1:] -= p[:, :-1]
    d2 = q.copy(); d2[1:, :] -= q[:-1, :]
    return d + d2

# 候选 symdiv 代码版
def sdiv_code(q1,q2,q3):
    dh = np.zeros_like(q1); dv = np.zeros_like(q2)
    dh[:,1:] += q1[:,:-1]; dh -= q1
    dv[1:,:] += q2[:-1,:]; dv -= q2
    a = np.zeros_like(q3); b = np.zeros_like(q3)
    a[:,1:] += 0.5*q3[:,:-1]; a -= 0.5*q3
    b[1:,:] += 0.5*q3[:-1,:]; b -= 0.5*q3
    dh += b; dv += a
    return dh, dv
# 候选 symdiv 正确版: -E^T
def sdiv_fix(q1,q2,q3):
    # -D_h^T q1  (h方向, 作用在w1)
    hh = q1.copy(); hh[:,1:] -= q1[:,:-1]
    # -D_v^T q2  (v方向, 作用在w2)
    vv = q2.copy(); vv[1:,:] -= q2[:-1,:]
    # E3 = 0.5(D_h w2 + D_v w1); 伴随贡献: 对w1: 0.5*(-D_v^T q3); 对w2: 0.5*(-D_h^T q3)
    t_v = q3.copy(); t_v[1:,:] -= q3[:-1,:]   # -D_v^T q3
    t_h = q3.copy(); t_h[:,1:] -= q3[:,:-1]   # -D_h^T q3
    out_h = hh + 0.5*t_v
    out_v = vv + 0.5*t_h
    return out_h, out_v

def mat_of_div(f):
    M = np.zeros((S, 2*S))
    for k in range(S):
        M[:S,k] = f(basis(k), np.zeros((N,N))).reshape(-1)
        M[:S,S+k] = f(np.zeros((N,N)), basis(k)).reshape(-1)
    return M
def mat_of_sdiv(f):
    M = np.zeros((2*S, 3*S))
    for c in range(3):
        for k in range(S):
            qs=[np.zeros((N,N)) for _ in range(3)]; qs[c]=basis(k)
            r=f(*qs); M[:S,c*S+k]=r[0].reshape(-1); M[S:,c*S+k]=r[1].reshape(-1)
    return M

print("=== div 是否等于 -D^T ===")
print("代码div:  ||DV_code + D^T|| =", np.abs(mat_of_div(div_code) + Dm.T).max())
print("修正div:  ||DV_fix  + D^T|| =", np.abs(mat_of_div(div_fix)  + Dm.T).max())
print("=== symdiv 是否等于 -E^T ===")
print("代码sdiv: ||SD_code + E^T|| =", np.abs(mat_of_sdiv(sdiv_code) + Em.T).max())
print("修正sdiv: ||SD_fix  + E^T|| =", np.abs(mat_of_sdiv(sdiv_fix)  + Em.T).max())
