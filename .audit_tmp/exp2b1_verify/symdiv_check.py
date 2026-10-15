import numpy as np
N = 4; size = N*N

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

def sym_div_code(q1, q2, q3):
    div_w_h = np.zeros_like(q1); div_w_v = np.zeros_like(q2)
    div_w_h[:, 1:] += q1[:, :-1]; div_w_h -= q1
    div_w_v[1:, :] += q2[:-1, :]; div_w_v -= q2
    dq3_dh = np.zeros_like(q3); dq3_dv = np.zeros_like(q3)
    dq3_dh[:, 1:] += 0.5*q3[:, :-1]; dq3_dh -= 0.5*q3
    dq3_dv[1:, :] += 0.5*q3[:-1, :]; dq3_dv -= 0.5*q3
    div_w_h += dq3_dv; div_w_v += dq3_dh
    return div_w_h, div_w_v

def sym_div_true(q1, q2, q3):
    # -E^T q : h分量 = (D_h^T q1) + 0.5(D_v^T q3); v分量 = (D_v^T q2) + 0.5(D_h^T q3)
    def DT1(p):  # 单分量前向差分的转置
        out = np.zeros_like(p)
        out[:, :-1] -= p[:, 1:]; out[:, -1] = -p[:, -1]
        out += p  # out[j] = p[j] - p[j+1]
        return out
    def DT2(p):
        out = np.zeros_like(p)
        out[:-1, :] -= p[1:, :]; out[-1, :] = -p[-1, :]
        out += p
        return out
    return DT1(q1) + 0.5*DT2(q3), DT2(q2) + 0.5*DT1(q3)

# 矩阵级伴随检验：E vs sym_div
Emat = np.zeros((3*size, 2*size)); Scode = np.zeros((2*size, 3*size)); Stru = np.zeros((2*size, 3*size))
for a in range(N):
    for b in range(N):
        idx = a*N+b
        e1 = np.zeros((N,N)); e1[a,b]=1.0
        E = sym_grad_fwd(e1, np.zeros((N,N)))
        Emat[:size, idx] = E[0].reshape(-1); Emat[size:2*size, idx] = E[1].reshape(-1); Emat[2*size:, idx] = E[2].reshape(-1)
        e2 = np.zeros((N,N)); e2[a,b]=1.0
        E = sym_grad_fwd(np.zeros((N,N)), e2)
        Emat[:size, size+idx] = E[0].reshape(-1); Emat[size:2*size, size+idx] = E[1].reshape(-1); Emat[2*size:, size+idx] = E[2].reshape(-1)
        for c in range(3):
            base = np.zeros((N,N)); base[a,b] = 1.0
            qs = [np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N))]
            qs[c] = base
            sc = sym_div_code(*qs); st = sym_div_true(*qs)
            Scode[:size, c*size+idx] = sc[0].reshape(-1); Scode[size:, c*size+idx] = sc[1].reshape(-1)
            Stru[:size, c*size+idx] = st[0].reshape(-1); Stru[size:, c*size+idx] = st[1].reshape(-1)
print("||sym_div_code + E^T|| =", np.abs(Scode + Emat.T).max(), "  (0 => 代码版是伴随)")
print("||sym_div_true + E^T|| =", np.abs(Stru + Emat.T).max(), "  (0 => 我的true版是伴随)")

# 若 true 版不对，则用随机搜索找正确组合方向
rng = np.random.default_rng(4)
for trial in range(3):
    w1 = rng.standard_normal((N,N)); w2 = rng.standard_normal((N,N))
    q1 = rng.standard_normal((N,N)); q2 = rng.standard_normal((N,N)); q3 = rng.standard_normal((N,N))
    E = sym_grad_fwd(w1, w2)
    lhs = np.sum(E[0]*q1)+np.sum(E[1]*q2)+np.sum(E[2]*q3)
    st = sym_div_true(q1,q2,q3)
    print(f"  trial{trial}: <Ew,q>={lhs:.6f} vs -<w,sym_div_true>={-(np.sum(w1*st[0])+np.sum(w2*st[1])):.6f}")
