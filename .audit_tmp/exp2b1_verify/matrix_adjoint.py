# 用显式矩阵构造算子，精确检验伴随性；并诊断 TGV 迭代发散的根源
import numpy as np

def gradient_forward(x):
    h, w = x.shape
    dx_h = np.zeros_like(x); dx_v = np.zeros_like(x)
    dx_h[:, :-1] = x[:, 1:] - x[:, :-1]
    dx_v[:-1, :] = x[1:, :] - x[:-1, :]
    return dx_h, dx_v

def divergence_backward(p_h, p_v):
    div = np.zeros(p_h.shape)
    div[:, 1:] += p_h[:, :-1]
    div -= p_h
    div[1:, :] += p_v[:-1, :]
    div -= p_v
    return div

def symmetrized_gradient_forward(w_h, w_v):
    Ew_1 = np.zeros_like(w_h); Ew_2 = np.zeros_like(w_v); Ew_3 = np.zeros_like(w_h)
    Ew_1[:, :-1] = w_h[:, 1:] - w_h[:, :-1]
    Ew_2[:-1, :] = w_v[1:, :] - w_v[:-1, :]
    d_wv_h = np.zeros_like(w_v); d_wh_v = np.zeros_like(w_h)
    d_wv_h[:, :-1] = w_v[:, 1:] - w_v[:, :-1]
    d_wh_v[:-1, :] = w_h[1:, :] - w_h[:-1, :]
    Ew_3 = 0.5 * (d_wv_h + d_wh_v)
    return Ew_1, Ew_2, Ew_3

def symmetrized_divergence_backward(q_1, q_2, q_3):
    div_w_h = np.zeros_like(q_1); div_w_v = np.zeros_like(q_2)
    div_w_h[:, 1:] += q_1[:, :-1]; div_w_h -= q_1
    div_w_v[1:, :] += q_2[:-1, :]; div_w_v -= q_2
    dq3_dh = np.zeros_like(q_3); dq3_dv = np.zeros_like(q_3)
    dq3_dh[:, 1:] += 0.5 * q_3[:, :-1]; dq3_dh -= 0.5 * q_3
    dq3_dv[1:, :] += 0.5 * q_3[:-1, :]; dq3_dv -= 0.5 * q_3
    div_w_h += dq3_dv; div_w_v += dq3_dh
    return div_w_h, div_w_v

N = 4
# 构造 D 矩阵：输入 x (N*N 展平)，输出 [Dx_h; Dx_v]
D = np.zeros((2*N*N, N*N))
for i in range(N):
    for j in range(N):
        idx = i*N + j
        if j < N-1:
            D[idx, idx+1] += 1; D[idx, idx] -= 1            # Dx_h 行 idx
            D[N*N+idx, idx] += 0; # placeholder
        if i < N-1:
            # Dx_v 行 N*N+idx:
            D[N*N+idx, idx+N] += 1; D[N*N+idx, idx] -= 1
# 修正 Dx_h 边界行保持0，已默认
# div 矩阵：输入 [p_h; p_v]，输出 div
Dv = np.zeros((N*N, 2*N*N))
for i in range(N):
    for j in range(N):
        idx = i*N + j
        # div = S_h(p_h) - p_h + S_v(p_v) - p_v
        # S_h(p)[i,j] = p[i,j-1] (j>=1)
        if j >= 1:
            Dv[idx, idx-1] += 1
        Dv[idx, idx] -= 1
        if i >= 1:
            Dv[idx, N*N + idx-N] += 1
        Dv[idx, N*N + idx] -= 1

err = np.max(np.abs(D.T + Dv))
print(f"[M1] ||D^T + div_matrix||_max = {err:.3e}  (应为0 => divergence_backward == -D^T)")

# E 矩阵与其伴随
E = np.zeros((3*N*N, 2*N*N)); Esym = np.zeros((2*N*N, 3*N*N))
for i in range(N):
    for j in range(N):
        idx = i*N + j
        if j < N-1:
            E[idx, idx+1] += 1; E[idx, idx] -= 1                    # E1 = ∂h w_h
        if i < N-1:
            E[N*N+idx, N*N+idx+N] += 1; E[N*N+idx, N*N+idx] -= 1    # E2 = ∂v w_v
        # E3 = 0.5(∂h w_v + ∂v w_h)
        if j < N-1:
            E[2*N*N+idx, N*N+idx+1] += 0.5; E[2*N*N+idx, N*N+idx] -= 0.5
        if i < N-1:
            E[2*N*N+idx, idx+N] += 0.5; E[2*N*N+idx, idx] -= 0.5
for i in range(N):
    for j in range(N):
        idx = i*N + j
        # 代码: div_w_h = S_h(q1)-q1 + [0.5(S_v(q3)-q3)]
        if j >= 1: Esym[idx, idx-1] += 1
        Esym[idx, idx] -= 1
        if i >= 1: Esym[idx, 2*N*N + idx-N] += 0.5
        Esym[idx, 2*N*N + idx] -= 0.5
        # div_w_v = S_v(q2)-q2 + [0.5(S_h(q3)-q3)]
        if i >= 1: Esym[N*N+idx, N*N+idx-N] += 1
        Esym[N*N+idx, N*N+idx] -= 1
        if j >= 1: Esym[N*N+idx, 2*N*N + idx-1] += 0.5
        Esym[N*N+idx, 2*N*N + idx] -= 0.5

err2 = np.max(np.abs(E.T + Esym))
print(f"[M2] ||E^T + symdiv_matrix||_max = {err2:.3e}  (应为0 => 代码symdiv == -E^T)")

# 函数级随机检验（防止矩阵构造与函数不一致，双保险）
rng = np.random.default_rng(1)
x = rng.standard_normal((N,N)); p1 = rng.standard_normal((N,N)); p2 = rng.standard_normal((N,N))
Dx = gradient_forward(x)
lhs = np.sum(Dx[0]*p1) + np.sum(Dx[1]*p2)
rhs = np.sum(x.reshape(-1) @ (D.T @ np.concatenate([p1.reshape(-1), p2.reshape(-1)])))
print(f"[M3] 函数 <Dx,p>={lhs:.6f} vs 矩阵 <x,D^T p>={rhs:.6f}; -<x,divp>={-np.sum(x*divergence_backward(p1,p2)):.6f}")

w1 = rng.standard_normal((N,N)); w2 = rng.standard_normal((N,N))
q1 = rng.standard_normal((N,N)); q2 = rng.standard_normal((N,N)); q3 = rng.standard_normal((N,N))
Em = symmetrized_gradient_forward(w1, w2)
lhsE = np.sum(Em[0]*q1) + np.sum(Em[1]*q2) + np.sum(Em[2]*q3)
dq = symmetrized_divergence_backward(q1, q2, q3)
rhsE = -np.sum(w1*dq[0] + w2*dq[1])
vec_test = np.concatenate([w1.reshape(-1),w2.reshape(-1)]) @ (E.T @ np.concatenate([q1.reshape(-1),q2.reshape(-1),q3.reshape(-1)]))
print(f"[M4] 函数 <Ew,q>={lhsE:.6f} vs -<w,symdiv q>={rhsE:.6f} vs 矩阵 <w,E^T q>={vec_test:.6f}")
