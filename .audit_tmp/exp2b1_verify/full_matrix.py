import numpy as np
N = 4; S = N*N

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
    dw_h = np.zeros_like(q1); dw_v = np.zeros_like(q2)
    dw_h[:, 1:] += q1[:, :-1]; dw_h -= q1
    dw_v[1:, :] += q2[:-1, :]; dw_v -= q2
    a = np.zeros_like(q3); b = np.zeros_like(q3)
    a[:, 1:] += 0.5*q3[:, :-1]; a -= 0.5*q3
    b[1:, :] += 0.5*q3[:-1, :]; b -= 0.5*q3
    dw_h += b; dw_v += a
    return dw_h, dw_v

def basis(k, shape=(N,N)):
    total = int(np.prod(shape))
    e = np.zeros(total); e[k] = 1.0
    return e.reshape(shape)

# D: (2S, S)
Dm = np.zeros((2*S, S))
for k in range(S):
    dh, dv = grad_fwd(basis(k))
    Dm[:S, k] = dh.reshape(-1); Dm[S:, k] = dv.reshape(-1)
# DIV: (S, 2S)
DV = np.zeros((S, 2*S))
for k in range(S):
    DV[:S, k] = div_code(basis(k), np.zeros((N,N))).reshape(-1)
    DV[:S, S+k] = div_code(np.zeros((N,N)), basis(k)).reshape(-1)
# E: (3S, 2S)
Em = np.zeros((3*S, 2*S))
for k in range(S):
    r = sym_grad_fwd(basis(k), np.zeros((N,N)))
    Em[:S, k], Em[S:2*S, k], Em[2*S:, k] = r[0].reshape(-1), r[1].reshape(-1), r[2].reshape(-1)
    r = sym_grad_fwd(np.zeros((N,N)), basis(k))
    Em[:S, S+k], Em[S:2*S, S+k], Em[2*S:, S+k] = r[0].reshape(-1), r[1].reshape(-1), r[2].reshape(-1)
# SDIV: (2S, 3S)
SD = np.zeros((2*S, 3*S))
for c in range(3):
    for k in range(S):
        qs = [np.zeros((N,N)) for _ in range(3)]; qs[c] = basis(k)
        r = sym_div_code(*qs)
        SD[:S, c*S+k] = r[0].reshape(-1); SD[S:, c*S+k] = r[1].reshape(-1)

print("== 抽样验证矩阵与函数一致 ==")
rng = np.random.default_rng(0)
x = rng.standard_normal((N,N)); p = rng.standard_normal((N,N)); q = rng.standard_normal((N,N))
w1 = rng.standard_normal((N,N)); w2 = rng.standard_normal((N,N))
q1 = rng.standard_normal((N,N)); q2 = rng.standard_normal((N,N)); q3 = rng.standard_normal((N,N))
xv = x.reshape(-1); pv = np.concatenate([p.reshape(-1), q.reshape(-1)])
dh, dv = grad_fwd(x)
print("D ok:", np.abs(Dm@xv - np.concatenate([dh.reshape(-1), dv.reshape(-1)])).max())
print("DIV ok:", np.abs(DV@pv - div_code(p,q).reshape(-1)).max())
wv = np.concatenate([w1.reshape(-1), w2.reshape(-1)]); qv = np.concatenate([q1.reshape(-1), q2.reshape(-1), q3.reshape(-1)])
e = sym_grad_fwd(w1,w2)
print("E ok:", np.abs(Em@wv - np.concatenate([e[0].reshape(-1), e[1].reshape(-1), e[2].reshape(-1)])).max())
s = sym_div_code(q1,q2,q3)
print("SDIV ok:", np.abs(SD@qv - np.concatenate([s[0].reshape(-1), s[1].reshape(-1)])).max())

print("== 伴随性 ==")
print("||DV + D^T|| =", np.abs(DV + Dm.T).max(), " (0 => 代码div == -D^T)")
print("||SD + E^T|| =", np.abs(SD + Em.T).max(), " (0 => 代码symdiv == -E^T)")

# 若失败，找出代码div与真伴随的差在哪
diff = DV + Dm.T
nz = np.argwhere(np.abs(diff) > 1e-12)
print("DV + D^T 非零元素位置(行=像素idx, 列=[p分量;q分量]):")
for r, c in nz[:12]:
    print(f"   px=({r//N},{r%N}) comp={'p_h' if c<S else 'p_v'}({(c%S)//N},{(c%S)%N}) val={diff[r,c]:+.1f}")
