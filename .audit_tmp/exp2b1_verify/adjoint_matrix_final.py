import numpy as np

N = 5
size = N * N

# 显式矩阵表示 D（按 gradient_forward 的行为）
D = np.zeros((2 * size, size))
for i in range(N):
    for j in range(N):
        idx = i * N + j
        if j < N - 1:
            D[idx, idx + 1] += 1.0
            D[idx, idx] -= 1.0
        if i < N - 1:
            D[size + idx, idx + N] += 1.0
            D[size + idx, idx] -= 1.0

rng = np.random.default_rng(11)
x = rng.standard_normal((N, N))
p = rng.standard_normal((N, N))
q = rng.standard_normal((N, N))

def D_fwd(x):
    dh = np.zeros_like(x); dv = np.zeros_like(x)
    dh[:, :-1] = x[:, 1:] - x[:, :-1]
    dv[:-1, :] = x[1:, :] - x[:-1, :]
    return dh, dv

# 验证矩阵 D 与函数一致
lhs_f = np.sum(D_fwd(x)[0] * p) + np.sum(D_fwd(x)[1] * q)
pv = np.concatenate([p.reshape(-1), q.reshape(-1)])
xv = x.reshape(-1)
lhs_m = pv @ (D @ xv)
print(f"矩阵/函数一致性: <Dx,p> func={lhs_f:.8f} mat={lhs_m:.8f}")

DT = D.T  # (2size,) -> size ; DT @ pv 是 -div 的真值（记 T = D^T p）
T = (DT @ pv).reshape(N, N)
print("真伴随 D^T p (等于算法里应使用的 -div):")
print(np.round(T, 3))

def div_code(p, q):
    div = np.zeros(p.shape)
    div[:, 1:] += p[:, :-1]
    div -= p
    div[1:, :] += q[:-1, :]
    div -= q
    return div

# 尝试各种候选，找与 D^T 匹配者
cands = {}
a = np.zeros((N, N)); a[:, :-1] -= p[:, 1:]; a += p; a[:, -1] -= p[:, -1]
a2 = np.zeros((N, N)); a2[:, :-1] -= p[:, 1:]; a2 += p  # 未补边界
b = np.zeros((N, N)); b[:, :-1] += p[:, 1:]; b -= p; b[:, -1] += p[:, -1]
cands["A: div[:, :-1] -= p[:,1:]; div += p"] = a2
cands["A+: 同A并补末列 -p[:,-1]"] = a
cands["B: 同A但符号整体相反补末列"] = b
for k, v in cands.items():
    vv = v.copy()
    print(f"[{k}] max|候选-真伴随| = {np.abs(vv - T).max():.3e}")

# 把 q 部分也纳入，完整对比 -div 候选:
def minus_div_full(p, q):
    out = np.zeros((N, N))
    out[:, :-1] -= p[:, 1:]
    out += p
    out[:, -1] -= p[:, -1]
    out[:-1, :] -= q[1:, :]
    out += q
    out[-1, :] -= q[-1, :]
    return out
print("候选(-div完整) vs 真伴随 max err =", np.abs(minus_div_full(p, q) - T).max())

def div_code_full(p, q):
    return -div_code(p, q)
print("代码 -div vs 真伴随 max err =", np.abs(div_code_full(p, q) - T).max())
