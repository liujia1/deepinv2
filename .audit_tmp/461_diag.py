# -*- coding: utf-8 -*-
"""实验4.6-1 复核诊断（修订版：稀疏求解）"""
import numpy as np
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys, io, builtins
_out = open(r'd:\works\deepinv\.audit_tmp\461_diag_out.txt', 'w', encoding='utf-8')
_raw_print = builtins.print
def print(*a, **k):
    _raw_print(*a, file=_out, **k); _out.flush()

def Du(u):
    M, N = u.shape
    d = np.zeros((2, M, N))
    d[0, :, :-1] = u[:, 1:] - u[:, :-1]
    d[1, :-1, :] = u[1:, :] - u[:-1, :]
    return d

def DTp(p):
    C, M, N = p.shape
    out = np.zeros((M, N))
    out[:, 1:] += p[0, :, :-1]
    out[:, :-1] -= p[0, :, :-1]
    out[1:, :] += p[1, :-1, :]
    out[:-1, :] -= p[1, :-1, :]
    return out

# ---- 1. 伴随性 ----
rng = np.random.default_rng(0)
u_t = rng.standard_normal((23, 31)); p_t = rng.standard_normal((2, 23, 31))
lhs = np.sum(Du(u_t) * p_t); rhs = np.sum(u_t * DTp(p_t))
print(f"[1] 伴随性: <Du,p>={lhs:.6f} <u,DTp>={rhs:.6f} 相等={np.isclose(lhs,rhs)}")

# ---- 2. 数据与参数 ----
N = 100
x = np.linspace(-1, 1, N); y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)
u1 = 0*X + 0.5
u2 = X + Y; u2 -= u2.min(); u2 /= u2.max()
u3 = np.clip(1 - ((1-X)**2 + (1-Y)**2), a_min=0, a_max=None); u3 -= u3.min(); u3 /= u3.max()
m1 = 0*X; m1[N//8:-N//8, N//8:-N//8] = 1
u = u1.copy(); u[m1==1] = u2[m1==1]*1.25
m2 = 0*X; m2[(X-1)**2 + (Y-1)**2 <= 1] = 1
u[m2==1] = u3[m2==1]
sigma = 0.1
np.random.seed(42)
f = u + np.random.randn(N, N) * sigma
lamb, eps, theta = 10.0, 1e-3, 0.99
L = lamb * 8 / eps + 1 / sigma**2
tau = 2 / L
print(f"[2] L={L:.1f} tau={tau:.3e} 噪声std/sigma=0.1 PSNR理论={10*np.log10(1/sigma**2):.2f} u范围=[{u.min():.3f},{u.max():.3f}]")

def grad_E(v):
    d = Du(v)
    return lamb * DTp(d / np.clip(np.abs(d), eps, None)) + (v - f) / sigma**2
def energy(v):
    return lamb * np.abs(Du(v)).sum() + np.sum((v - f)**2) / (2 * sigma**2)
def psnr(a_, b_):
    mse = np.mean((a_ - b_)**2)
    return 10*np.log10(1.0/mse) if mse > 0 else float('inf')

def run_map(maxit=10000):
    u_map = f.copy(); u_old = f.copy()
    for it in range(maxit):
        g = grad_E(u_map)
        u_i = u_map + theta*(u_map - u_old); u_old = u_map.copy()
        u_map = u_i - tau*g
    return u_map

def run_ila(maxit=10000, burn=1000, seed=42, keep_chain=False):
    g_t = torch.Generator().manual_seed(seed)
    u_ila = f.copy(); u_old = f.copy()
    s1 = np.zeros((N, N)); s2 = np.zeros((N, N))
    chain = [] if keep_chain else None
    for it in range(maxit):
        g = grad_E(u_ila)
        u_i = u_ila + theta*(u_ila - u_old); u_old = u_ila.copy()
        noise = torch.randn(N, N, generator=g_t).numpy() * np.sqrt(2*tau*(1-theta))
        u_ila = u_i - tau*g + noise
        if it >= burn:
            s1 += u_ila; s2 += u_ila**2
            if chain is not None: chain.append(u_ila.copy())
    n = maxit - burn
    avg = s1/n; var = s2/n - avg**2
    return avg, var, (np.stack(chain) if chain is not None else None)

u_map = run_map()
e_map = energy(u_map)
print(f"[2] MAP(重球10k): energy={e_map:.4f} psnr={psnr(u_map,u):.2f}")

# 收敛判据：接着再跑10k步看能量是否漂移
u_c = u_map.copy(); u_o = u_map.copy()
# 需要一致的动量状态：用前一步近似=当前步
for it in range(10000):
    g = grad_E(u_c)
    u_i = u_c + theta*(u_c - u_o); u_o = u_c.copy()
    u_c = u_i - tau*g
print(f"[2] MAP再延10k: energy={energy(u_c):.4f} (drift={energy(u_c)-e_map:+.4f}) psnr={psnr(u_c,u):.2f}")

# 无动量慢GD参考
v = f.copy()
for it in range(200000):
    v = v - 0.6 * grad_E(v) / L
print(f"[2] GD参考(200k步,0.6/L): energy={energy(v):.4f} psnr={psnr(v,u):.2f} ||v-u_map||/||u_map||={np.linalg.norm(v-u_map)/np.linalg.norm(u_map):.3e}")

# ---- 3. 高斯近似精确后验（稀疏）----
d_map = Du(u_map)
w = lamb / np.clip(np.abs(d_map), eps, None)   # shape (2,N,N)
w_col = w[0]  # 权重对应列向前差分 u[:,j+1]-u[:,j]
w_row = w[1]  # 权重对应行向前差分 u[i+1,:]-u[i,:]
D1 = sp.diags([-np.ones(N-1), np.ones(N-1)], [0, 1], shape=(N-1, N), format='csr')
IN = sp.eye(N, format='csr')
# 列差分算子（沿内层j）: kron(I_i, D1_j) -> 输出 index i*(N-1)+j
Bc = sp.kron(IN, D1, format='csr')
Wc = sp.diags(w_col[:, :-1].ravel(order='C'), 0, format='csr')
H1 = Bc.T @ Wc @ Bc
# 行差分算子（沿外层i）: kron(D1_i, I_j) -> 输出 index i*N+col
Br = sp.kron(D1, IN, format='csr')
Wr = sp.diags(w_row[:-1, :].ravel(order='C'), 0, format='csr')
H2 = Br.T @ Wr @ Br
H = (H1 + H2 + sp.eye(N*N, format='csr') / sigma**2).tocsc()
# 数值校验: 与逐列算子作用对比
Et = np.eye(N*N).ravel()
Hcol = np.zeros((N*N, 6))
for jj, j in enumerate([0, 1234, 5050, 9999, 42, 7777]):
    e_j = Et[j*(N*N):(j+1)*(N*N)].reshape(N, N)
    Hcol[:, jj] = (DTp(w * Du(e_j)) + e_j / sigma**2).ravel()
err = max(abs(H[:, j2].toarray().ravel() - Hcol[:, k]).max() for k, j2 in enumerate([0, 1234, 5050, 9999, 42, 7777]))
print(f"[3] H稀疏构造 vs 逐列算子 最大误差={err:.2e}")
print(f"[3] H nnz={H.nnz}, 检查Cholesky可行性...")
lu = spla.splu(H)
I_np = np.eye(N*N)
def Hsolve(B):
    if B.ndim == 1: return lu.solve(B)
    return np.column_stack([lu.solve(B[:, j]) for j in range(B.shape[1])])
g_map = grad_E(u_map)
b = g_map.ravel() - H @ u_map.ravel()
u_gmean = u_map.ravel() + Hsolve(Hsolve(b))
u_gmean = u_gmean.reshape(N, N)
Z = Hsolve(I_np)
cov_diag = np.einsum('ij,ij->j', Z, Z)
post_std_exact = np.sqrt(cov_diag).reshape(N, N)
print(f"[3] 高斯均值 vs MAP: ||Δ||/||MAP||={np.linalg.norm(u_gmean-u_map)/np.linalg.norm(u_map):.4f} psnr_gmean={psnr(u_gmean,u):.2f} psnr_map={psnr(u_map,u):.2f}")
print(f"[3] 精确后验std: 全图均值={post_std_exact.mean():.4f} 平坦区(40:60)={post_std_exact[40:60,40:60].mean():.4f} 最大值={post_std_exact.max():.4f}")

# ---- 4. ILA复现与统计诊断 ----
avg, var, chain = run_ila(10000, 1000, 42, keep_chain=True)
psnr_ila = psnr(avg, u)
print(f"[4] ILA: psnr_ila={psnr_ila:.2f} (讲义29.71) psnr_map={psnr(u_map,u):.2f} (讲义32.16)")
print(f"[4] ||avg-MAP||/||MAP||={np.linalg.norm(avg-u_map)/np.linalg.norm(u_map):.4f}  ||avg-高斯均值||/||·||={np.linalg.norm(avg-u_gmean)/np.linalg.norm(u_gmean):.4f}")
samp_std = np.sqrt(np.clip(var, 0, None))
print(f"[4] 采样std均值={samp_std.mean():.4f} vs 精确后验std均值={post_std_exact.mean():.4f}")
print(f"[4] 空间相关(采样std vs 精确std)={np.corrcoef(samp_std.ravel(), post_std_exact.ravel())[0,1]:.3f}")

dev = chain - chain.mean(axis=0, keepdims=True)
for lag in [1, 100, 400, 800, 1500]:
    a = np.mean([np.corrcoef(dev[:-lag, i, j], dev[lag:, i, j])[0,1] for i, j in [(30,30),(70,70),(50,20),(20,80),(5,95)]])
    print(f"[4] 自相关 lag={lag:5d}: {a:.3f}")
# 均值估计的MC误差理论值: var_exact/ESS
s2_flat = cov_diag.mean()
print(f"[4] 单像素后验var均值={s2_flat:.2e}; 若ESS=9000/τ_int，均值估计误差std=sqrt(var/ESS)")

# ---- 5. burn-in 充分性：分段均值PSNR ----
for name, sl in [("1000-3000", slice(0,2000)), ("3000-6000", slice(2000,5000)), ("6000-9000", slice(5000,8000)), ("全部9000", slice(0,9000))]:
    seg = chain[sl].mean(axis=0)
    print(f"[5] 分段均值({name:>10}): psnr={psnr(seg,u):.2f} ||seg-高斯均值||/||·||={np.linalg.norm(seg-u_gmean)/np.linalg.norm(u_gmean):.4f}")

# ---- 6. 多seed ----
print("[6] seed变化:")
for s in [1, 7, 2024]:
    a2, v2, _ = run_ila(10000, 1000, s)
    print(f"    seed={s:5d}: psnr_ila={psnr(a2,u):.2f}")
