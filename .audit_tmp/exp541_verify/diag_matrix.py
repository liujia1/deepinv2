# -*- coding: utf-8 -*-
"""诊断4：用显式矩阵算子做无歧义的伴随核验 + 真prox对比"""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sampling_tools.chambolle_prox_TV import GradientIm, DivergenceIm

torch.set_default_dtype(torch.float64)
device = torch.device('cpu')
H = W = 6

# ---- 用基向量构造 GradientIm 的矩阵表示 G: (2*N, N) ----
N = H * W
G_cols = []
for i in range(N):
    u = torch.zeros(H, W)
    u.view(-1)[i] = 1.0
    gx, gy = GradientIm(u, device)
    G_cols.append(torch.cat([gx.reshape(-1), gy.reshape(-1)]))
G = torch.stack(G_cols, dim=1)  # N x 2N, G @ u = grad u

def div_as_matrix(Dop):
    cols = []
    for j in range(2 * N):
        p = torch.zeros(2 * N)
        p[j] = 1.0
        px = p[:N].reshape(H, W); py = p[N:].reshape(H, W)
        cols.append(Dop(px, py).reshape(-1))
    return torch.stack(cols, dim=1)  # N x 2N

D_code = div_as_matrix(DivergenceIm)

def div_true(px, py):
    """-G^T 的显式实现（与GradientIm严格配对）"""
    out = torch.zeros(H, W)
    out[0, :] = px[0, :]
    out[1:-1, :] = px[1:-1, :] - px[:-2, :]
    out[-1, :] = -px[-2, :]
    out[:, 0] += py[:, 0]
    out[:, 1:-1] += py[:, 1:-1] - py[:, :-2]
    out[:, -1] += -py[:, -2]
    return out

D_true = div_as_matrix(lambda px, py: div_true(px, py))

print("||D_code - (-G^T)||_max =", (D_code + G.T).abs().max().item())
print("||D_true - (-G^T)||_max =", (D_true + G.T).abs().max().item())

# ---- 真 Chambolle（正确配对） vs 库版本 ----
def Chambolle(g, lam, iters, div_op):
    px = torch.zeros_like(g); py = torch.zeros_like(g)
    tau = 0.249
    for k in range(iters):
        u_ = div_op(px, py) - g / lam
        upx, upy = GradientIm(u_, device)
        tmp = torch.sqrt(upx**2 + upy**2)
        px = (px + tau * upx) / (1 + tau * tmp)
        py = (py + tau * upy) / (1 + tau * tmp)
    return g - lam * div_op(px, py)

def tv(x): return torch.diff(x, dim=0).abs().sum() + torch.diff(x, dim=1).abs().sum()
def obj(x, g, lam): return (tv(x) + 0.5 * ((g - x)**2).sum() / lam).item()

rng = np.random.default_rng(7)
g6 = torch.from_numpy(rng.standard_normal((H, W)) + 0.5 * (np.abs(rng.standard_normal((H, W))) > 1.5)).double()
lam = 0.1
x_true = Chambolle(g6, lam, 20000, div_true)
x_code = Chambolle(g6, lam, 20000, DivergenceIm)
print(f"\n小问题(H=W=6): obj真解={obj(x_true,g6,lam):.10f}  obj库解={obj(x_code,g6,lam):.10f}")
print(f"||x_code - x_true||_inf = {(x_code-x_true).abs().max().item():.3e}")

# ---- 真解下 FD-vs-理论 误差（小问题, 全部像素） ----
g_th = (g6 - x_true) / lam
def env_true(v):
    p = Chambolle(v, lam, 20000, div_true)
    return obj(p, v, lam)
eps = 1e-5
errs = []
for i in range(N):
    e = torch.zeros_like(g6); e.view(-1)[i] = 1.0
    fd = (env_true(g6 + eps*e) - env_true(g6 - eps*e)) / (2*eps)
    errs.append(abs(fd - g_th.view(-1)[i].item()))
print(f"真prox的FD验证: mean_err={np.mean(errs):.3e}  max_err={np.max(errs):.3e}  rel={np.mean(errs)/g_th.abs().mean().item():.2%}")

# ---- 库版本prox(作为'函数')的FD梯度 vs (g-x_code)/lam ----
def env_code(v):
    p = Chambolle(v, lam, 20000, DivergenceIm)
    return obj(p, v, lam)
g_code = (g6 - x_code) / lam
errs2 = []
for i in range(N):
    e = torch.zeros_like(g6); e.view(-1)[i] = 1.0
    fd = (env_code(g6 + eps*e) - env_code(g6 - eps*e)) / (2*eps)
    errs2.append(abs(fd - g_code.view(-1)[i].item()))
print(f"库prox的FD验证: mean_err={np.mean(errs2):.3e}  max_err={np.max(errs2):.3e}  rel={np.mean(errs2)/g_code.abs().mean().item():.2%}")
