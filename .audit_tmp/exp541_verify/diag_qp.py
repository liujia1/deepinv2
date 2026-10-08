# -*- coding: utf-8 -*-
"""诊断5：用对偶QP精确求解TV prox，裁决FD误差的真实来源
问题: min_x 0.5||x-g||^2 + lam*TV(x), TV(x)=<Grad x, p>, ||p||inf<=1
对偶: max_{||q||<=1} -0.5lam^2||G^T q||^2 + lam <G g, q>  (G=GradientIm矩阵)
精确解: x* = g - lam G^T q*
"""
import os, sys, torch, numpy as np
from scipy.optimize import minimize
from scipy import sparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sampling_tools.chambolle_prox_TV import GradientIm, DivergenceIm

torch.set_default_dtype(torch.float64)
device = torch.device('cpu')
H = W = 6
N = H * W
lam = 0.1

# 构造 G (2N x N):  GradientIm(u) -> (dux, duy), 拼接为 2N 向量
cols = []
for i in range(N):
    u = torch.zeros(H, W); u.view(-1)[i] = 1.0
    gx, gy = GradientIm(u, device)
    cols.append(torch.cat([gx.reshape(-1), gy.reshape(-1)]).numpy())
G = np.stack(cols, axis=1)  # 2N x N, G@u = grad u
GT = G.T                     # N x 2N

rng = np.random.default_rng(7)
g6 = rng.standard_normal((H, W)) + 0.5 * (np.abs(rng.standard_normal((H, W))) > 1.5)
gv = g6.reshape(-1)

def solve_prox_qp(gv):
    # minimize 0.5*lam^2 * ||GT q||^2 - lam * (G g)·q   s.t. |q|<=1
    Gg = G @ gv
    A = sparse.csr_matrix(GT.T @ GT)  # 2N x 2N? (G G^T)
    def fun(q):
        r = GT @ q           # N-vector: div-like
        return 0.5 * lam * lam * (r @ r) - lam * (Gg @ q)
    def jac(q):
        r = GT @ q
        return lam * lam * (G @ r) - lam * Gg
    q0 = np.zeros(2 * N)
    res = minimize(fun, q0, jac=jac, method='L-BFGS-B',
                   bounds=[(-1, 1)] * (2 * N),
                   options=dict(maxiter=5000, ftol=1e-16, gtol=1e-12))
    x = gv - lam * (GT @ res.x)
    return x.reshape(H, W), res

def tv(x): return torch.diff(x, dim=0).abs().sum() + torch.diff(x, dim=1).abs().sum()
def obj(x, g, lam=0.1): return (tv(x) + 0.5 * ((g - x) ** 2).sum() / lam).item()

x_qp, res = solve_prox_qp(gv)
x_qp_t = torch.from_numpy(x_qp)
print(f"QP求解成功: {res.success}, nit={res.nit}")
print(f"obj(x_exact_QP) = {obj(x_qp_t, torch.from_numpy(g6)):.10f}")
# 对偶值证书: dual(q*) = -0.5*lam^2||G^T q*||^2 + lam <G g, q*>
q_star = res.x
dual_val = -0.5 * lam * lam * ((GT @ q_star) @ (GT @ q_star)) + lam * (G @ gv) @ q_star
print(f"dual(q*) = {dual_val:.10f}  (强对偶下应等于 primal min)")

# 对比 Chambolle 两个版本
def Chambolle(g, div_op, iters, ret_p=False):
    g = g.clone(); px = torch.zeros_like(g); py = torch.zeros_like(g); tau = 0.249
    for k in range(iters):
        u_ = div_op(px, py) - g / lam
        upx, upy = GradientIm(u_, device)
        tmp = torch.sqrt(upx ** 2 + upy ** 2)
        px = (px + tau * upx) / (1 + tau * tmp)
        py = (py + tau * upy) / (1 + tau * tmp)
    x = g - lam * div_op(px, py)
    return (x, px, py) if ret_p else x

def div_true(px, py):
    out = torch.zeros(H, W)
    out[0, :] = px[0, :]; out[1:-1, :] = px[1:-1, :] - px[:-2, :]; out[-1, :] = -px[-2, :]
    out[:, 0] += py[:, 0]; out[:, 1:-1] += py[:, 1:-1] - py[:, :-2]; out[:, -1] += -py[:, -2]
    return out

g6t = torch.from_numpy(g6)
for iters in [200, 2000, 20000]:
    x_lib = Chambolle(g6t, DivergenceIm, iters)
    x_fx = Chambolle(g6t, div_true, iters)
    print(f"iters={iters:6d}  obj_库={obj(x_lib, g6t):.8f}  obj_修正={obj(x_fx, g6t):.8f}")

# KKT检验：修正版求解器的不动点是否满足真最优性条件
x_fx200, px, py = Chambolle(g6t, div_true, 200, ret_p=True)
gx_, gy_ = GradientIm(x_fx200, device)
align = ((-gx_.sign() * (gx_ != 0) - px).abs().max() + (-gy_.sign() * (gy_ != 0) - py).abs().max()).item()
print(f"\nKKT检验(修正版,200iter): max|p - (-sign(Gx))| = {align:.3e}")
# 对偶值（用p）: 若 |p|=1几乎处处 且对齐成立，则 x_fx 应为最优
w = (g6t - x_fx200) / lam
dual_p = (G @ gv) @ torch.cat([(-px).reshape(-1), (-py).reshape(-1)]).numpy() * lam - 0.5 * lam * lam * ((GT @ torch.cat([(-px).reshape(-1), (-py).reshape(-1)]).numpy()) ** 2).sum()
print(f"obj(x_fx200)={obj(x_fx200, g6t):.8f}  dual(q=-p)={dual_p:.8f}")

# 精确prox下的FD-vs-理论（验证定理本身）
print("\n=== 精确QP prox 的 FD 验证 ===")
g_th = (g6t - x_qp_t) / lam
def env_exact(vn):
    p, _ = solve_prox_qp(vn.reshape(-1))
    return obj(torch.from_numpy(p), torch.from_numpy(vn))
eps = 1e-5
errs = []
for i in range(N):
    e = torch.zeros(H, W); e.view(-1)[i] = 1.0
    fd = (env_exact((g6t + eps * e).numpy()) - env_exact((g6t - eps * e).numpy())) / (2 * eps)
    errs.append(abs(fd - g_th.view(-1)[i].item()))
print(f"mean_err={np.mean(errs):.3e}  max_err={np.max(errs):.3e}  rel={np.mean(errs) / g_th.abs().mean().item():.3%}")

# 库版本prox 的 FD 验证（重现原实验现象）
x_lib200 = Chambolle(g6t, DivergenceIm, 200)
g_lib = (g6t - x_lib200) / lam
def env_lib(v):
    p = Chambolle(v, DivergenceIm, 200)
    return obj(p, v)
errs2 = []
for i in range(N):
    e = torch.zeros(H, W); e.view(-1)[i] = 1.0
    fd = (env_lib(g6t + eps * e) - env_lib(g6t - eps * e)) / (2 * eps)
    errs2.append(abs(fd - g_lib.view(-1)[i].item()))
print(f"库prox(200iter): mean_err={np.mean(errs2):.3e}  rel={np.mean(errs2) / g_lib.abs().mean().item():.3%}")
print(f"||x_lib200 - x_exact||_inf = {(x_lib200 - x_qp_t).abs().max().item():.3e}")
