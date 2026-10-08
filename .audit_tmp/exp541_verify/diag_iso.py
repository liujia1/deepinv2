# -*- coding: utf-8 -*-
"""诊断6：验证假设——chambolle_prox_TV 求解的是【各向同性】TV 的 prox，
而实验5.4-1步骤2的 compute_tv_norm 用的是【各向异性】TV。
R 不一致 → Moreau 包络梯度恒等式对不上 → 26%误差与'边缘非光滑'无关。
"""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sampling_tools.chambolle_prox_TV import GradientIm, DivergenceIm, chambolle_prox_TV

torch.set_default_dtype(torch.float64)
device = torch.device('cpu')
lam = 0.1

test_image = np.zeros((64, 64))
test_image[10:20, 10:20] = 1.0
test_image[40:50, 40:50] = 0.8
torch.manual_seed(42)
y = torch.from_numpy(test_image) + 0.2 * torch.randn(64, 64, dtype=torch.float64)

def tv_aniso(x):
    dx = torch.diff(x, dim=0); dy = torch.diff(x, dim=1)
    return dx.abs().sum() + dy.abs().sum()

def tv_iso(x):
    dx = torch.diff(x, dim=0); dy = torch.diff(x, dim=1)
    # 与 GradientIm 的 sqrt(dux²+duy²) 配对一致：在 (i,j), i≤H-2, j≤W-2 处配对；
    # 边界处单边项：j=W-1 只剩 |dx|，i=H-1 只剩 |dy|
    a = dx[:, :-1]; b = dy[:-1, :]
    core = torch.sqrt(a**2 + b**2).sum()
    edge_x = dx[:, -1].abs().sum()   # 最后一列只有行方向差
    edge_y = dy[-1, :].abs().sum()   # 最后一行只有列方向差
    return core + edge_x + edge_y

def obj_with(tvfun, x, g):
    return (tvfun(x) + 0.5 * ((g - x)**2).sum() / lam).item()

# 库解在两种 R 下的目标值：若库解最小化的是 iso-TV prox，则 obj_iso(库解) ≤ obj_iso(任意扰动)
x_lib = chambolle_prox_TV(y, device, {'lambda': lam, 'maxiter': 2000})
print(f"tv_aniso(x_lib)={obj_with(tv_aniso, x_lib, y):.6f}   tv_iso(x_lib)={obj_with(tv_iso, x_lib, y):.6f}")

def fd_check(proxfun, tvfun, eps=1e-5, n=30, tag=""):
    prox = proxfun(y)
    g_th = (y - prox) / lam
    def env(v):
        p = proxfun(v)
        return (tvfun(p) + 0.5 * ((v - p)**2).sum() / lam).item()
    torch.manual_seed(42)
    idx = torch.randint(0, 64*64, (n,))
    errs = []
    for i in range(n):
        h, w = (idx[i]//64).item(), (idx[i]%64).item()
        e = torch.zeros_like(y); e[h, w] = 1.0
        fd = (env(y + eps*e) - env(y - eps*e)) / (2*eps)
        errs.append(abs(fd - g_th[h, w].item()))
    rel = np.mean(errs) / g_th.abs().mean().item()
    print(f"{tag}: mean_err={np.mean(errs):.3e}  max_err={np.max(errs):.3e}  rel={rel:.3%}")

prox_lib = lambda v: chambolle_prox_TV(v, device, {'lambda': lam, 'maxiter': 2000})
print("\n--- 库prox + 不同R 的 FD-vs-理论 检查 (float64, 2000iter) ---")
fd_check(prox_lib, tv_aniso, tag="库prox + 各向异性R(实验现状)", n=12)
fd_check(prox_lib, tv_iso,   tag="库prox + 各向同性R(假设)", n=12)

# 修正伴随 + 各向同性R
def div_true(px, py):
    out = torch.zeros_like(px)
    out[0, :] = px[0, :]; out[1:-1, :] = px[1:-1, :] - px[:-2, :]; out[-1, :] = -px[-2, :]
    out[:, 0] += py[:, 0]; out[:, 1:-1] += py[:, 1:-1] - py[:, :-2]; out[:, -1] += -py[:, -2]
    return out

def prox_chambolle_divtrue(g, iters=20000):
    g = g.clone(); px = torch.zeros_like(g); py = torch.zeros_like(g); tau = 0.249
    for k in range(iters):
        u_ = div_true(px, py) - g / lam
        upx, upy = GradientIm(u_, device)
        tmp = torch.sqrt(upx**2 + upy**2)
        px = (px + tau*upx) / (1 + tau*tmp)
        py = (py + tau*upy) / (1 + tau*tmp)
    return g - lam * div_true(px, py)

prox_fix = lambda v: prox_chambolle_divtrue(v, 3000)
print("\n--- 修正伴随版求解器 ---")
print(f"obj_iso(x_fix)={obj_with(tv_iso, prox_fix(y), y):.6f}  (库解: {obj_with(tv_iso, x_lib, y):.6f})")
print(f"obj_aniso(x_fix)={obj_with(tv_aniso, prox_fix(y), y):.6f}  (库解: {obj_with(tv_aniso, x_lib, y):.6f})")
fd_check(prox_fix, tv_iso, tag="修正prox + 各向同性R", n=12)
fd_check(prox_fix, tv_aniso, tag="修正prox + 各向异性R", n=12)
