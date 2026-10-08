# -*- coding: utf-8 -*-
"""诊断3：
A) 用已知闭式解的 R(x)=0.5||x||^2 验证FD流程本身无误（管线校验）
B) 核查 chambolle_prox_TV 的 DivergenceIm 是否真是 GradientIm 的负伴随
C) 一阶最优性检验：chambolle返回的x是否最小化真TV目标
D) 修正伴随后的求解器：FD-vs理论 误差是否消失
"""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sampling_tools import chambolle_prox_TV
from sampling_tools.chambolle_prox_TV import GradientIm

torch.set_default_dtype(torch.float64)
device = torch.device('cpu')
DT = torch.float64

# ---------- A) 管线校验：R=0.5||x||^2 闭式 ----------
print("=== A) FD管线校验（R=0.5||x||^2, prox=y/(1+lam) 闭式）===")
torch.manual_seed(0)
y = torch.randn(32, 32, dtype=DT)
lam = 0.1
def prox_quad(v): return v / (1 + lam)
def env_quad(v):
    p = prox_quad(v)
    return (0.5*(p**2).sum() + 0.5*((v-p)**2).sum()/lam).item()
eps = 1e-4
errs = []
g_th = (y - prox_quad(y)) / lam
for (h, w) in [(3,5), (10,20), (31,31), (0,0)]:
    e = torch.zeros_like(y); e[h,w] = 1.0
    fd = (env_quad(y+eps*e) - env_quad(y-eps*e)) / (2*eps)
    errs.append(abs(fd - g_th[h,w].item()))
print(f"mean_abs_err = {np.mean(errs):.3e}  (应≈0，验证FD方法本身正确)")

# ---------- B) 伴随性核查 ----------
print("\n=== B) DivergenceIm 是否为 GradientIm 的负伴随 ===")
torch.manual_seed(1)
H = W = 8
u = torch.randn(H, W, dtype=DT)
px = torch.randn(H, W, dtype=DT); py = torch.randn(H, W, dtype=DT)
# 把 px 最后一行/ py 最后一列置零（模拟实际迭代中它们恒为0的情形也测一般情形）
dux, duy = GradientIm(u, device)
lhs = (dux*px).sum() + (duy*py).sum()          # <Grad u, p>
from sampling_tools.chambolle_prox_TV import DivergenceIm
rhs = (DivergenceIm(px, py) * u).sum()          # <div p, u>
print(f"<Grad u, p> = {lhs.item():.6f}")
print(f"<div p, u>  = {rhs.item():.6f}   (伴随要求 = {-lhs.item():.6f})")
print(f"=> 伴随成立: {torch.allclose(lhs, -rhs)}")

# ---------- C) 一阶最优性：原求解器的x是否最小化真目标 ----------
print("\n=== C) chambolle_prox_TV 解的一阶最优性检验 ===")
test_image = np.zeros((64, 64))
test_image[10:20, 10:20] = 1.0
test_image[40:50, 40:50] = 0.8
torch.manual_seed(42)
y64 = torch.from_numpy(test_image).to(DT) + 0.2*torch.randn(64, 64, dtype=DT)

def tv(x): return torch.diff(x, dim=0).abs().sum() + torch.diff(x, dim=1).abs().sum()
def obj(x, g, lam=0.1): return (tv(x) + 0.5*((g-x)**2).sum()/lam).item()

x_orig = chambolle_prox_TV(y64.to(torch.float32), device, {'lambda': 0.1, 'maxiter': 2000}).to(DT)
print(f"obj(x_orig, 真TV目标) = {obj(x_orig, y64):.8f}")
# 随机扰动下降检验
rng = np.random.default_rng(0)
neg = 0
for t in range(50):
    d = torch.from_numpy(rng.standard_normal((64,64))).to(DT)
    d = d / d.norm() * 0.05
    if obj(x_orig + d, y64) < obj(x_orig, y64) - 1e-10:
        neg += 1
print(f"50次随机扰动中 obj 下降次数: {neg} (>0 说明 x_orig 不是真TV目标的极小点)")

# ---------- D) 修正伴随版 Chambolle ----------
print("\n=== D) 修正散度(末行用p[-2]/末列用p[:,-2])后的求解器 ===")
def DivergenceIm_fixed(p1, p2):
    z = p2[:, 1:-1] - p2[:, :-2]
    shape2 = list(p2.shape); shape2[1] = 1
    v = torch.hstack([p2[:, 0].reshape(shape2), z, -p2[:, -2].reshape(shape2)])
    shape1 = list(p1.shape); shape1[0] = 1
    z = p1[1:-1, :] - p1[:-2, :]
    u = torch.vstack([p1[0, :].reshape(shape1), z, -p1[-2, :].reshape(shape1)])
    return v + u

def prox_tv_fixed(g, lam=0.1, maxiter=5000, tol=1e-12):
    g = g.clone()
    px = torch.zeros_like(g); py = torch.zeros_like(g)
    tau = 0.249
    for k in range(maxiter):
        divp = DivergenceIm_fixed(px, py)
        u_ = divp - g / lam
        upx, upy = GradientIm(u_, device)
        tmp = torch.sqrt(upx*upx + upy*upy)
        px = (px + tau*upx) / (1 + tau*tmp)
        py = (py + tau*upy) / (1 + tau*tmp)
    return g - lam * DivergenceIm_fixed(px, py)

x_fix = prox_tv_fixed(y64)
print(f"obj(x_fixed) = {obj(x_fix, y64):.8f}   (对比 x_orig: {obj(x_orig, y64):.8f})")
neg = 0
for t in range(50):
    d = torch.from_numpy(rng.standard_normal((64,64))).to(DT)
    d = d / d.norm() * 0.05
    if obj(x_fix + d, y64) < obj(x_fix, y64) - 1e-10:
        neg += 1
print(f"x_fixed 50次扰动 obj 下降次数: {neg}")
print(f"||x_orig - x_fixed||_inf = {(x_orig-x_fix).abs().max().item():.3e}")

# 用修正求解器重做FD验证
lam = 0.1
g_th = (y64 - x_fix) / lam
def env_fixed(v):
    p = prox_tv_fixed(v)
    return (tv(p) + 0.5*((v-p)**2).sum()/lam).item()
torch.manual_seed(42)
idx = torch.randint(0, 64*64, (40,))
for eps in [1e-4, 1e-3]:
    errs = []
    for i in range(40):
        h, w = (idx[i]//64).item(), (idx[i]%64).item()
        e = torch.zeros_like(y64); e[h,w] = 1.0
        fd = (env_fixed(y64+eps*e) - env_fixed(y64-eps*e)) / (2*eps)
        errs.append(abs(fd - g_th[h,w].item()))
    rel = np.mean(errs)/g_th.abs().mean().item()
    print(f"eps={eps:.0e}  mean_err={np.mean(errs):.3e}  max_err={np.max(errs):.3e}  rel={rel:.2%}")
