# -*- coding: utf-8 -*-
"""诊断2：Chambolle求解器收敛性 + 更强求解器下的FD-vs-理论误差"""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sampling_tools import chambolle_prox_TV

device = torch.device('cpu')
test_image = np.zeros((64, 64))
test_image[10:20, 10:20] = 1.0
test_image[40:50, 40:50] = 0.8

def tv_norm(x):
    return torch.diff(x, dim=0).abs().sum() + torch.diff(x, dim=1).abs().sum()

def objective(x, g, lam):
    return tv_norm(x) + 0.5 * ((g - x) ** 2).sum() / lam

torch.manual_seed(42)
y = torch.from_numpy(test_image).double() + 0.2 * torch.randn(64, 64, dtype=torch.float64)
lam = 0.1

print("=== 求解器收敛性：不同maxiter下的原问题目标值 ===")
prev = None
for mi in [200, 1000, 5000, 20000, 50000]:
    p = chambolle_prox_TV(y, device, {'lambda': lam, 'maxiter': mi, 'tol': 1e-14})
    obj = objective(p, y, lam).item()
    msg = "" if prev is None else f"  Δ={obj-prev:+.3e}"
    print(f"maxiter={mi:6d}  obj={obj:.10f}{msg}")
    prev = obj

p_ref = chambolle_prox_TV(y, device, {'lambda': lam, 'maxiter': 50000, 'tol': 1e-14})
p_200 = chambolle_prox_TV(y, device, {'lambda': lam, 'maxiter': 200})
print(f"\n||prox_200 - prox_50000||_inf = {(p_200-p_ref).abs().max().item():.3e}")
print(f"||prox_200 - prox_50000||_2  = {(p_200-p_ref).norm().item():.3e}")

print("\n=== maxiter=50000 求解器下重做FD验证 (float64) ===")
g_th = (y - p_ref) / lam
def envelope(v):
    p = chambolle_prox_TV(v, device, {'lambda': lam, 'maxiter': 50000, 'tol': 1e-14})
    return objective(p, v, lam).item()

torch.manual_seed(42)
idx = torch.randint(0, 64*64, (40,))
for eps in [1e-3, 1e-5]:
    errs = []
    for i in range(40):
        h, w = (idx[i] // 64).item(), (idx[i] % 64).item()
        e_i = torch.zeros_like(y); e_i[h, w] = 1.0
        fd = (envelope(y + eps*e_i) - envelope(y - eps*e_i)) / (2*eps)
        errs.append(abs(fd - g_th[h, w].item()))
    print(f"eps={eps:.0e}  mean_err={np.mean(errs):.4e}  max_err={np.max(errs):.4e}  rel={np.mean(errs)/(g_th.abs().mean().item()):.2%}")
