# -*- coding: utf-8 -*-
"""诊断5.4-1步骤2：32%相对误差的来源
对比维度：dtype(float32/float64) x 求解器精度(maxiter/tol) x 有限差分步长eps
"""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sampling_tools import chambolle_prox_TV

torch.manual_seed(42); np.random.seed(42)
device = torch.device('cpu')

test_image = np.zeros((64, 64))
test_image[10:20, 10:20] = 1.0
test_image[40:50, 40:50] = 0.8

def compute_tv_norm(x):
    dx = torch.diff(x, dim=0); dy = torch.diff(x, dim=1)
    return torch.sum(torch.abs(dx)) + torch.sum(torch.abs(dy))

def run_case(dtype, maxiter, tol, eps, n_samples=60, lam=0.1):
    ti = torch.from_numpy(test_image).to(dtype)
    torch.manual_seed(42)
    y = ti + 0.2 * torch.randn_like(ti)
    prox = chambolle_prox_TV(y, device, {'lambda': lam, 'maxiter': maxiter, 'tol': tol})
    g_th = (y - prox) / lam

    def envelope(v):
        p = chambolle_prox_TV(v, device, {'lambda': lam, 'maxiter': maxiter, 'tol': tol})
        r = v - p
        return compute_tv_norm(p) + 0.5 * torch.sum(r ** 2) / lam

    torch.manual_seed(42)
    H, W = y.shape
    idx = torch.randint(0, H * W, (n_samples,))
    errs = []
    for i in range(n_samples):
        h, w = (idx[i] // W).item(), (idx[i] % W).item()
        e_i = torch.zeros_like(y); e_i[h, w] = 1.0
        fd = (envelope(y + eps * e_i) - envelope(y - eps * e_i)) / (2 * eps)
        errs.append(abs(fd.item() - g_th[h, w].item()))
    mean_err = float(np.mean(errs)); max_err = float(np.max(errs))
    rel = mean_err / (torch.mean(torch.abs(g_th)).item() + 1e-12)
    # 检查envelope值对精度敏感性：同一点float32 vs float64的envelope差
    return dict(dtype=str(dtype).replace('torch.',''), maxiter=maxiter, tol=tol, eps=eps,
                mean_abs_grad_th=float(torch.mean(torch.abs(g_th)).item()),
                mean_err=mean_err, max_err=max_err, rel=rel)

cases = [
    (torch.float32, 200, 1e-3, 1e-4),   # 原实验设置
    (torch.float32, 2000, 1e-8, 1e-4),  # 只提求解器精度
    (torch.float64, 200, 1e-3, 1e-4),   # 只提dtype
    (torch.float64, 2000, 1e-8, 1e-4),  # 双管齐下
    (torch.float64, 2000, 1e-8, 1e-3),  # 再放大步长
    (torch.float64, 2000, 1e-8, 1e-2),
]
print(f"{'dtype':8}{'maxiter':9}{'tol':8}{'eps':8}{'mean|g|':10}{'mean_err':10}{'max_err':10}{'rel':8}")
for dt, mi, tl, ep in cases:
    r = run_case(dt, mi, tl, ep)
    print(f"{r['dtype']:8}{mi:<9}{tl:<8.0e}{ep:<8.0e}{r['mean_abs_grad_th']:<10.4f}{r['mean_err']:<10.4e}{r['max_err']:<10.4e}{r['rel']:<8.2%}")
