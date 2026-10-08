# -*- coding: utf-8 -*-
"""诊断7：定位步骤2中FD误差最大的像素位置，判断是边界bug还是固有kink"""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sampling_tools import chambolle_prox_TV, GradientIm

device = torch.device('cpu')
torch.manual_seed(42); np.random.seed(42)
test_image = np.zeros((64, 64))
test_image[10:20, 10:20] = 1.0
test_image[40:50, 40:50] = 0.8
t = torch.from_numpy(test_image).float()
noisy = t + 0.2 * torch.randn_like(t)
y = noisy.double()
lam = 0.1
opts = {'lambda': lam, 'maxiter': 2000, 'tol': 1e-14}

def tv_iso(x):
    dux, duy = GradientIm(x, x.device)
    return torch.sqrt(dux*dux + duy*duy).sum()

prox = chambolle_prox_TV(y, device, opts)
g_th = (y - prox) / lam
def env(v):
    p = chambolle_prox_TV(v, device, opts)
    return (tv_iso(p) + 0.5*((v-p)**2).sum()/lam).item()

torch.manual_seed(42)
idx = torch.randint(0, 64*64, (100,))
recs = []
for i in range(100):
    h, w = (idx[i]//64).item(), (idx[i]%64).item()
    e = torch.zeros_like(y); e[h, w] = 1.0
    fd = (env(y + 1e-5*e) - env(y - 1e-5*e)) / (2e-5)
    recs.append((abs(fd - g_th[h, w].item()), h, w, g_th[h, w].item()))
recs.sort(reverse=True)
print("top-12 errors (|err|, h, w, g_th):")
for r in recs[:12]:
    print(f"  {r[0]:.4f}  ({r[1]:2d},{r[2]:2d})  g_th={r[3]:+.3f}")
errs = np.array([r[0] for r in recs])
print(f"mean={errs.mean():.4e}  median={np.median(errs):.4e}")
# 去掉 top-5 后的均值
print(f"mean without top5 = {np.sort(errs)[:-5].mean():.4e}")

# 用修正伴随求解器对照同样100个像素
def div_true(px, py):
    out = torch.zeros_like(px)
    out[0, :] = px[0, :]; out[1:-1, :] = px[1:-1, :] - px[:-2, :]; out[-1, :] = -px[-2, :]
    out[:, 0] += py[:, 0]; out[:, 1:-1] += py[:, 1:-1] - py[:, :-2]; out[:, -1] += -py[:, -2]
    return out
def prox_fixed(g, iters=2000):
    px = torch.zeros_like(g); py = torch.zeros_like(g); tau = 0.249
    for k in range(iters):
        u_ = div_true(px, py) - g / lam
        upx, upy = GradientIm(u_, g.device)
        tmp = torch.sqrt(upx*upx + upy*upy)
        px = (px + tau*upx) / (1 + tau*tmp)
        py = (py + tau*upy) / (1 + tau*tmp)
    return g - lam * div_true(px, py)
prox2 = prox_fixed(y)
g_th2 = (y - prox2) / lam
def env2(v):
    p = prox_fixed(v)
    return (tv_iso(p) + 0.5*((v-p)**2).sum()/lam).item()
recs2 = []
for i in range(100):
    h, w = (idx[i]//64).item(), (idx[i]%64).item()
    e = torch.zeros_like(y); e[h, w] = 1.0
    fd = (env2(y + 1e-5*e) - env2(y - 1e-5*e)) / (2e-5)
    recs2.append((abs(fd - g_th2[h, w].item()), h, w))
recs2.sort(reverse=True)
errs2 = np.array([r[0] for r in recs2])
print(f"\n修正伴随求解器: mean={errs2.mean():.4e}  max={errs2.max():.4e}")
print("top-6:", [(f"{r[0]:.4f}", r[1], r[2]) for r in recs2[:6]])
