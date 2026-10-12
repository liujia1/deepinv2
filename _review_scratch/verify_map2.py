# -*- coding: utf-8 -*-
"""D2: 高精度 TV-MAP（强 FISTA + 收敛曲线）与 beta 扫描，量化重建天花板"""
import sys, math, json, time
import numpy as np
import torch
from skimage.data import camera
from skimage.transform import resize

EXP_DIR = r"d:\works\deepinv\第4章 MCMC与ULA算法\第四章 配套实验\实验4.4-1"
sys.path.insert(0, EXP_DIR)
from sampling_tools import blur_operators, chambolle_prox_TV

torch.set_num_threads(8)
dev = torch.device('cpu')

torch.manual_seed(42)
np.random.seed(42)
u = resize(camera().astype(np.float64), (128, 128), anti_aliasing=True)
x_true = torch.tensor(u, dtype=torch.float32)
A, AT, AAT = blur_operators([9, 9], (128, 128), "uniform", dev)
Ax = A(x_true)
sigma = torch.linalg.norm(Ax - Ax.mean()) / math.sqrt(128*128*10**2)
y = Ax + sigma*torch.randn_like(x_true)
sigma2 = sigma.item()**2
Lf = float(AAT/sigma2)

def psnr(ref, est):
    mse = torch.mean((ref-est)**2).item()
    return 10*math.log10(255.0**2/mse)

def fista(beta, n_outer=1500, inner=200):
    step = 1.0/Lf
    tau_tv = step*beta
    x = y.clone(); z = x.clone(); t = 1.0
    curve = []
    for j in range(n_outer):
        xn = chambolle_prox_TV(z - step*AT(A(z)-y)/sigma2, dev,
                               {'lambda': tau_tv, 'MaxIter': inner})
        tn = (1+math.sqrt(1+4*t*t))/2
        z = xn + ((t-1)/tn)*(xn-x)
        if (j+1) % 100 == 0 or j == 0:
            curve.append((j+1, psnr(x_true, xn), float(torch.norm(xn-x)/torch.norm(xn))))
        x = xn; t = tn
    return x, curve

out = {'psnr_obs': psnr(x_true, y), 'Lf': Lf, 'runs': {}}
for beta in [0.03, 0.1, 0.3]:
    t0 = time.time()
    xm, curve = fista(beta)
    out['runs'][f'beta={beta}'] = {'psnr_map': psnr(x_true, xm),
                                   'range': [xm.min().item(), xm.max().item()],
                                   'curve': curve, 'wall': time.time()-t0}
    print(f"beta={beta}: MAP={psnr(x_true,xm):.2f} dB (obs {out['psnr_obs']:.2f}), "
          f"末端相对步长={curve[-1][2]:.2e}, 用时={time.time()-t0:.0f}s")
    print("  收敛曲线(iter,psnr,reldx):", [(a, round(b,3), f'{c:.1e}') for a,b,c in curve])

with open(r'd:\works\deepinv\_review_scratch\chains_D2.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("saved chains_D2.json")
