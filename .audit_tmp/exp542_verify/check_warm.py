# -*- coding: utf-8 -*-
"""验证 WarmStartTV(K=10) 在真实链轨迹上的 prox 精度：
每隔若干步把热启动 prox 输出与库 chambolle 冷启动 300 次的参考解比较。"""
import os, sys, math, importlib.util
import torch, numpy as np
from PIL import Image

EXP = r"d:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.4-2"
sys.path.insert(0, EXP)
import torch
from sampling_tools import chambolle_prox_TV, blur_operators, PSNR

# 从主脚本加载 WarmStartTV / _gradient_im / _divergence_im（不执行主体）
spec = importlib.util.spec_from_file_location("exp542", os.path.join(EXP, "5.4-2.py"))
# 直接读源码提取类定义代价更小：用 runpy 会执行全脚本，故此处复制导入方式——
# 简单起见：exec 主脚本中函数定义前的片段太复杂；改为从模块对象导入。
# 主脚本顶层会执行（依赖环境变量），用 QUICK+OUTDIR 会跑全流程，不合适。
# 因此直接内联实现一份相同的热启动类（与主脚本逐行一致）用于本验证。
def _gradient_im(u):
    sx = list(u.shape); sx[0] = 1
    dux = torch.vstack([u[1:, :] - u[:-1, :], torch.zeros(sx)])
    sy = list(u.shape); sy[1] = 1
    duy = torch.hstack([u[:, 1:] - u[:, :-1], torch.zeros(sy)])
    return dux, duy
def _divergence_im(p1, p2):
    z = p2[:, 1:-1] - p2[:, :-2]
    s2 = list(p2.shape); s2[1] = 1
    v = torch.hstack([p2[:, 0].reshape(s2), z, -p2[:, -1].reshape(s2)])
    z = p1[1:-1, :] - p1[:-2, :]
    s1 = list(p1.shape); s1[0] = 1
    u = torch.vstack([p1[0, :].reshape(s1), z, -p1[-1, :].reshape(s1)])
    return v + u
class WarmStartTV:
    def __init__(self, lam, n_inner=10, tau=0.249):
        self.lam, self.K, self.tau, self.px, self.py = float(lam), int(n_inner), tau, None, None
    def reset(self, shape, device):
        self.px = torch.zeros(shape, device=device); self.py = torch.zeros(shape, device=device)
    def prox(self, g):
        if self.px is None: self.reset(g.shape, g.device)
        lam, tau, K = self.lam, self.tau, self.K
        px, py = self.px, self.py
        for _ in range(K):
            u = _divergence_im(px, py) - g / lam
            upx, upy = _gradient_im(u)
            tmp = torch.sqrt(upx*upx + upy*upy)
            px = (px + tau*upx)/(1+tau*tmp); py = (py + tau*upy)/(1+tau*tmp)
        self.px, self.py = px, py
        return g - lam*_divergence_im(px, py)

torch.manual_seed(42); np.random.seed(42)
dev = torch.device('cpu')
im = np.array(Image.open(os.path.join(EXP, "cman.png")))
C = 128; h0 = (im.shape[0]-C)//2; w0 = (im.shape[1]-C)//2
im = im[h0:h0+C, w0:w0+C]
x = torch.Tensor(im/255.)
A, AT, AAT_norm = blur_operators([5,5], [C,C], "uniform", dev)
y0 = A(x)
sigma = torch.linalg.matrix_norm(A(x)-torch.mean(A(x)), ord='fro')/math.sqrt(torch.numel(x)*10**(40/10))
y = y0 + sigma*torch.randn_like(x)
Ly = float(AAT_norm/sigma**2)
beta, gamma = 20.0, 0.0025
lam = beta*gamma
delta = 0.99/(1/gamma + Ly)
gradf = lambda X: AT(A(X)-y)/sigma**2
proj = lambda z: torch.clamp(z,0,1)

w = WarmStartTV(lam, n_inner=10)
X = y.clone()
errs = []
for k in range(0):
    prox_w = w.prox(X)
    if k in (1, 10, 50, 100, 300, 600):
        with torch.no_grad():
            prox_ref = chambolle_prox_TV(X, dev, {'lambda': lam, 'maxiter': 300, 'tol': 1e-14})
        emax = float((prox_w-prox_ref).abs().max())
        erms = float(torch.sqrt(((prox_w-prox_ref)**2).mean()))
        errs.append((k, emax, erms))
    drift = gradf(X) + (X-prox_w)/gamma
    with torch.no_grad():
        X = proj(X - delta*drift + math.sqrt(2*delta)*torch.randn_like(X))
for k, emax, erms in errs:
    print(f"step {k:4d}: |prox_warm10 - prox_cold300| max={emax:.3e} rms={erms:.3e}, "
          f"对应先验漂移max误差={emax/gamma:.3e}（似然Lipschitz={Ly:.1f}）")

# 不同内迭代数的稳态误差扫描（复用同一条链的状态轨迹）
print("\n--- n_inner 稳态精度扫描（step 100/300/600）---")
for K in (40, 50):
    w2 = WarmStartTV(lam, n_inner=K)
    X2 = y.clone()
    row = []
    for k in range(401):
        prox_w = w2.prox(X2)
        if k in (200, 400):
            with torch.no_grad():
                prox_ref = chambolle_prox_TV(X2, dev, {'lambda': lam, 'maxiter': 300, 'tol': 1e-14})
            row.append(float(torch.sqrt(((prox_w-prox_ref)**2).mean())))
        with torch.no_grad():
            X2 = proj(X2 - delta*(gradf(X2) + (X2-prox_w)/gamma) + math.sqrt(2*delta)*torch.randn_like(X2))
    print(f"K={K:2d}: rms误差 = {['%.2e' % r for r in row]}（σ={float(sigma):.2e}，γ={gamma}）")
