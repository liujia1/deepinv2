# -*- coding: utf-8 -*-
"""4.4-1 复核用临时脚本：计时与快速基准（不写入实验目录）"""
import os, sys, time, math
import numpy as np
import torch
from skimage.data import camera
from skimage.transform import resize

EXP_DIR = r"d:\works\deepinv\第4章 MCMC与ULA算法\第四章 配套实验\实验4.4-1"
sys.path.insert(0, EXP_DIR)
from sampling_tools import blur_operators, chambolle_prox_TV

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cpu')

u_np = camera().astype(np.float64)
u_np = resize(u_np, (128, 128), anti_aliasing=True)
x_true = torch.Tensor(u_np).to(device)
nx, ny = x_true.shape
dimx = nx * ny

t0 = time.time()
A, AT, AAT_norm = blur_operators([9, 9], (nx, ny), "uniform", device)
print(f"setup operators: {time.time()-t0:.2f}s, AAT_norm={AAT_norm:.6f}")

Ax = A(x_true)
BSNR = 20
sigma = torch.linalg.matrix_norm(Ax - torch.mean(Ax), ord='fro') / math.sqrt(dimx * 10 ** (BSNR / 10))
y = Ax + sigma * torch.randn_like(x_true)
sigma2 = sigma.item()**2
print(f"sigma={sigma.item():.6f} sigma2={sigma2:.6f}")
print(f"x_true range [{x_true.min():.2f},{x_true.max():.2f}], PSNR obs check")

mse = torch.mean((x_true-y)**2).item()
print(f"PSNR obs = {10*math.log10(255.0**2/mse):.4f} dB")

# timing: 100 MYULA iterations
lam_beta = 0.3
gamma = 0.98/(AAT_norm/sigma2 + 1.0)
X = y.clone()
t0 = time.time()
for i in range(100):
    gl = AT(A(X)-y)/sigma2
    prx = chambolle_prox_TV(X, device, {'lambda': lam_beta, 'MaxIter': 10})
    gp = (X - prx)/1.0
    X = X - gamma*(gl+gp) + math.sqrt(2*gamma)*torch.randn_like(X)
dt = time.time()-t0
print(f"100 MYULA iters (MaxIter=10): {dt:.2f}s -> est 10000 iters: {dt*100:.0f}s = {dt*100/60:.1f} min")
