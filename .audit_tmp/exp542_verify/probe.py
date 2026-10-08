# -*- coding: utf-8 -*-
"""定位 TV 链 NaN 来源"""
import os, sys, math
import torch
import numpy as np
from PIL import Image

torch.set_num_threads(os.cpu_count())
EXP = r'd:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.4-2'
sys.path.insert(0, EXP)
from sampling_tools import chambolle_prox_TV, blur_operators

im = np.array(Image.open(os.path.join(EXP, 'cman.png')))
h0, w0 = (256 - 128) // 2, (256 - 128) // 2
x = torch.Tensor(im[h0:h0 + 128, w0:w0 + 128] / 255.)
torch.manual_seed(42)
A, AT, AAT_norm = blur_operators([5, 5], [128, 128], 'uniform', torch.device('cpu'))
y0 = A(x)
sigma = torch.linalg.matrix_norm(A(x) - torch.mean(A(x)), ord='fro') / math.sqrt(torch.numel(x) * 10 ** 4)
y = y0 + sigma * torch.randn_like(x)
L_y = float(AAT_norm / sigma ** 2)
print('L_y', L_y, 'finite y', torch.isfinite(y).all().item())
gradf = lambda X: AT(A(X) - y) / sigma ** 2

gamma = 0.05
delta = 0.99 / (1.0 / gamma + L_y)
print('delta', delta)
X = y.clone()
for k in range(200):
    g = gradf(X)
    pr = chambolle_prox_TV(X, torch.device('cpu'), {'lambda': 0.05, 'maxiter': 200})
    drift = (X - pr) / gamma
    X = torch.clamp(X - delta * (g + drift) + math.sqrt(2 * delta) * torch.randn_like(X), 0, 1)
    if not torch.isfinite(X).all() or not torch.isfinite(pr).all() or not torch.isfinite(g).all():
        print('NaN at iter', k)
        print('finite g', torch.isfinite(g).all().item(),
              'finite pr', torch.isfinite(pr).all().item(),
              'finite X_new', torch.isfinite(X).all().item())
        print('g absmax', np.nanmax(np.abs(g.numpy())))
        break
    if k % 20 == 0:
        print(k, 'ok  X range', float(X.min()), float(X.max()),
              'gmax', float(g.abs().max()), 'prmax', float(pr.max()))
