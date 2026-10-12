# -*- coding: utf-8 -*-
"""检验 chambolle 对偶步长 tau=0.249 在最坏输入（棋盘格最高频）下是否稳定收敛"""
import sys
import torch
import numpy as np

EXP_DIR = r"d:\works\deepinv\第4章 MCMC与ULA算法\第四章 配套实验\实验4.4-1"
sys.path.insert(0, EXP_DIR)
from sampling_tools.chambolle_prox_TV import chambolle_prox_TV

dev = torch.device('cpu')
n = 128
ii, jj = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
checker = ((ii + jj) % 2 * 2 - 1) * 100.0 + 128.0  # ±100 最高频
noise = torch.randn(n, n) * 30 + 128.0

# 用手工实现以便扫描不同 tau_c
def grad(u):
    z = u[1:, :] - u[:-1, :]
    dux = torch.vstack([z, torch.zeros(1, n)])
    z = u[:, 1:] - u[:, :-1]
    duy = torch.hstack([z, torch.zeros(n, 1)])
    return dux, duy

def div(p1, p2):
    z = p2[:, 1:-1] - p2[:, :-2]
    v = torch.hstack([p2[:, :1], z, -p2[:, -2:-1]])
    z = p1[1:-1, :] - p1[:-2, :]
    u = torch.vstack([p1[:1, :], z, -p1[-2:-1, :]])
    return v + u

def prox(g, tau_tv, tau_c, maxiter):
    px = torch.zeros_like(g); py = torch.zeros_like(g)
    for _ in range(maxiter):
        u = div(px, py) - g / tau_tv
        upx, upy = grad(u)
        tmp = torch.sqrt(upx * upx + upy * upy)
        px = (px + tau_c * upx) / (1 + tau_c * tmp)
        py = (py + tau_c * upy) / (1 + tau_c * tmp)
    return g - tau_tv * div(px, py)

for name, g in [('checkerboard', checker), ('white_noise', noise), ('smooth', torch.tensor(
        np.kron(np.random.randn(8, 8), np.ones((16, 16)))).float() * 20 + 128.0)]:
    ref = prox(g, 0.3, 0.125, 20000)
    print(f"--- {name} ---")
    for tc in [0.125, 0.249, 0.2499]:
        errs = []
        for m in [10, 100, 1000, 5000]:
            p = prox(g, 0.3, tc, m)
            errs.append((m, float(torch.norm(p - ref) / torch.norm(ref))))
        print(f"  tau_c={tc}: " + " ".join(f"m{m}={e:.2e}" for m, e in errs))

# 确认 D 的算子范数（平方）
x = torch.randn(n, n); x /= x.norm()
lam = 1.0
for _ in range(3000):
    dx, dy = grad(x)
    y = -div(dx, dy)
    v = y.norm().item(); x = y / v
print(f"\n-D*D 最大特征值 = {v:.4f} (理论界 8)，投影梯度稳定界 2/L = {2/v:.4f}")
