# -*- coding: utf-8 -*-
"""实验4.4-2 快速静态核查（秒级）：
(1) 复现确定性量：true sigma2、AAT_norm、SNR(y)，与 results_summary.json 比对
(2) TVnorm 转置怪癖的实际影响：lib(.T) vs 正确各向同性TV，在多幅代表图像上的相对差
"""
import sys, os, math, json
import numpy as np
import torch

sys.path.insert(0, r"d:/works/deepinv/第4章 MCMC与ULA算法/第四章 配套实验/实验4.4-2")
from skimage.data import camera
from sampling_tools import blur_operators, chambolle_prox_TV, TVnorm

dev = torch.device('cpu')
torch.manual_seed(42)

x = torch.Tensor(camera().astype(np.float64))
nx, ny = x.shape
d = nx * ny
A, AT, AAT = blur_operators([9, 9], (nx, ny), 'uniform', dev)
Ax = A(x)
sigma = torch.linalg.matrix_norm(Ax - torch.mean(Ax), ord='fro') / math.sqrt(d * 10 ** 3)
y = Ax + sigma * torch.randn_like(x)
s2 = float(sigma ** 2)

def snr(ref, est):
    return 10 * torch.log10(torch.sum(ref ** 2) / torch.sum((ref - est) ** 2)).item()

print("== (1) 确定性量 ==")
print(f"AAT_norm = {AAT:.6f} (均匀9x9核DC增益=1, 应≈1)")
print(f"true_sigma2 = {s2:.6f}  (JSON: 4.975089)")
print(f"SNR(y) = {snr(x, y):.2f}  (JSON: 18.84)")

def tv_correct(im):
    dh = torch.zeros_like(im); dh[:, :-1] = im[:, 1:] - im[:, :-1]
    dv = torch.zeros_like(im); dv[:-1, :] = im[1:, :] - im[:-1, :]
    return torch.sum(torch.sqrt(dh ** 2 + dv ** 2)).item()

def tv_lib(im):
    return TVnorm(im).item()

print("\n== (2) TVnorm 定义差异 (lib 的 diffv 带 .T) ==")
imgs = {'camera': x, 'y(obs)': y}
# 模拟后验样本族：不同强度 TV 去噪 + 噪声
proxg = lambda z, lam: chambolle_prox_TV(z, dev, {'lambda': lam, 'MaxIter': 50})
for lam in [0.5, 2.0, 8.0]:
    imgs[f'prox_lam{lam}'] = proxg(y, lam)
rng = torch.Generator().manual_seed(0)
imgs['white_noise'] = torch.randn(x.shape, generator=rng) * 5 + 128
smooth = torch.Tensor(np.kron(camera()[::16, ::16].astype(np.float64), np.ones((16, 16))))
imgs['smooth_blocky'] = smooth

for name, im in imgs.items():
    a, b = tv_lib(im), tv_correct(im)
    print(f"  {name:16s} lib={a:12.1f} correct={b:12.1f} rel_diff={abs(a-b)/b*100:6.3f}%")

# theta 均衡关系的传播：theta* = d / E[TV]，TV定义差 -> theta 差同比例
print("\n== 含义：Fisher 均衡 theta=d/E[TV]，TV 相对差即 theta 系统差的上界 ==")
