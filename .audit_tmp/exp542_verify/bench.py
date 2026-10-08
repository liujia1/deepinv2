# -*- coding: utf-8 -*-
"""5.4-2 复核：CPU 小规模计时基准"""
import os, sys, math, time
import torch
import numpy as np
from PIL import Image

torch.set_num_threads(os.cpu_count())
EXP = r'd:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.4-2'
sys.path.insert(0, EXP)
from sampling_tools import chambolle_prox_TV, blur_operators  # noqa

im = np.array(Image.open(os.path.join(EXP, 'cman.png')))
H, W = im.shape
h0, w0 = (H - 128) // 2, (W - 128) // 2
imc = im[h0:h0 + 128, w0:w0 + 128]
x = torch.Tensor(imc / 255.)
device = torch.device('cpu')

A, AT, AAT_norm = blur_operators([5, 5], [128, 128], 'uniform', device)
torch.manual_seed(42)
y0 = A(x)
BSNRdb = 40
sigma = torch.linalg.matrix_norm(A(x) - torch.mean(A(x)), ord='fro') / math.sqrt(torch.numel(x) * 10 ** (BSNRdb / 10))
y = y0 + sigma * torch.randn_like(x)
print('sigma=', float(sigma), 'AAT_norm=', float(AAT_norm), 'L_y=', float(AAT_norm / sigma ** 2))

# chambolle 计时（maxiter=100 与 200）
for mi in (100, 200):
    t = time.time()
    for _ in range(5):
        pr = chambolle_prox_TV(y, device, {'lambda': 0.05, 'maxiter': mi})
    print(f'chambolle maxiter={mi}: {(time.time()-t)/5*1000:.1f} ms/call')

# DnCNN 前向计时（CPU）
sys.path.insert(0, os.path.join(EXP, 'sampling_tools'))
from load_model import DnCNN  # noqa
net = DnCNN(channels=1, num_of_layers=17)
sd = torch.load(os.path.join(EXP, 'Pretrained_models', 'RealSN_DnCNN_noise5.pth'), map_location='cpu', weights_only=True)
sd = {k.replace('module.', ''): v for k, v in sd.items()}
net.load_state_dict(sd)
net.eval()
with torch.no_grad():
    for n in (128, 256):
        z = torch.zeros(1, 1, n, n)
        # warmup
        net(z)
        t = time.time()
        for _ in range(3):
            net(z)
        print(f'DnCNN forward {n}x{n}: {(time.time()-t)/3*1000:.1f} ms')
