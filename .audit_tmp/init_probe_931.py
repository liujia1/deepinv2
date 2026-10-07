# -*- coding: utf-8 -*-
"""9.3-1 初始化时刻（未做任何参数更新）的每维KL与活跃维度数
复刻 9.3-1.py 的随机流顺序：np.seed -> torch.seed -> Encoder -> Decoder -> shuffle loader
"""
import io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
np.random.seed(42)
import torch
torch.manual_seed(42)
import torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BASE = r'D:\works\deepinv\第9章 VAE与重参数化\第九章 配套实验\实验9.3-1'


class Encoder(nn.Module):
    def __init__(s, di=784, hi=400, lz=50):
        super().__init__(); s.fc1 = nn.Linear(di, hi); s.fc_mu = nn.Linear(hi, lz); s.fc_logvar = nn.Linear(hi, lz)
    def forward(s, x):
        h = F.relu(s.fc1(x)); return s.fc_mu(h), s.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(s, lz=50, hi=400, di=784):
        super().__init__(); s.fc1 = nn.Linear(lz, hi); s.fc2 = nn.Linear(hi, di)
    def forward(s, z):
        return torch.sigmoid(s.fc2(F.relu(s.fc1(z))))


enc = Encoder(lz=50)
dec = Decoder(lz=50)   # 与脚本一致：先建 encoder 再建 decoder（消耗同一段随机流）
ds = datasets.MNIST(os.path.join(BASE, 'data'), train=True, download=False, transform=transforms.ToTensor())
loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)
x, _ = next(iter(loader))
mu, logvar = enc(x.view(-1, 784))
kdm = (0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)).mean(dim=0).detach().numpy()
print('init total KL/sample = %.4f nats, per-dim mean = %.5f' % (kdm.sum(), kdm.mean()))
print('init active dims (>0.01) = %d/50' % (kdm > 0.01).sum())
s = np.sort(kdm)[::-1]
print('init per-dim KL desc top8 :', np.array2string(s[:8], precision=4))
print('init per-dim KL min/max   : %.5f / %.5f' % (s.min(), s.max()))
print('mu abs mean = %.4f, sigma2 mean = %.4f' % (mu.abs().mean().item(), logvar.exp().mean().item()))
