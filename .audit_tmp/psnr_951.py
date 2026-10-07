# -*- coding: utf-8 -*-
"""测量 9.5-1 三个 β checkpoint (dz=20) 的测试集 BCE / KL / PSNR / 活跃维度，
为 9.3-1 的 β 归因提供跨实验对照。"""
import io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

BASE = r'D:\works\deepinv\第9章 VAE与重参数化\第九章 配套实验\实验9.5-1'


class Encoder(nn.Module):
    def __init__(s, di=784, hi=400, lz=20):
        super().__init__(); s.fc1 = nn.Linear(di, hi); s.fc_mu = nn.Linear(hi, lz); s.fc_logvar = nn.Linear(hi, lz)
    def forward(s, x):
        h = F.relu(s.fc1(x)); return s.fc_mu(h), s.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(s, lz=20, hi=400, di=784):
        super().__init__(); s.fc1 = nn.Linear(lz, hi); s.fc2 = nn.Linear(hi, di)
    def forward(s, z):
        return torch.sigmoid(s.fc2(F.relu(s.fc1(z))))


tf = transforms.Compose([transforms.ToTensor()])
test = datasets.MNIST(os.path.join(BASE, 'data'), train=False, download=False, transform=tf)
loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False, num_workers=0)

for tag, beta in [('beta0', 0.0), ('beta1', 1.0), ('beta4', 4.0)]:
    enc, dec = Encoder(), Decoder()
    ck = torch.load(os.path.join(BASE, 'vae_%s_checkpoint.pth' % tag), map_location='cpu', weights_only=False)
    enc.load_state_dict(ck['encoder_state_dict']); dec.load_state_dict(ck['decoder_state_dict'])
    enc.eval(); dec.eval()
    kl_sum = torch.zeros(20); n = 0; bce = 0.0; kld = 0.0; mse_l = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.view(-1, 784)
            mu, lv = enc(x)
            z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
            xr = dec(z)
            kdim = 0.5 * (mu.pow(2) + lv.exp() - lv - 1)
            kl_sum += kdim.sum(dim=0); n += x.shape[0]
            bce += F.binary_cross_entropy(xr, x, reduction='sum').item()
            kld += kdim.sum().item()
            mse_l.append(((xr - x) ** 2).mean(dim=1))
    mse = torch.cat(mse_l)
    psnr = (10 * torch.log10(1.0 / mse.clamp(min=1e-12)))
    kd = (kl_sum / n).numpy()
    print('%s (beta=%.1f, dz=20): BCE=%.2f KL=%.2f loss=%.2f active=%d/20 PSNR=%.2f dB  top5KL=%s'
          % (tag, beta, bce / n, kld / n, (bce + beta * kld) / n,
             int((kd > 0.01).sum()), psnr.mean().item(),
             np.array2string(np.sort(kd)[::-1][:5], precision=3)), flush=True)
