# -*- coding: utf-8 -*-
"""诊断: 9.3-1 checkpoint 的 per-dim KL / 重建质量 / 初始 KL 是否为 0"""
import io, sys, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

BASE = r'D:\works\deepinv\第9章 VAE与重参数化\第九章 配套实验\实验9.3-1'
DEV = torch.device('cpu')
torch.manual_seed(0)

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

enc, dec = Encoder(), Decoder()
ck = torch.load(os.path.join(BASE, 'vae_mnist_checkpoint.pth'), map_location=DEV, weights_only=False)
enc.load_state_dict(ck['encoder_state_dict']); dec.load_state_dict(ck['decoder_state_dict'])
enc.eval(); dec.eval()

tf = transforms.Compose([transforms.ToTensor()])
test = datasets.MNIST(os.path.join(BASE, 'data'), train=False, download=False, transform=tf)
loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False, num_workers=0)

BETA = 4.0
kl_sum = torch.zeros(50)
n = 0
psnr_list, mse_list = [], []
bce_sum = kld_sum = 0.0
with torch.no_grad():
    for x, _ in loader:
        x = x.view(-1, 784)
        mu, logvar = enc(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        xr = dec(z)
        kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
        kl_sum += kl_dim.sum(dim=0); n += x.shape[0]
        bce_sum += F.binary_cross_entropy(xr, x, reduction='sum').item()
        kld_sum += kl_dim.sum().item()
        mse = ((xr - x) ** 2).mean(dim=1)
        mse_list.append(mse)
mse = torch.cat(mse_list)
psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-12))
kl_per_dim = (kl_sum / n).numpy()
print('test samples:', n)
print('test BCE/sample = %.3f, KL/sample = %.3f, loss = %.3f' % (bce_sum / n, kld_sum / n, (bce_sum + BETA * kld_sum) / n))
print('recon PSNR: mean %.2f dB, median %.2f dB' % (psnr.mean().item(), psnr.median().item()))
print('per-dim KL sorted desc (top 20):')
print(np.array2string(np.sort(kl_per_dim)[::-1][:20], precision=4))
print('per-dim KL sorted asc (bottom 20):')
print(np.array2string(np.sort(kl_per_dim)[:20], precision=6))
for th in (0.01, 0.02, 0.05, 0.1):
    print('  active dims @ threshold %.2f : %d/50' % (th, int((kl_per_dim > th).sum())))
print('sum per-dim mean KL = %.3f (应等于 KL/sample)' % kl_per_dim.sum())

# 初始 KL 是否接近 0（未训练的新初始化模型）
kl0_list = []
for seed in range(5):
    torch.manual_seed(seed)
    e2 = Encoder()
    with torch.no_grad():
        for x, _ in loader:
            mu, logvar = e2(x.view(-1, 784))
            kl0 = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
            kl0_list.append((seed, kl0.mean().item(), kl0.std().item()))
            break
print('fresh-init KL/sample (first test batch, 5 seeds):')
for s, m, sd in kl0_list:
    print('  seed %d: mean %.4f  std %.4f' % (s, m, sd))

# 隐空间诊断：μ 的每维标准差（跨数据），衡量该维是否真的携带 x 的信息
mus = []
with torch.no_grad():
    for x, _ in loader:
        mu, _ = enc(x.view(-1, 784)); mus.append(mu)
mu_all = torch.cat(mus)
mu_std = mu_all.std(dim=0).numpy()
print('μ per-dim std sorted desc (top 15):', np.array2string(np.sort(mu_std)[::-1][:15], precision=3))
print('μ per-dim std sorted asc (bottom 10):', np.array2string(np.sort(mu_std)[:10], precision=5))
print('corr(active, mu_std>0.1) dims:', int(((kl_per_dim > 0.01) == (mu_std > 0.1)).sum()), '/50')
