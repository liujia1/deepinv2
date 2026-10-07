# -*- coding: utf-8 -*-
"""定量检验：AE(β=0) vs VAE(β=1) 插值是否真的存在'空洞/崩坏'差异"""
import os, sys, io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

EXP = r'd:\works\deepinv\第9章 VAE与重参数化\第九章 配套实验\实验9.5-1'

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400); self.fc_mu = nn.Linear(400, 20); self.fc_logvar = nn.Linear(400, 20)
    def forward(self, x):
        h = F.relu(self.fc1(x)); return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 400); self.fc2 = nn.Linear(400, 784)
    def forward(self, z):
        return torch.sigmoid(self.fc2(F.relu(self.fc1(z))))

transform = transforms.ToTensor()
train_set = datasets.MNIST(os.path.join(EXP, 'data'), train=True, download=False, transform=transform)
test_set = datasets.MNIST(os.path.join(EXP, 'data'), train=False, download=False, transform=transform)

models = {}
for beta in [0, 1, 4]:
    ck = torch.load(os.path.join(EXP, f'vae_beta{beta}_checkpoint.pth'), map_location='cpu', weights_only=False)
    enc, dec = Encoder(), Decoder()
    enc.load_state_dict(ck['encoder_state_dict']); dec.load_state_dict(ck['decoder_state_dict'])
    enc.eval(); dec.eval()
    models[beta] = (enc, dec)

with torch.no_grad():
    # 训练集全部 μ（用于流形距离）
    train_mu = {}
    for beta in [0, 1]:
        enc, _ = models[beta]
        mus = []
        for x, y in DataLoader(train_set, batch_size=1000):
            mus.append(enc(x.view(-1, 784))[0])
        train_mu[beta] = torch.cat(mus)

    test_x = test_set.data.float().view(-1, 784) / 255.0
    test_y = test_set.targets

torch.manual_seed(0)
rng = np.random.default_rng(0)
# 随机抽取不同数字对
pairs = []
while len(pairs) < 30:
    i, j = rng.integers(0, 10000, 2)
    if test_y[i] != test_y[j]:
        pairs.append((int(i), int(j)))

def interp_metrics(beta):
    enc, dec = models[beta]
    mid_blur, mid_nn, ep_psnr = [], [], []
    for i, j in pairs:
        x1, x2 = test_x[i:i+1], test_x[j:j+1]
        mu1 = enc(x1)[0]; mu2 = enc(x2)[0]
        zm = 0.5 * (mu1 + mu2)
        xm = dec(zm)
        # 模糊度：二值化交叉熵（越低越清晰）
        p = xm.clamp(1e-6, 1-1e-6)
        # 用中点解码与两端原图的平均像素差衡量"过渡合理性"；再算对比度代理
        contrast = (xm.max() - xm.mean()).item()
        # 中点与最近训练样本μ的距离 / 端点与最近训练样本μ距离
        d_mid = torch.cdist(zm, train_mu[beta]).min().item()
        d1 = torch.cdist(mu1, train_mu[beta]).min().item()
        d2 = torch.cdist(mu2, train_mu[beta]).min().item()
        mid_nn.append(d_mid / max(1e-8, 0.5*(d1+d2)))
        mid_blur.append(contrast)
        ep_psnr.append((-torch.log2(((dec(mu1)-x1)**2).mean()).item()))
    return np.array(mid_blur), np.array(mid_nn)

for beta in [0, 1]:
    blur, nn_ratio = interp_metrics(beta)
    print(f'beta={beta} ({"AE" if beta==0 else "VAE"}): 中点亮度跨度均值={blur.mean():.4f}, 中点NN距离/端点NN距离 比值均值={nn_ratio.mean():.3f}, 中点样本标准差={blur.std():.4f}')

# 检查讲义图所用的7→8对的插值路径是否经过低密度区
with torch.no_grad():
    for beta in [0, 1]:
        enc, dec = models[beta]
        # 复现图2样本对: batch0第5个(7)与batch1第0个(8)
        x1 = test_x[:128][5:6]; x2 = test_x[128:256][0:1]
        mu1, mu2 = enc(x1)[0], enc(x2)[0]
        alphas = np.linspace(0, 1, 11)
        zs = (1-alphas)[:, None] * mu1.numpy() + alphas[:, None] * mu2.numpy()
        zt = torch.tensor(zs, dtype=train_mu[beta].dtype)
        dnn = torch.cdist(zt, train_mu[beta]).min(dim=1).values.numpy()
        d_ep = 0.5 * (torch.cdist(mu1.to(train_mu[beta].dtype), train_mu[beta]).min().item()
                      + torch.cdist(mu2.to(train_mu[beta].dtype), train_mu[beta]).min().item())
        print(f'beta={beta} 7→8插值路径 各α到最近训练μ距离: {np.round(dnn,2).tolist()} (端点平均NN距离={d_ep:.2f})')
