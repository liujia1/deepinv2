# -*- coding: utf-8 -*-
"""对照实验（复核9.3-1的"β>1促进坍缩"因果归因是否成立）
  C1: dz=50, beta=1.0   —— 同架构弱KL，检查活跃维度是否恢复到接近50
  C2: dz=50, beta=4.0 + KL退火(warmup 10 epochs) —— 检查坍缩是否可逆（固有属性 vs 训练动力学陷阱）
其余设置（架构、lr、batch、epochs、seed）与 9.3-1 完全一致。
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BASE = r'D:\works\deepinv\第9章 VAE与重参数化\第九章 配套实验\实验9.3-1'
OUT = r'D:\works\deepinv\.audit_tmp\exp931_control'
os.makedirs(OUT, exist_ok=True)
DEV = torch.device('cpu')
NUM_EPOCHS = int(os.environ.get('CTRL_EPOCHS', '20'))


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


tf = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST(os.path.join(BASE, 'data'), train=True, download=False, transform=tf)


def run(tag, beta, anneal_epochs=0):
    np.random.seed(42); torch.manual_seed(42)
    loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    enc, dec = Encoder(), Decoder()
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    hist = {'loss': [], 'bce': [], 'kld': [], 'active': []}
    t0 = time.time()
    for ep in range(NUM_EPOCHS):
        b_eff = beta * min(1.0, (ep + 1) / anneal_epochs) if anneal_epochs > 0 else beta
        tot = [0.0, 0.0, 0.0]; kl_dim_sum = None; nb = 0
        for x, _ in loader:
            x = x.view(-1, 784)
            mu, logvar = enc(x)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            xr = dec(z)
            bce = F.binary_cross_entropy(xr, x, reduction='sum')
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = bce + b_eff * kld
            opt.zero_grad(); loss.backward(); opt.step()
            tot[0] += loss.item(); tot[1] += bce.item(); tot[2] += kld.item()
            d = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(dim=0).detach()
            kl_dim_sum = d if kl_dim_sum is None else kl_dim_sum + d
            nb += 1
        n = len(train_set)
        kdm = (kl_dim_sum / nb).numpy()
        hist['loss'].append(tot[0] / n); hist['bce'].append(tot[1] / n)
        hist['kld'].append(tot[2] / n); hist['active'].append(int((kdm > 0.01).sum()))
        print('  [%s] ep %2d/%d beta_eff=%.2f Loss=%.2f BCE=%.2f KL=%.2f Active=%d/50'
              % (tag, ep + 1, NUM_EPOCHS, b_eff, hist['loss'][-1], hist['bce'][-1],
                 hist['kld'][-1], hist['active'][-1]), flush=True)
    print('  [%s] per-dim KL desc:' % tag, np.array2string(np.sort(kdm)[::-1][:15], precision=3))
    print('  [%s] elapsed %.1fs' % (tag, time.time() - t0), flush=True)
    with open(os.path.join(OUT, 'ctrl_%s.json' % tag), 'w', encoding='utf-8') as f:
        json.dump({'beta': beta, 'anneal_epochs': anneal_epochs, 'epochs': NUM_EPOCHS,
                   'final_active': hist['active'][-1], 'final_kld': hist['kld'][-1],
                   'final_bce': hist['bce'][-1], 'final_kl_per_dim_top15': np.sort(kdm)[::-1][:15].tolist(),
                   'history': hist}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    for tag, beta, an in [('b1_dz50', 1.0, 0), ('b4_anneal', 4.0, 10)]:
        if os.environ.get('CTRL_ONLY') and os.environ['CTRL_ONLY'] != tag:
            continue
        run(tag, beta, an)
