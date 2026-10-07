# -*- coding: utf-8 -*-
"""实验9.5-1 复核脚本：解析checkpoint、复算指标、核对讲义预期"""
import os, sys, io, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

EXP = r'd:\works\deepinv\第9章 VAE与重参数化\第九章 配套实验\实验9.5-1'
device = torch.device('cpu')

class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(os.path.join(EXP, 'data'), train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(os.path.join(EXP, 'data'), train=False, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

torch.manual_seed(42)
np.random.seed(42)

betas = [0, 1, 4]
out = {}
for beta in betas:
    path = os.path.join(EXP, f'vae_beta{beta}_checkpoint.pth')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    info = {k: (v if not isinstance(v, dict) else f'<dict keys={list(v.keys())}>')
            for k, v in ck.items() if k != 'history'}
    h = ck.get('history', {})
    print('='*70)
    print(f'beta={beta}: checkpoint 元信息')
    for k, v in info.items():
        print(f'  {k}: {v}')
    print(f'  history 长度: loss={len(h.get("loss",[]))}, bce={len(h.get("bce",[]))}, kld={len(h.get("kld",[]))}, active={len(h.get("active",[]))}')
    print(f'  loss曲线:   {[round(x,2) for x in h["loss"]]}')
    print(f'  bce曲线:    {[round(x,2) for x in h["bce"]]}')
    print(f'  kld曲线:    {[round(x,1) for x in h["kld"]]}')
    print(f'  active曲线: {h["active"]}')

    # 加载模型并复算测试集指标
    enc = Encoder(); dec = Decoder()
    enc.load_state_dict(ck['encoder_state_dict'])
    dec.load_state_dict(ck['decoder_state_dict'])
    enc.eval(); dec.eval()

    all_mu, all_lv, tot_bce, tot_kld, n = [], [], 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(-1, 784)
            mu, lv = enc(x)
            zr = mu + torch.exp(0.5*lv)*torch.randn_like(mu)
            xr = dec(zr)
            bce = F.binary_cross_entropy(xr, x, reduction='sum').item()
            kld = (-0.5*torch.sum(1+lv-mu.pow(2)-lv.exp())).item()
            tot_bce += bce; tot_kld += kld; n += x.shape[0]
            all_mu.append(mu); all_lv.append(lv)
    all_mu = torch.cat(all_mu); all_lv = torch.cat(all_lv)
    kl_per_dim = 0.5*(all_mu.pow(2)+all_lv.exp()-all_lv-1).mean(dim=0).numpy()
    active_test = int((kl_per_dim > 0.01).sum())
    # 逐样本平均重建PSNR
    with torch.no_grad():
        x = test_dataset.data[:200].float().view(-1,784)/255.0
        mu, lv = enc(x)
        xr = dec(mu)  # 用均值解码（确定性）
        mse = F.mse_loss(xr, x, reduction='mean').item()
        psnr = 10*np.log10(1.0/mse) if mse > 0 else 99

    print(f'  [复算-测试集] BCE/样本={tot_bce/n:.2f}, KL/样本={tot_kld/n:.2f}, 复算Loss={tot_bce/n+beta*tot_kld/n:.2f}')
    print(f'  [复算-测试集] 活跃维度={active_test}/20, μ解码MSE={mse:.5f} PSNR≈{psnr:.2f}dB')
    print(f'  [复算] kl_per_dim(测试集) = {np.round(kl_per_dim,4).tolist()}')
    out[f'beta_{beta}'] = {
        'ckpt_epoch': ck['epoch'], 'is_final': ck.get('is_final'),
        'final_loss': h['loss'][-1], 'final_bce': h['bce'][-1], 'final_kld': h['kld'][-1],
        'final_active_train': h['active'][-1],
        'test_bce_per_sample': tot_bce/n, 'test_kld_per_sample': tot_kld/n,
        'test_active': active_test, 'kl_per_dim_test': kl_per_dim.tolist(),
        'psnr_mu_decode': psnr,
        'loss_curve': h['loss'], 'bce_curve': h['bce'], 'kld_curve': h['kld'], 'active_curve': h['active'],
    }

# 插值样本对复现（与主代码相同逻辑）
enc_ae = Encoder(); dec_ae = Decoder()
ck0 = torch.load(os.path.join(EXP, 'vae_beta0_checkpoint.pth'), map_location='cpu', weights_only=False)
enc_ae.load_state_dict(ck0['encoder_state_dict']); dec_ae.load_state_dict(ck0['decoder_state_dict'])
ti = iter(test_loader)
imgs1, labels1 = next(ti); imgs2, labels2 = next(ti)
idx1, idx2 = next((i, j) for i in range(len(labels1)) for j in range(len(labels2)) if labels1[i] != labels2[j])
print('='*70)
print(f'插值样本对: labels1[{idx1}]={labels1[idx1].item()} → labels2[{idx2}]={labels2[idx2].item()} (batch0/batch1)')

with open(r'd:\works\deepinv\.audit_tmp\exp951_verify\verify_out.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('结果已存 verify_out.json')
