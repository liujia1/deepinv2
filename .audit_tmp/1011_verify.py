# -*- coding: utf-8 -*-
"""复核10.1-1: 用已保存checkpoint重算步骤3 ELBO逐项 + 重建质量定量指标"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cpu')

base = r"d:\works\deepinv\第10章 层级VAE与扩散的变分推导\第十章 配套实验\实验10.1-1"

# ---- 与10.1-1.py完全一致的模型定义 ----
class EncoderL1(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

class EncoderL2(nn.Module):
    def __init__(self, latent1_dim=20, hidden_dim=200, latent2_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(latent1_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent2_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent2_dim)
    def forward(self, z1):
        h = F.relu(self.fc1(z1))
        return self.fc_mu(h), self.fc_logvar(h)

class DecoderL1(nn.Module):
    def __init__(self, latent2_dim=10, hidden_dim=200, latent1_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(latent2_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent1_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent1_dim)
    def forward(self, z2):
        h = F.relu(self.fc1(z2))
        return self.fc_mu(h), self.fc_logvar(h)

class DecoderL0(nn.Module):
    def __init__(self, latent1_dim=20, hidden_dim=400, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent1_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, z1):
        h = F.relu(self.fc1(z1))
        return torch.sigmoid(self.fc2(h))

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# ---- 加载数据 ----
data_dir = os.path.join(base, 'data')
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(data_dir, train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

# ---- 加载checkpoint ----
ckh = torch.load(os.path.join(base, 'hvae_mnist_checkpoint.pth'), map_location=device, weights_only=False)
cks = torch.load(os.path.join(base, 'svae_mnist_checkpoint.pth'), map_location=device, weights_only=False)
enc1 = EncoderL1(); enc2 = EncoderL2(); dec1 = DecoderL1(); dec0 = DecoderL0()
enc1.load_state_dict(ckh['enc1_state_dict']); enc2.load_state_dict(ckh['enc2_state_dict'])
dec1.load_state_dict(ckh['dec1_state_dict']); dec0.load_state_dict(ckh['dec0_state_dict'])
enc_s = EncoderL1(); dec_s = DecoderL0()
enc_s.load_state_dict(cks['encoder_state_dict']); dec_s.load_state_dict(cks['decoder_state_dict'])
for m in [enc1, enc2, dec1, dec0, enc_s, dec_s]:
    m.eval()

# ---- 步骤3重算: 层级ELBO逐项 (随机采样路径, 与讲义一致) ----
total_recon, total_kl1, total_kl2, n = 0, 0, 0, 0
with torch.no_grad():
    for x, _ in test_loader:
        x = x.view(-1, 784)
        mu1, logvar1 = enc1(x)
        z1 = reparameterize(mu1, logvar1)
        mu2, logvar2 = enc2(z1)
        z2 = reparameterize(mu2, logvar2)
        mu1d, logvar1d = dec1(z2)
        z1d = reparameterize(mu1d, logvar1d)
        xr = dec0(z1d)
        total_recon += F.binary_cross_entropy(xr, x, reduction='sum').item()
        total_kl1 += kl_divergence(mu1, logvar1).item()
        total_kl2 += kl_divergence(mu2, logvar2).item()
        n += x.shape[0]
ar, ak1, ak2 = total_recon/n, total_kl1/n, total_kl2/n
print(f"[层级VAE 测试集 随机采样] recon={ar:.2f} kl1={ak1:.2f} kl2={ak2:.2f} kl_total={ak1+ak2:.2f} ELBO={-(ar+ak1+ak2):.2f} kl1/kl2={ak1/ak2:.3f}")

# ---- 单次前向的波动: 换seed重跑一遍看数值稳定性 ----
torch.manual_seed(7)
total_recon, total_kl1, total_kl2, n = 0, 0, 0, 0
with torch.no_grad():
    for x, _ in test_loader:
        x = x.view(-1, 784)
        mu1, logvar1 = enc1(x); z1 = reparameterize(mu1, logvar1)
        mu2, logvar2 = enc2(z1); z2 = reparameterize(mu2, logvar2)
        mu1d, logvar1d = dec1(z2); z1d = reparameterize(mu1d, logvar1d)
        xr = dec0(z1d)
        total_recon += F.binary_cross_entropy(xr, x, reduction='sum').item()
        total_kl1 += kl_divergence(mu1, logvar1).item()
        total_kl2 += kl_divergence(mu2, logvar2).item()
        n += x.shape[0]
print(f"[换seed=7] recon={total_recon/n:.2f} kl1={total_kl1/n:.2f} kl2={total_kl2/n:.2f}")

# ---- 单层VAE测试集 ----
total_recon_s, total_kl_s, n = 0, 0, 0
with torch.no_grad():
    for x, _ in test_loader:
        x = x.view(-1, 784)
        mu, logvar = enc_s(x)
        z = reparameterize(mu, logvar)
        xr = dec_s(z)
        total_recon_s += F.binary_cross_entropy(xr, x, reduction='sum').item()
        total_kl_s += kl_divergence(mu, logvar).item()
        n += x.shape[0]
ars, aks = total_recon_s/n, total_kl_s/n
print(f"[单层VAE 测试集] recon={ars:.2f} kl={aks:.2f} ELBO={-(ars+aks):.2f}")

# ---- 定量重建质量: 全测试集, 均值路径(与图2一致) ----
def recon_metrics(enc_fwd, tag):
    mse_sum, psnrs = 0.0, []
    with torch.no_grad():
        for x, _ in test_loader:
            xr = enc_fwd(x.view(-1, 784)).clamp(1e-6, 1 - 1e-6)
            x0 = x.view(-1, 784)
            mse = ((xr - x0) ** 2).mean(dim=1)
            mse_sum += mse.sum().item()
            psnrs.append((-10 * torch.log10(mse + 1e-12)).numpy())
    ps = np.concatenate(psnrs)
    print(f"[{tag}] 全测试集 MSE={mse_sum/len(test_dataset):.5f} 平均PSNR={ps.mean():.2f} dB")

recon_metrics(lambda x: dec0(enc1(x)[0]), "层级VAE-均值路径(图2同款)")
recon_metrics(lambda x: dec_s(enc_s(x)[0]), "单层VAE-均值路径(图2同款)")
# 训练目标同款路径: z1 -> z2 -> dec1采样 -> dec0
def hvae_train_path(x):
    mu1, lv1 = enc1(x); z1 = reparameterize(mu1, lv1)
    mu2, lv2 = enc2(z1); z2 = reparameterize(mu2, lv2)
    mu1d, lv1d = dec1(z2); z1d = reparameterize(mu1d, lv1d)
    return dec0(z1d)
recon_metrics(hvae_train_path, "层级VAE-训练随机路径")

# ---- KL1 vs KL2 逐epoch关系 (checkpoint history) ----
h = ckh['history']
kl1_last5 = h['kl1'][-5:]; kl2_last5 = h['kl2'][-5:]
print(f"[训练末5epoch] kl1={['%.3f'%v for v in kl1_last5]} kl2={['%.3f'%v for v in kl2_last5]}")
cross = [i+1 for i in range(len(h['kl1'])) if h['kl1'][i] < h['kl2'][i]]
print(f"[KL1<KL2的epoch] {cross} (共{len(cross)}/20)")
print(f"[训练末epoch] kl1={h['kl1'][-1]:.3f} kl2={h['kl2'][-1]:.3f} ratio={h['kl1'][-1]/h['kl2'][-1]:.3f}")

# ---- 先验采样质量粗检: 层级 vs 单层 的采样图与训练图是否一致(不可逐位复现, 仅形态) ----
with torch.no_grad():
    z2p = torch.randn(10, 10)
    mu1p, lv1p = dec1(z2p); z1p = reparameterize(mu1p, lv1p)
    sh = dec0(z1p)
    zp = torch.randn(10, 20)
    ss = dec_s(zp)
print(f"[采样均值] 层级={sh.mean().item():.3f} 单层={ss.mean().item():.3f}")
