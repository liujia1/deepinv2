# -*- coding: utf-8 -*-
"""复核10.1-1(新权重): 诊断dec1是否对z₂输入不敏感(先验退化为无条件单高斯) + 采样多样性"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(0)
base = r"d:\works\deepinv\第10章 层级VAE与扩散的变分推导\第十章 配套实验\实验10.1-1"

class EncoderL1(nn.Module):
    def __init__(s):
        super().__init__(); s.fc1=nn.Linear(784,400); s.fc_mu=nn.Linear(400,20); s.fc_logvar=nn.Linear(400,20)
    def forward(s,x): h=F.relu(s.fc1(x)); return s.fc_mu(h), s.fc_logvar(h)
class EncoderL2(nn.Module):
    def __init__(s):
        super().__init__(); s.fc1=nn.Linear(20,200); s.fc_mu=nn.Linear(200,10); s.fc_logvar=nn.Linear(200,10)
    def forward(s,z1): h=F.relu(s.fc1(z1)); return s.fc_mu(h), s.fc_logvar(h)
class DecoderL1(nn.Module):
    def __init__(s):
        super().__init__(); s.fc1=nn.Linear(10,200); s.fc_mu=nn.Linear(200,20); s.fc_logvar=nn.Linear(200,20)
    def forward(s,z2): h=F.relu(s.fc1(z2)); return s.fc_mu(h), s.fc_logvar(h)
class DecoderL0(nn.Module):
    def __init__(s):
        super().__init__(); s.fc1=nn.Linear(20,400); s.fc2=nn.Linear(400,784)
    def forward(s,z1): return torch.sigmoid(s.fc2(F.relu(s.fc1(z1))))

ckh = torch.load(os.path.join(base,'hvae_mnist_checkpoint.pth'), map_location='cpu', weights_only=False)
cks = torch.load(os.path.join(base,'svae_mnist_checkpoint.pth'), map_location='cpu', weights_only=False)
enc1,enc2,dec1,dec0 = EncoderL1(),EncoderL2(),DecoderL1(),DecoderL0()
enc1.load_state_dict(ckh['enc1_state_dict']); enc2.load_state_dict(ckh['enc2_state_dict'])
dec1.load_state_dict(ckh['dec1_state_dict']); dec0.load_state_dict(ckh['dec0_state_dict'])
enc_s, dec_s = EncoderL1(), DecoderL0()
enc_s.load_state_dict(cks['encoder_state_dict']); dec_s.load_state_dict(cks['decoder_state_dict'])
for m in [enc1,enc2,dec1,dec0,enc_s,dec_s]: m.eval()

# 1) dec1 对 z₂ 输入的敏感性: mu1_dec 的跨样本标准差(输入N(0,I)时)
with torch.no_grad():
    z2 = torch.randn(2000, 10)
    mu1d, lv1d = dec1(z2)
    print(f"[dec1敏感性] 先验输入z₂~N(0,I)时 mu1_dec 各维std的均值 = {mu1d.std(dim=0).mean():.4f} (若≈0则dec1忽略输入)")
    print(f"[dec1敏感性] mu1_dec 全局std = {mu1d.std():.4f}; logvar1_dec 均值 = {lv1d.mean():.3f} std = {lv1d.std():.4f}")
    # 与聚合后验对比: q(z1|x)的边缘std
    x_all = []
    for x,_ in DataLoader(datasets.MNIST(os.path.join(base,'data'),train=False,download=False,transform=transforms.ToTensor()),batch_size=10000):
        x_all.append(x.view(-1,784))
    xa = torch.cat(x_all)[:5000]
    mu1q,_ = enc1(xa)
    print(f"[聚合后验] q(z₁|x) 边缘std均值 = {mu1q.std(dim=0).mean():.4f} (z₁实际信息尺度)")

# 2) 采样多样性: 100个先验样本的平均成对距离
def reparameterize(mu, logvar):
    return mu + torch.exp(0.5*logvar)*torch.randn_like(mu)

with torch.no_grad():
    def diversity(sample_fn, tag):
        ims = []
        for _ in range(5):
            ims.append(sample_fn(20))
        ims = torch.cat(ims)  # 100 x 784
        flat = ims
        # 成对L1距离均值
        d = torch.cdist(flat, flat, p=1).mean().item() / 784
        # 像素边际std
        print(f"[{tag}] 100样本平均成对L1距离/像素 = {d:.5f}, 像素std = {flat.std():.4f}")
    diversity(lambda n: dec0(reparameterize(*dec1(torch.randn(n,10)))), "层级VAE先验采样")
    diversity(lambda n: dec_s(torch.randn(n,20)), "单层VAE先验采样")

# 3) 用正确函数重测多样性
with torch.no_grad():
    def sample_h(n):
        mu1d, lv1d = dec1(torch.randn(n,10))
        return dec0(reparameterize(mu1d, lv1d))
    def sample_s(n):
        return dec_s(torch.randn(n,20))
    sh = torch.cat([sample_h(20) for _ in range(5)])
    ss = torch.cat([sample_s(20) for _ in range(5)])
    print(f"[多样性] 层级: 成对L1={torch.cdist(sh,sh,p=1).mean().item()/784:.5f} 像素std={sh.std():.4f}")
    print(f"[多样性] 单层: 成对L1={torch.cdist(ss,ss,p=1).mean().item()/784:.5f} 像素std={ss.std():.4f}")

# 4) 全测试集重建PSNR复核(均值路径) + BCE
test_loader = DataLoader(datasets.MNIST(os.path.join(base,'data'),train=False,download=False,transform=transforms.ToTensor()),batch_size=1000)
def eval_recon(fwd, tag):
    mse_l, bce_l = [], []
    with torch.no_grad():
        for x,_ in test_loader:
            x = x.view(-1,784); r = fwd(x).clamp(1e-6,1-1e-6)
            mse_l.append(((r-x)**2).mean(dim=1))
            bce_l.append(F.binary_cross_entropy(r, x, reduction='none').sum(dim=1))
    mse = torch.cat(mse_l); bce = torch.cat(bce_l)
    print(f"[{tag}] PSNR={(-10*torch.log10(mse+1e-12)).mean():.2f}dB  BCE/样本={bce.mean():.2f}")
eval_recon(lambda x: dec0(enc1(x)[0]), "层级VAE均值路径")
eval_recon(lambda x: dec_s(enc_s(x)[0]), "单层VAE均值路径")
