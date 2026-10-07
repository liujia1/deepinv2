# -*- coding: utf-8 -*-
import os, sys, io, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

BASE = r'd:\works\deepinv\第9章 VAE与重参数化\第九章 配套实验\实验9.1-1'
dev = torch.device('cpu')

class Encoder(nn.Module):
    def __init__(s, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__(); s.fc1=nn.Linear(input_dim,hidden_dim); s.fc_mu=nn.Linear(hidden_dim,latent_dim); s.fc_logvar=nn.Linear(hidden_dim,latent_dim)
    def forward(s,x):
        h=F.relu(s.fc1(x)); return s.fc_mu(h), s.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(s, latent_dim=20, hidden_dim=400, output_dim=784):
        super().__init__(); s.fc1=nn.Linear(latent_dim,hidden_dim); s.fc2=nn.Linear(hidden_dim,output_dim)
    def forward(s,z):
        return torch.sigmoid(s.fc2(F.relu(s.fc1(z))))

def load(tag, path):
    ck = torch.load(path, map_location=dev, weights_only=False)
    print(f"\n=== {tag} checkpoint ===")
    print("  keys:", list(ck.keys()))
    print("  epoch:", ck.get('epoch'), " beta:", ck.get('beta'), " is_final:", ck.get('is_final'))
    print("  final loss:", ck.get('loss'))
    tl = ck.get('train_losses'); print("  len(train_losses):", len(tl) if tl else None, " last:", tl[-1] if tl else None)
    e=Encoder().to(dev); d=Decoder().to(dev)
    e.load_state_dict(ck['encoder_state_dict']); d.load_state_dict(ck['decoder_state_dict'])
    e.eval(); d.eval()
    return e,d,ck

vae_e, vae_d, vae_ck = load('VAE', os.path.join(BASE,'vae_mnist_checkpoint.pth'))
ae_e, ae_d, ae_ck = load('AE', os.path.join(BASE,'ae_mnist_checkpoint.pth'))

tf = transforms.Compose([transforms.ToTensor()])
test = datasets.MNIST(os.path.join(BASE,'data'), train=False, download=False, transform=tf)
tl = DataLoader(test, batch_size=128, shuffle=False, num_workers=0)

def psnr(a,b):
    mse = torch.mean((a-b)**2).item()
    return 10*np.log10(1.0/mse) if mse>0 else float('inf')

with torch.no_grad():
    imgs,_ = next(iter(tl)); x = imgs.view(-1,784)
    vmu,vlv = vae_e(x); vmu,vlv = vae_e(x)
    vrec = vae_d(vmu)
    amu,alv = ae_e(x); arec = ae_d(amu)
    print("\n=== Reconstruction PSNR (first 128 test imgs, using mu) ===")
    print(f"  VAE recon PSNR: {psnr(vrec,x):.3f} dB")
    print(f"  AE  recon PSNR: {psnr(arec,x):.3f} dB")
    print("\n=== logvar statistics over latent dims (test batch) ===")
    print(f"  VAE mean logvar: {vlv.mean().item():.3f}  (min {vlv.min().item():.2f}, max {vlv.max().item():.2f})")
    print(f"  AE  mean logvar: {alv.mean().item():.3f}  (min {alv.min().item():.2f}, max {alv.max().item():.2f})")
    # latent mu scale: VAE should be ~N(0,1)-ish; AE unconstrained
    print(f"  VAE mu std: {vmu.std().item():.3f}  AE mu std: {amu.std().item():.3f}")
    print(f"  VAE mu absmean: {vmu.abs().mean().item():.3f}  AE mu absmean: {amu.abs().mean().item():.3f}")

# prior sampling: decode N(0,I) and check mean brightness / structure
with torch.no_grad():
    torch.manual_seed(0)
    z = torch.randn(2000,20)
    vs = vae_d(z); as_ = ae_d(z)
    print("\n=== Prior-sample decode statistics (2000 z~N(0,I)) ===")
    print(f"  VAE samples: mean {vs.mean().item():.3f} std {vs.std().item():.3f}")
    print(f"  AE  samples: mean {as_.mean().item():.3f} std {as_.std().item():.3f}")
