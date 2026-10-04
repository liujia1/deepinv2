# -*- coding: utf-8 -*-
"""去模糊能否通过更大zeta超过观测? 并给维纳反卷积/纯先验基线定位难度。"""
EXPDIR = r'd:\works\deepinv\第13章 条件生成与逆问题求解\第十三章 配套实验\实验13.6-2'
import os, sys, numpy as np, torch
import torch.nn.functional as F
import torch.nn as nn
device = torch.device('cpu')
T = 200
betas = torch.linspace(1e-4, 0.02, T); alphas = 1 - betas
ab = torch.cumprod(alphas, 0); abp = torch.cat([torch.ones(1), ab[:-1]])
sab = torch.sqrt(ab); s1mab = torch.sqrt(1 - ab)
pvar = betas * (1 - abp) / (1 - ab); sra = 1 / torch.sqrt(alphas)
bos = betas / s1mab

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2; emb = np.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], -1)
class ConvBlock(nn.Module):
    def __init__(self, ic, oc, td):
        super().__init__(); g = min(4, oc)
        self.conv1 = nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1), nn.GroupNorm(g, oc), nn.SiLU())
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(td, oc))
        self.conv2 = nn.Sequential(nn.Conv2d(oc, oc, 3, padding=1), nn.GroupNorm(g, oc), nn.SiLU())
        self.shortcut = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()
    def forward(self, x, te):
        h = self.conv1(x); h = h + self.time_proj(te)[:, :, None, None]; h = self.conv2(h)
        return h + self.shortcut(x)
class SmallUNet(nn.Module):
    def __init__(self, td=64):
        super().__init__(); ch = [1,16,32,64]
        self.time_mlp = nn.Sequential(SinusoidalTimeEmbedding(td), nn.Linear(td, td), nn.SiLU())
        self.down1 = ConvBlock(ch[0],ch[1],td); self.down2 = ConvBlock(ch[1],ch[2],td); self.down3 = ConvBlock(ch[2],ch[3],td)
        self.bottleneck = ConvBlock(ch[3],ch[3],td); self.up3 = ConvBlock(ch[3]+ch[2],ch[2],td); self.up2 = ConvBlock(ch[2]+ch[1],ch[1],td); self.up1 = ConvBlock(ch[1]+ch[0],ch[0],td)
        self.out_conv = nn.Conv2d(ch[0],1,1); self.pool = nn.MaxPool2d(2)
    def forward(self, x_t, t):
        te = self.time_mlp(t); h1 = self.down1(x_t,te); h2 = self.down2(self.pool(h1),te); h3 = self.down3(self.pool(h2),te)
        h = self.bottleneck(h3,te)
        h = F.interpolate(h,(14,14),mode='nearest'); h = self.up3(torch.cat([h,h2],1),te)
        h = F.interpolate(h,(28,28),mode='nearest'); h = self.up2(torch.cat([h,h1],1),te)
        h = self.up1(torch.cat([h,x_t],1),te); return self.out_conv(h)

model = SmallUNet(); ck = torch.load(os.path.join(EXPDIR,'dps_inpaint_checkpoint.pth'), map_location='cpu', weights_only=False)
model.load_state_dict(ck['model_state_dict']); model.eval()

from torchvision import datasets, transforms
tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x*2-1)])
ds = datasets.MNIST(os.path.join(EXPDIR,'data'), train=False, download=False, transform=tf)
X = torch.stack([ds[i][0] for i in range(4)])

class Blur:
    def __init__(s, ks=7, sigma=2.0):
        c = torch.arange(ks,dtype=torch.float32)-ks//2; g = torch.exp(-(c**2)/(2*sigma**2)); k = torch.outer(g,g); k/=k.sum()
        s.k = k.unsqueeze(0).unsqueeze(0); s.p = ks//2; s.ks=ks; s.sigma=sigma
    def __call__(s, x): return F.conv2d(x, s.k, padding=s.p)
blur = Blur()

def psnr(p, t):
    p=(p+1)/2; t=(t+1)/2; return 10*np.log10(1.0/(torch.mean((p-t)**2).item()+1e-10))

torch.manual_seed(42); np.random.seed(42)
y = blur(X) + torch.randn_like(X)*0.02
print('obs PSNR:', round(psnr(y, X),2))

# 维纳反卷积基线(频域, 已知核)
def wiener(y, ks, sigma, noise=0.001):
    from numpy.fft import fft2, ifft2, fftshift
    c = np.arange(ks)-ks//2; g = np.exp(-(c**2)/(2*sigma**2)); k = np.outer(g,g); k/=k.sum()
    K = fft2(k, s=y.shape[-2:]); Y = fft2(y.numpy()[0,0]); H=K; Hc=np.conj(K)
    W = Hc/(np.abs(H)**2 + noise)
    r = np.real(ifft2(W*Y)); r = (r-r.min())/(r.max()-r.min()+1e-9)*2-1
    return torch.tensor(r, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
w = wiener(y, 7, 2.0, noise=0.02)
print('wiener PSNR:', round(psnr(w, X[:1]),2), '(仅第1张, 参考)')

def dps(mode_zeta_list):
    for zeta in mode_zeta_list:
        x = torch.randn(X.shape)
        for ti in reversed(range(T)):
            t = torch.full((X.shape[0],), ti, dtype=torch.long)
            with torch.no_grad(): ep = model(x, t)
            xg = x.detach().requires_grad_(True)
            eg = model(xg, t); x0 = (xg - s1mab[ti]*eg)/sab[ti]
            loss = torch.sum((y - blur(x0))**2)
            g = torch.autograd.grad(loss, xg)[0]
            bn = g.view(g.shape[0],-1).norm(dim=1).clamp(min=1e-8); g = g/bn.view(-1,1,1,1)
            x = x.detach(); ep = ep.detach()
            epc = ep + zeta*s1mab[ti]*g
            with torch.no_grad():
                mean = sra[ti]*(x - bos[ti]*epc)
                x = mean if ti==0 else mean + torch.sqrt(pvar[ti])*torch.randn_like(x)
        print(f'  deblur zeta={zeta}: PSNR={psnr(x.clamp(-1,1), X):.2f}')
torch.manual_seed(100); dps([5, 8, 12, 20, 40])
