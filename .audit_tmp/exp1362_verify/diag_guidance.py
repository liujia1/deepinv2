# -*- coding: utf-8 -*-
"""诊断：13.6-2 DPS去模糊低PSNR是固有属性还是引导缩放配置问题？
复用已训练checkpoint, 对比不同引导缩放:
  A) 现行: eps + zeta * sigma_t * unit_grad   (sigma_t=sqrt(1-abar), t->0时消失)
  B) 去掉sigma_t: eps + zeta * unit_grad        (引导不随t衰减)
  C) 真DPS(不单位化): eps + zeta * grad          (保留残差量级, zeta~1/sigma_y^2)
"""
EXPDIR = r'd:\works\deepinv\第13章 条件生成与逆问题求解\第十三章 配套实验\实验13.6-2'
import os, sys, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
device = torch.device('cpu')

T = 200
beta_min, beta_max = 1e-4, 0.02
betas = torch.linspace(beta_min, beta_max, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, 0)
alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
posterior_var = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)
sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)
beta_over_sqrt_1m_ab = betas / sqrt_one_minus_alpha_bars

import importlib.util
spec = importlib.util.spec_from_file_location("m", os.path.join(os.path.dirname(os.path.abspath(__file__)), "13.6-2.py"))
# 不能直接import主脚本(会重跑训练/采样)。改为手动复制最小模型定义。
import torch.nn as nn
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb = np.log(10000) / (half - 1)
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
        super().__init__(); ch = [1, 16, 32, 64]
        self.time_mlp = nn.Sequential(SinusoidalTimeEmbedding(td), nn.Linear(td, td), nn.SiLU())
        self.down1 = ConvBlock(ch[0], ch[1], td); self.down2 = ConvBlock(ch[1], ch[2], td); self.down3 = ConvBlock(ch[2], ch[3], td)
        self.bottleneck = ConvBlock(ch[3], ch[3], td); self.up3 = ConvBlock(ch[3]+ch[2], ch[2], td); self.up2 = ConvBlock(ch[2]+ch[1], ch[1], td); self.up1 = ConvBlock(ch[1]+ch[0], ch[0], td)
        self.out_conv = nn.Conv2d(ch[0], 1, 1); self.pool = nn.MaxPool2d(2)
    def forward(self, x_t, t):
        te = self.time_mlp(t); h1 = self.down1(x_t, te); h2 = self.down2(self.pool(h1), te); h3 = self.down3(self.pool(h2), te)
        h = self.bottleneck(h3, te)
        h = F.interpolate(h, size=(14, 14), mode='nearest'); h = self.up3(torch.cat([h, h2], 1), te)
        h = F.interpolate(h, size=(28, 28), mode='nearest'); h = self.up2(torch.cat([h, h1], 1), te)
        h = self.up1(torch.cat([h, x_t], 1), te); return self.out_conv(h)

model = SmallUNet()
ck = torch.load(os.path.join(EXPDIR, 'dps_inpaint_checkpoint.pth'), map_location='cpu', weights_only=False)
model.load_state_dict(ck['model_state_dict']); model.eval()

from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x*2-1)])
test = datasets.MNIST(os.path.join(EXPDIR, 'data'), train=False, download=False, transform=transform)
test_images = torch.stack([test[i][0] for i in range(4)]).to(device)

class GaussianBlurOperator:
    def __init__(self, ks=5, sigma=1.5):
        c = torch.arange(ks, dtype=torch.float32) - ks//2
        g = torch.exp(-(c**2)/(2*sigma**2)); k = torch.outer(g, g); k = k/k.sum()
        self.kernel = k.unsqueeze(0).unsqueeze(0); self.pad = ks//2
    def __call__(self, x): return F.conv2d(x, self.kernel, padding=self.pad)

def psnr(pred, target):
    p = (pred+1)/2; t = (target+1)/2
    return 10*np.log10(1.0/(torch.mean((p-t)**2).item()+1e-10))

def dps_sample(model, y, fwd, shape, zeta, mode):
    x = torch.randn(shape)
    for ti in reversed(range(T)):
        t = torch.full((shape[0],), ti, dtype=torch.long)
        sat = sqrt_one_minus_alpha_bars[ti]
        with torch.no_grad(): eps_pred = model(x, t)
        x = x.detach().requires_grad_(True)
        epsg = model(x, t)
        x0 = (x - sat*epsg)/sqrt_alpha_bars[ti]
        loss = torch.sum((y - fwd(x0))**2)
        g = torch.autograd.grad(loss, x)[0]
        if mode in ('A', 'B'):
            bn = g.view(g.shape[0], -1).norm(dim=1).clamp(min=1e-8)
            g = g/bn.view(-1, 1, 1, 1)
        x = x.detach(); eps_pred = eps_pred.detach()
        if mode == 'A':   corr = zeta * sat * g
        elif mode == 'B': corr = zeta * g
        elif mode == 'C': corr = zeta * g
        eps_c = eps_pred + corr
        with torch.no_grad():
            mean = sqrt_recip_alphas[ti]*(x - beta_over_sqrt_1m_ab[ti]*eps_c)
            x = mean if ti == 0 else mean + torch.sqrt(posterior_var[ti])*torch.randn_like(x)
    return x.clamp(-1, 1)

blur = GaussianBlurOperator(7, 2.0)
torch.manual_seed(42); np.random.seed(42)
sigma_y = 0.02
y_blur = blur(test_images) + torch.randn_like(test_images)*sigma_y
print('obs PSNR (blur):', round(psnr(y_blur, test_images), 2))
for mode, zs in [('A', [1,3,5]), ('B', [0.1,0.3,1.0]), ('C', [1e-3,1e-2,5e-2])]:
    for z in zs:
        torch.manual_seed(123)
        xh = dps_sample(model, y_blur, blur, test_images.shape, z, mode)
        print(f'  mode {mode} zeta={z}: PSNR={psnr(xh, test_images):.2f}')
