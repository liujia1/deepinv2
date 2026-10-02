# -*- coding: utf-8 -*-
"""扩展扫描(审计13.4-3续): A归一化版 zeta={15,30}, B chung版 zeta={5,10,30}"""
import os, sys, json
import numpy as np
import torch

device = torch.device('cpu')
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SAVE_DIR)

np.random.seed(42)
torch.manual_seed(42)

T = 1000
beta_min, beta_max = 1e-4, 0.02
betas = torch.linspace(beta_min, beta_max, T).to(device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
alpha_bars_prev = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
posterior_var = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)
sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)
beta_over_sqrt_1m_ab = betas / sqrt_one_minus_alpha_bars

import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        gn_groups = min(4, out_ch)
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(gn_groups, out_ch), nn.SiLU())
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.conv2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(gn_groups, out_ch), nn.SiLU())
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)

class SmallUNet(nn.Module):
    def __init__(self, time_dim=64):
        super().__init__()
        ch = [1, 16, 32, 64]
        self.time_mlp = nn.Sequential(SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU())
        self.down1 = ConvBlock(ch[0], ch[1], time_dim)
        self.down2 = ConvBlock(ch[1], ch[2], time_dim)
        self.down3 = ConvBlock(ch[2], ch[3], time_dim)
        self.bottleneck = ConvBlock(ch[3], ch[3], time_dim)
        self.up3 = ConvBlock(ch[3] + ch[2], ch[2], time_dim)
        self.up2 = ConvBlock(ch[2] + ch[1], ch[1], time_dim)
        self.up1 = ConvBlock(ch[1] + ch[0], ch[0], time_dim)
        self.out_conv = nn.Conv2d(ch[0], 1, 1)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x_t, t):
        t_emb = self.time_mlp(t)
        h1 = self.down1(x_t, t_emb)
        h2 = self.down2(self.pool(h1), t_emb)
        h3 = self.down3(self.pool(h2), t_emb)
        h = self.bottleneck(h3, t_emb)
        h = F.interpolate(h, size=(14, 14), mode='nearest')
        h = self.up3(torch.cat([h, h2], dim=1), t_emb)
        h = F.interpolate(h, size=(28, 28), mode='nearest')
        h = self.up2(torch.cat([h, h1], dim=1), t_emb)
        h = self.up1(torch.cat([h, x_t], dim=1), t_emb)
        return self.out_conv(h)


def dps_sample_norm(model, y, shape, zeta=1.0, seed=42):
    model.eval()
    torch.manual_seed(seed)
    x = torch.randn(shape, device=device)
    for t_idx in reversed(range(T)):
        t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        sqrt_ab_t = sqrt_alpha_bars[t_idx]
        sqrt_1mab_t = sqrt_one_minus_alpha_bars[t_idx]
        x = x.detach().requires_grad_(True)
        eps_pred = model(x, t)
        x0_hat = (x - sqrt_1mab_t * eps_pred) / sqrt_ab_t
        r = y - x0_hat
        loss = torch.sum(r ** 2)
        g = torch.autograd.grad(loss, x)[0]
        gn = g.norm()
        if gn > 1e-8:
            g = g / gn
        with torch.no_grad():
            eps_corr = eps_pred.detach() + zeta * sqrt_1mab_t * g
            mean = sqrt_recip_alphas[t_idx] * (x - beta_over_sqrt_1m_ab[t_idx] * eps_corr)
            x = mean if t_idx == 0 else mean + torch.sqrt(posterior_var[t_idx]) * torch.randn_like(x)
    return x.clamp(-1, 1)


def dps_sample_chung(model, y, shape, zeta=1.0, sigma_y=0.3, seed=42):
    model.eval()
    torch.manual_seed(seed)
    x = torch.randn(shape, device=device)
    for t_idx in reversed(range(T)):
        t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        sqrt_ab_t = sqrt_alpha_bars[t_idx]
        sqrt_1mab_t = sqrt_one_minus_alpha_bars[t_idx]
        x = x.detach().requires_grad_(True)
        eps_pred = model(x, t)
        x0_hat = (x - sqrt_1mab_t * eps_pred) / sqrt_ab_t
        r = y - x0_hat
        rnorm = torch.sqrt(torch.sum(r ** 2) + 1e-12)
        g = torch.autograd.grad(rnorm, x)[0]
        with torch.no_grad():
            mean = sqrt_recip_alphas[t_idx] * (x.detach() - beta_over_sqrt_1m_ab[t_idx] * eps_pred.detach())
            if t_idx > 0:
                x_prev = mean + torch.sqrt(posterior_var[t_idx]) * torch.randn_like(x)
            else:
                x_prev = mean
            step = zeta * posterior_var[t_idx] / (sigma_y ** 2 * sqrt_ab_t * rnorm)
            x = x_prev - step * g
    return x.clamp(-1, 1)


from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2 - 1)])
test_dataset = datasets.MNIST(os.path.join(SAVE_DIR, 'data'), train=False, download=False, transform=transform)
single_img = test_dataset[0][0].unsqueeze(0).to(device)
g_obs = torch.Generator(device='cpu').manual_seed(2026)
sigma_y = 0.3
y = single_img + torch.randn(single_img.shape, generator=g_obs, device=device) * sigma_y

model = SmallUNet().to(device)
ck = torch.load(os.path.join(SAVE_DIR, 'zeta_psnr_checkpoint.pth'), map_location=device, weights_only=False)
model.load_state_dict(ck['model_state_dict'])

def compute_psnr(pred, target):
    mse = torch.mean(((pred + 1) / 2 - (target + 1) / 2) ** 2).item()
    return 10 * np.log10(1.0 / (mse + 1e-10))

print(f"PSNR(y)={compute_psnr(y, single_img):.2f}", flush=True)
results = {}
for name, fn, zetas in [
    ("A_normalized", dps_sample_norm, [15.0, 30.0]),
    ("B_chung", dps_sample_chung, [5.0, 10.0, 30.0]),
]:
    res = {}
    for z in zetas:
        xh = fn(model, y, single_img.shape, z, seed=42)
        p = compute_psnr(xh, single_img)
        res[str(z)] = round(p, 2)
        print(f"[{name}] zeta={z}: PSNR={p:.2f} dB", flush=True)
    results[name] = res
with open(os.path.join(SAVE_DIR, 'ctrl_results_ext.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("done", flush=True)
