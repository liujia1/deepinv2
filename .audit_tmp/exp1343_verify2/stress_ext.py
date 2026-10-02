# -*- coding: utf-8 -*-
"""审计13.4-3压力测试: 归一化实现 zeta={30(锚点),50,100}, Chung原版 zeta={100}
检验"倒U下降段被结构性抑制"是否随zeta继续增大而失效(免责裁决加固)。
采样函数与主脚本逐行一致(CRN seed=42)。"""
import os, sys, json
import numpy as np
import torch

device = torch.device('cpu')
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

T = 1000
beta_min, beta_max = 1e-4, 0.02
betas = torch.linspace(beta_min, beta_max, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])
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


class IdentityOperator:
    def __call__(self, x):
        return x


def dps_sample(model, y, forward_op, shape, zeta=1.0, seed=None):
    """与主脚本 dps_sample 完全一致"""
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(shape, device=device)
    for t_idx in reversed(range(T)):
        t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        sqrt_ab_t = sqrt_alpha_bars[t_idx]
        sqrt_1mab_t = sqrt_one_minus_alpha_bars[t_idx]
        x = x.detach().requires_grad_(True)
        eps_pred = model(x, t)
        x0_hat = (x - sqrt_1mab_t * eps_pred) / sqrt_ab_t
        Ax0_hat = forward_op(x0_hat)
        likelihood_loss = torch.sum((y - Ax0_hat) ** 2)
        likelihood_grad = torch.autograd.grad(likelihood_loss, x)[0]
        grad_norm = likelihood_grad.norm()
        if grad_norm > 1e-8:
            likelihood_grad = likelihood_grad / grad_norm
        with torch.no_grad():
            eps_corrected = eps_pred.detach() + zeta * sqrt_1mab_t * likelihood_grad
            model_mean = sqrt_recip_alphas[t_idx] * (
                x - beta_over_sqrt_1m_ab[t_idx] * eps_corrected
            )
            if t_idx == 0:
                x = model_mean
            else:
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(posterior_var[t_idx]) * noise
    return x.clamp(-1, 1)


def dps_sample_chung(model, y, forward_op, shape, zeta=1.0, sigma_y=0.3, seed=None):
    """与主脚本 dps_sample_chung 完全一致"""
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(shape, device=device)
    for t_idx in reversed(range(T)):
        t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        sqrt_ab_t = sqrt_alpha_bars[t_idx]
        sqrt_1mab_t = sqrt_one_minus_alpha_bars[t_idx]
        x = x.detach().requires_grad_(True)
        eps_pred = model(x, t)
        x0_hat = (x - sqrt_1mab_t * eps_pred) / sqrt_ab_t
        r = y - forward_op(x0_hat)
        rnorm = torch.sqrt(torch.sum(r ** 2) + 1e-12)
        grad = torch.autograd.grad(rnorm, x)[0]
        with torch.no_grad():
            model_mean = sqrt_recip_alphas[t_idx] * (
                x.detach() - beta_over_sqrt_1m_ab[t_idx] * eps_pred.detach()
            )
            if t_idx > 0:
                x_prev = model_mean + torch.sqrt(posterior_var[t_idx]) * torch.randn_like(x)
            else:
                x_prev = model_mean
            step = zeta * posterior_var[t_idx] / (sigma_y ** 2 * sqrt_ab_t * rnorm)
            x = x_prev - step * grad
    return x.clamp(-1, 1)


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2 - 1)])
test_dataset = datasets.MNIST(os.path.join(SAVE_DIR, 'data'), train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
single_img = next(iter(test_loader))[0][:1].to(device)

_gen_obs = torch.Generator().manual_seed(2026)
sigma_y_denoise = 0.3
y_single = single_img + torch.randn(single_img.shape, generator=_gen_obs).to(device) * sigma_y_denoise

model = SmallUNet().to(device)
ck = torch.load(os.path.join(SAVE_DIR, 'zeta_psnr_checkpoint.pth'), map_location=device, weights_only=False)
model.load_state_dict(ck['model_state_dict'])


def compute_psnr(pred, target):
    pred_01 = (pred + 1) / 2
    target_01 = (target + 1) / 2
    mse = torch.mean((pred_01 - target_01)**2).item()
    return 10 * np.log10(1.0 / (mse + 1e-10))


identity_op = IdentityOperator()
print(f"PSNR(y)={compute_psnr(y_single, single_img):.2f}", flush=True)
results = {}
plan = [
    ("A_normalized", dps_sample, [30.0, 50.0, 100.0]),
    ("B_chung", dps_sample_chung, [100.0]),
]
for name, fn, zetas in plan:
    res = {}
    for z in zetas:
        xh = fn(model, y_single, identity_op, single_img.shape, zeta=z, seed=42)
        p = compute_psnr(xh, single_img)
        res[str(z)] = round(p, 2)
        print(f"[{name}] zeta={z}: PSNR={p:.2f} dB", flush=True)
    results[name] = res
with open(os.path.join(SAVE_DIR, 'stress_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("done", flush=True)
