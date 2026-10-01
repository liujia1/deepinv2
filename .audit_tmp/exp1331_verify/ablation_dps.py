# -*- coding: utf-8 -*-
"""
实验13.3-1 复核对照实验：
裁决"PSNR低于含噪观测"的根因——
  B组: 不归一化梯度(讲义输入表所声称的做法), ζ∈{1,3,5}
  C组: 归一化梯度但更大ζ∈{15,30,60}
  D组: ζ=0 无条件采样基线(检验ζ=1是否≈无条件)
使用与 13.3-1.py 完全相同的 RNG 消耗顺序, 保证观测 y 逐位一致。
"""
import sys, io, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging, warnings
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SAVE_DIR, '.chinese'))
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=os.path.join(SAVE_DIR, '.chinese'))

np.random.seed(42)
import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cpu')

# ---- 与 13.3-1.py 相同的调度 ----
T = 200
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


def dps_sample_v(model, y, shape, zeta, normalize=True):
    """与13.3-1.py相同, 但可切换 归一化/不归一化 梯度"""
    model.eval()
    x = torch.randn(shape, device=device)
    for t_idx in reversed(range(T)):
        t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        sqrt_ab_t = sqrt_alpha_bars[t_idx]
        sqrt_1mab_t = sqrt_one_minus_alpha_bars[t_idx]
        with torch.no_grad():
            eps_pred = model(x, t)
        x = x.detach().requires_grad_(True)
        eps_pred_grad = model(x, t)
        x0_hat = (x - sqrt_1mab_t * eps_pred_grad) / sqrt_ab_t
        likelihood_loss = torch.sum((y - x0_hat) ** 2)
        g = torch.autograd.grad(likelihood_loss, x)[0]
        if normalize:
            bs = g.shape[0]
            gn = g.view(bs, -1).norm(dim=1).clamp(min=1e-8)
            g = g / gn.view(bs, 1, 1, 1)
        x = x.detach()
        eps_corrected = eps_pred.detach() + zeta * sqrt_1mab_t * g
        with torch.no_grad():
            model_mean = sqrt_recip_alphas[t_idx] * (x - beta_over_sqrt_1m_ab[t_idx] * eps_corrected)
            if t_idx == 0:
                x = model_mean
            else:
                x = model_mean + torch.sqrt(posterior_var[t_idx]) * torch.randn_like(x)
    return x.clamp(-1, 1)


def compute_psnr(pred, target):
    pred_01 = (pred + 1) / 2
    target_01 = (target + 1) / 2
    mse = torch.mean((pred_01 - target_01) ** 2).item()
    return 10 * np.log10(1.0 / (mse + 1e-10))


# ---- 数据与RNG消耗顺序与 13.3-1.py 完全一致 ----
print("加载MNIST...")
data_dir = os.path.join(SAVE_DIR, 'data')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1),
])
train_dataset = datasets.MNIST(data_dir, train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(data_dir, train=False, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 原脚本 train_model 中先创建 SmallUNet()(消耗RNG) 再加载checkpoint
model = SmallUNet().to(device)
ckpt = torch.load(os.path.join(SAVE_DIR, 'dps_denoise_checkpoint.pth'), map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
print(f"checkpoint: epoch={ckpt['epoch']}, is_final={ckpt['is_final']}")

test_images = next(iter(test_loader))[0][:4].to(device)
sigma_y = 0.3
noise_obs = torch.randn_like(test_images) * sigma_y
y_denoise = test_images + noise_obs
print(f"含噪观测PSNR: {compute_psnr(y_denoise, test_images):.2f} dB")

results = {}

# D组: 无条件基线 (zeta=0)
print("\n[D] 无条件采样 (zeta=0)...")
x_d = dps_sample_v(model, y_denoise, test_images.shape, zeta=0.0)
results['D_zeta0_uncond'] = compute_psnr(x_d, test_images)
print(f"  PSNR={results['D_zeta0_uncond']:.2f} dB")

# B组: 不归一化(讲义输入表声称的做法), zeta∈{1,3,5}
for z in [1.0, 3.0, 5.0]:
    print(f"[B] 不归一化 zeta={z}...")
    xb = dps_sample_v(model, y_denoise, test_images.shape, zeta=z, normalize=False)
    results[f'B_unnorm_zeta{z}'] = compute_psnr(xb, test_images)
    print(f"  PSNR={results[f'B_unnorm_zeta{z}']:.2f} dB")

# C组: 归一化但更大zeta
for z in [15.0, 30.0, 60.0]:
    print(f"[C] 归一化 zeta={z}...")
    xc = dps_sample_v(model, y_denoise, test_images.shape, zeta=z, normalize=True)
    results[f'C_norm_zeta{z}'] = compute_psnr(xc, test_images)
    print(f"  PSNR={results[f'C_norm_zeta{z}']:.2f} dB")

import json
with open(os.path.join(SAVE_DIR, 'ablation_results.json'), 'w', encoding='utf-8') as f:
    json.dump({k: round(v, 2) for k, v in results.items()}, f, ensure_ascii=False, indent=2)
print("\n汇总:", json.dumps({k: round(v, 2) for k, v in results.items()}, ensure_ascii=False))
