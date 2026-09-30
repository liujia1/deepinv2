# -*- coding: utf-8 -*-
"""14.3-3 独立核查：
1) 复现主脚本的 RNG 顺序，确认样本一致（距离逐位吻合）
2) 用独立证据裁决"RF 1步质量较差"是固有属性还是免责话术：
   - 多样性（生成批内平均两两L2距离）
   - 模糊度（样本均值图的对比度 / 单样本能量）
   - 真实样本基线的最近邻距离
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FOLDER = r"d:\works\deepinv\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.3-3"
sys.path.insert(0, FOLDER)

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cpu')

# ---- 与主脚本相同的模型定义（从 14.3-3.py 复制，保证结构一致）----
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

T = 200
beta_min, beta_max = 1e-4, 0.02
betas = torch.linspace(beta_min, beta_max, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
posterior_var = betas * (1 - torch.cat([torch.ones(1), alpha_bars[:-1]])) / (1 - alpha_bars)
sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)
beta_over_sqrt_1m_ab = betas / torch.sqrt(1 - alpha_bars)

rf_model = SmallUNet()
rf_model.load_state_dict(torch.load(os.path.join(FOLDER, 'rf_final.pth'), map_location='cpu', weights_only=False)['model_state_dict'])
rf_model.eval()
ddpm_model = SmallUNet()
ddpm_model.load_state_dict(torch.load(os.path.join(FOLDER, 'ddpm_final.pth'), map_location='cpu', weights_only=False)['model_state_dict'])
ddpm_model.eval()

from torchvision import datasets, transforms
test_dataset = datasets.MNIST(os.path.join(FOLDER, 'data'), train=False, download=False, transform=transforms.ToTensor())

shape = (8, 1, 28, 28)

@torch.no_grad()
def flow_ode_sample(model, n_steps):
    x = torch.randn(shape)
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t_val = step / n_steps
        t_int = torch.full((shape[0],), int((1 - t_val) * (T - 1)), dtype=torch.long)
        v = model(x, t_int)
        x = x + v * dt
    return x.clamp(0, 1)

@torch.no_grad()
def ddpm_sample(model):
    x = torch.randn(shape)
    for t_idx in reversed(range(T)):
        t = torch.full((shape[0],), t_idx, dtype=torch.long)
        pred = model(x, t)
        mean = sqrt_recip_alphas[t_idx] * (x - beta_over_sqrt_1m_ab[t_idx] * pred)
        if t_idx == 0:
            x = mean
        else:
            x = mean + torch.sqrt(posterior_var[t_idx]) * torch.randn_like(x)
    return x.clamp(0, 1)

# 复现主脚本 RNG 顺序: step2 循环 [1,5,10,50]，然后 DDPM，再 rf50，再 rf10
s1 = flow_ode_sample(rf_model, 1)
s5 = flow_ode_sample(rf_model, 5)
s10a = flow_ode_sample(rf_model, 10)
s50a = flow_ode_sample(rf_model, 50)
ddpm_s = ddpm_sample(ddpm_model)
rf50 = flow_ode_sample(rf_model, 50)
rf10 = flow_ode_sample(rf_model, 10)

test_samples = torch.stack([test_dataset[i][0] for i in range(100)])

def nn_dist(gen):
    g = gen.view(gen.shape[0], -1)
    r = test_samples.view(test_samples.shape[0], -1)
    return torch.cdist(g, r).min(dim=1)[0].mean().item()

def diversity(gen):
    g = gen.view(gen.shape[0], -1)
    d = torch.cdist(g, g)
    iu = torch.triu_indices(g.shape[0], g.shape[0], offset=1)
    return d[iu[0], iu[1]].mean().item()

# 真实样本基线（未出现在参考集100张中的 test[100:108]）
real8 = torch.stack([test_dataset[i][0] for i in range(100, 108)])

print("== 复现主脚本指标（应与讲义/主脚本一致）==")
print(f"dist_ddpm={nn_dist(ddpm_s):.4f} (讲义 6.0491)")
print(f"dist_rf50 ={nn_dist(rf50):.4f} (讲义 7.2078)")
print(f"dist_rf10 ={nn_dist(rf10):.4f} (讲义 6.8741)")
print(f"dist_rf1  ={nn_dist(s1):.4f} (讲义 5.5797)")

print("\n== 独立质量证据 ==")
print(f"{'方法':<12}{'最近邻距离':>10}{'批内多样性':>12}{'批均图标准差':>14}{'单样本平均std':>14}")
for name, gen in [('RF 1步', s1), ('RF 10步', rf10), ('RF 50步', rf50), ('DDPM 200步', ddpm_s), ('真实样本', real8)]:
    g = gen.view(gen.shape[0], -1)
    batch_mean_std = gen.mean(dim=0).std().item()   # 8张平均后仍有的结构强度（模糊塌缩→低）
    per_sample_std = g.std(dim=1).mean().item()     # 单样本对比度
    print(f"{name:<12}{nn_dist(gen):>10.4f}{diversity(gen):>12.2f}{batch_mean_std:>14.4f}{per_sample_std:>14.4f}")

# 1步输出是否≈全局条件均值（多样性塌缩检验）：两两距离占样本能量比例
g1 = s1.view(8, -1)
print(f"\nRF1步: 样本两两距离/样本平均范数 = {diversity(s1)/g1.norm(dim=1).mean().item():.3f}")
g2 = rf10.view(8, -1)
print(f"RF10步: 样本两两距离/样本平均范数 = {diversity(rf10)/g2.norm(dim=1).mean().item():.3f}")
