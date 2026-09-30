# -*- coding: utf-8 -*-
"""14.4-1 独立裁决：批内多样性 + 真实样本基线 + 清晰度代理
复现主脚本采样顺序：seed 42 → 1rf-1step → 2rf-1step → 1rf-50step → 1rf-10step
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

FOLDER = r"d:\works\deepinv\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.4-1"
device = torch.device('cpu')
T = 200


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


@torch.no_grad()
def flow_ode_sample(model, shape, n_steps=50):
    model.eval()
    x = torch.randn(shape, device=device)
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t_val = step / n_steps
        t_int = torch.full((shape[0],), int((1 - t_val) * (T - 1)), device=device, dtype=torch.long)
        v = model(x, t_int)
        x = x + v * dt
    return x.clamp(0, 1)


def compute_nn_distance(generated, reference):
    gen_flat = generated.view(generated.shape[0], -1)
    ref_flat = reference.view(reference.shape[0], -1)
    return torch.cdist(gen_flat, ref_flat).min(dim=1)[0].mean().item()


def batch_div(g):
    g = g.view(g.shape[0], -1)
    d = torch.cdist(g, g)
    iu = torch.triu_indices(g.shape[0], g.shape[0], offset=1)
    return d[iu[0], iu[1]].mean().item()


def pixel_std(g):
    """每张图内部像素标准差（模糊的条件均值图像素方差低）"""
    return g.view(g.shape[0], -1).std(dim=1).mean().item()


rf_model = SmallUNet().to(device)
rf_model.load_state_dict(torch.load(os.path.join(FOLDER, 'rf_final.pth'), map_location=device, weights_only=False)['model_state_dict'])
model_2rf = SmallUNet().to(device)
model_2rf.load_state_dict(torch.load(os.path.join(FOLDER, '2rf_final.pth'), map_location=device, weights_only=False)['model_state_dict'])

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(os.path.join(FOLDER, 'data'), train=False, download=False, transform=transform)

torch.manual_seed(42)
n_samples = 8
shape = (n_samples, 1, 28, 28)
# 与主脚本一致的消耗顺序
s_1rf_1 = flow_ode_sample(rf_model, shape, n_steps=1)
s_2rf_1 = flow_ode_sample(model_2rf, shape, n_steps=1)
s_1rf_50 = flow_ode_sample(rf_model, shape, n_steps=50)
s_1rf_10 = flow_ode_sample(rf_model, shape, n_steps=10)

test_samples = torch.stack([test_dataset[i][0] for i in range(100)]).to(device)
real_samples = torch.stack([test_dataset[i][0] for i in range(100, 100 + n_samples)]).to(device)

print("== 复现主脚本四项距离 ==")
print(f"1-RF 1步 : {compute_nn_distance(s_1rf_1, test_samples):.4f}  (期望 5.9394)")
print(f"2-RF 1步 : {compute_nn_distance(s_2rf_1, test_samples):.4f}  (期望 6.8389)")
print(f"1-RF 10步: {compute_nn_distance(s_1rf_10, test_samples):.4f}  (期望 6.0490)")
print(f"1-RF 50步: {compute_nn_distance(s_1rf_50, test_samples):.4f}  (期望 7.1966)")

print("\n== 新增裁决指标 ==")
print(f"真实样本基线距离: {compute_nn_distance(real_samples, test_samples):.4f}")
for name, s in [('1-RF 1步', s_1rf_1), ('2-RF 1步', s_2rf_1), ('1-RF 10步', s_1rf_10), ('1-RF 50步', s_1rf_50), ('真实样本', real_samples)]:
    print(f"{name:10s} 批内多样性={batch_div(s):7.3f}  图内像素std={pixel_std(s):.4f}")
