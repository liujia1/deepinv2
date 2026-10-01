# -*- coding: utf-8 -*-
"""实验14.5-1 复核 B+C（缩减规模，CPU可承受）
B) 分时间片速度场误差：同一测度下比较两模型在端点/中间的预测误差
C) 1步输出与"平均数字"的距离（NN指标偏好模糊平均脸的机理）
"""
import sys, io, os
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

EXP = r"d:\works\deepinv\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.5-1"
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

def load_model(pth):
    m = SmallUNet().to(device)
    ck = torch.load(pth, map_location=device, weights_only=False)
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m

model_u = load_model(os.path.join(EXP, 'rf_uniform_final.pth'))
model_l = load_model(os.path.join(EXP, 'rf_logit_final.pth'))

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(os.path.join(EXP, 'data'), train=False, download=False, transform=transform)

@torch.no_grad()
def flow_ode_sample(model, shape, n_steps, x_init):
    x = x_init.clone()
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t_val = step / n_steps
        t_int = torch.full((shape[0],), int((1 - t_val) * (T - 1)), device=device, dtype=torch.long)
        v = model(x, t_int)
        x = x + v * dt
    return x.clamp(0, 1)

# ============ B) 分时间片速度场误差（缩减：128数字×16噪声=2048/bin） ============
print("=" * 66)
print("B) 分时间片条件速度场误差 E||v_pred-(x0-z)||²（同一测度, 2048样本/bin）")
print("=" * 66)
bins = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.45), (0.45, 0.55), (0.55, 0.75), (0.75, 0.9), (0.9, 1.0)]
g = torch.Generator(device=device).manual_seed(7)
n_data, n_noise = 128, 16
x0 = test_dataset.data[:n_data].float()[:, None] / 255.0
err_u = np.zeros(len(bins)); err_l = np.zeros(len(bins))
with torch.no_grad():
    for bi, (lo, hi) in enumerate(bins):
        N = n_data * n_noise
        t = (torch.rand(N, generator=g) * (hi - lo) + lo)
        xb = x0.repeat_interleave(n_noise, dim=0)
        z = torch.randn(N, 1, 28, 28, generator=g)
        t4 = t[:, None, None, None]
        x_t = (1 - t4) * z + t4 * xb
        v_tgt = xb - z
        t_int = ((1 - t) * (T - 1)).long().clamp(0, T - 1)
        err_u[bi] = ((model_u(x_t, t_int) - v_tgt) ** 2).mean().item()
        err_l[bi] = ((model_l(x_t, t_int) - v_tgt) ** 2).mean().item()
print(f"{'t区间':>13} | {'均匀模型':>9} | {'LN模型':>9} | LN相对误差")
for bi, (lo, hi) in enumerate(bins):
    rel = (err_l[bi] - err_u[bi]) / err_u[bi] * 100
    print(f"  [{lo:.2f},{hi:.2f}) | {err_u[bi]:9.5f} | {err_l[bi]:9.5f} | {rel:+7.1f}%")

# ============ C) 1步输出与均值图距离 ============
print("\n" + "=" * 66)
print("C) 1步采样输出：到MNIST测试均值图的距离 & 像素方差")
print("=" * 66)
mean_img = x0.mean(dim=0, keepdim=True)
for name, m in [('uniform', model_u), ('logit  ', model_l)]:
    g = torch.Generator(device=device).manual_seed(1000)
    x_init = torch.randn(64, 1, 28, 28, generator=g)
    out1 = flow_ode_sample(m, x_init.shape, 1, x_init)
    d_mean = (out1 - mean_img).flatten(1).norm(dim=1).mean().item()
    var = out1.flatten(1).var(dim=1).mean().item()
    print(f"  {name}: 到均值图距离={d_mean:.3f}, 平均像素方差={var:.4f}")
print("  （到均值图越近=越糊成平均脸, NN代理指标系统性偏好这种输出）")
