# -*- coding: utf-8 -*-
"""
实验13.3-2：DPS图像去噪端到端
对应章节：13.3.2节 DPS深度剖析

素材来源：实验13.2-步骤1+2

实验内容：
  - 训练UNet扩散模型（MNIST, ε-prediction）
  - DPS算法求解图像去噪逆问题
  - 端到端验证：从训练扩散模型到DPS求解逆问题

注意：本实验使用MNIST训练50轮DDPM并执行DPS，CPU上运行较慢。
如需加速，建议使用GPU。
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import warnings

# ====== 解决中文乱码的核心代码 ======
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
import platform
from matplotlib.font_manager import FontManager

def _find_chinese_font():
    candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong'] if platform.system() == 'Windows' else ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei']
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available:
            return font
    import os as _os, re
    for f in fm.ttflist:
        for pat in ['cjk', 'wqy', 'noto.*cjk', 'simhei']:
            if re.search(pat, f.name.lower()):
                return f.name
    return None

_cn_font = _find_chinese_font()
if _cn_font:
    plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['font.family'] = 'sans-serif'
    print(f"[Font] 已检测到中文字体: {_cn_font}")
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
# ========================================================

np.random.seed(42)
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()


# ============================================================
# 噪声调度
# ============================================================
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


def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    return (
        sqrt_alpha_bars[t][:, None, None, None] * x_0 +
        sqrt_one_minus_alpha_bars[t][:, None, None, None] * noise
    )


# ============================================================
# 去噪网络: 小型UNet
# ============================================================
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(gn_groups, out_ch),
            nn.SiLU(),
        )
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(gn_groups, out_ch),
            nn.SiLU(),
        )
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
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
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


# ============================================================
# DPS采样算法
# ============================================================
def dps_sample(model, y, forward_op, sigma_y, shape, zeta=1.0, n_steps=None):
    model.eval()
    if n_steps is None:
        n_steps = T

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

        Ax0_hat = forward_op(x0_hat)
        likelihood_loss = torch.sum((y - Ax0_hat) ** 2)
        likelihood_grad = torch.autograd.grad(likelihood_loss, x)[0]

        grad_norm = likelihood_grad.norm()
        if grad_norm > 1e-8:
            likelihood_grad = likelihood_grad / grad_norm

        x = x.detach()
        eps_pred = eps_pred.detach()

        eps_corrected = eps_pred - zeta * sqrt_1mab_t * likelihood_grad

        with torch.no_grad():
            model_mean = sqrt_recip_alphas[t_idx] * (
                x - beta_over_sqrt_1m_ab[t_idx] * eps_corrected
            )
            if t_idx == 0:
                x = model_mean
            else:
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(posterior_var[t_idx]) * noise

    return x.clamp(0, 1)


@torch.no_grad()
def ddpm_sample(model, shape):
    model.eval()
    x = torch.randn(shape, device=device)
    for t_idx in reversed(range(T)):
        t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        pred = model(x, t)
        model_mean = sqrt_recip_alphas[t_idx] * (
            x - beta_over_sqrt_1m_ab[t_idx] * pred
        )
        if t_idx == 0:
            x = model_mean
        else:
            noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(posterior_var[t_idx]) * noise
    return x.clamp(0, 1)


class IdentityOperator:
    def __call__(self, x):
        return x


# ============================================================
# 数据加载
# ============================================================
print("加载MNIST数据集...")
data_dir = os.path.join(SAVE_DIR, 'data')
os.makedirs(data_dir, exist_ok=True)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
print(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")


# ============================================================
# 步骤1：训练UNet扩散模型
# ============================================================
print(f"\n{'='*60}")
print("步骤1：训练UNet扩散模型（ε-prediction, MNIST）")
print("=" * 60)

num_epochs = 50
model = SmallUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x, _ in train_loader:
        x = x.to(device)
        batch = x.shape[0]
        t = torch.randint(0, T, (batch,), device=device)
        noise = torch.randn_like(x)
        x_t = q_sample(x, t, noise)
        pred = model(x_t, t)
        loss = F.mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch
    if (epoch + 1) % 10 == 0 or epoch == 0:
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"  Epoch {epoch+1:3d}/{num_epochs}  Loss={avg_loss:.6f}")

print("训练完成！")


# ============================================================
# 步骤2：DPS求解图像去噪逆问题
# ============================================================
print(f"\n{'='*60}")
print("步骤2：DPS求解图像去噪逆问题（A=I, 13.3.2节）")
print("=" * 60)

print("""
逆问题: y = x + n, n ~ N(0, σ_y^2)
DPS算法: 在逆向DDPM采样中注入似然梯度
  先验得分: -ε̂_θ(x_t, t) / √(1-ᾱ_t)
  似然梯度: (y - x̂_{0|t}) / (σ_y^2 · √ᾱ_t)
  修正得分: 先验得分 + ζ · 似然梯度
""")

test_images = next(iter(test_loader))[0][:4].to(device)

sigma_y_denoise = 0.3
noise_obs = torch.randn_like(test_images) * sigma_y_denoise
y_denoise = test_images + noise_obs

identity_op = IdentityOperator()
print("DPS去噪采样中...")
x_hat_denoise = dps_sample(model, y_denoise, identity_op, sigma_y_denoise,
                            shape=test_images.shape, zeta=1.0)

print("无条件DDPM采样中...")
x_uncond = ddpm_sample(model, test_images.shape)

def compute_psnr(pred, target):
    mse = torch.mean((pred - target)**2).item()
    return 10 * np.log10(1.0 / (mse + 1e-10))

psnr_denoise = compute_psnr(x_hat_denoise, test_images)
psnr_noisy = compute_psnr(y_denoise, test_images)
print(f"  含噪观测PSNR: {psnr_noisy:.2f} dB")
print(f"  DPS重建PSNR:   {psnr_denoise:.2f} dB")

# 可视化
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for i in range(4):
    axes[0, i].imshow(test_images[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, i].axis('off')
    if i == 0: axes[0, i].set_ylabel('真实x₀', fontsize=12, rotation=0, labelpad=50)

    axes[1, i].imshow(y_denoise[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1, i].axis('off')
    if i == 0: axes[1, i].set_ylabel(f'观测y=x+n\nσ_y={sigma_y_denoise}', fontsize=11, rotation=0, labelpad=60)

    axes[2, i].imshow(x_hat_denoise[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[2, i].axis('off')
    if i == 0: axes[2, i].set_ylabel('DPS重建', fontsize=12, rotation=0, labelpad=50)

plt.suptitle(f'DPS图像去噪 (PSNR: {psnr_noisy:.1f}→{psnr_denoise:.1f} dB)', fontsize=14, y=1.01)
plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, 'DPS图像去噪端到端.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图已保存: {fig_path}")

print(f"\n{'='*60}")
print("实验13.3-2 完成!")
print("=" * 60)
print(f"""
关键结论:
1. DPS算法实践（13.3.2节）
   - 在自训练的MNIST DDPM上成功实现DPS
   - 去噪PSNR: {psnr_noisy:.1f} → {psnr_denoise:.1f} dB
   - 端到端流程：训练扩散模型 → DPS求解逆问题

2. 实践要点
   - autograd自动处理任意正向算子A的Jacobian
   - 梯度归一化（likelihood_grad / grad_norm）稳定不同时间步的修正幅度
   - DDPM采样步使用修正后的ε̂，无需重新推导
""")
