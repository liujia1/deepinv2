# -*- coding: utf-8 -*-
"""
实验13.4-2：引导权重ζ对PSNR的影响
对应章节：13.4.3节 引导权重与质量-多样性权衡

素材来源：实验13.2-步骤4

★ 原创设计：固定测试图像，用不同ζ执行DPS去噪，对比PSNR和视觉效果

实验内容：
  - 训练UNet扩散模型（MNIST, ε-prediction）
  - 用不同ζ值执行DPS去噪
  - 绘制ζ-PSNR曲线

注意：本实验训练50轮DDPM并执行多次DPS采样，CPU上运行较慢。
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


class IdentityOperator:
    def __call__(self, x):
        return x


# ============================================================
# 数据加载与训练
# ============================================================
print("加载MNIST数据集...")
data_dir = os.path.join(SAVE_DIR, 'data')
os.makedirs(data_dir, exist_ok=True)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 训练
print(f"\n{'='*60}")
print("训练UNet扩散模型...")
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
# 步骤：引导权重ζ对重建质量的影响
# ============================================================
print(f"\n{'='*60}")
print("★ 步骤：引导权重ζ对重建质量的影响（13.4.3节）")
print("=" * 60)

print("""
13.4.3节：ζ控制先验与似然的相对强度
  ζ=0: 纯先验（无条件采样，忽略观测）
  ζ=1: 标准DPS（平衡先验与似然）
  ζ>1: 强数据一致性（类似MAP，多样性低）
""")

zeta_values = [0.0, 0.3, 0.7, 1.0, 1.5, 3.0]
psnr_results = []

test_images = next(iter(test_loader))[0][:4].to(device)
single_img = test_images[:1]
sigma_y_denoise = 0.3
y_single = single_img + torch.randn_like(single_img) * sigma_y_denoise

identity_op = IdentityOperator()

def compute_psnr(pred, target):
    mse = torch.mean((pred - target)**2).item()
    return 10 * np.log10(1.0 / (mse + 1e-10))

for zeta in zeta_values:
    x_hat = dps_sample(model, y_single, identity_op, sigma_y_denoise,
                        shape=single_img.shape, zeta=zeta)
    psnr = compute_psnr(x_hat, single_img)
    psnr_results.append(psnr)
    print(f"  ζ={zeta:4.1f}: PSNR={psnr:.2f} dB")

# 可视化
fig, axes = plt.subplots(2, len(zeta_values), figsize=(20, 6))

for idx, zeta in enumerate(zeta_values):
    x_hat = dps_sample(model, y_single, identity_op, sigma_y_denoise,
                        shape=single_img.shape, zeta=zeta)
    axes[0, idx].imshow(x_hat[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, idx].axis('off')
    label = "无条件" if zeta == 0 else f"ζ={zeta}"
    axes[0, idx].set_title(f'{label}\nPSNR={psnr_results[idx]:.1f}dB', fontsize=10)

# ζ-PSNR曲线
axes[1, 0].plot(zeta_values, psnr_results, 'ro-', markersize=8, lw=2)
axes[1, 0].set_xlabel('引导权重 ζ', fontsize=12)
axes[1, 0].set_ylabel('PSNR (dB)', fontsize=12)
axes[1, 0].set_title('ζ-重建质量权衡（13.4.3节）', fontsize=13)
axes[1, 0].grid(alpha=0.3)
axes[1, 0].annotate('ζ=0: 无条件采样\nζ=1: 标准DPS\nζ>1: 强数据一致性',
                    xy=(0.55, 0.3), xycoords='axes fraction', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#dfe6e9', alpha=0.8))

# 隐藏空子图
for i in range(1, len(zeta_values)):
    axes[1, i].axis('off')

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '引导权重ζ对PSNR的影响.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图已保存: {fig_path}")

print(f"\n{'='*60}")
print("实验13.4-2 完成!")
print("=" * 60)
print(f"""
关键结论:
1. ζ-重建质量权衡（13.4.3节）
   - ζ=0: 无条件采样（忽略观测，PSNR低）
   - ζ=1: 标准DPS（平衡先验与似然，通常PSNR最高）
   - ζ过大: 过度拟合观测噪声，PSNR下降
   - 最优ζ通常在0.5-1.5之间

2. 实践要点
   - 单一图像的最优ζ可能因图而异
   - 工程上常用 ζ=1 作为默认起点
   - ζ-PSNR曲线呈倒U形——这是质量-多样性权衡的具体表现
""")
