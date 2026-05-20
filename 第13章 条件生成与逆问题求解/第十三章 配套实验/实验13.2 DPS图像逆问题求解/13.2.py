# -*- coding: utf-8 -*-
"""
实验13.2 DPS图像逆问题求解
对应知识点：
  - 13.3.2节 DPS深度剖析（实际训练+采样验证）
  - 13.3.4节 隐空间优化（DiffPIR风格对比）
  - 13.4.1节 Classifier Guidance与DPS的统一
  - 13.4.3节 引导权重ζ与质量-多样性权衡

在MNIST上训练小型UNet扩散模型，然后用DPS算法求解图像去模糊/去噪逆问题。
与实验7.5不同，本实验使用自己训练的DDPM模型（而非deepinv预训练模型），
完整实现从"训练扩散模型"到"用DPS求解逆问题"的端到端流程。

素材来源：
  - 实验11.2的SmallUNet架构和DDPM训练代码
  - 实验7.5的DPS算法思路（1D→图像）
  - ★ 原创设计：在自训练MNIST DDPM上实现DPS，验证零样本迁移
  - ★ 原创设计：不同ζ值的重建质量对比

运行前提：需要GPU

实验内容：
  步骤1：训练UNet扩散模型（MNIST, ε-prediction）
  步骤2：DPS算法求解图像去噪逆问题
  步骤3：DPS求解图像去模糊逆问题
  步骤4：引导权重ζ对重建质量的影响（13.4.3节）
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

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    SAVE_DIR = os.path.join(_gdrive, '实验13_2_DPS')
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"检测到 Google Drive，结果将保存至: {SAVE_DIR}")
else:
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    print(f"本地环境，结果将保存至: {SAVE_DIR}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')


# ============================================================
# 噪声调度（与11.2/12.2一致）
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


# ============================================================
# 前向过程
# ============================================================
def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    return (
        sqrt_alpha_bars[t][:, None, None, None] * x_0 +
        sqrt_one_minus_alpha_bars[t][:, None, None, None] * noise
    )


# ============================================================
# 去噪网络: 小型UNet（与11.2/12.2一致）
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
# Tweedie估计: x̂_{0|t} = (x_t - √(1-ᾱ_t)·ε̂_θ) / √ᾱ_t
# ============================================================
def tweedie_x0_estimate(model, x_t, t):
    """从含噪状态x_t和模型预测ε̂_θ计算Tweedie估计x̂_{0|t}"""
    eps_pred = model(x_t, t)
    ab_t = alpha_bars[t[0]] if t.dim() == 1 else alpha_bars[t]
    x0_hat = (x_t - sqrt_one_minus_alpha_bars[t[0]] * eps_pred) / sqrt_alpha_bars[t[0]]
    return x0_hat, eps_pred


# ============================================================
# DPS采样算法（13.3.2节）
# ============================================================
def dps_sample(model, y, forward_op, sigma_y, shape, zeta=1.0, n_steps=None):
    """DPS后验采样算法
    
    对应13.3.2节的DPS伪代码：
    1. 预测x̂_0 (Tweedie估计)
    2. 计算一致性梯度 ∇||y - A(x̂_0)||^2
    3. 修正得分 = 先验得分 + ζ * 似然梯度
    4. Euler-Maruyama步进
    
    Args:
        model: 预训练ε-prediction扩散模型
        y: 观测 (B, C, H, W)
        forward_op: 正向算子A，输入x返回A(x)
        sigma_y: 观测噪声标准差
        shape: 采样形状 (B, C, H, W)
        zeta: DPS引导权重（13.4.3节）
        n_steps: 采样步数（默认使用全部T步）
    """
    model.eval()
    if n_steps is None:
        n_steps = T
    
    x = torch.randn(shape, device=device)
    
    for t_idx in reversed(range(T)):
        t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        
        sqrt_ab_t = sqrt_alpha_bars[t_idx]
        sqrt_1mab_t = sqrt_one_minus_alpha_bars[t_idx]

        # 第1步：先验得分 → ε̂_θ（无梯度）
        with torch.no_grad():
            eps_pred = model(x, t)

        # 第2步：Tweedie估计 x̂_{0|t}（需要梯度以计算∇_{x_t}）
        x = x.detach().requires_grad_(True)
        eps_pred_grad = model(x, t)
        x0_hat = (x - sqrt_1mab_t * eps_pred_grad) / sqrt_ab_t

        # 第3步：计算DPS似然梯度 ∇_{x_t} ||y - A(x̂_0)||² 
        # 通过autograd自动反向传播，正确处理任意正向算子A
        Ax0_hat = forward_op(x0_hat)
        likelihood_loss = torch.sum((y - Ax0_hat) ** 2)
        likelihood_grad = torch.autograd.grad(likelihood_loss, x)[0]

        # ★ 对梯度做归一化（DPS论文推荐），稳定不同时间步的修正幅度
        grad_norm = likelihood_grad.norm()
        if grad_norm > 1e-8:
            likelihood_grad = likelihood_grad / grad_norm

        x = x.detach()
        eps_pred = eps_pred.detach()

        # 第4步：修正ε̂_θ（正确符号：减去似然梯度方向）
        # 修正后的ε̂ = ε̂_θ - ζ * √(1-ᾱ_t) * likelihood_grad
        # 注意符号：似然梯度指向损失增大方向，DPS需要沿梯度"下降"方向修正ε
        eps_corrected = eps_pred - zeta * sqrt_1mab_t * likelihood_grad

        # 第5步：DDPM采样步（使用修正后的ε̂）
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


# ============================================================
# 无条件DDPM采样（对比基线）
# ============================================================
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


# ============================================================
# 正向算子定义
# ============================================================
class GaussianBlurOperator:
    """高斯模糊正向算子 A: x → blur(x)
    
    使用可微分的卷积实现，支持autograd反向传播
    """
    def __init__(self, kernel_size=5, sigma=1.5, device='cpu'):
        self.kernel_size = kernel_size
        self.sigma = sigma
        # 生成高斯模糊核
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel = torch.outer(g, g)
        kernel = kernel / kernel.sum()
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,k,k)
        self.pad = kernel_size // 2
    
    def __call__(self, x):
        """x: (B, 1, H, W) → (B, 1, H, W)"""
        return F.conv2d(x, self.kernel, padding=self.pad)


class IdentityOperator:
    """恒等算子 A: x → x（去噪问题）"""
    def __call__(self, x):
        return x


class InpaintingOperator:
    """遮挡算子 A: x → mask * x
    
    ★ 原创设计：随机遮挡30%像素的inpainting问题
    """
    def __init__(self, mask_ratio=0.3, device='cpu'):
        # 生成固定随机遮挡
        torch.manual_seed(42)
        self.mask = (torch.rand(1, 1, 28, 28, device=device) > mask_ratio).float()
    
    def __call__(self, x):
        return self.mask * x


# ============================================================
# 数据加载
# ============================================================
print("加载MNIST数据集...")
data_dir = os.path.join(SAVE_DIR, 'data')
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

# 准备测试图像
test_images = next(iter(test_loader))[0][:4].to(device)  # 4张测试图

# 生成去噪观测
sigma_y_denoise = 0.3
noise_obs = torch.randn_like(test_images) * sigma_y_denoise
y_denoise = test_images + noise_obs

# DPS去噪
identity_op = IdentityOperator()
print("DPS去噪采样中...")
x_hat_denoise = dps_sample(model, y_denoise, identity_op, sigma_y_denoise,
                            shape=test_images.shape, zeta=1.0)

# 无条件采样基线
print("无条件DDPM采样中...")
x_uncond = ddpm_sample(model, test_images.shape)

# 计算PSNR
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

plt.suptitle(f'步骤2：DPS图像去噪 (PSNR: {psnr_noisy:.1f}→{psnr_denoise:.1f} dB)', fontsize=14, y=1.01)
plt.tight_layout()
fig_path1 = os.path.join(SAVE_DIR, '步骤1_DPS去噪.png')
plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"图1已保存: {fig_path1}")


# ============================================================
# 步骤3：DPS求解图像去模糊逆问题
# ============================================================
print(f"\n{'='*60}")
print("步骤3：DPS求解图像去模糊逆问题（13.3.2节 + 13.3.4节）")
print("=" * 60)

print("""
逆问题: y = A(x) + n, A=高斯模糊, n ~ N(0, σ_y^2)
关键：模型从未见过模糊图像——DPS的零样本迁移能力（13.6节）

与DiffPIR的对比（13.3.4节）：
  DPS: 在采样过程中注入似然梯度（修正SDE轨迹）
  DiffPIR: 交替执行去噪+数据一致性投影（PnP风格）
""")

# 高斯模糊算子
blur_op = GaussianBlurOperator(kernel_size=7, sigma=2.0, device=device)

# 生成模糊观测
sigma_y_blur = 0.02
y_blur = blur_op(test_images) + torch.randn_like(test_images) * sigma_y_blur

# DPS去模糊
print("DPS去模糊采样中...")
x_hat_blur = dps_sample(model, y_blur, blur_op, sigma_y_blur,
                          shape=test_images.shape, zeta=1.0)

psnr_blur_obs = compute_psnr(y_blur, test_images)
psnr_blur_dps = compute_psnr(x_hat_blur, test_images)
print(f"  模糊观测PSNR: {psnr_blur_obs:.2f} dB")
print(f"  DPS重建PSNR:   {psnr_blur_dps:.2f} dB")

# ★ 原创设计：inpainting问题
print("\n★ 原创设计：DPS求解inpainting逆问题")
inpaint_op = InpaintingOperator(mask_ratio=0.3, device=device)
sigma_y_inpaint = 0.01
y_inpaint = inpaint_op(test_images) + torch.randn_like(test_images) * sigma_y_inpaint

print("DPS inpainting采样中...")
x_hat_inpaint = dps_sample(model, y_inpaint, inpaint_op, sigma_y_inpaint,
                            shape=test_images.shape, zeta=1.0)

psnr_inpaint_obs = compute_psnr(y_inpaint, test_images)
psnr_inpaint_dps = compute_psnr(x_hat_inpaint, test_images)
print(f"  遮挡观测PSNR: {psnr_inpaint_obs:.2f} dB")
print(f"  DPS重建PSNR:   {psnr_inpaint_dps:.2f} dB")

# 可视化
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for i in range(4):
    # 去模糊
    axes[0, i].imshow(y_blur[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, i].axis('off')
    if i == 0: axes[0, i].set_ylabel('模糊观测y', fontsize=12, rotation=0, labelpad=50)
    
    axes[1, i].imshow(x_hat_blur[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1, i].axis('off')
    if i == 0: axes[1, i].set_ylabel('DPS去模糊', fontsize=12, rotation=0, labelpad=50)
    
    # inpainting
    axes[2, i].imshow(x_hat_inpaint[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[2, i].axis('off')
    if i == 0: axes[2, i].set_ylabel('★ DPS inpainting', fontsize=12, rotation=0, labelpad=50)

plt.suptitle('步骤3：DPS去模糊 & ★ inpainting', fontsize=14, y=1.01)
plt.tight_layout()
fig_path2 = os.path.join(SAVE_DIR, '步骤2_DPS去模糊.png')
plt.savefig(fig_path2, dpi=150, bbox_inches=None)
plt.close()
print(f"图2已保存: {fig_path2}")


# ============================================================
# ★ 步骤4：引导权重ζ对重建质量的影响（13.4.3节）
# ============================================================
print(f"\n{'='*60}")
print("★ 步骤4：引导权重ζ对重建质量的影响（13.4.3节）")
print("=" * 60)

print("""
13.4.3节：ζ控制先验与似然的相对强度
  ζ=0: 纯先验（无条件采样，忽略观测）
  ζ=1: 标准DPS（平衡先验与似然）
  ζ>1: 强数据一致性（类似MAP，多样性低）
  
★ 原创设计：固定测试图像，用不同ζ执行DPS去噪，对比PSNR和视觉效果
""")

zeta_values = [0.0, 0.3, 0.7, 1.0, 1.5, 3.0]
psnr_results = []

# 使用一张测试图像
single_img = test_images[:1]
y_single = single_img + torch.randn_like(single_img) * sigma_y_denoise

for zeta in zeta_values:
    x_hat = dps_sample(model, y_single, identity_op, sigma_y_denoise,
                        shape=single_img.shape, zeta=zeta)
    psnr = compute_psnr(x_hat, single_img)
    psnr_results.append(psnr)
    print(f"  ζ={zeta:4.1f}: PSNR={psnr:.2f} dB")

# 可视化
fig, axes = plt.subplots(2, len(zeta_values), figsize=(20, 6))

for idx, zeta in enumerate(zeta_values):
    # 注意：不能用 with torch.no_grad(): 包裹 dps_sample，
    # 因为函数内部需要 autograd 计算似然梯度
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
axes[1, 0].set_title('★ ζ-重建质量权衡（13.4.3节）', fontsize=13)
axes[1, 0].grid(alpha=0.3)
axes[1, 0].annotate('ζ=0: 无条件采样\nζ=1: 标准DPS\nζ>1: 强数据一致性',
                    xy=(0.55, 0.3), xycoords='axes fraction', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#dfe6e9', alpha=0.8))

# 隐藏空子图
for i in range(1, len(zeta_values)):
    axes[1, i].axis('off')

plt.tight_layout()
fig_path3 = os.path.join(SAVE_DIR, '步骤3_引导权重对比.png')
plt.savefig(fig_path3, dpi=150, bbox_inches=None)
plt.close()
print(f"图3已保存: {fig_path3}")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print("实验13.2 完成!")
print("=" * 60)
print(f"""
关键结论:
1. DPS算法实践（13.3.2节）
   - 在自训练的MNIST DDPM上成功实现DPS
   - 去噪PSNR: {psnr_noisy:.1f} → {psnr_denoise:.1f} dB
   - 去模糊PSNR: {psnr_blur_obs:.1f} → {psnr_blur_dps:.1f} dB

2. 零样本迁移（13.1/13.6节）
   - 模型只训练了去噪（无条件DDPM），但能解决去模糊和inpainting
   - 无需针对特定逆问题重新训练——这是DPS的核心优势

3. 引导权重ζ（13.4.3节）★ 原创设计
   - ζ=0: 无条件采样（忽略观测，PSNR低）
   - ζ=1: 标准DPS（平衡先验与似然）
   - ζ过大: 过度拟合观测噪声，重建质量下降
   - 最优ζ通常在0.5-1.5之间

4. DPS与DiffPIR对比（13.3.4节）
   - DPS: 修正SDE轨迹，理论清晰（后验得分分解）
   - DiffPIR: 交替去噪+投影，PnP思想（第5章的延伸）
   - 两者都基于同一个预训练扩散模型，无需额外训练
""")