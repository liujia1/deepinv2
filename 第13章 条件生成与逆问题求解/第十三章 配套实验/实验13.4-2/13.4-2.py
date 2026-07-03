# -*- coding: utf-8 -*-
"""
实验13.4-2：引导权重zeta对PSNR的影响
对应章节：13.4.3节 引导权重与质量-多样性权衡

★ 原创设计：固定测试图像，用不同zeta执行DPS去噪，对比PSNR和视觉效果

实验内容：
  - 训练UNet扩散模型（MNIST, epsilon-prediction）
  - 用不同zeta值执行DPS去噪
  - 绘制zeta-PSNR曲线

注意：本实验训练50轮DDPM并执行多次DPS采样。
"""

import sys
import io
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import logging
import warnings

# 设置控制台输出为 UTF-8 (Windows下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# 静默matplotlib相关警告
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验13.4-2')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(_chinese_path, exist_ok=True)
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

print("\n" + "=" * 60)
print("实验13.4-2: 引导权重zeta对PSNR的影响")
print("=" * 60)
print("对应章节: 13.4.3节 引导权重与质量-多样性权衡")
print("知识点: 引导权重zeta, PSNR倒U形曲线, 实际权重选择")


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
    return x.clamp(-1, 1)


class IdentityOperator:
    def __call__(self, x):
        return x


# ============================================================
# 训练函数（含checkpoint resume）
# ============================================================
def train_model(checkpoint_path, num_epochs=50):
    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    start_epoch = 0
    is_final = False

    if os.path.exists(checkpoint_path):
        print(f"\n检测到已保存的模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get('is_final', False):
            print(f"已检测到最终训练完成的模型, 直接加载, 跳过训练过程")
            print(f"  训练轮数: {checkpoint['epoch']+1}")
            print(f"  最终损失: {checkpoint['loss']:.6f}")
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                is_final = True
            except RuntimeError as e:
                print(f"警告: checkpoint与当前模型架构不兼容, 删除后重新训练")
                os.remove(checkpoint_path)
                is_final = False
        else:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"检测到未完成的训练, 从第 {start_epoch+1} 轮继续")
            except RuntimeError as e:
                print(f"警告: checkpoint与当前模型架构不兼容, 删除后重新训练")
                os.remove(checkpoint_path)
                start_epoch = 0

    if not is_final:
        if start_epoch >= num_epochs:
            print(f"  注意: start_epoch({start_epoch}) >= num_epochs({num_epochs}), 无需继续训练")
            is_final = True
        else:
            print(f"\n开始训练 epsilon-prediction DDPM (T={T}, epochs={num_epochs})...")
            print("-" * 75)
            for epoch in range(start_epoch, num_epochs):
                model.train()
                total_loss = 0
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ascii=True, leave=False)
                for x, _ in pbar:
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
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                avg_loss = total_loss / len(train_loader.dataset)
                print(f"Epoch {epoch+1:3d}/{num_epochs}  Loss={avg_loss:.6f}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'is_final': False
                }, checkpoint_path)
            torch.save({
                'epoch': num_epochs - 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'is_final': True
            }, checkpoint_path)
            print(f"模型已保存: {checkpoint_path}")
    else:
        print(f"\n使用已训练完成的模型, 跳过训练过程")

    return model


# ============================================================
# 数据加载
# ============================================================
print("\n加载MNIST数据集...")
data_dir = os.path.join(SAVE_DIR, 'data')
os.makedirs(data_dir, exist_ok=True)
# MNIST数据归一化到[-1,1] (与11.4-1、13.3-2修复一致):
# 训练时网络看到的 x_T 分布(由 [-1,1] 数据前向扩散得到)与采样起点
# torch.randn(标准高斯) 在统计意义上更匹配,避免系统性均值偏移拖累重建质量。
# 采样函数末尾的 clamp 同步改为 [-1,1];PSNR 与可视化在转换回 [0,1] 空间后计算。
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1),  # [0,1] -> [-1,1]
])
train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# ============================================================
# 训练
# ============================================================
print(f"\n{'='*60}")
print("训练UNet扩散模型...")
print("=" * 60)

CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'zeta_psnr_checkpoint.pth')
num_epochs = 50
model = train_model(CHECKPOINT_PATH, num_epochs=num_epochs)


# ============================================================
# 步骤：引导权重zeta对重建质量的影响
# ============================================================
print(f"\n{'='*60}")
print("步骤：引导权重zeta对重建质量的影响（13.4.3节）")
print("=" * 60)

print("""
13.4.3节：zeta控制先验与似然的相对强度
  zeta=0: 纯先验（无条件采样，忽略观测）
  zeta=1: 标准DPS（平衡先验与似然）
  zeta>1: 强数据一致性（类似MAP，多样性低）
""")

zeta_values = [0.0, 0.3, 0.7, 1.0, 1.5, 3.0]
psnr_results = []

test_images = next(iter(test_loader))[0][:4].to(device)
single_img = test_images[:1]
sigma_y_denoise = 0.3
y_single = single_img + torch.randn_like(single_img) * sigma_y_denoise

identity_op = IdentityOperator()

def compute_psnr(pred, target):
    """pred/target: 在 [-1,1] 空间,统一转换到 [0,1] 再用 MAX=1 计算 PSNR"""
    pred_01 = (pred + 1) / 2
    target_01 = (target + 1) / 2
    mse = torch.mean((pred_01 - target_01)**2).item()
    return 10 * np.log10(1.0 / (mse + 1e-10))

for zeta in zeta_values:
    x_hat = dps_sample(model, y_single, identity_op, sigma_y_denoise,
                        shape=single_img.shape, zeta=zeta)
    psnr = compute_psnr(x_hat, single_img)
    psnr_results.append(psnr)
    print(f"  zeta={zeta:4.1f}: PSNR={psnr:.2f} dB")

# 可视化
fig, axes = plt.subplots(2, len(zeta_values), figsize=(20, 6))

for idx, zeta in enumerate(zeta_values):
    x_hat = dps_sample(model, y_single, identity_op, sigma_y_denoise,
                        shape=single_img.shape, zeta=zeta)
    axes[0, idx].imshow(((x_hat[0, 0] + 1) / 2).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, idx].axis('off')
    label = "无条件" if zeta == 0 else r"$\zeta={}$".format(zeta)
    axes[0, idx].set_title(f'{label}\nPSNR={psnr_results[idx]:.1f}dB', fontsize=10)

# zeta-PSNR曲线
axes[1, 0].plot(zeta_values, psnr_results, 'ro-', markersize=8, lw=2)
axes[1, 0].set_xlabel(r'引导权重 $\zeta$', fontsize=12)
axes[1, 0].set_ylabel('PSNR (dB)', fontsize=12)
axes[1, 0].set_title(r'$\zeta$-重建质量权衡（13.4.3节）', fontsize=13)
axes[1, 0].grid(alpha=0.3)
axes[1, 0].annotate('zeta=0: 无条件采样\nzeta=1: 标准DPS\nzeta>1: 强数据一致性',
                    xy=(0.55, 0.3), xycoords='axes fraction', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#dfe6e9', alpha=0.8))

# 隐藏空子图
for i in range(1, len(zeta_values)):
    axes[1, i].axis('off')

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '引导权重zeta对PSNR的影响.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图已保存: {fig_path}")

print("\n" + "=" * 60)
print("实验13.4-2 完成!")
print("=" * 60)
print(f"""
关键结论:
1. zeta-重建质量权衡（13.4.3节）
   - zeta=0: 无条件采样（忽略观测，PSNR低）
   - zeta=1: 标准DPS（平衡先验与似然，通常PSNR最高）
   - zeta过大: 过度拟合观测噪声，PSNR下降
   - 最优zeta通常在0.5-1.5之间

2. 实践要点
   - 单一图像的最优zeta可能因图而异
   - 工程上常用 zeta=1 作为默认起点
   - zeta-PSNR曲线呈倒U形——这是质量-多样性权衡的具体表现
""")
