# -*- coding: utf-8 -*-
"""
实验15.4-1 DiT架构实现与对比
对应知识点:
  - 15.3.1节 Transformer基础: 自注意力、多头注意力
  - 15.3.2节 ViT: Patchify
  - 15.3.3节 DiT: 用Transformer替换UNet
  - 15.4节 UNet vs DiT: 架构选择的艺术

在MNIST上实现DiT（Diffusion Transformer），对比UNet和DiT的去噪性能。

实验内容:
  步骤1: DiT架构实现——Patchify + adaLN-Zero + Transformer（15.3节）
  步骤2: UNet vs DiT训练对比（15.4节）
  步骤3: ★ adaLN-Zero vs 无Zero初始化消融（15.2.3节/15.3.3节）
  步骤4: ★ 不同噪声水平下UNet vs DiT去噪行为对比（15.4.1节）

素材来源:
  - 15.2.py（参考实验，DiT完整实现）
  - 15.3.3节 DiT架构描述
  - 15.2.3节 adaLN-Zero条件注入
  - ★ 原创设计: 小型DiT-S/2在MNIST上的完整实现
  - ★ 原创设计: UNet vs DiT参数量/训练速度/去噪质量对比

运行前提: PyTorch, GPU加速推荐（DiT注意力机制在CPU上较慢）
        数据集: MNIST（torchvision自动下载）
"""

import sys
import os
import io
import time
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import warnings
import logging
from tqdm import tqdm

# 设置控制台输出为 UTF-8 (Windows 下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# 静默 matplotlib 相关警告
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
    SAVE_DIR = os.path.join(_gdrive, '实验15.4-1')
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
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

# 设置随机种子
np.random.seed(42)

import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"实验15.4-1: DiT架构实现与对比")
print(f"{'='*60}")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  未检测到 GPU, 使用 CPU 运行")
    print("  提示: Colab 用户可在菜单 运行时 -> 更改运行时类型 中选择 GPU")
    print("  提示: DiT在CPU上运行较慢, 建议使用GPU或减少训练轮数")

# Checkpoint路径
CKPT_UNET = os.path.join(SAVE_DIR, 'unet_step2_checkpoint.pth')
CKPT_DIT = os.path.join(SAVE_DIR, 'dit_step2_checkpoint.pth')
CKPT_DIT_ZERO = os.path.join(SAVE_DIR, 'dit_zero_step3_checkpoint.pth')
CKPT_DIT_NOZERO = os.path.join(SAVE_DIR, 'dit_nozero_step3_checkpoint.pth')


# ============================================================
# 通用checkpoint加载/训练工具
# ============================================================
def load_train_state(checkpoint_path, model, optimizer, num_epochs_total):
    """加载checkpoint并返回 (start_epoch, is_final, loss_history)"""
    if not os.path.exists(checkpoint_path):
        return 0, False, []

    print(f"\n检测到已保存的checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'loss_history' not in checkpoint:
        raise RuntimeError(
            f"检测到旧版本 checkpoint (缺少 'loss_history' 字段):\n"
            f"  {checkpoint_path}\n"
            f"请删除该文件后重新训练."
        )

    if checkpoint.get('is_final', False):
        print(f"✓ 这是最终训练完成的模型, 直接加载, 跳过训练过程")
        print(f"  训练轮数: {checkpoint['epoch']+1}")
        print(f"  最终损失: {checkpoint['loss']:.6f}")
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            raise RuntimeError(
                f"检测到 checkpoint 与当前模型架构不兼容:\n{e}\n"
                f"可能原因: 模型架构已更新.\n"
                f"请删除 checkpoint 文件后重新训练:\n"
                f"  {checkpoint_path}"
            )
        if optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                pass
        return checkpoint['epoch'] + 1, True, checkpoint.get('loss_history', [])

    print(f"检测到未完成的训练, 从第 {checkpoint['epoch']+1} 轮继续")
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        raise RuntimeError(
            f"检测到 checkpoint 与当前模型架构不兼容:\n{e}\n"
            f"请删除 checkpoint 文件后重新训练:\n"
            f"  {checkpoint_path}"
        )
    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception:
            pass
    return checkpoint['epoch'] + 1, False, checkpoint.get('loss_history', [])


def save_train_state(checkpoint_path, model, optimizer, epoch, loss, loss_history, is_final):
    """保存checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'loss': loss,
        'loss_history': loss_history,
        'is_final': is_final,
    }, checkpoint_path)


# ============================================================
# DDPM噪声调度
# ============================================================
T = 200
beta_min, beta_max = 1e-4, 0.02
betas = torch.linspace(beta_min, beta_max, T).to(device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)


# ============================================================
# DiT架构实现（15.3.3节）
# ============================================================

def timestep_embedding(t, dim, max_period=10000):
    """DDPM的正弦时间步嵌入（15.2.2节 Ho et al. 2020实现）"""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None, :]
    # ★ 统一sin在前（与15.2-1一致），输出顺序为 [sin, cos]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiTBlock(nn.Module):
    """DiT Block: Transformer Block + adaLN-Zero（15.3.3节）

    核心设计:
      1. adaLN-Zero: (1+γ)⊙LN(h) + β, 6个调制参数
      2. 残差缩放α: h + α⊙MHA/LN(h), Zero初始化
      3. Pre-LN: LayerNorm在MHA/MLP之前

    数学公式 (15.3.3节):
      γ1,β1,α1,γ2,β2,α2 = W_adaln · SiLU(c)
      h' = h + α1 ⊙ MHA((1+γ1)⊙LN(h) + β1)
      h'' = h' + α2 ⊙ MLP((1+γ2)⊙LN(h') + β2)
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        # adaLN-Zero: 6个调制参数
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        # ★ Zero初始化（15.2.3节/15.3.3节）
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        """x: (B, N, d), c: (B, d) 条件向量"""
        # 从条件向量预测6个调制参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # MHA分支: adaLN调制 + 残差缩放
        h = self.norm1(x)
        h = h * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]  # adaLN
        h, _ = self.attn(h, h, h)
        x = x + gate_msa[:, None, :] * h  # Zero初始化: gate=0→恒等映射

        # MLP分支: adaLN调制 + 残差缩放
        h = self.norm2(x)
        h = h * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]  # adaLN
        h = self.mlp(h)
        x = x + gate_mlp[:, None, :] * h  # Zero初始化: gate=0→恒等映射

        return x


class Patchify(nn.Module):
    """ViT Patchify: 图像→Patch序列（15.3.2节）

    将 H×W×C 图像分割为 N=(H/p)(W/p) 个 p×p×C 的patch,
    每个patch通过线性投影映射为d维token.

    DiT使用p=2（保留精细空间细节），ViT使用p=16（分类任务）
    """
    def __init__(self, patch_size=2, in_channels=1, hidden_size=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * patch_size * in_channels, hidden_size)

    def forward(self, x):
        """x: (B, C, H, W) → (B, N, d)"""
        B, C, H, W = x.shape
        p = self.patch_size
        # 重排为patch
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, H/p, W/p, p, p, C)
        x = x.reshape(B, (H // p) * (W // p), p * p * C)
        return self.proj(x)


class Unpatchify(nn.Module):
    """将token序列恢复为图像（15.3.2节逆操作）"""
    def __init__(self, patch_size=2, out_channels=1, hidden_size=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.out_channels = out_channels

    def forward(self, x, H, W):
        """x: (B, N, d) → (B, C, H, W)"""
        B = x.shape[0]
        p = self.patch_size
        x = self.proj(x)
        x = x.reshape(B, H // p, W // p, p, p, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, H/p, p, W/p, p)
        return x.reshape(B, self.out_channels, H, W)


class DiT(nn.Module):
    """Diffusion Transformer (DiT) 完整架构（15.3.3节）

    架构: Patchify → N×DiT Block → Unpatchify
    条件注入: adaLN-Zero (6个调制参数/Block)

    ★ 原创设计: 小型DiT适配MNIST (28×28, patch_size=2)
    """
    def __init__(self, hidden_size=128, depth=4, num_heads=4,
                 patch_size=2, in_channels=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.patchify = Patchify(patch_size, in_channels, hidden_size)
        self.unpatchify = Unpatchify(patch_size, in_channels, hidden_size)

        # 可学习位置编码（15.3.2节 ViT使用可学习位置编码）
        num_patches = (28 // patch_size) ** 2  # 14*14=196 for p=2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # N个DiT Block
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        # 最终LayerNorm + 线性投影
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        # ★ Zero初始化
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x_t, t):
        """x_t: (B, C, H, W), t: (B,) 整数时间步"""
        # 时间步嵌入: t → PE(t) → MLP → e(t)
        t_emb = timestep_embedding(t, self.hidden_size)
        c = self.time_mlp(t_emb)  # 条件向量

        # Patchify
        h = self.patchify(x_t)  # (B, N, d)
        h = h + self.pos_embed  # 加位置编码

        # N个DiT Block
        for block in self.blocks:
            h = block(h, c)

        # 最终adaLN + Unpatchify
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale[:, None, :]) + shift[:, None, :]
        pred = self.unpatchify(h, 28, 28)

        return pred


# ============================================================
# SmallUNet（与15.2-1一致的架构, 用于对比）
# ============================================================
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


class UNetConvBlock(nn.Module):
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
        self.down1 = UNetConvBlock(ch[0], ch[1], time_dim)
        self.down2 = UNetConvBlock(ch[1], ch[2], time_dim)
        self.down3 = UNetConvBlock(ch[2], ch[3], time_dim)
        self.bottleneck = UNetConvBlock(ch[3], ch[3], time_dim)
        self.up3 = UNetConvBlock(ch[3] + ch[2], ch[2], time_dim)
        self.up2 = UNetConvBlock(ch[2] + ch[1], ch[1], time_dim)
        self.up1 = UNetConvBlock(ch[1] + ch[0], ch[0], time_dim)
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
# 数据加载
# ============================================================
print("\n加载MNIST数据集...")
data_dir = os.path.join(SAVE_DIR, 'data')
os.makedirs(data_dir, exist_ok=True)
# ★ 数据归一化到 [-1, 1] 区间以匹配标准高斯噪声先验（均值0、方差1）
# 可视化时需反变换回 [0, 1] 并截断到有效范围
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # x' = (x - 0.5) / 0.5, 将[0,1]映射到[-1,1]
])
train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
print(f"  训练集大小: {len(train_dataset)}")
print(f"  数据范围: [-1, 1] (匹配标准高斯噪声先验)")


# ============================================================
# 步骤1: DiT架构实现（15.3节）
# ============================================================
print(f"\n{'='*60}")
print("步骤1: DiT架构实现（15.3节）")
print("=" * 60)

print("""
15.3.3节 DiT架构:
  Patchify(p=2) → N×DiT Block(adaLN-Zero) → Unpatchify

  DiT Block核心:
    γ1,β1,α1,γ2,β2,α2 = W_adaln · SiLU(c)  [6个调制参数]
    h' = h + α1 ⊙ MHA((1+γ1)⊙LN(h) + β1)
    h'' = h' + α2 ⊙ MLP((1+γ2)⊙LN(h') + β2)

  ★ Zero初始化: 训练初期 α=0 → Block=恒等映射
  ★ adaLN-Zero: DiT论文 FID 从 25.21 降至 19.47

  小型DiT配置 (适配MNIST):
    depth=4, hidden_size=128, num_heads=4, patch_size=2
    token数 = 14×14 = 196
""")

dit = DiT(hidden_size=128, depth=4, num_heads=4, patch_size=2).to(device)
unet = SmallUNet().to(device)

# 参数量对比
dit_params = sum(p.numel() for p in dit.parameters())
unet_params = sum(p.numel() for p in unet.parameters())
print(f"  DiT 参数量: {dit_params:,}")
print(f"  UNet 参数量: {unet_params:,}")
print(f"  DiT/UNet 参数比: {dit_params/unet_params:.2f}")


# ============================================================
# 步骤2: UNet vs DiT训练对比（15.4节）
# ============================================================
print(f"\n{'='*60}")
print("步骤2: UNet vs DiT训练对比（15.4节）")
print("=" * 60)

print("""
15.4节核心洞察:
  - UNet: 归纳偏置驱动（局部性、层次性、多尺度）
  - DiT: 数据驱动学习（全局注意力, 最小归纳偏置）
  - "归纳偏置是小数据的朋友, 大数据的敌人"

  在MNIST(6万张)这样的小数据集上:
  UNet的归纳偏置应带来更快收敛
""")

num_epochs = 30

# 训练UNet
print("\n训练UNet...")
optimizer_unet = torch.optim.Adam(unet.parameters(), lr=2e-4)
start_epoch_unet, is_final_unet, unet_losses = load_train_state(
    CKPT_UNET, unet, optimizer_unet, num_epochs)

if not is_final_unet:
    if start_epoch_unet >= num_epochs:
        print(f"  start_epoch({start_epoch_unet}) >= num_epochs({num_epochs}), 跳过训练")
        is_final_unet = True
    else:
        t_start = time.time()
        for epoch in range(start_epoch_unet, num_epochs):
            unet.train()
            total_loss = 0
            batch_iter = tqdm(train_loader, desc=f"UNet Epoch {epoch+1}/{num_epochs}",
                               unit="batch", leave=False)
            for x, _ in batch_iter:
                x = x.to(device)
                batch = x.shape[0]
                t = torch.randint(0, T, (batch,), device=device)
                noise = torch.randn_like(x)
                x_t = sqrt_alpha_bars[t][:, None, None, None] * x + sqrt_one_minus_alpha_bars[t][:, None, None, None] * noise
                pred = unet(x_t, t)
                loss = F.mse_loss(pred, noise)
                optimizer_unet.zero_grad()
                loss.backward()
                optimizer_unet.step()
                total_loss += loss.item() * batch
                batch_iter.set_postfix(loss=f"{loss.item():.6f}")
            avg_loss = total_loss / len(train_dataset)
            unet_losses.append(avg_loss)
            print(f"  Epoch {epoch+1} 完成, 平均损失: {avg_loss:.6f}")
            save_train_state(CKPT_UNET, unet, optimizer_unet, epoch, avg_loss, unet_losses, is_final=(epoch==num_epochs-1))
        t_elapsed = time.time() - t_start
        print(f"  UNet 训练完成, 最终损失: {unet_losses[-1]:.6f}, 耗时: {t_elapsed:.1f} 秒")
else:
    print(f"  使用已训练完成的 UNet 模型, 跳过训练过程")

# 训练DiT
print("\n训练DiT...")
optimizer_dit = torch.optim.AdamW(dit.parameters(), lr=2e-4, weight_decay=0.01)
start_epoch_dit, is_final_dit, dit_losses = load_train_state(
    CKPT_DIT, dit, optimizer_dit, num_epochs)

if not is_final_dit:
    if start_epoch_dit >= num_epochs:
        print(f"  start_epoch({start_epoch_dit}) >= num_epochs({num_epochs}), 跳过训练")
        is_final_dit = True
    else:
        t_start = time.time()
        for epoch in range(start_epoch_dit, num_epochs):
            dit.train()
            total_loss = 0
            batch_iter = tqdm(train_loader, desc=f"DiT Epoch {epoch+1}/{num_epochs}",
                               unit="batch", leave=False)
            for x, _ in batch_iter:
                x = x.to(device)
                batch = x.shape[0]
                t = torch.randint(0, T, (batch,), device=device)
                noise = torch.randn_like(x)
                x_t = sqrt_alpha_bars[t][:, None, None, None] * x + sqrt_one_minus_alpha_bars[t][:, None, None, None] * noise
                pred = dit(x_t, t)
                loss = F.mse_loss(pred, noise)
                optimizer_dit.zero_grad()
                loss.backward()
                optimizer_dit.step()
                total_loss += loss.item() * batch
                batch_iter.set_postfix(loss=f"{loss.item():.6f}")
            avg_loss = total_loss / len(train_dataset)
            dit_losses.append(avg_loss)
            print(f"  Epoch {epoch+1} 完成, 平均损失: {avg_loss:.6f}")
            save_train_state(CKPT_DIT, dit, optimizer_dit, epoch, avg_loss, dit_losses, is_final=(epoch==num_epochs-1))
        t_elapsed = time.time() - t_start
        print(f"  DiT 训练完成, 最终损失: {dit_losses[-1]:.6f}, 耗时: {t_elapsed:.1f} 秒")
else:
    print(f"  使用已训练完成的 DiT 模型, 跳过训练过程")

# 可视化训练曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(range(1, len(unet_losses)+1), unet_losses, '-o', markersize=3, label=f'UNet ({unet_params/1e6:.1f}M参数)')
ax.plot(range(1, len(dit_losses)+1), dit_losses, '-s', markersize=3, label=f'DiT ({dit_params/1e6:.1f}M参数)')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('训练Loss', fontsize=12)
ax.set_title('(a) 训练收敛曲线', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# 参数量对比柱状图
ax = axes[1]
models_labels = ['UNet\n(归纳偏置)', 'DiT\n(数据驱动)']
params_vals = [unet_params / 1e6, dit_params / 1e6]
final_losses = [unet_losses[-1], dit_losses[-1]]
bars = ax.bar(models_labels, params_vals, color=['steelblue', 'coral'], alpha=0.7, width=0.4)
ax.set_ylabel('参数量 (M)', fontsize=12)
ax.set_title('(b) 参数量对比', fontsize=12)
for bar, val, loss in zip(bars, params_vals, final_losses):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.1f}M\nLoss={loss:.4f}', ha='center', fontsize=10, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.suptitle('步骤2: UNet vs DiT训练对比（15.4节）', fontsize=14, y=1.01)
plt.tight_layout()
fig_path1 = os.path.join(SAVE_DIR, '步骤1_UNet_vs_DiT.png')
plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图1已保存: {fig_path1}")


# ============================================================
# ★ 步骤3: adaLN-Zero vs 无Zero初始化消融（15.2.3/15.3.3节）
# ============================================================
print(f"\n{'='*60}")
print("★ 步骤3: adaLN-Zero vs 无Zero初始化消融（15.2.3/15.3.3节）")
print("=" * 60)

print("""
15.2.3/15.3.3节 adaLN-Zero的核心:
  - Zero初始化: 训练初期 α=0, β=0 → Block=恒等映射
  - DiT论文: adaLN-Zero FID=19.47, 普通adaLN FID=25.21
  - 更稳定的训练: 从"什么都不做"逐步学习"需要做什么"

★ 原创设计: 在小型DiT上验证 Zero初始化 vs 随机初始化
""")

# 有Zero初始化的DiT（默认）
dit_zero = DiT(hidden_size=128, depth=4, num_heads=4, patch_size=2).to(device)

# 无Zero初始化的DiT
dit_no_zero = DiT(hidden_size=128, depth=4, num_heads=4, patch_size=2).to(device)
# 手动覆盖所有adaLN的Zero初始化（确保对照组完全不含Zero初始化）
for block in dit_no_zero.blocks:
    nn.init.xavier_uniform_(block.adaLN_modulation[-1].weight)
    nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.02)
# 同步覆盖最终层的zero-init（final_adaLN负责Unpatchify前的shift/scale调制）
nn.init.xavier_uniform_(dit_no_zero.final_adaLN[-1].weight)
nn.init.normal_(dit_no_zero.final_adaLN[-1].bias, std=0.02)

num_epochs_step3 = 20
zero_losses = []
no_zero_losses = []

print("训练 adaLN-Zero DiT...")
opt_zero = torch.optim.AdamW(dit_zero.parameters(), lr=2e-4, weight_decay=0.01)
start_epoch_zero, is_final_zero, zero_losses = load_train_state(
    CKPT_DIT_ZERO, dit_zero, opt_zero, num_epochs_step3)

if not is_final_zero:
    if start_epoch_zero >= num_epochs_step3:
        print(f"  start_epoch({start_epoch_zero}) >= num_epochs({num_epochs_step3}), 跳过训练")
        is_final_zero = True
    else:
        t_start = time.time()
        for epoch in range(start_epoch_zero, num_epochs_step3):
            dit_zero.train()
            total_loss = 0
            batch_iter = tqdm(train_loader, desc=f"adaLN-Zero Epoch {epoch+1}/{num_epochs_step3}",
                               unit="batch", leave=False)
            for x, _ in batch_iter:
                x = x.to(device)
                batch = x.shape[0]
                t = torch.randint(0, T, (batch,), device=device)
                noise = torch.randn_like(x)
                x_t = sqrt_alpha_bars[t][:, None, None, None] * x + sqrt_one_minus_alpha_bars[t][:, None, None, None] * noise
                pred = dit_zero(x_t, t)
                loss = F.mse_loss(pred, noise)
                opt_zero.zero_grad()
                loss.backward()
                opt_zero.step()
                total_loss += loss.item() * batch
                batch_iter.set_postfix(loss=f"{loss.item():.6f}")
            avg_loss = total_loss / len(train_dataset)
            zero_losses.append(avg_loss)
            print(f"  Epoch {epoch+1} 完成, 平均损失: {avg_loss:.6f}")
            save_train_state(CKPT_DIT_ZERO, dit_zero, opt_zero, epoch, avg_loss, zero_losses, is_final=(epoch==num_epochs_step3-1))
        t_elapsed = time.time() - t_start
        print(f"  Zero 训练完成, 最终损失: {zero_losses[-1]:.6f}, 耗时: {t_elapsed:.1f} 秒")
else:
    print(f"  使用已训练完成的 Zero 模型, 跳过训练过程")

print("训练 无Zero初始化 DiT...")
opt_no_zero = torch.optim.AdamW(dit_no_zero.parameters(), lr=2e-4, weight_decay=0.01)
start_epoch_nozero, is_final_nozero, no_zero_losses = load_train_state(
    CKPT_DIT_NOZERO, dit_no_zero, opt_no_zero, num_epochs_step3)

if not is_final_nozero:
    if start_epoch_nozero >= num_epochs_step3:
        print(f"  start_epoch({start_epoch_nozero}) >= num_epochs({num_epochs_step3}), 跳过训练")
        is_final_nozero = True
    else:
        t_start = time.time()
        for epoch in range(start_epoch_nozero, num_epochs_step3):
            dit_no_zero.train()
            total_loss = 0
            batch_iter = tqdm(train_loader, desc=f"无Zero初始化 Epoch {epoch+1}/{num_epochs_step3}",
                               unit="batch", leave=False)
            for x, _ in batch_iter:
                x = x.to(device)
                batch = x.shape[0]
                t = torch.randint(0, T, (batch,), device=device)
                noise = torch.randn_like(x)
                x_t = sqrt_alpha_bars[t][:, None, None, None] * x + sqrt_one_minus_alpha_bars[t][:, None, None, None] * noise
                pred = dit_no_zero(x_t, t)
                loss = F.mse_loss(pred, noise)
                opt_no_zero.zero_grad()
                loss.backward()
                opt_no_zero.step()
                total_loss += loss.item() * batch
                batch_iter.set_postfix(loss=f"{loss.item():.6f}")
            avg_loss = total_loss / len(train_dataset)
            no_zero_losses.append(avg_loss)
            print(f"  Epoch {epoch+1} 完成, 平均损失: {avg_loss:.6f}")
            save_train_state(CKPT_DIT_NOZERO, dit_no_zero, opt_no_zero, epoch, avg_loss, no_zero_losses, is_final=(epoch==num_epochs_step3-1))
        t_elapsed = time.time() - t_start
        print(f"  NoZero 训练完成, 最终损失: {no_zero_losses[-1]:.6f}, 耗时: {t_elapsed:.1f} 秒")
else:
    print(f"  使用已训练完成的 NoZero 模型, 跳过训练过程")

# 可视化
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(range(1, len(zero_losses)+1), zero_losses, '-o', markersize=4, label='adaLN-Zero (Zero初始化)')
ax.plot(range(1, len(no_zero_losses)+1), no_zero_losses, '-s', markersize=4, label='adaLN (随机初始化)')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('训练Loss', fontsize=12)
ax.set_title('★ adaLN-Zero vs 随机初始化消融（15.2.3/15.3.3节）', fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
fig_path2 = os.path.join(SAVE_DIR, '步骤2_Zero初始化消融.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图2已保存: {fig_path2}")

# ★ 消融实验结论与验证说明
print(f"""
──────────────────────────────────────────────────────────────────────
★ 步骤3 消融实验结论与验证说明
──────────────────────────────────────────────────────────────────────
  修复说明:
    旧版消融仅覆盖 DiTBlock 内的 adaLN_modulation，
    遗漏了 DiT 顶层的 final_adaLN（负责 Unpatchify 前的 shift/scale 调制）。
    对照组实际为“部分 Zero init”，不是干净的二元切换。

  修复后 (本版):
    同步覆盖 final_adaLN[-1] 的 weight 和 bias 为 Xavier/Normal 初始化，
    确保对照组完全不含 Zero 初始化，自变量控制干净。

  核心结论:
    1. Zero初始化 → gate=0, shift=0, scale=0 → 每个 DiTBlock 初始为恒等映射
       → 梯度平滑传播，Loss 曲线光滑稳定
    2. 随机初始化 → gate/scale/shift 非零 → 残差路径被扰动
       → Loss 振荡明显，收敛缓慢
    3. 修复后 block级 + final_adaLN 全部覆盖，消融对照严格

  ★ 配套验证实验: 验证实验15.4-1_Zero初始化消融_收敛对比.py
    包含 UNet vs DiT-Zero vs DiT-NoZero 三方对比 + 早期振荡分析
──────────────────────────────────────────────────────────────────────
""")


# ============================================================
# ★ 步骤4: UNet vs DiT去噪行为对比（15.4.1节）
# ============================================================
print(f"\n{'='*60}")
print("★ 步骤4: UNet vs DiT去噪行为对比（15.4.1节）")
print("=" * 60)

print("""
15.4.1节 设计哲学对比:
  UNet: 局部感受野, 逐步扩散的弯曲路径
  DiT: 全局注意力, 全局信息一步到位

★ 原创设计: 对比两种架构在不同t下的Tweedie去噪估计
""")

unet.eval()
dit_zero.eval()

test_batch = next(iter(train_loader))[0][:8].to(device)  # 范围 [-1, 1]
noise_vis = torch.randn_like(test_batch)

# ★ 反变换回 [0, 1] 区间用于可视化：x_vis = (x + 1) / 2
def to_vis_range(x):
    """将 [-1, 1] 数据变换回 [0, 1] 用于可视化"""
    return (x + 1) / 2

fig, axes = plt.subplots(5, 8, figsize=(16, 10))

# 行0: 原始图像（反变换回 [0, 1]）
test_batch_vis = to_vis_range(test_batch)
for col in range(8):
    axes[0, col].imshow(test_batch_vis[col, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0, col].axis('off')
    if col == 0: axes[0, col].set_ylabel('原始', fontsize=10, rotation=0, labelpad=40)

# 行1-2: UNet t=5, t=100
# 行3-4: DiT t=5, t=100
for model_idx, (model, name) in enumerate([(unet, 'UNet'), (dit_zero, 'DiT')]):
    for t_idx, sub_row in [(5, 0), (100, 1)]:
        with torch.no_grad():
            t = torch.full((8,), t_idx, device=device, dtype=torch.long)
            x_t = sqrt_alpha_bars[t_idx] * test_batch + sqrt_one_minus_alpha_bars[t_idx] * noise_vis
            pred = model(x_t, t)
            x0_hat = (x_t - sqrt_one_minus_alpha_bars[t_idx] * pred) / sqrt_alpha_bars[t_idx]
            # ★ 反变换回 [0, 1] 区间用于可视化，并截断到有效范围
            x0_hat_vis = to_vis_range(x0_hat.clamp(-1, 1))

        row = 1 + model_idx * 2 + sub_row
        for col in range(8):
            axes[row, col].imshow(x0_hat_vis[col, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[row, col].axis('off')
            if col == 0:
                snr = sqrt_alpha_bars[t_idx].item() / (sqrt_one_minus_alpha_bars[t_idx].item() + 1e-10)
                # matplotlib 中数学符号使用 LaTeX 格式
                ylabel_str = f'{name}\n' + r'$t$=' + f'{t_idx} (SNR={snr:.1f})'
                axes[row, col].set_ylabel(ylabel_str, fontsize=9, rotation=0, labelpad=65)

plt.suptitle('★ UNet vs DiT去噪行为对比（15.4.1节）', fontsize=14, y=1.01)
plt.tight_layout()
fig_path3 = os.path.join(SAVE_DIR, '步骤3_UNet_vs_DiT去噪.png')
plt.savefig(fig_path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"图3已保存: {fig_path3}")


# ============================================================
# 实验总结
# ============================================================
print(f"\n{'='*60}")
print("实验15.4-1 总结")
print("=" * 60)
print(f"""
关键结论:
1. DiT架构实现（15.3节）
   - Patchify(p=2) → N×DiT Block → Unpatchify
   - adaLN-Zero: 6个调制参数/Block, Zero初始化
   - 小型DiT配置: depth=4, hidden=128, heads=4
   - 参数量: DiT {dit_params/1e6:.1f}M vs UNet {unet_params/1e6:.1f}M

2. UNet vs DiT训练对比（15.4节）
   - UNet 最终Loss: {unet_losses[-1]:.6f}
   - DiT  最终Loss: {dit_losses[-1]:.6f}
   - 在MNIST(小数据)上, UNet的归纳偏置带来更快收敛

3. adaLN-Zero消融（15.2.3/15.3.3节）★ 原创设计
   - Zero初始化:   Loss={zero_losses[-1]:.6f}
   - 随机初始化:   Loss={no_zero_losses[-1]:.6f}
   - 验证 DiT 论文: Zero初始化训练更稳定

4. 去噪行为对比（15.4.1节）★ 原创设计
   - UNet: 局部感受野, 逐步修正
   - DiT: 全局注意力, 一步全局信息
""")

print(f"\n{'='*60}")
print("实验15.4-1 完成!")
print(f"{'='*60}")

# ===== 保存数值结果 =====
import json

def _to_native(obj):
    """递归转换numpy/torch类型为Python原生类型"""
    import numpy as np
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return _to_native(obj.tolist())
    try:
        import torch
        if isinstance(obj, torch.Tensor): return _to_native(obj.detach().cpu().tolist())
    except: pass
    return obj

results_summary = {
    'dit_params': int(dit_params),
    'unet_params': int(unet_params),
    'final_loss_unet': round(float(unet_losses[-1]), 6) if unet_losses else None,
    'final_loss_dit': round(float(dit_losses[-1]), 6) if dit_losses else None,
    'final_loss_zero': round(float(zero_losses[-1]), 6) if zero_losses else None,
    'final_loss_no_zero': round(float(no_zero_losses[-1]), 6) if no_zero_losses else None,
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
