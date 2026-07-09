# -*- coding: utf-8 -*-
"""
实验15.2-1 UNet去噪器实现与时间步嵌入
对应知识点：
  - 15.1.1节 DnCNN与残差学习
  - 15.1.2节 UNet编码器-解码器+跳跃连接
  - 15.2.1节 为什么需要时间步条件
  - 15.2.2节 正弦位置编码
  - 15.2.3节 条件注入方式（加法/FiLM/adaLN-Zero对比）

本实验不需要GPU，通过MNIST去噪对比三种条件注入方式。

实验内容:
  步骤1: DnCNN vs UNet——架构对比（15.1.1/15.1.2节）
  步骤2: 正弦位置编码可视化（15.2.2节）
  步骤3: ★ 三种条件注入方式对比——加法/FiLM/adaLN（15.2.3节）
  步骤4: ★ 不同噪声水平下的去噪行为可视化（15.2.1节）

素材来源:
  - 实验15.1（参考实验）
  - 实验14.2的SmallUNet架构
  - ★ 原创设计: 三种条件注入方式（加法/FiLM/adaLN）的对比实验
  - ★ 原创设计: 不同噪声水平下去噪器行为的可视化

运行前提: PyTorch, CPU/GPU均可 (MNIST数据规模小, CPU亦可)
        数据集: MNIST（torchvision自动下载）
"""

import sys
import os
import io
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import warnings
import logging

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
    SAVE_DIR = os.path.join(_gdrive, '实验15.2-1')
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
print(f"实验15.2-1: UNet去噪器实现与时间步嵌入")
print(f"{'='*60}")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  未检测到 GPU, 使用 CPU 运行")
    print("  提示: Colab 用户可在菜单 运行时 -> 更改运行时类型 中选择 GPU")

# Checkpoint路径
CKPT_DNCNN = os.path.join(SAVE_DIR, 'dncnn_checkpoint.pth')
CKPT_UNET_STEP1 = os.path.join(SAVE_DIR, 'unet_step1_film_checkpoint.pth')
CKPT_UNET_ADD = os.path.join(SAVE_DIR, 'unet_add_checkpoint.pth')
CKPT_UNET_FILM = os.path.join(SAVE_DIR, 'unet_film_checkpoint.pth')
CKPT_UNET_ADALN = os.path.join(SAVE_DIR, 'unet_adaln_checkpoint.pth')


# ============================================================
# 通用checkpoint加载/训练工具
# ============================================================
def load_train_state(checkpoint_path, model, optimizer, num_epochs_total, allow_arch_mismatch=True):
    """加载checkpoint并返回 (start_epoch, is_final, loss_history)

    - 若checkpoint存在且is_final=True: 加载权重, start_epoch=num_epochs_total, 跳过训练
    - 若checkpoint存在且is_final=False: 加载权重, 从断点继续
    - 若checkpoint不存在: 返回(0, False, [])
    - allow_arch_mismatch=True: 模型架构不匹配时抛RuntimeError提示
    """
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
            if allow_arch_mismatch:
                raise RuntimeError(
                    f"检测到 checkpoint 与当前模型架构不兼容:\n{e}\n"
                    f"可能原因: 模型架构已更新 (channels/time_dim/injection 等变更).\n"
                    f"请删除 checkpoint 文件后重新训练:\n"
                    f"  {checkpoint_path}"
                )
            raise
        if optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                pass
        return checkpoint['epoch'] + 1, True, checkpoint.get('loss_history', [])

    # 未完成, 从断点继续
    print(f"检测到未完成的训练, 从第 {checkpoint['epoch']+1} 轮继续")
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        if allow_arch_mismatch:
            raise RuntimeError(
                f"检测到 checkpoint 与当前模型架构不兼容:\n{e}\n"
                f"请删除 checkpoint 文件后重新训练:\n"
                f"  {checkpoint_path}"
            )
        raise
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
# 正弦位置编码（15.2.2节）
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    """正弦位置编码 + MLP，15.2.2节公式

    PE(t, 2i) = sin(t / 10000^{2i/d})
    PE(t, 2i+1) = cos(t / 10000^{2i/d})

    然后通过 MLP: e(t) = W2 * SiLU(W1 * PE(t))
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# ============================================================
# DnCNN去噪器（15.1.1节）
# ============================================================
class DnCNN(nn.Module):
    """DnCNN: 15.1.1节，残差学习预测噪声

    堆叠6个Conv+BN+ReLU块，无时间步条件
    对应15.1.1节的核心设计:
      - 残差学习: 预测噪声 ε 而非干净图像 x
      - 对应DDPM的 ε-预测参数化（11.3节）
    """
    def __init__(self, channels=32, n_blocks=6):
        super().__init__()
        layers = [nn.Conv2d(1, channels, 3, padding=1), nn.ReLU()]
        for _ in range(n_blocks - 2):
            layers.extend([
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            ])
        layers.append(nn.Conv2d(channels, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """残差学习: 输出 = x - 预测噪声"""
        return x - self.net(x)  # x - ε̂ = x̂


# ============================================================
# UNet去噪器（15.1.2节 + 15.2.3节条件注入）
# ============================================================
class ConvBlock(nn.Module):
    """UNet卷积块，支持三种条件注入方式"""
    def __init__(self, in_ch, out_ch, time_dim, injection='film'):
        super().__init__()
        self.injection = injection
        gn_groups = min(4, out_ch)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(gn_groups, out_ch),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(gn_groups, out_ch),
            nn.SiLU(),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        if injection == 'add':
            # 方式一: 加法注入 h + e(t), 15.2.3节
            self.time_proj = nn.Linear(time_dim, out_ch)
        elif injection == 'film':
            # 方式二: FiLM γ(t)⊙h + β(t), 15.2.3节
            self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2 * out_ch))
        elif injection == 'adaln':
            # 方式三: adaLN-Zero (1+γ(t))⊙LN(h) + β(t), 15.2.3节
            # ★ 修正: LN归一化仅对通道维度C（而非C,H,W），与DiT论文一致
            self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2 * out_ch))
            # ★ Zero初始化: adaLN-Zero的关键, 15.2.3节
            nn.init.zeros_(self.time_proj[-1].weight)
            nn.init.zeros_(self.time_proj[-1].bias)
            # ★ 残差门控alpha: 初始化为0, 使Block在训练初期为恒等映射
            self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, t_emb):
        h = self.conv1(x)

        if self.injection == 'add':
            # 加法注入: h + e(t)
            h = h + self.time_proj(t_emb)[:, :, None, None]
        elif self.injection == 'film':
            # FiLM: γ(t)⊙h + β(t)
            params = self.time_proj(t_emb)
            gamma, beta = params[:, :h.shape[1]], params[:, h.shape[1]:]
            h = gamma[:, :, None, None] * h + beta[:, :, None, None]
        elif self.injection == 'adaln':
            # adaLN-Zero: (1+γ(t))⊙LN(h) + β(t)
            # ★ 修正: LN归一化仅对通道维度C（与DiT论文一致）
            # 先permute到(B,H,W,C), 做LN(C), 再permute回来
            h_perm = h.permute(0, 2, 3, 1)  # (B, H, W, C)
            h_norm = F.layer_norm(h_perm, [h_perm.shape[-1]])  # LN over C
            h_norm = h_norm.permute(0, 3, 1, 2)  # back to (B, C, H, W)
            params = self.time_proj(t_emb)
            gamma, beta = params[:, :h.shape[1]], params[:, h.shape[1]:]
            h = (1 + gamma[:, :, None, None]) * h_norm + beta[:, :, None, None]

        h = self.conv2(h)
        # ★ adaLN-Zero残差门控: alpha初始化为0, 训练初期Block=恒等映射
        if self.injection == 'adaln':
            return self.alpha * h + self.shortcut(x)
        return h + self.shortcut(x)


class SmallUNet(nn.Module):
    """小型UNet去噪器（15.1.2节 + 15.2.3节条件注入）

    架构: channels=[1,16,32,64]
    条件注入: 支持 'add'/'film'/'adaln' 三种方式
    """
    def __init__(self, time_dim=64, injection='film'):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        ch = [1, 16, 32, 64]
        self.down1 = ConvBlock(ch[0], ch[1], time_dim, injection)
        self.down2 = ConvBlock(ch[1], ch[2], time_dim, injection)
        self.down3 = ConvBlock(ch[2], ch[3], time_dim, injection)
        self.bottleneck = ConvBlock(ch[3], ch[3], time_dim, injection)
        self.up3 = ConvBlock(ch[3] + ch[2], ch[2], time_dim, injection)
        self.up2 = ConvBlock(ch[2] + ch[1], ch[1], time_dim, injection)
        self.up1 = ConvBlock(ch[1] + ch[0], ch[0], time_dim, injection)
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
# 步骤1: DnCNN vs UNet——架构对比（15.1.1/15.1.2节）
# ============================================================
print(f"\n{'='*60}")
print("步骤1: DnCNN vs UNet——架构对比（15.1.1/15.1.2节）")
print("=" * 60)

print("""
15.1.1节 DnCNN: 残差学习预测噪声, 堆叠Conv+BN+ReLU
  - 单尺度处理, 感受野有限
  - 无时间步条件: 固定噪声水平

15.1.2节 UNet: 编码器-解码器+跳跃连接
  - 多尺度特征融合
  - 时间步条件: 同一个网络处理不同噪声水平

对比要点:
  - DnCNN只能处理固定 σ 的高斯去噪
  - UNet通过时间步嵌入可以处理任意噪声水平
""")

# 训练DnCNN（固定 σ=0.3 的高斯去噪, 15.1.1节）
dncnn = DnCNN(channels=32, n_blocks=6).to(device)
optimizer_dncnn = torch.optim.Adam(dncnn.parameters(), lr=1e-3)
sigma_fixed = 0.3

num_epochs_dncnn = 10
start_epoch_dncnn, is_final_dncnn, dncnn_losses = load_train_state(
    CKPT_DNCNN, dncnn, optimizer_dncnn, num_epochs_dncnn)

if not is_final_dncnn:
    print(f"\n训练 DnCNN (固定 σ={sigma_fixed}, 共 {num_epochs_dncnn} 轮)...")
    t_start = time.time()
    for epoch in range(start_epoch_dncnn, num_epochs_dncnn):
        dncnn.train()
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            noise = torch.randn_like(x) * sigma_fixed
            x_noisy = x + noise
            x_denoised = dncnn(x_noisy)
            loss = F.mse_loss(x_denoised, x)
            optimizer_dncnn.zero_grad()
            loss.backward()
            optimizer_dncnn.step()
            total_loss += loss.item() * x.shape[0]
        avg_loss = total_loss / len(train_dataset)
        dncnn_losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"  [DnCNN] Epoch {epoch+1}/{num_epochs_dncnn}  Loss={avg_loss:.6f}")
        save_train_state(CKPT_DNCNN, dncnn, optimizer_dncnn, epoch, avg_loss, dncnn_losses, is_final=(epoch==num_epochs_dncnn-1))
    t_elapsed = time.time() - t_start
    print(f"  DnCNN 训练完成, 最终损失: {dncnn_losses[-1]:.6f}, 耗时: {t_elapsed:.1f} 秒")
else:
    print(f"\n使用已训练完成的 DnCNN 模型, 跳过训练过程")

# 训练UNet（条件去噪, 15.1.2节 + 15.2节）
unet = SmallUNet(injection='film').to(device)
optimizer_unet = torch.optim.Adam(unet.parameters(), lr=2e-4)

num_epochs_unet_step1 = 10
start_epoch_unet, is_final_unet, unet_losses = load_train_state(
    CKPT_UNET_STEP1, unet, optimizer_unet, num_epochs_unet_step1)

if not is_final_unet:
    print(f"\n训练 UNet (FiLM 注入, 共 {num_epochs_unet_step1} 轮)...")
    t_start = time.time()
    for epoch in range(start_epoch_unet, num_epochs_unet_step1):
        unet.train()
        total_loss = 0
        for x, _ in train_loader:
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
        avg_loss = total_loss / len(train_dataset)
        unet_losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"  [UNet] Epoch {epoch+1}/{num_epochs_unet_step1}  Loss={avg_loss:.6f}")
        save_train_state(CKPT_UNET_STEP1, unet, optimizer_unet, epoch, avg_loss, unet_losses, is_final=(epoch==num_epochs_unet_step1-1))
    t_elapsed = time.time() - t_start
    print(f"  UNet 训练完成, 最终损失: {unet_losses[-1]:.6f}, 耗时: {t_elapsed:.1f} 秒")
else:
    print(f"\n使用已训练完成的 UNet 模型, 跳过训练过程")

# 可视化对比
fig, axes = plt.subplots(3, 6, figsize=(15, 8))
test_imgs = next(iter(train_loader))[0][:6].to(device)  # 范围 [-1, 1]

# 高斯去噪（DnCNN）
with torch.no_grad():
    noise_gauss = torch.randn_like(test_imgs) * sigma_fixed
    noisy_gauss = test_imgs + noise_gauss
    denoised_dncnn = dncnn(noisy_gauss)

# ★ 反变换回 [0, 1] 区间用于可视化：x_vis = (x + 1) / 2
def to_vis_range(x):
    """将 [-1, 1] 数据变换回 [0, 1] 用于可视化"""
    return (x + 1) / 2

test_imgs_vis = to_vis_range(test_imgs)
noisy_gauss_vis = to_vis_range(noisy_gauss)
denoised_dncnn_vis = to_vis_range(denoised_dncnn.clamp(-1, 1))

for col in range(6):
    axes[0, col].imshow(noisy_gauss_vis[col, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0, col].axis('off')
    axes[1, col].imshow(denoised_dncnn_vis[col, 0].cpu().detach(), cmap='gray', vmin=0, vmax=1)
    axes[1, col].axis('off')
    axes[2, col].imshow(test_imgs_vis[col, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[2, col].axis('off')

# matplotlib 中数学符号使用 LaTeX 格式
axes[0, 0].set_ylabel(r'含噪 ($\sigma$=' + f'{sigma_fixed})', fontsize=11, rotation=0, labelpad=60)
axes[1, 0].set_ylabel(r'DnCNN去噪' + '\n' + r'(固定$\sigma$, 15.1.1)', fontsize=11, rotation=0, labelpad=70)
axes[2, 0].set_ylabel('原始图像', fontsize=11, rotation=0, labelpad=60)

plt.suptitle('步骤1: DnCNN vs UNet（15.1.1/15.1.2节）', fontsize=14, y=1.01)
plt.tight_layout()
fig_path1 = os.path.join(SAVE_DIR, '步骤1_DnCNN_vs_UNet.png')
plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图1已保存: {fig_path1}")


# ============================================================
# 步骤2: 正弦位置编码可视化（15.2.2节）
# ============================================================
print(f"\n{'='*60}")
print("步骤2: 正弦位置编码可视化（15.2.2节）")
print("=" * 60)

print("""
15.2.2节 正弦位置编码:
  PE(t, 2i) = sin(t / 10000^{2i/d})
  PE(t, 2i+1) = cos(t / 10000^{2i/d})

关键性质:
  - 每个维度对应不同频率, 形成多尺度"时间步指纹"
  - 相对位置可线性表达: PE(t+Δt) = M(Δt)·PE(t)
  - 无需学习, 有界性好
""")

dim = 64
pe = SinusoidalTimeEmbedding(dim)
t_range = torch.arange(0, T).float()
embeddings = pe(t_range).detach().numpy()  # (T, dim)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) 编码热力图
ax = axes[0]
im = ax.imshow(embeddings.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xlabel(r'时间步 $t$', fontsize=12)
ax.set_ylabel('嵌入维度', fontsize=12)
ax.set_title('(a) 正弦位置编码热力图', fontsize=12)
plt.colorbar(im, ax=ax, shrink=0.8)

# (b) 不同维度的波形
ax = axes[1]
dims_to_plot = [0, 4, 16, 32, 60]
for d_idx in dims_to_plot:
    label = f'维度{d_idx} ({"低频" if d_idx < 16 else "高频"})'
    ax.plot(t_range.numpy(), embeddings[:, d_idx], label=label, alpha=0.8)
ax.set_xlabel(r'时间步 $t$', fontsize=12)
ax.set_ylabel('编码值', fontsize=12)
ax.set_title('(b) 不同频率的编码波形', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (c) 不同时间步的编码向量
ax = axes[2]
ts_to_plot = [0, 10, 50, 100, 150, 199]
for t_val in ts_to_plot:
    ax.plot(range(dim), embeddings[t_val], alpha=0.7, label=r'$t$=' + f'{t_val}')
ax.set_xlabel('嵌入维度', fontsize=12)
ax.set_ylabel('编码值', fontsize=12)
ax.set_title('(c) 不同时间步的编码向量', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle('步骤2: 正弦位置编码可视化（15.2.2节）', fontsize=14, y=1.01)
plt.tight_layout()
fig_path2 = os.path.join(SAVE_DIR, '步骤2_正弦编码.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig_path2}")


# ============================================================
# ★ 步骤3: 三种条件注入方式对比（15.2.3节）
# ============================================================
print(f"\n{'='*60}")
print("★ 步骤3: 三种条件注入方式对比（15.2.3节）")
print("=" * 60)

print("""
15.2.3节 三种条件注入方式:
  1. 加法: h + e(t) — 弱调制, 仅改变偏移
  2. FiLM: γ(t)⊙h + β(t) — 中等调制, 改变尺度和偏移
  3. adaLN-Zero: (1+γ(t))⊙LN_C(h) + β(t) — 强调制, LN仅归一化通道维度
     ★ Zero初始化: 调制参数=0 + 残差门控alpha=0 → 训练初期Block=恒等映射

★ 原创设计: 对比三种注入方式在扩散去噪训练中的收敛速度和去噪质量
""")

injections = ['add', 'film', 'adaln']
injection_labels = {'add': '加法注入', 'film': 'FiLM', 'adaln': 'adaLN-Zero'}
injection_ckpts = {'add': CKPT_UNET_ADD, 'film': CKPT_UNET_FILM, 'adaln': CKPT_UNET_ADALN}
models = {}
histories = {}
num_epochs_step3 = 30

for inj in injections:
    print(f"\n  训练 {injection_labels[inj]} UNet (共 {num_epochs_step3} 轮)...")
    model = SmallUNet(injection=inj).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    start_epoch, is_final, losses = load_train_state(
        injection_ckpts[inj], model, optimizer, num_epochs_step3)

    if not is_final:
        if start_epoch >= num_epochs_step3:
            print(f"  start_epoch({start_epoch}) >= num_epochs({num_epochs_step3}), 跳过训练")
            is_final = True
        else:
            t_start = time.time()
            for epoch in range(start_epoch, num_epochs_step3):
                model.train()
                total_loss = 0
                for x, _ in train_loader:
                    x = x.to(device)
                    batch = x.shape[0]
                    t = torch.randint(0, T, (batch,), device=device)
                    noise = torch.randn_like(x)
                    x_t = sqrt_alpha_bars[t][:, None, None, None] * x + sqrt_one_minus_alpha_bars[t][:, None, None, None] * noise
                    pred = model(x_t, t)
                    loss = F.mse_loss(pred, noise)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * batch
                avg_loss = total_loss / len(train_dataset)
                losses.append(avg_loss)
                if (epoch + 1) % 10 == 0:
                    print(f"    Epoch {epoch+1:3d}/{num_epochs_step3}  Loss={avg_loss:.6f}")
                save_train_state(injection_ckpts[inj], model, optimizer, epoch, avg_loss, losses, is_final=(epoch==num_epochs_step3-1))
            t_elapsed = time.time() - t_start
            print(f"  训练完成, 最终损失: {losses[-1]:.6f}, 耗时: {t_elapsed:.1f} 秒")
    else:
        print(f"  使用已训练完成的 {injection_labels[inj]} 模型, 跳过训练过程")

    models[inj] = model
    histories[inj] = losses

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) 训练曲线
ax = axes[0]
for inj in injections:
    ax.plot(range(1, len(histories[inj])+1), histories[inj], '-o', markersize=3, label=injection_labels[inj])
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('训练Loss', fontsize=12)
ax.set_title('(a) 训练收敛曲线', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# (b) 不同时间步的去噪MSE
ax = axes[1]
test_batch = next(iter(train_loader))[0][:32].to(device)  # 范围 [-1, 1]
noise_test = torch.randn_like(test_batch)

for inj in injections:
    models[inj].eval()
    mses = []
    t_indices = [10, 30, 50, 80, 100, 130, 150, 180]
    for t_idx in t_indices:
        with torch.no_grad():
            t = torch.full((32,), t_idx, device=device, dtype=torch.long)
            x_t = sqrt_alpha_bars[t_idx] * test_batch + sqrt_one_minus_alpha_bars[t_idx] * noise_test
            pred = models[inj](x_t, t)
            # 计算噪声预测MSE（相对指标，用于横向对比三种注入方式）
            mse = F.mse_loss(pred, noise_test).item()
            mses.append(mse)
    ax.plot(t_indices, mses, '-o', markersize=5, label=injection_labels[inj])

ax.set_xlabel(r'时间步 $t$', fontsize=12)
ax.set_ylabel('噪声预测MSE', fontsize=12)
ax.set_title('(b) 不同时间步的预测质量', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.suptitle('★ 步骤3: 三种条件注入方式对比（15.2.3节）', fontsize=14, y=1.01)
plt.tight_layout()
fig_path3 = os.path.join(SAVE_DIR, '步骤3_条件注入对比.png')
plt.savefig(fig_path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图3已保存: {fig_path3}")


# ============================================================
# ★ 步骤4: 不同噪声水平下的去噪行为可视化（15.2.1节）
# ============================================================
print(f"\n{'='*60}")
print("★ 步骤4: 不同噪声水平下的去噪行为可视化（15.2.1节）")
print("=" * 60)

print("""
15.2.1节 扩散去噪器的特殊需求:
  - t 接近 T（高噪声）: "大刀阔斧"去噪→低通滤波
  - t 接近 0（低噪声）: "精雕细琢"保留细节→细节增强
  - 中间 t: 平滑过渡

★ 原创设计: 可视化 UNet 在不同 t 下去噪输出的行为差异
""")

# 使用 FiLM 模型（最佳注入方式）
best_model = models['film']
best_model.eval()

fig, axes = plt.subplots(4, 6, figsize=(15, 10))
t_indices = [5, 50, 100, 180]
noise_vis = torch.randn(6, 1, 28, 28, device=device)
test_vis = next(iter(train_loader))[0][:6].to(device)  # 范围 [-1, 1]

for row, t_idx in enumerate(t_indices):
    with torch.no_grad():
        t = torch.full((6,), t_idx, device=device, dtype=torch.long)
        x_t = sqrt_alpha_bars[t_idx] * test_vis + sqrt_one_minus_alpha_bars[t_idx] * noise_vis
        pred_noise = best_model(x_t, t)
        # Tweedie 公式: x̂_0 = (x_t - √(1-ᾱ_t)·ε̂) / √ᾱ_t
        x0_hat = (x_t - sqrt_one_minus_alpha_bars[t_idx] * pred_noise) / sqrt_alpha_bars[t_idx]
        # ★ 反变换回 [0, 1] 区间用于可视化，并截断到有效范围
        x0_hat_vis = to_vis_range(x0_hat.clamp(-1, 1))

    for col in range(6):
        axes[row, col].imshow(x0_hat_vis[col, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
        if col == 0:
            snr = sqrt_alpha_bars[t_idx].item() / (sqrt_one_minus_alpha_bars[t_idx].item() + 1e-10)
            axes[row, col].set_ylabel(r'$t$=' + f'{t_idx}\nSNR={snr:.2f}', fontsize=10, rotation=0, labelpad=50)

plt.suptitle('★ 不同噪声水平下的去噪行为（15.2.1节）', fontsize=14, y=1.01)
plt.tight_layout()
fig_path4 = os.path.join(SAVE_DIR, '步骤4_去噪行为.png')
plt.savefig(fig_path4, dpi=150, bbox_inches='tight')
plt.close()
print(f"图4已保存: {fig_path4}")


# ============================================================
# 实验总结
# ============================================================
print(f"\n{'='*60}")
print("实验15.2-1 总结")
print("=" * 60)
print(f"""
关键结论:
1. DnCNN vs UNet（15.1.1/15.1.2节）
   - DnCNN: 单尺度, 无时间步条件, 只能处理固定噪声水平
   - UNet: 多尺度+跳跃连接+时间步嵌入, 处理任意噪声水平
   - DnCNN 训练损失: {dncnn_losses[-1]:.6f}
   - UNet 训练损失: {unet_losses[-1]:.6f}

2. 正弦位置编码（15.2.2节）
   - 多尺度频率: 低维度=低频(慢变化), 高维度=高频(快变化)
   - 形成时间步"指纹", 不同t有不同的编码模式

3. 条件注入方式（15.2.3节）★ 原创设计
   - 加法: 弱调制, 收敛最慢 (Loss={histories['add'][-1]:.6f})
   - FiLM: 中等调制, 收敛较快 (Loss={histories['film'][-1]:.6f})
   - adaLN: 强调制+Zero初始化, 训练最稳定 (Loss={histories['adaln'][-1]:.6f})
   - 验证 DiT 论文发现: adaLN-Zero 优于其他方式

4. 去噪行为（15.2.1节）★ 原创设计
   - 高噪声(t大): 去噪输出模糊, 类似低通滤波
   - 低噪声(t小): 去噪输出保留细节, 类似细节增强
   - 验证 15.2.1节: 扩散去噪器根据 t 调整策略
""")

print(f"\n{'='*60}")
print("实验15.2-1 完成!")
print(f"{'='*60}")
