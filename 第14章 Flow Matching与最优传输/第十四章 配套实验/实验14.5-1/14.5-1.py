# -*- coding: utf-8 -*-
"""
实验14.5-1 Logit-Normal时间采样
对应章节: 14.5.1节

知识点:
  - 均匀时间采样: t ~ U[0,1]，对所有时间步同等对待
  - Logit-Normal采样: π_ln(t; m, s)，使中间时间步采样概率更高
  - SD3的60种轨迹对比实验: Rectified Flow + Logit-Normal采样效果最优
  - 参数m控制重心位置，s控制集中程度

实验内容:
  步骤1: 可视化均匀采样 vs Logit-Normal采样的时间分布
  步骤2: 在MNIST上训练Rectified Flow（均匀时间采样）
  步骤3: 在MNIST上训练Rectified Flow（Logit-Normal时间采样）
  步骤4: 对比两种采样策略的生成质量和训练收敛

数据集: MNIST，GPU推荐但CPU也可

素材来源:
  - Esser et al. (2024) SD3 Scaling Rectified Flow Transformers
  - 14.3-2中的SmallUNet架构

运行前提: PyTorch, torchvision, CPU/GPU均可
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import logging
import warnings
import time
from tqdm import tqdm

# 静默 matplotlib 相关警告
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", message=".*Glyph.*")
warnings.filterwarnings("ignore", message=".*cmap.*")

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验14.5-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(_chinese_path, exist_ok=True)

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)

import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"实验14.5-1: Logit-Normal时间采样")
print(f"{'='*60}")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  未检测到 GPU, 使用 CPU 训练")
    print("  提示: Colab 用户可在菜单 运行时 -> 更改运行时类型 中选择 GPU")

# Checkpoint路径
CHECKPOINT_PATH_UNIFORM = os.path.join(SAVE_DIR, 'rf_uniform_checkpoint.pth')
FINAL_CHECKPOINT_PATH_UNIFORM = os.path.join(SAVE_DIR, 'rf_uniform_final.pth')
CHECKPOINT_PATH_LOGIT = os.path.join(SAVE_DIR, 'rf_logit_checkpoint.pth')
FINAL_CHECKPOINT_PATH_LOGIT = os.path.join(SAVE_DIR, 'rf_logit_final.pth')

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ============================================================
# Logit-Normal时间采样
# ============================================================
def logit_normal_sample(batch_size, m=0.0, s=1.0, device='cpu'):
    """Logit-Normal时间采样（14.5.1节）

    π_ln(t; m, s) = 1/(s*sqrt(2π)) * 1/(t*(1-t)) * exp(-(logit(t)-m)^2 / (2s^2))

    采样方法:
    1. 从 N(m, s^2) 采样 u
    2. t = sigmoid(u) = 1/(1+exp(-u))

    参数:
      m: 重心位置（m=0时重心在t=0.5，m>0时偏右）
      s: 集中程度（s越小越集中，s→∞时退化为均匀分布）
    """
    u = torch.randn(batch_size, device=device) * s + m
    t = torch.sigmoid(u)
    return t.clamp(1e-5, 1 - 1e-5)  # 数值保护，避免t=0或t=1


def logit_normal_pdf(t, m=0.0, s=1.0):
    """Logit-Normal概率密度函数"""
    t = np.clip(t, 1e-5, 1 - 1e-5)
    logit_t = np.log(t / (1 - t))
    pdf = (1.0 / (s * np.sqrt(2 * np.pi)) *
           1.0 / (t * (1 - t)) *
           np.exp(-(logit_t - m) ** 2 / (2 * s ** 2)))
    return pdf


# ============================================================
# 去噪网络: 小型UNet（与14.3-2一致）
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
    """小型UNet——适配Rectified Flow的速度预测"""
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
        """预测速度场v_θ(x_t, t)，t为整数时间步t∈{0,...,T-1}"""
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
# 时间离散化粒度（用于将连续t映射到正弦嵌入的整数输入）
# ============================================================
T = 200


# ============================================================
# Flow ODE采样器
# ============================================================
@torch.no_grad()
def flow_ode_sample(model, shape, n_steps=50):
    """从Flow ODE采样: dx/dt = v_θ(x, t), t: 0→1"""
    model.eval()
    x = torch.randn(shape, device=device)
    dt = 1.0 / n_steps

    for step in range(n_steps):
        t_val = step / n_steps
        t_int = torch.full((shape[0],), int((1 - t_val) * (T - 1)), device=device, dtype=torch.long)
        v = model(x, t_int)
        x = x + v * dt

    return x.clamp(0, 1)


# ============================================================
# 数据加载
# ============================================================
print("\n加载MNIST数据集...")
data_dir = os.path.join(SAVE_DIR, 'data')
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
print(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")


# ============================================================
# 训练函数（带Resume能力）
# ============================================================
def train_model(model, optimizer, train_loader, time_sampling='uniform',
                num_epochs=50, checkpoint_path=None, final_checkpoint_path=None,
                progress_bar=True):
    """训练Rectified Flow模型

    time_sampling: 'uniform' 或 'logit_normal'
    """
    start_epoch = 0
    train_losses = []

    # Logit-Normal参数（14.5.1节推荐值）
    ln_m = 0.0   # 重心在t=0.5
    ln_s = 1.0   # 集中程度

    # 检查最终权重
    if final_checkpoint_path and os.path.exists(final_checkpoint_path):
        print(f"检测到最终权重: {final_checkpoint_path}")
        print("直接加载，跳过训练过程")
        checkpoint = torch.load(final_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint.get('train_losses', [])
        return train_losses

    # 检查中间权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"检测到中间权重: {checkpoint_path}")
        print("继续训练...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        train_losses = checkpoint.get('train_losses', [])

    sampling_label = '均匀采样' if time_sampling == 'uniform' else r'Logit-Normal采样'
    print(f"训练 RF ({sampling_label})，从 epoch {start_epoch} 开始...")
    t_start = time.time()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"[{sampling_label}] Epoch {epoch+1}/{num_epochs}", leave=False) \
            if progress_bar else train_loader
        for x, _ in pbar:
            x = x.to(device)
            batch = x.shape[0]

            # 采样噪声z ~ N(0,I)
            z = torch.randn_like(x)

            # 时间采样
            if time_sampling == 'uniform':
                t_continuous = torch.rand(batch, device=device)
            else:
                t_continuous = logit_normal_sample(batch, m=ln_m, s=ln_s, device=device)

            # 映射到整数时间步（正弦嵌入）
            t_int = ((1 - t_continuous) * (T - 1)).long()
            t_4d = t_continuous[:, None, None, None]

            # 线性插值: x_t = (1-t)z + t*x_0
            x_t = (1 - t_4d) * z + t_4d * x

            # 速度目标: v = x_0 - z
            v_target = x - z
            v_pred = model(x_t, t_int)

            # RF损失
            loss = F.mse_loss(v_pred, v_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch

            if progress_bar:
                pbar.set_postfix({'loss': f'{total_loss / len(train_loader.dataset):.6f}'})

        if progress_bar:
            pbar.close()

        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{sampling_label}] Epoch {epoch+1:3d}/{num_epochs}  Loss={avg_loss:.6f}")

        # 保存中间checkpoint
        if checkpoint_path and (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
            }, checkpoint_path)

    # 保存最终权重
    if final_checkpoint_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
        }, final_checkpoint_path)
        print(f"最终权重已保存: {final_checkpoint_path}")

    t_elapsed = time.time() - t_start
    print(f"训练完成，耗时: {t_elapsed:.1f}s")

    return train_losses


# ============================================================
# 步骤1：可视化时间采样分布
# ============================================================
print("\n" + "="*60)
print("步骤1: 可视化均匀采样 vs Logit-Normal采样")
print("="*60)
print("14.5.1节: Logit-Normal时间采样")
print("  均匀采样: t ~ U[0,1]，所有时间步等概率")
print("  Logit-Normal: 中间时间步(t≈0.5)采样概率更高")
print("  参数m=0, s=1: 重心在0.5，适度集中")

t_range = np.linspace(0.001, 0.999, 500)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) 均匀采样
ax = axes[0]
ax.plot(t_range, np.ones_like(t_range), 'b-', lw=2, label=r'均匀: $t \sim U[0,1]$')
ax.fill_between(t_range, 0, np.ones_like(t_range), alpha=0.2, color='blue')
ax.set_xlabel(r'时间步 $t$', fontsize=12)
ax.set_ylabel(r'概率密度', fontsize=12)
ax.set_title(r'(a) 均匀时间采样', fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)

# (b) Logit-Normal采样（不同s）
ax = axes[1]
for s_val in [0.5, 1.0, 2.0]:
    pdf = logit_normal_pdf(t_range, m=0.0, s=s_val)
    ax.plot(t_range, pdf, lw=2, label=rf'Logit-Normal $s$={s_val}')
ax.fill_between(t_range, 0, logit_normal_pdf(t_range, m=0.0, s=1.0), alpha=0.2, color='green')
ax.set_xlabel(r'时间步 $t$', fontsize=12)
ax.set_ylabel(r'概率密度', fontsize=12)
ax.set_title(r'(b) Logit-Normal采样（$m$=0, 不同$s$）', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)

# (c) Logit-Normal采样（不同m）
ax = axes[2]
for m_val in [-0.5, 0.0, 0.5]:
    pdf = logit_normal_pdf(t_range, m=m_val, s=1.0)
    ax.plot(t_range, pdf, lw=2, label=rf'Logit-Normal $m$={m_val}')
ax.fill_between(t_range, 0, logit_normal_pdf(t_range, m=0.0, s=1.0), alpha=0.2, color='green')
ax.set_xlabel(r'时间步 $t$', fontsize=12)
ax.set_ylabel(r'概率密度', fontsize=12)
ax.set_title(r'(c) Logit-Normal采样（$s$=1, 不同$m$）', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)

plt.suptitle(r'实验14.5-1：时间采样策略对比（14.5.1节）', fontsize=14, y=1.02)
plt.tight_layout()
fig_path1 = os.path.join(SAVE_DIR, '步骤1_时间采样分布.png')
plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"图1已保存: {fig_path1}")

# 生成采样直方图对比
n_hist = 10000
t_uniform = np.random.uniform(0, 1, n_hist)
t_logit = logit_normal_sample(n_hist, m=0.0, s=1.0, device='cpu').numpy()

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.hist(t_uniform, bins=50, density=True, alpha=0.5, label=r'均匀采样', color='blue')
ax.hist(t_logit, bins=50, density=True, alpha=0.5, label=r'Logit-Normal采样 ($m$=0, $s$=1)', color='green')
ax.plot(t_range, np.ones_like(t_range), 'b--', lw=1.5, alpha=0.7)
ax.plot(t_range, logit_normal_pdf(t_range, m=0.0, s=1.0), 'g--', lw=1.5, alpha=0.7)
ax.set_xlabel(r'时间步 $t$', fontsize=12)
ax.set_ylabel(r'概率密度', fontsize=12)
ax.set_title(r'实验14.5-1：采样分布直方图对比', fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
plt.tight_layout()
fig_path_hist = os.path.join(SAVE_DIR, '步骤1_采样直方图.png')
plt.savefig(fig_path_hist, dpi=150, bbox_inches='tight')
plt.close()
print(f"补充图已保存: {fig_path_hist}")


# ============================================================
# 步骤2：训练RF（均匀时间采样）
# ============================================================
print("\n" + "="*60)
print("步骤2: 训练 Rectified Flow（均匀时间采样）")
print("="*60)

num_epochs = 50

model_uniform = SmallUNet().to(device)
optimizer_uniform = torch.optim.Adam(model_uniform.parameters(), lr=2e-4)
losses_uniform = train_model(model_uniform, optimizer_uniform, train_loader,
                              time_sampling='uniform', num_epochs=num_epochs,
                              checkpoint_path=CHECKPOINT_PATH_UNIFORM,
                              final_checkpoint_path=FINAL_CHECKPOINT_PATH_UNIFORM,
                              progress_bar=True)


# ============================================================
# 步骤3：训练RF（Logit-Normal时间采样）
# ============================================================
print("\n" + "="*60)
print("步骤3: 训练 Rectified Flow（Logit-Normal时间采样）")
print("="*60)
print("14.5.1节: SD3推荐Logit-Normal采样，少步采样FID更优")
print("  参数: m=0.0 (重心在t=0.5), s=1.0 (适度集中)")

model_logit = SmallUNet().to(device)
optimizer_logit = torch.optim.Adam(model_logit.parameters(), lr=2e-4)
losses_logit = train_model(model_logit, optimizer_logit, train_loader,
                            time_sampling='logit_normal', num_epochs=num_epochs,
                            checkpoint_path=CHECKPOINT_PATH_LOGIT,
                            final_checkpoint_path=FINAL_CHECKPOINT_PATH_LOGIT,
                            progress_bar=True)


# ============================================================
# 步骤4：对比两种采样策略
# ============================================================
print("\n" + "="*60)
print("步骤4: 均匀采样 vs Logit-Normal采样 对比")
print("="*60)

# 4.1 收敛曲线对比
print("\n收敛曲线对比:")
if losses_uniform and losses_logit:
    print(f"  均匀采样 最终Loss: {losses_uniform[-1]:.6f}")
    print(f"  Logit-Normal 最终Loss: {losses_logit[-1]:.6f}")

# 4.2 采样质量对比
n_samples = 8
sample_shape = (n_samples, 1, 28, 28)
step_counts = [1, 5, 10, 50]

print("\n少步采样对比...")
samples_uniform = {}
samples_logit = {}

for n_steps in step_counts:
    samples_uniform[n_steps] = flow_ode_sample(model_uniform, sample_shape, n_steps=n_steps)
    samples_logit[n_steps] = flow_ode_sample(model_logit, sample_shape, n_steps=n_steps)

# 计算量化指标：生成样本与最近邻真实样本的平均像素距离
print("\n计算量化指标（生成样本与最近邻真实样本的平均像素距离）...")
test_samples = torch.stack([test_dataset[i][0] for i in range(min(100, len(test_dataset)))]).to(device)


def compute_nn_distance(generated, reference):
    """计算生成样本与参考集中最近邻的平均像素距离"""
    gen_flat = generated.view(generated.shape[0], -1)
    ref_flat = reference.view(reference.shape[0], -1)
    distances = torch.cdist(gen_flat, ref_flat)
    min_distances = distances.min(dim=1)[0]
    return min_distances.mean().item()


dist_uniform = {}
dist_logit = {}

for n_steps in step_counts:
    dist_uniform[n_steps] = compute_nn_distance(samples_uniform[n_steps], test_samples)
    dist_logit[n_steps] = compute_nn_distance(samples_logit[n_steps], test_samples)
    improvement = (dist_uniform[n_steps] - dist_logit[n_steps]) / dist_uniform[n_steps] * 100
    print(f"  {n_steps:3d}步: 均匀={dist_uniform[n_steps]:.4f}, "
          f"Logit-Normal={dist_logit[n_steps]:.4f}, "
          f"改善={improvement:+.1f}%")

# ============================================================
# 可视化
# ============================================================

# 图2: 收敛曲线对比
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
if losses_uniform:
    ax.plot(losses_uniform, label=r'均匀采样 $t \sim U[0,1]$', alpha=0.8, lw=2)
if losses_logit:
    ax.plot(losses_logit, label=r'Logit-Normal采样 ($m$=0, $s$=1)', alpha=0.8, lw=2)
ax.set_xlabel(r'Epoch', fontsize=12)
ax.set_ylabel(r'训练损失 $L_{\mathrm{CFM}}$', fontsize=12)
ax.set_title(r'实验14.5-1：训练收敛对比（14.5.1节）', fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
fig_path2 = os.path.join(SAVE_DIR, '步骤2_收敛曲线.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图2已保存: {fig_path2}")

# 图3: 少步采样质量对比
fig, axes = plt.subplots(len(step_counts), 2 * n_samples, figsize=(2 * n_samples * 1.2, len(step_counts) * 2.5))

for row, n_steps in enumerate(step_counts):
    # 均匀采样结果
    for col in range(n_samples):
        ax = axes[row, col]
        ax.imshow(samples_uniform[n_steps][col, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if row == 0 and col == 0:
            ax.set_title(r'均匀采样', fontsize=11, pad=10)

    # Logit-Normal采样结果
    for col in range(n_samples):
        ax = axes[row, n_samples + col]
        ax.imshow(samples_logit[n_steps][col, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if row == 0 and n_samples == 0:
            ax.set_title(r'Logit-Normal', fontsize=11, pad=10)

    # 行标签
    axes[row, 0].set_ylabel(f'{n_steps}步', fontsize=12, rotation=0, labelpad=40)

# 添加列标题
fig.text(0.25, 0.98, r'均匀采样', fontsize=13, ha='center', va='top')
fig.text(0.75, 0.98, r'Logit-Normal采样', fontsize=13, ha='center', va='top')

plt.suptitle(r'实验14.5-1：少步采样质量对比（14.5.1节）', fontsize=14, y=1.03)
plt.tight_layout()
fig_path3 = os.path.join(SAVE_DIR, '步骤3_少步采样对比.png')
plt.savefig(fig_path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"图3已保存: {fig_path3}")

# 图4: 量化指标对比
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

x_pos = np.arange(len(step_counts))
width = 0.35

bars1 = ax.bar(x_pos - width/2, [dist_uniform[s] for s in step_counts],
               width, label=r'均匀采样', color='steelblue', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, [dist_logit[s] for s in step_counts],
               width, label=r'Logit-Normal采样', color='forestgreen', alpha=0.8)

# 添加数值标注
for bar_group in [bars1, bars2]:
    for bar in bar_group:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

ax.set_xlabel(r'采样步数', fontsize=12)
ax.set_ylabel(r'平均最近邻距离', fontsize=12)
ax.set_title(r'实验14.5-1：采样质量量化对比（14.5.1节）', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{s}步' for s in step_counts])
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
fig_path4 = os.path.join(SAVE_DIR, '步骤4_量化指标对比.png')
plt.savefig(fig_path4, dpi=150, bbox_inches='tight')
plt.close()
print(f"图4已保存: {fig_path4}")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print("实验14.5-1 完成!")
print("=" * 60)
print(f"""
关键结论:
1. 时间采样策略对比（14.5.1节）
   - 均匀采样: t ~ U[0,1]，所有时间步等概率
   - Logit-Normal采样: 中间时间步(t≈0.5)概率更高
   - 参数m控制重心位置，s控制集中程度

2. 训练收敛对比
   - 均匀采样最终Loss: {losses_uniform[-1]:.6f}
   - Logit-Normal最终Loss: {losses_logit[-1]:.6f}
   - Logit-Normal可能收敛更快（中间时间步对训练更关键）

3. 少步采样质量对比
   采样步数 | 均匀采样距离 | Logit-Normal距离 | 改善
   {'-'*55}""")

for n_steps in step_counts:
    improvement = (dist_uniform[n_steps] - dist_logit[n_steps]) / dist_uniform[n_steps] * 100
    print(f"   {n_steps:5d}步  |  {dist_uniform[n_steps]:.4f}     |  {dist_logit[n_steps]:.4f}          | {improvement:+.1f}%")

print(f"""
4. SD3实践启示（14.5.1节）
   - SD3在60种轨迹对比中，Rectified Flow + Logit-Normal采样最优
   - 少步采样(4步/8步)时优势更明显
   - 中间时间步对向量场学习更关键，集中采样提升效率
   - 本实验在MNIST上的效果可能不如SD3在ImageNet上显著
     （MNIST数据简单，均匀采样已接近饱和）
""")
