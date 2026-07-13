# -*- coding: utf-8 -*-
"""
实验15.5-1 训练最佳实践系统对比
对应章节: 15.5.1 优化器与学习率调度

知识点:
  - Adam vs AdamW: 解耦权重衰减的一致正则化
  - Warmup + Cosine Decay vs Constant LR: 训练稳定性与收敛质量
  - 2x2因子设计: 优化器 x 调度策略, 系统分离各因素贡献

实验内容:
  步骤1: Adam vs AdamW对比（固定Constant LR, 15.5.1节）
  步骤2: 学习率调度对比（固定AdamW, Constant vs Warmup+Cosine, 15.5.1节）
  步骤3: 综合对比（4种配置训练曲线 + 最终Loss柱状图）
  步骤4: 采样质量对比（最佳 vs 最差配置的DDPM采样结果）

素材来源:
  - 15.5.1节 优化器与学习率调度
  - 15.4-1.py SmallUNet架构
  - ★ 原创设计: 2x2因子实验分离优化器和调度策略的独立贡献

运行前提: PyTorch, GPU加速推荐
        数据集: MNIST（torchvision自动下载）
"""

import sys
import os
import io
import time
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import logging
from tqdm import tqdm
from copy import deepcopy

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验15.5-1')
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

np.random.seed(42)

import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"实验15.5-1: 训练最佳实践系统对比")
print(f"{'='*60}")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  未检测到 GPU, 使用 CPU 训练")
    print("  提示: Colab 用户可在菜单 运行时 -> 更改运行时类型 中选择 GPU")


# ============================================================
# Checkpoint工具
# ============================================================
def load_train_state(checkpoint_path, model, optimizer, num_epochs_total, scheduler=None):
    """加载checkpoint, 返回 (start_epoch, is_final, loss_history)"""
    if not os.path.exists(checkpoint_path):
        return 0, False, []

    print(f"\n检测到已保存的checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'loss_history' not in checkpoint:
        raise RuntimeError(
            f"检测到旧版本 checkpoint (缺少 'loss_history' 字段):\n"
            f"  {checkpoint_path}\n请删除该文件后重新训练."
        )

    if checkpoint.get('is_final', False):
        print(f"✓ 这是最终训练完成的模型, 直接加载, 跳过训练过程")
        print(f"  训练轮数: {checkpoint['epoch']+1}")
        print(f"  最终损失: {checkpoint['loss']:.6f}")
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            raise RuntimeError(
                f"checkpoint 与模型架构不兼容:\n{e}\n"
                f"请删除后重新训练: {checkpoint_path}"
            )
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                pass
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception:
                pass
        return checkpoint['epoch'] + 1, True, checkpoint.get('loss_history', [])

    print(f"检测到未完成的训练, 从第 {checkpoint['epoch']+1} 轮继续")
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        raise RuntimeError(
            f"checkpoint 与模型架构不兼容:\n{e}\n"
            f"请删除后重新训练: {checkpoint_path}"
        )
    if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception:
            pass
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception:
            pass
    return checkpoint['epoch'] + 1, False, checkpoint.get('loss_history', [])


def save_train_state(checkpoint_path, model, optimizer, epoch, loss, loss_history, is_final, scheduler=None):
    """保存checkpoint, is_final时不存optimizer和scheduler"""
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'loss_history': loss_history,
        'is_final': is_final,
    }
    if not is_final:
        ckpt['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            ckpt['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(ckpt, checkpoint_path)


# ============================================================
# SmallUNet架构（与15.4-1一致）
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class UNetConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        gn_groups = min(4, out_ch)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(gn_groups, out_ch),
            nn.SiLU(),
        )
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
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
# DDPM噪声调度
# ============================================================
T = 200
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)


# ============================================================
# 学习率调度: Warmup + Cosine Decay (15.5.1节)
# ============================================================
def compute_warmup_cosine_factor(step, warmup_steps, total_steps):
    """计算 Warmup + Cosine Decay 在第 step 步的 lr 缩放因子 (15.5.1节)

    Warmup阶段: 线性从 0 增至 1
    Cosine阶段: 按余弦曲线从 1 衰减至 0

    返回值 ∈ [0, 1], 实际 lr = base_lr * factor
    """
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    """Warmup + Cosine Decay 调度器 (15.5.1节)"""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: compute_warmup_cosine_factor(step, warmup_steps, total_steps)
    )


# ============================================================
# 数据加载
# ============================================================
print("\n加载MNIST数据集...")
data_dir = os.path.join(SAVE_DIR, 'data')
os.makedirs(data_dir, exist_ok=True)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
print(f"  训练集大小: {len(train_dataset)}")
print(f"  每轮批次数: {len(train_loader)}")


# ============================================================
# 训练配置 (2x2因子设计)
# ============================================================
num_epochs = 20
batches_per_epoch = len(train_loader)
total_steps = num_epochs * batches_per_epoch
warmup_steps = min(500, total_steps // 10)

configs = {
    'adam_const': {
        'optimizer_cls': torch.optim.Adam,
        'optimizer_kwargs': {'lr': 2e-4},
        'use_scheduler': False,
        'label': 'Adam + Constant',
        'color': '#e74c3c',
    },
    'adamw_const': {
        'optimizer_cls': torch.optim.AdamW,
        'optimizer_kwargs': {'lr': 2e-4, 'weight_decay': 0.01},
        'use_scheduler': False,
        'label': 'AdamW + Constant',
        'color': '#3498db',
    },
    'adam_cosine': {
        'optimizer_cls': torch.optim.Adam,
        'optimizer_kwargs': {'lr': 2e-4},
        'use_scheduler': True,
        'label': 'Adam + Warmup+Cosine',
        'color': '#e67e22',
    },
    'adamw_cosine': {
        'optimizer_cls': torch.optim.AdamW,
        'optimizer_kwargs': {'lr': 2e-4, 'weight_decay': 0.01},
        'use_scheduler': True,
        'label': 'AdamW + Warmup+Cosine',
        'color': '#2ecc71',
    },
}

checkpoint_paths = {
    name: os.path.join(SAVE_DIR, f'{name}_checkpoint.pth')
    for name in configs
}

# 所有模型从相同初始权重出发, 确保公平对比
torch.manual_seed(42)
base_model = SmallUNet().to(device)
base_state_dict = deepcopy(base_model.state_dict())
base_params = sum(p.numel() for p in base_model.parameters())
print(f"  SmallUNet 参数量: {base_params:,} ({base_params/1e6:.1f}M)")
del base_model


# ============================================================
# 通用训练函数
# ============================================================
def train_model(config_name, config, checkpoint_path):
    """训练单个配置, 返回 (model, loss_history)"""
    # 修复: 在每次配置训练前重置全局随机状态, 确保4个配置在相同的数据顺序
    # 和DDPM噪声(t, eps)序列下训练, 严格保证2x2因子实验的公平对比前提
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = SmallUNet().to(device)
    model.load_state_dict(deepcopy(base_state_dict))
    optimizer = config['optimizer_cls'](model.parameters(), **config['optimizer_kwargs'])

    scheduler = None
    if config['use_scheduler']:
        scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    start_epoch, is_final, loss_history = load_train_state(
        checkpoint_path, model, optimizer, num_epochs, scheduler)

    if is_final:
        return model, loss_history

    if start_epoch >= num_epochs:
        print(f"  start_epoch({start_epoch}) >= num_epochs({num_epochs}), 跳过训练")
        return model, loss_history

    print(f"\n  训练 {config['label']} (共 {num_epochs} 轮, 从第 {start_epoch+1} 轮开始)...")
    t_start = time.time()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader,
                     desc=f"{config['label']} Epoch {epoch+1}/{num_epochs}",
                     leave=False, unit='batch')
        for x, _ in pbar:
            x = x.to(device)
            b = x.size(0)
            t = torch.randint(0, T, (b,), device=device)
            eps = torch.randn_like(x)
            x_t = (sqrt_alpha_bars[t].view(b, 1, 1, 1) * x +
                   sqrt_one_minus_alpha_bars[t].view(b, 1, 1, 1) * eps)
            pred = model(x_t, t)
            loss = F.mse_loss(pred, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss.item() * b
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_dataset)
        loss_history.append(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  {config['label']} Epoch {epoch+1}/{num_epochs} 完成, "
              f"Avg Loss={avg_loss:.6f}, LR={current_lr:.2e}")
        save_train_state(checkpoint_path, model, optimizer, epoch,
                         avg_loss, loss_history, is_final=(epoch == num_epochs - 1),
                         scheduler=scheduler)

    t_elapsed = time.time() - t_start
    print(f"  {config['label']} 训练完成, 最终Loss: {loss_history[-1]:.6f}, 耗时: {t_elapsed:.1f}s")
    return model, loss_history


# ============================================================
# 训练所有配置
# ============================================================
print(f"\n{'='*60}")
print("训练所有配置 (2x2因子设计)")
print(f"{'='*60}")
print(f"  优化器: Adam vs AdamW (weight_decay=0.01)")
print(f"  调度器: Constant LR vs Warmup({warmup_steps}步)+Cosine")
print(f"  训练轮数: {num_epochs}, 总步数: {total_steps}")

trained_models = {}
trained_histories = {}

for name, config in configs.items():
    print(f"\n{'─'*40}")
    model, history = train_model(name, config, checkpoint_paths[name])
    trained_models[name] = model
    trained_histories[name] = history


# ============================================================
# 步骤1: Adam vs AdamW对比 (固定Constant LR)
# ============================================================
print(f"\n{'='*60}")
print("步骤1: Adam vs AdamW对比 (固定Constant LR, 15.5.1节)")
print(f"{'='*60}")
print("""
15.5.1节 核心区别:
  Adam:  权重衰减与自适应学习率耦合, 正则化效果不一致
  AdamW: 解耦权重衰减, 正则化强度一致
  实践建议: 扩散模型训练首选 AdamW (weight_decay=0.01)
""")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for name in ['adam_const', 'adamw_const']:
    h = trained_histories[name]
    ax.plot(range(1, len(h)+1), h, '-o', markersize=3,
            color=configs[name]['color'], label=configs[name]['label'])
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel(r'训练 Loss', fontsize=12)
ax.set_title('(a) 训练收敛对比', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

ax = axes[1]
names_step1 = ['adam_const', 'adamw_const']
losses_step1 = [trained_histories[n][-1] for n in names_step1]
bars = ax.bar([configs[n]['label'] for n in names_step1], losses_step1,
              color=[configs[n]['color'] for n in names_step1], alpha=0.7, width=0.4)
# 用相对偏移避免文字贴柱顶/被柱体覆盖
text_offset = max(losses_step1) * 0.02
for bar, val in zip(bars, losses_step1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + text_offset,
            f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel(r'最终训练 Loss', fontsize=12)
ax.set_title('(b) 最终Loss对比', fontsize=12)
ax.grid(alpha=0.3, axis='y')

plt.suptitle(r'步骤1: Adam vs AdamW (Constant LR, 15.5.1节)', fontsize=14, y=1.01)
plt.tight_layout()
fig_path1 = os.path.join(SAVE_DIR, '步骤1_Adam_vs_AdamW.png')
plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"图1已保存: {fig_path1}")

loss_adam = trained_histories['adam_const'][-1]
loss_adamw = trained_histories['adamw_const'][-1]
improve = (loss_adam - loss_adamw) / loss_adam * 100
print(f"  Adam  最终Loss: {loss_adam:.6f}")
print(f"  AdamW 最终Loss: {loss_adamw:.6f}")
if improve > 0:
    print(f"  AdamW 相对改善: {improve:.2f}% (解耦权重衰减的正则化优势)")
else:
    print(f"  注意: 在当前设置下AdamW未显示优势, 可能因模型较小或训练轮数不足")


# ============================================================
# 步骤2: 学习率调度对比 (固定AdamW)
# ============================================================
print(f"\n{'='*60}")
print("步骤2: 学习率调度对比 (固定AdamW, 15.5.1节)")
print(f"{'='*60}")
print(f"""
15.5.1节 学习率调度:
  Constant LR: 固定学习率, 简单但后期可能震荡
  Warmup+Cosine: 前期稳定 + 后期精细调整
  Warmup步数: {warmup_steps} (总步数{total_steps}的{warmup_steps/total_steps*100:.1f}%)
""")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
steps_arr = np.arange(total_steps)
lr_const = np.full(total_steps, 2e-4)

# 复用 compute_warmup_cosine_factor 避免逻辑重复
lr_cosine = np.array([2e-4 * compute_warmup_cosine_factor(s, warmup_steps, total_steps)
                      for s in steps_arr])
ax.plot(steps_arr, lr_const, label='Constant LR', color='#3498db', linewidth=1.5)
ax.plot(steps_arr, lr_cosine, label='Warmup + Cosine', color='#2ecc71', linewidth=1.5)
ax.axvline(x=warmup_steps, color='gray', linestyle='--', alpha=0.5,
           label=f'Warmup结束 (step={warmup_steps})')
ax.set_xlabel('训练步数', fontsize=12)
ax.set_ylabel(r'学习率 $\eta$', fontsize=12)
ax.set_title('(a) 学习率调度曲线', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
for name in ['adamw_const', 'adamw_cosine']:
    h = trained_histories[name]
    ax.plot(range(1, len(h)+1), h, '-o', markersize=3,
            color=configs[name]['color'], label=configs[name]['label'])
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel(r'训练 Loss', fontsize=12)
ax.set_title('(b) 训练收敛对比 (AdamW)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

ax = axes[2]
names_step2 = ['adamw_const', 'adamw_cosine']
losses_step2 = [trained_histories[n][-1] for n in names_step2]
bars = ax.bar([configs[n]['label'].replace('AdamW + ', '') for n in names_step2],
              losses_step2,
              color=[configs[n]['color'] for n in names_step2], alpha=0.7, width=0.4)
# 用相对偏移避免文字贴柱顶
text_offset = max(losses_step2) * 0.02
for bar, val in zip(bars, losses_step2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + text_offset,
            f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel(r'最终训练 Loss', fontsize=12)
ax.set_title('(c) 最终Loss对比 (AdamW)', fontsize=12)
ax.grid(alpha=0.3, axis='y')

plt.suptitle(r'步骤2: 学习率调度对比 (AdamW, 15.5.1节)', fontsize=14, y=1.01)
plt.tight_layout()
fig_path2 = os.path.join(SAVE_DIR, '步骤2_学习率调度对比.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig_path2}")

loss_const = trained_histories['adamw_const'][-1]
loss_cosine = trained_histories['adamw_cosine'][-1]
improve_lr = (loss_const - loss_cosine) / loss_const * 100
print(f"  Constant 最终Loss: {loss_const:.6f}")
print(f"  Cosine   最终Loss: {loss_cosine:.6f}")
if improve_lr > 0:
    print(f"  Cosine相对改善: {improve_lr:.2f}% (余弦衰减的精细调整优势)")
else:
    print(f"  注意: 在当前设置下Cosine调度未显示明显优势, 可能因训练轮数较少")


# ============================================================
# 步骤3: 综合对比 (4种配置)
# ============================================================
print(f"\n{'='*60}")
print("步骤3: 综合对比 (4种配置, 15.5.1节)")
print(f"{'='*60}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for name, config in configs.items():
    h = trained_histories[name]
    ax.plot(range(1, len(h)+1), h, '-o', markersize=2,
            color=config['color'], label=config['label'], alpha=0.85)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel(r'训练 Loss', fontsize=12)
ax.set_title('(a) 4种配置训练曲线', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
names_all = list(configs.keys())
labels_all = [configs[n]['label'] for n in names_all]
losses_all = [trained_histories[n][-1] for n in names_all]
colors_all = [configs[n]['color'] for n in names_all]
bars = ax.barh(labels_all, losses_all, color=colors_all, alpha=0.7, height=0.5)
# 用相对偏移避免文字贴柱顶
text_offset = max(losses_all) * 0.02
for bar, val in zip(bars, losses_all):
    ax.text(bar.get_width() + text_offset, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
ax.set_xlabel(r'最终训练 Loss', fontsize=12)
ax.set_title('(b) 最终Loss排名', fontsize=12)
ax.invert_yaxis()
ax.grid(alpha=0.3, axis='x')

plt.suptitle(r'步骤3: 综合对比 (4种配置, 15.5.1节)', fontsize=14, y=1.01)
plt.tight_layout()
fig_path3 = os.path.join(SAVE_DIR, '步骤3_综合对比.png')
plt.savefig(fig_path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"图3已保存: {fig_path3}")

ranking = sorted(names_all, key=lambda n: trained_histories[n][-1])
print("\n  最终Loss排名 (低→好):")
for i, name in enumerate(ranking):
    print(f"    {i+1}. {configs[name]['label']}: {trained_histories[name][-1]:.6f}")


# ============================================================
# 步骤4: 采样质量对比 (最佳 vs 最差配置)
# ============================================================
print(f"\n{'='*60}")
print("步骤4: 采样质量对比 (最佳 vs 最差配置)")
print(f"{'='*60}")

def to_vis_range(x):
    return (x + 1) / 2

best_name = ranking[0]
worst_name = ranking[-1]
print(f"  最佳配置: {configs[best_name]['label']} (Loss={trained_histories[best_name][-1]:.6f})")
print(f"  最差配置: {configs[worst_name]['label']} (Loss={trained_histories[worst_name][-1]:.6f})")

n_samples = 8

@torch.no_grad()
def sample_ddpm(model, n_samples):
    model.eval()
    x = torch.randn(n_samples, 1, 28, 28, device=device)
    for t_val in reversed(range(T)):
        t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
        pred = model(x, t)
        alpha_bar_t = alpha_bars[t_val]
        alpha_bar_prev = alpha_bars[t_val - 1] if t_val > 0 else torch.tensor(1.0, device=device)
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * pred) / torch.sqrt(alpha_bar_t)
        x0_pred = x0_pred.clamp(-1, 1)
        if t_val > 0:
            sigma_t = torch.sqrt(betas[t_val])
            dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma_t**2, min=0)) * pred
            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma_t * torch.randn_like(x)
        else:
            x = x0_pred
    return x

print(f"\n  DDPM采样 ({T}步, 各{n_samples}张)...")
samples_best = sample_ddpm(trained_models[best_name], n_samples)
samples_worst = sample_ddpm(trained_models[worst_name], n_samples)

fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4.5))
for col in range(n_samples):
    axes[0, col].imshow(to_vis_range(samples_best[col, 0].cpu()).clamp(0, 1),
                         cmap='gray', vmin=0, vmax=1)
    axes[0, col].axis('off')
    axes[1, col].imshow(to_vis_range(samples_worst[col, 0].cpu()).clamp(0, 1),
                         cmap='gray', vmin=0, vmax=1)
    axes[1, col].axis('off')
axes[0, 0].set_ylabel(f'最佳: {configs[best_name]["label"]}',
                       fontsize=9, rotation=0, labelpad=120)
axes[1, 0].set_ylabel(f'最差: {configs[worst_name]["label"]}',
                       fontsize=9, rotation=0, labelpad=120)

plt.suptitle(r'步骤4: 采样质量对比 (最佳 vs 最差配置)', fontsize=14, y=1.01)
plt.tight_layout()
fig_path4 = os.path.join(SAVE_DIR, '步骤4_采样质量对比.png')
plt.savefig(fig_path4, dpi=150, bbox_inches='tight')
plt.close()
print(f"图4已保存: {fig_path4}")


# ============================================================
# 实验总结
# ============================================================
print(f"\n{'='*60}")
print("实验15.5-1 总结")
print(f"{'='*60}")
print(f"""
关键结论:
1. 优化器对比 (步骤1, 15.5.1节)
   - Adam  最终Loss: {trained_histories['adam_const'][-1]:.6f}
   - AdamW 最终Loss: {trained_histories['adamw_const'][-1]:.6f}
   - AdamW解耦权重衰减, 正则化更一致

2. 学习率调度对比 (步骤2, 15.5.1节)
   - Constant 最终Loss: {trained_histories['adamw_const'][-1]:.6f}
   - Cosine   最终Loss: {trained_histories['adamw_cosine'][-1]:.6f}
   - Warmup+Cosine提供更平滑的训练过程

3. 综合对比 (步骤3)
   - 最佳配置: {configs[ranking[0]]['label']} (Loss={trained_histories[ranking[0]][-1]:.6f})
   - 最差配置: {configs[ranking[-1]]['label']} (Loss={trained_histories[ranking[-1]][-1]:.6f})
   - 优化器和调度器的贡献可叠加

4. 实践建议 (15.5.1节)
   - 首选: AdamW (weight_decay=0.01) + Warmup + Cosine Decay
   - AdamW的解耦权重衰减在大模型上优势更明显
   - Warmup防止训练初期梯度不稳定, Cosine加速后期收敛
""")

print(f"\n{'='*60}")
print("实验15.5-1 完成!")
print(f"{'='*60}")