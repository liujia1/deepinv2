# -*- coding: utf-8 -*-
"""
实验15.5-1 延长验证版: 100轮训练 + 验证损失监控
目的: 裁决 15.5-1 原实验中"实测与预期相反"的矛盾——
      Cosine 调度在短程 train loss 下落后于 Constant, 究竟是
      (a) 评估指标/训练时长不匹配的固有偏差, 还是
      (b) "Cosine 更优"这一工业经验本身不成立?

设计 (与 15.5-1 同一 2x2 因子框架, 仅改两处):
  1. num_epochs: 20 -> 100 (总步数 9380 -> 46900)
  2. 每轮结束在 MNIST 测试集(10000张)上计算固定协议的验证 L_simple:
     每张图分配固定时间步 t = index % T, 固定噪声 eps (独立种子生成),
     因此验证 loss 无采样噪声, 4 个配置之间、100 轮之间完全可比。

预期判读:
  - 若随训练延长, cosine 的 val loss 追平/反超 constant (差值曲线穿零),
    则原实验的反差源于"短程 + train loss 评估"的系统性偏差, 工业经验成立;
  - 若到 100 轮 cosine 仍稳定落后, 则说明该结论在 MNIST/小模型上确实不成立.

运行: Colab GPU (T4即可, 预计30~60分钟) 或本地 (CPU很慢, 不推荐)
      支持断点续训: checkpoint 以 extend_ 前缀保存, 不影响原 20 轮实验产物.
"""

import sys
import os
import io
import time
import math
import json
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
print(f"实验15.5-1 延长验证: 100轮 + 验证损失监控")
print(f"{'='*60}")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  未检测到 GPU: 100轮x4配置在CPU上需数小时, 强烈建议在 Colab 选择 GPU 运行时")


# ============================================================
# Checkpoint工具 (extend版: 额外携带 val_history)
# ============================================================
def load_train_state(checkpoint_path, model, optimizer, num_epochs_total, scheduler=None):
    """加载checkpoint, 返回 (start_epoch, is_final, loss_history, val_history)"""
    if not os.path.exists(checkpoint_path):
        return 0, False, [], []

    print(f"\n检测到已保存的checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'loss_history' not in checkpoint or 'val_history' not in checkpoint:
        raise RuntimeError(
            f"检测到不兼容的 checkpoint (缺少 'loss_history'/'val_history' 字段):\n"
            f"  {checkpoint_path}\n请删除该文件后重新训练."
        )

    def _restore_state():
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            raise RuntimeError(f"checkpoint 与模型架构不兼容:\n{e}\n请删除后重新训练: {checkpoint_path}")
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
        # 恢复 RNG 状态, 使续训的噪声流与一次性训练严格一致
        if 'rng_state' in checkpoint:
            try:
                torch.set_rng_state(checkpoint['rng_state'])
                if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint:
                    torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
            except Exception:
                pass

    if checkpoint.get('is_final', False):
        print(f"✓ 这是最终训练完成的模型, 直接加载, 跳过训练过程")
        print(f"  训练轮数: {checkpoint['epoch']+1}")
        print(f"  最终验证Loss: {checkpoint['val_history'][-1]:.6f}")
        _restore_state()
        return checkpoint['epoch'] + 1, True, checkpoint['loss_history'], checkpoint['val_history']

    print(f"检测到未完成的训练, 从第 {checkpoint['epoch']+1} 轮继续")
    _restore_state()
    return checkpoint['epoch'] + 1, False, checkpoint['loss_history'], checkpoint['val_history']


def save_train_state(checkpoint_path, model, optimizer, epoch, loss,
                     loss_history, val_history, is_final, scheduler=None):
    """保存checkpoint, is_final时不存optimizer和scheduler (成品最小化)"""
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'loss_history': loss_history,
        'val_history': val_history,
        'is_final': is_final,
    }
    if not is_final:
        ckpt['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            ckpt['scheduler_state_dict'] = scheduler.state_dict()
        # 保存 RNG 状态: 续训时从中恢复, 保证噪声流与一次性训练一致(不重放)
        ckpt['rng_state'] = torch.get_rng_state()
        if torch.cuda.is_available():
            ckpt['cuda_rng_state'] = torch.cuda.get_rng_state_all()
    torch.save(ckpt, checkpoint_path)


# ============================================================
# SmallUNet架构 (与15.5-1完全一致)
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
# DDPM噪声调度 (与15.5-1一致)
# ============================================================
T = 200
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)


# ============================================================
# 学习率调度: Warmup + Cosine Decay
# ============================================================
def compute_warmup_cosine_factor(step, warmup_steps, total_steps):
    """Warmup线性0->1, 之后cosine 1->0 (15.5.1节)"""
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: compute_warmup_cosine_factor(step, warmup_steps, total_steps)
    )


# ============================================================
# 数据加载: 训练集 + 固定协议验证集
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
test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
print(f"  训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
print(f"  每轮批次数: {len(train_loader)}")

# 验证协议: 全部10000张测试图, 固定时间步 t = index % T (每个t覆盖50张),
# 固定噪声 eps 由独立种子(1234)生成 -> 验证loss完全确定, 跨配置/跨轮严格可比。
# x_t 只依赖固定数据, 一次性预计算, 每轮验证仅需40次前向.
print("构建固定协议验证集...")
_val_gen = torch.Generator(device='cpu').manual_seed(1234)
_x_all = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(device)
_t_all = (torch.arange(len(test_dataset), device=device) % T)
_eps_all = torch.randn(len(test_dataset), 1, 28, 28, generator=_val_gen).to(device)
_xv_all = (sqrt_alpha_bars[_t_all].view(-1, 1, 1, 1) * _x_all +
           sqrt_one_minus_alpha_bars[_t_all].view(-1, 1, 1, 1) * _eps_all)
VAL_CHUNK = 250

@torch.no_grad()
def evaluate_val_loss(model):
    """固定协议验证 L_simple (eps-MSE), 无采样噪声"""
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(_xv_all), VAL_CHUNK):
        xb = _xv_all[i:i+VAL_CHUNK]
        tb = _t_all[i:i+VAL_CHUNK]
        eb = _eps_all[i:i+VAL_CHUNK]
        mse = F.mse_loss(model(xb, tb), eb, reduction='mean')
        total += mse.item() * xb.size(0)
        n += xb.size(0)
    model.train()
    return total / n


# ============================================================
# 训练配置 (2x2因子设计, 与15.5-1一致, 仅轮数延长)
# ============================================================
num_epochs = 100
batches_per_epoch = len(train_loader)
total_steps = num_epochs * batches_per_epoch
# warmup绝对步数保持500: 占比从原实验的5.3%降到1.1%, 更接近工业比例(DDPM约0.5%)
warmup_steps = 500

configs = {
    # 注: Adam 与 AdamW 均设置 weight_decay=0.01 (与15.5-1.py口径一致):
    #     Adam 以 L2 形式把 wd 耦合进梯度, AdamW 解耦 —— 这样两侧同 λ,
    #     才是 15.5.1 节"耦合 vs 解耦"的干净对照(而非"有正则 vs 无正则")
    'adam_const': {
        'optimizer_cls': torch.optim.Adam,
        'optimizer_kwargs': {'lr': 2e-4, 'weight_decay': 0.01},
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
        'optimizer_kwargs': {'lr': 2e-4, 'weight_decay': 0.01},
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
    name: os.path.join(SAVE_DIR, f'extend_{name}_checkpoint.pth')
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
# 通用训练函数 (每轮末计算验证loss)
# ============================================================
def train_model(config_name, config, checkpoint_path):
    """训练单个配置, 返回 (model, loss_history, val_history)"""
    model = SmallUNet().to(device)
    model.load_state_dict(deepcopy(base_state_dict))
    optimizer = config['optimizer_cls'](model.parameters(), **config['optimizer_kwargs'])

    scheduler = None
    if config['use_scheduler']:
        scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    start_epoch, is_final, loss_history, val_history = load_train_state(
        checkpoint_path, model, optimizer, num_epochs, scheduler)

    if is_final:
        return model, loss_history, val_history

    # 公平性协议: 仅从零开始训练(start_epoch==0)时重置全局随机状态,
    # 保证4个配置的数据顺序与(t, eps)噪声序列完全一致;
    # 续训(start_epoch>0)时沿用 checkpoint 中保存的 RNG 状态,
    # 避免无条件重新播种导致后续 epoch 重放 epoch0 的噪声流
    if start_epoch == 0:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

    if start_epoch >= num_epochs:
        print(f"  start_epoch({start_epoch}) >= num_epochs({num_epochs}), 跳过训练")
        return model, loss_history, val_history

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
        val_loss = evaluate_val_loss(model)
        loss_history.append(avg_loss)
        val_history.append(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  {config['label']} Epoch {epoch+1}/{num_epochs} 完成, "
              f"Train={avg_loss:.6f}, Val={val_loss:.6f}, LR={current_lr:.2e}")
        save_train_state(checkpoint_path, model, optimizer, epoch,
                         avg_loss, loss_history, val_history,
                         is_final=(epoch == num_epochs - 1), scheduler=scheduler)

    t_elapsed = time.time() - t_start
    print(f"  {config['label']} 训练完成, 最终Val Loss: {val_history[-1]:.6f}, 耗时: {t_elapsed:.1f}s")
    return model, loss_history, val_history


# ============================================================
# 训练所有配置
# ============================================================
print(f"\n{'='*60}")
print("训练所有配置 (2x2因子设计, 100轮)")
print(f"{'='*60}")
print(f"  优化器: Adam(耦合wd=0.01) vs AdamW(解耦wd=0.01)")
print(f"  调度器: Constant LR vs Warmup({warmup_steps}步, 占{warmup_steps/total_steps*100:.1f}%)+Cosine")
print(f"  训练轮数: {num_epochs}, 总步数: {total_steps}")

trained_models = {}
trained_histories = {}
val_histories = {}

for name, config in configs.items():
    print(f"\n{'─'*40}")
    model, history, vhistory = train_model(name, config, checkpoint_paths[name])
    trained_models[name] = model
    trained_histories[name] = history
    val_histories[name] = vhistory


# ============================================================
# 图E1: train/val 双口径曲线 (4配置, 100轮)
# ============================================================
print(f"\n{'='*60}")
print("作图: train/val 双口径曲线")
print(f"{'='*60}")

epochs_arr = np.arange(1, num_epochs + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for name, config in configs.items():
    h = trained_histories[name]
    ax.plot(epochs_arr[:len(h)], h, color=config['color'], label=config['label'], linewidth=1.2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel(r'训练 Loss', fontsize=12)
ax.set_title(f'(a) 训练 Loss ({num_epochs}轮)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
for name, config in configs.items():
    h = val_histories[name]
    ax.plot(epochs_arr[:len(h)], h, color=config['color'], label=config['label'], linewidth=1.2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel(r'验证 Loss ($L_{\mathrm{simple}}$, 固定协议)', fontsize=12)
ax.set_title(f'(b) 验证 Loss ({num_epochs}轮) \u2014 裁决口径', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle(f'图E1: {num_epochs}轮 train/val 双口径收敛曲线', fontsize=14, y=1.01)
plt.tight_layout()
fig_pathE1 = os.path.join(SAVE_DIR, '延长验证_E1_双口径曲线.png')
plt.savefig(fig_pathE1, dpi=150, bbox_inches='tight')
plt.close()
print(f"图E1已保存: {fig_pathE1}")


# ============================================================
# 图E2: cosine−constant 差值曲线 (找交叉点) + 最终val排名
# ============================================================
print(f"\n{'='*60}")
print("作图: cosine−constant 验证差值与最终排名")
print(f"{'='*60}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
crossover_epochs = {}
for opt_name, opt_label in [('adam', 'Adam'), ('adamw', 'AdamW')]:
    c = val_histories[f'{opt_name}_const']
    o = val_histories[f'{opt_name}_cosine']
    diff = np.array(o) - np.array(c)   # >0: cosine更差; <0: cosine更好
    n = len(diff)
    ax.plot(epochs_arr[:n], diff, color=configs[f'{opt_name}_cosine']['color'],
            label=f'{opt_label}: Cosine - Constant', linewidth=1.4)
    neg = np.where(diff < 0)[0]
    if len(neg) > 0:
        ce = int(neg[0]) + 1
        crossover_epochs[opt_label] = ce
        ax.axvline(x=ce, color='gray', linestyle=':', alpha=0.7)
        ax.annotate(f'{opt_label} 交叉@Epoch{ce}', xy=(ce, diff[ce-1]),
                    xytext=(ce + 3, diff.max() * 0.3), fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='gray'))
    else:
        crossover_epochs[opt_label] = None
ax.axhline(y=0, color='black', linewidth=1)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel(r'验证 Loss 差值', fontsize=12)
ax.set_title('(a) 差值穿零 = cosine 开始反超', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
names_all = list(configs.keys())
val_final = [val_histories[n][-1] for n in names_all]
order = np.argsort(val_final)
labels_s = [configs[names_all[i]]['label'] for i in order]
vals_s = [val_final[i] for i in order]
colors_s = [configs[names_all[i]]['color'] for i in order]
bars = ax.barh(labels_s, vals_s, color=colors_s, alpha=0.75, height=0.5)
offset = max(vals_s) * 0.01
for bar, val in zip(bars, vals_s):
    ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
ax.set_xlabel(r'最终验证 Loss', fontsize=12)
ax.set_title(f'(b) {num_epochs}轮最终验证Loss排名 (低\u2192上为优)', fontsize=12)
ax.invert_yaxis()
ax.grid(alpha=0.3, axis='x')

plt.suptitle(r'图E2: cosine-constant 交叉分析与最终验证排名', fontsize=14, y=1.01)
plt.tight_layout()
fig_pathE2 = os.path.join(SAVE_DIR, '延长验证_E2_交叉与排名.png')
plt.savefig(fig_pathE2, dpi=150, bbox_inches='tight')
plt.close()
print(f"图E2已保存: {fig_pathE2}")


# ============================================================
# 图E3: 4配置采样对比 (同一固定初始噪声, 受控对比)
# ============================================================
print(f"\n{'='*60}")
print("作图: 4配置采样对比 (相同初始噪声)")
print(f"{'='*60}")

def to_vis_range(x):
    return (x + 1) / 2

n_samples = 8
# 先固定一次初始x_T与逐步噪声, 4配置共用 -> 采样差异只来自模型本身
_sample_gen = torch.Generator(device=device).manual_seed(999)
_x_shared = torch.randn(n_samples, 1, 28, 28, device=device, generator=_sample_gen)
_noise_shared = [torch.randn(n_samples, 1, 28, 28, device=device, generator=_sample_gen)
                 for _ in range(T)]

@torch.no_grad()
def sample_shared_noise(model):
    """与sample_ddpm同公式, 但逐步噪声用预生成序列, 保证4配置噪声轨迹相同"""
    model.eval()
    x = _x_shared.clone()
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
            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma_t * _noise_shared[t_val]
        else:
            x = x0_pred
    model.train()
    return x

order_val = sorted(names_all, key=lambda n: val_histories[n][-1])
fig, axes = plt.subplots(4, n_samples, figsize=(n_samples * 2, 8.5))
for row, name in enumerate(order_val):
    s = sample_shared_noise(trained_models[name])
    for col in range(n_samples):
        axes[row, col].imshow(to_vis_range(s[col, 0].cpu()).clamp(0, 1),
                              cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
    axes[row, 0].text(-0.12, 0.5,
                      f"{configs[name]['label']}\nVal={val_histories[name][-1]:.4f}",
                      fontsize=9, ha='right', va='center',
                      transform=axes[row, 0].transAxes)

plt.suptitle(f'图E3: {num_epochs}轮后4配置采样 (相同初始噪声, 按验证Loss排名)', fontsize=14, y=1.0)
plt.tight_layout()
fig_pathE3 = os.path.join(SAVE_DIR, '延长验证_E3_采样对比.png')
plt.savefig(fig_pathE3, dpi=150, bbox_inches='tight')
plt.close()
print(f"图E3已保存: {fig_pathE3}")


# ============================================================
# 裁决总结
# ============================================================
print(f"\n{'='*60}")
print("延长验证裁决")
print(f"{'='*60}")

val_ranking = sorted(names_all, key=lambda n: val_histories[n][-1])
print("\n  最终验证Loss排名 (低→好):")
for i, name in enumerate(val_ranking):
    print(f"    {i+1}. {configs[name]['label']}: Val={val_histories[name][-1]:.6f}, "
          f"Train={trained_histories[name][-1]:.6f}")

print(f"\n  cosine−constant 验证差值交叉点: {crossover_epochs}")

cos_better = all(
    val_histories[f'{o}_cosine'][-1] < val_histories[f'{o}_const'][-1]
    for o in ['adam', 'adamw'])
if cos_better:
    print(f"""
  裁决: {num_epochs}轮 + 验证口径下, Cosine 已反超 Constant ——
  原实验(20轮, train loss)的反差确为评估口径与训练时长的系统性偏差,
  15.5.1 节工业经验在其适用条件下成立。""")
else:
    print(f"""
  裁决: {num_epochs}轮 + 验证口径下, Cosine 仍未全面反超 Constant ——
  在 MNIST/小模型规模上, 15.5.1 节"Cosine更优"的经验不能无条件照搬,
  应以实测排名为准。""")


# ===== 保存数值结果 =====
def _to_native(obj):
    """递归转换numpy/torch类型为Python原生类型"""
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return _to_native(obj.tolist())
    try:
        if isinstance(obj, torch.Tensor): return _to_native(obj.detach().cpu().tolist())
    except: pass
    return obj

results_summary = {
    'num_epochs': num_epochs,
    'total_steps': total_steps,
    'warmup_steps': warmup_steps,
    'final_train_losses': {n: round(float(trained_histories[n][-1]), 6) for n in configs},
    'final_val_losses': {n: round(float(val_histories[n][-1]), 6) for n in configs},
    'val_ranking_best_to_worst': [configs[n]['label'] for n in val_ranking],
    'crossover_epoch_cosine_minus_constant': crossover_epochs,
    'verdict_cosine_overtakes_on_val': bool(cos_better),
}
results_summary = _to_native(results_summary)
out_json = os.path.join(SAVE_DIR, 'results_summary_extend100.json')
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {out_json}")

print(f"\n{'='*60}")
print("延长验证完成! 将本目录的 results_summary_extend100.json 与三张图取回即可更新讲义")
print(f"{'='*60}")
