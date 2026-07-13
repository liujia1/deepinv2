# -*- coding: utf-8 -*-
"""
实验15.4-2 DiT缩放规律实验（简化版）
对应章节: 15.3.3 DiT架构 / 15.4.2 DiT缩放规律

知识点:
  - DiT缩放性: 参数量增加时生成质量遵循可预测提升曲线
  - 三种规模配置: DiT-S / DiT-M / DiT-B
  - 幂律拟合: Loss ~ a * Params^b
  - 与15.4.2节 DiT论文缩放规律的对应

实验内容:
  步骤1: 三种DiT规模架构对比（参数量、token数）
  步骤2: 训练收敛对比（参数量↑ → Loss↓）
  步骤3: DDPM采样质量对比
  步骤4: 缩放规律可视化（参数量 vs Loss, 幂律拟合）

素材来源:
  - 15.4-1.py DiT架构实现
  - 15.4.2节 DiT缩放规律
  - ★ 原创设计: MNIST上的小规模缩放规律验证

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
    SAVE_DIR = os.path.join(_gdrive, '实验15.4-2')
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
print(f"实验15.4-2: DiT缩放规律实验（简化版）")
print(f"{'='*60}")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  未检测到 GPU, 使用 CPU 训练")
    print("  提示: Colab 用户可在菜单 运行时 -> 更改运行时类型 中选择 GPU")
    print("  提示: DiT在CPU上运行较慢, 建议使用GPU")


# ============================================================
# Checkpoint工具
# ============================================================
def load_train_state(checkpoint_path, model, optimizer, num_epochs_total, scheduler=None):
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
# DiT架构 (与15.4-1一致, 支持可配置规模)
# ============================================================
def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiTBlock(nn.Module):
    """DiT Block: Transformer + adaLN-Zero (15.3.3节)"""
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
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        h = self.norm1(x)
        h = h * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        h, _ = self.attn(h, h, h)
        x = x + gate_msa[:, None, :] * h
        h = self.norm2(x)
        h = h * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        h = self.mlp(h)
        x = x + gate_mlp[:, None, :] * h
        return x


class Patchify(nn.Module):
    def __init__(self, patch_size=2, in_channels=1, hidden_size=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * patch_size * in_channels, hidden_size)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, (H // p) * (W // p), p * p * C)
        return self.proj(x)


class Unpatchify(nn.Module):
    def __init__(self, patch_size=2, out_channels=1, hidden_size=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.out_channels = out_channels

    def forward(self, x, H, W):
        B = x.shape[0]
        p = self.patch_size
        x = self.proj(x)
        x = x.reshape(B, H // p, W // p, p, p, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, self.out_channels, H, W)
        return x


class DiT(nn.Module):
    """Diffusion Transformer (15.3.3节), 支持可配置规模"""
    def __init__(self, hidden_size=128, depth=4, num_heads=4,
                 patch_size=2, in_channels=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.patchify = Patchify(patch_size, in_channels, hidden_size)
        self.unpatchify = Unpatchify(patch_size, in_channels, hidden_size)
        num_patches = (28 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x_t, t):
        t_emb = timestep_embedding(t, self.hidden_size)
        c = self.time_mlp(t_emb)
        h = self.patchify(x_t)
        h = h + self.pos_embed
        for block in self.blocks:
            h = block(h, c)
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale[:, None, :]) + shift[:, None, :]
        pred = self.unpatchify(h, 28, 28)
        return pred


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


# ============================================================
# DiT缩放配置 (15.4.2节)
# ============================================================
dit_configs = {
    'dit_s': {
        'hidden_size': 64,
        'depth': 4,
        'num_heads': 4,
        'label': 'DiT-S',
        'color': '#3498db',
    },
    'dit_m': {
        'hidden_size': 96,
        'depth': 5,
        'num_heads': 4,
        'label': 'DiT-M',
        'color': '#e67e22',
    },
    'dit_b': {
        'hidden_size': 128,
        'depth': 6,
        'num_heads': 4,
        'label': 'DiT-B',
        'color': '#e74c3c',
    },
}

checkpoint_paths = {
    name: os.path.join(SAVE_DIR, f'{name}_checkpoint.pth')
    for name in dit_configs
}

num_epochs = 25


# ============================================================
# 步骤1: 三种DiT规模架构对比 (15.3.3/15.4.2节)
# ============================================================
print(f"\n{'='*60}")
print("步骤1: 三种DiT规模架构对比 (15.3.3/15.4.2节)")
print(f"{'='*60}")
print("""
15.4.2节 DiT缩放规律:
  DiT核心优势: 参数量增加时, 生成质量遵循可预测提升曲线
  DiT论文: S(33M) → B(130M) → L(458M) → XL(675M), FID持续下降

  本实验简化版 (MNIST):
    DiT-S: hidden=64,  depth=4, heads=4
    DiT-M: hidden=96,  depth=5, heads=4
    DiT-B: hidden=128, depth=6, heads=4
""")

dit_models = {}
param_counts = {}

for name, cfg in dit_configs.items():
    model = DiT(
        hidden_size=cfg['hidden_size'],
        depth=cfg['depth'],
        num_heads=cfg['num_heads'],
        patch_size=2,
        in_channels=1,
    ).to(device)
    dit_models[name] = model
    n_params = sum(p.numel() for p in model.parameters())
    param_counts[name] = n_params
    print(f"  {cfg['label']}: hidden={cfg['hidden_size']}, depth={cfg['depth']}, "
          f"heads={cfg['num_heads']}, 参数量={n_params:,} ({n_params/1e6:.2f}M)")


# ============================================================
# 步骤2: 训练收敛对比
# ============================================================
print(f"\n{'='*60}")
print("步骤2: 训练收敛对比 (参数量↑ → Loss↓, 15.4.2节)")
print(f"{'='*60}")

trained_models = {}
trained_histories = {}

for name, cfg in dit_configs.items():
    # 修复: 在每种DiT规模训练前重置全局随机状态, 确保三种规模在相同的数据顺序
    # 和DDPM噪声(t, eps)序列下训练, 严格保证缩放规律实验的公平对比前提
    # 注意: 若训练中途打断后恢复(start_epoch > 0), 续训后的随机序列与"一次性跑完"不一致,
    #       但三个模型处理方式相同, 跨模型比较仍公平
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = dit_models[name]
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    start_epoch, is_final, loss_history = load_train_state(
        checkpoint_paths[name], model, optimizer, num_epochs)

    if is_final:
        trained_models[name] = model
        trained_histories[name] = loss_history
        continue

    if start_epoch >= num_epochs:
        print(f"  {cfg['label']}: start_epoch({start_epoch}) >= num_epochs({num_epochs}), 跳过训练")
        trained_models[name] = model
        trained_histories[name] = loss_history
        continue

    print(f"\n  训练 {cfg['label']} (共 {num_epochs} 轮, 从第 {start_epoch+1} 轮开始)...")
    t_start = time.time()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader,
                     desc=f"{cfg['label']} Epoch {epoch+1}/{num_epochs}",
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
            epoch_loss += loss.item() * b
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_dataset)
        loss_history.append(avg_loss)
        print(f"  {cfg['label']} Epoch {epoch+1}/{num_epochs} 完成, Avg Loss={avg_loss:.6f}")
        save_train_state(checkpoint_paths[name], model, optimizer, epoch,
                         avg_loss, loss_history, is_final=(epoch == num_epochs - 1))

    t_elapsed = time.time() - t_start
    print(f"  {cfg['label']} 训练完成, 最终Loss: {loss_history[-1]:.6f}, 耗时: {t_elapsed:.1f}s")
    trained_models[name] = model
    trained_histories[name] = loss_history

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for name, cfg in dit_configs.items():
    h = trained_histories[name]
    ax.plot(range(1, len(h)+1), h, '-o', markersize=3,
            color=cfg['color'], label=f"{cfg['label']} ({param_counts[name]/1e6:.2f}M)")
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel(r'训练 Loss', fontsize=12)
ax.set_title('(a) 训练收敛曲线', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

ax = axes[1]
names_list = list(dit_configs.keys())
labels_list = [dit_configs[n]['label'] for n in names_list]
final_losses = [trained_histories[n][-1] for n in names_list]
colors_list = [dit_configs[n]['color'] for n in names_list]
bars = ax.bar(labels_list, final_losses, color=colors_list, alpha=0.7, width=0.5)
# 用相对偏移避免文字贴柱顶
text_offset = max(final_losses) * 0.02
for bar, val, pc in zip(bars, final_losses, [param_counts[n] for n in names_list]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + text_offset,
            f'{val:.4f}\n({pc/1e6:.2f}M)', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel(r'最终训练 Loss', fontsize=12)
ax.set_title('(b) 最终Loss对比', fontsize=12)
ax.grid(alpha=0.3, axis='y')

plt.suptitle(r'步骤2: DiT缩放与训练收敛 (15.4.2节)', fontsize=14, y=1.01)
plt.tight_layout()
fig_path2 = os.path.join(SAVE_DIR, '步骤2_训练收敛对比.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图2已保存: {fig_path2}")


# ============================================================
# 步骤3: DDPM采样质量对比
# ============================================================
print(f"\n{'='*60}")
print("步骤3: DDPM采样质量对比")
print(f"{'='*60}")

def to_vis_range(x):
    return (x + 1) / 2

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

fig, axes = plt.subplots(len(dit_configs), n_samples,
                          figsize=(n_samples * 1.8, len(dit_configs) * 2.2))

for row, (name, cfg) in enumerate(dit_configs.items()):
    print(f"  采样 {cfg['label']}...")
    samples = sample_ddpm(trained_models[name], n_samples)
    for col in range(n_samples):
        axes[row, col].imshow(to_vis_range(samples[col, 0].cpu()).clamp(0, 1),
                               cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
    axes[row, 0].set_ylabel(f"{cfg['label']}\n({param_counts[name]/1e6:.2f}M)",
                             fontsize=10, rotation=0, labelpad=55)

plt.suptitle(r'步骤3: DiT缩放采样质量对比 (DDPM, $T$=200步)', fontsize=14, y=1.01)
plt.tight_layout()
fig_path3 = os.path.join(SAVE_DIR, '步骤3_采样质量对比.png')
plt.savefig(fig_path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"图3已保存: {fig_path3}")


# ============================================================
# 步骤4: 缩放规律可视化 (15.4.2节)
# ============================================================
print(f"\n{'='*60}")
print("步骤4: 缩放规律可视化 (15.4.2节)")
print(f"{'='*60}")
print("""
15.4.2节 缩放规律:
  DiT论文验证: 验证损失与FID正相关
  参数量每增加10倍, 验证损失约降低 Δ
  本实验: 在MNIST上验证 Loss ~ a * Params^b 的幂律关系
""")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

params_M = np.array([param_counts[n] / 1e6 for n in dit_configs])
final_loss_arr = np.array([trained_histories[n][-1] for n in dit_configs])
labels_arr = [dit_configs[n]['label'] for n in dit_configs]
colors_arr = [dit_configs[n]['color'] for n in dit_configs]

ax = axes[0]
for i, (px, ly, cl, lb) in enumerate(zip(params_M, final_loss_arr, colors_arr, labels_arr)):
    ax.scatter(px, ly, color=cl, s=150, zorder=5, label=lb, edgecolors='black', linewidths=1)

# 修复说明: 3点幂律拟合的统计效力不足(残差自由度=1, 无法评估拟合优度),
# 此处仅作"示意性"幂律拟合, 不应作为严格统计结论
fit_warning = ""
log_params = np.log10(params_M)
log_losses = np.log10(final_loss_arr)
coeffs = np.polyfit(log_params, log_losses, 1)
b_fit, a_fit = coeffs
a_lin = 10 ** a_fit
x_fit = np.logspace(np.log10(params_M.min() * 0.5), np.log10(params_M.max() * 2), 100)
y_fit = a_lin * x_fit ** b_fit
ax.plot(x_fit, y_fit, 'k--', alpha=0.4, linewidth=1.5,
        label=rf'示意性拟合: Loss $\propto$ Params$^{{{b_fit:.2f}}}$')
fit_warning = f"\n  ⚠ 提示: 3点拟合残差自由度=1, 该指数仅作示意, 不可作为统计结论"

ax.set_xscale('log')
ax.set_xlabel(r'参数量 (M, 对数坐标)', fontsize=12)
ax.set_ylabel(r'最终训练 Loss', fontsize=12)
ax.set_title('(a) 参数量 vs Loss (缩放规律)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which='both')

ax = axes[1]
x_pos = np.arange(len(dit_configs))
bars = ax.bar(x_pos, final_loss_arr, color=colors_arr, alpha=0.7, width=0.5)
# 用相对偏移避免文字贴柱顶
text_offset = max(final_loss_arr) * 0.02
for bar, val, pc in zip(bars, final_loss_arr, params_M):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + text_offset,
            f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{lb}\n({pc:.2f}M)' for lb, pc in zip(labels_arr, params_M)], fontsize=10)
ax.set_ylabel(r'最终训练 Loss', fontsize=12)
ax.set_title('(b) 各规模Loss对比', fontsize=12)
ax.grid(alpha=0.3, axis='y')

plt.suptitle(r'步骤4: DiT缩放规律 (15.4.2节)', fontsize=14, y=1.01)
plt.tight_layout()
fig_path4 = os.path.join(SAVE_DIR, '步骤4_缩放规律.png')
plt.savefig(fig_path4, dpi=150, bbox_inches='tight')
plt.close()
print(f"图4已保存: {fig_path4}")

print("\n  缩放规律总结:")
for name, cfg in dit_configs.items():
    print(f"    {cfg['label']}: {param_counts[name]/1e6:.2f}M 参数, "
          f"最终Loss = {trained_histories[name][-1]:.6f}")

print(f"\n  示意性拟合: Loss ∝ Params^({b_fit:.2f})")
if b_fit < 0:
    print(f"    指数为负({b_fit:.2f}), 方向上与'参数量↑→Loss↓'一致")
    print(f"    提示: 3点拟合的指数仅作示意, 严格幂律需更多规模(DiT原论文用6+规模)")
else:
    print(f"    指数非负({b_fit:.2f}), 缩放规律不明显")
    print(f"    可能原因: 训练轮数不足或模型规模差异较小")
print(fit_warning)

ratio_sb = param_counts['dit_b'] / param_counts['dit_s']
loss_improve = (trained_histories['dit_s'][-1] - trained_histories['dit_b'][-1]) / trained_histories['dit_s'][-1] * 100
print(f"\n  DiT-B/DiT-S 参数量比: {ratio_sb:.1f}x")
print(f"  DiT-B 相对 DiT-S 的Loss改善: {loss_improve:.1f}%")


# ============================================================
# 实验总结
# ============================================================
print(f"\n{'='*60}")
print("实验15.4-2 总结")
losses_monotonic = (trained_histories['dit_s'][-1] > trained_histories['dit_m'][-1]
                     > trained_histories['dit_b'][-1])
trend_desc = "观察到参数量↑→Loss↓的单调趋势" if losses_monotonic else \
             "未观察到严格单调趋势(可能受训练轮数/规模差异影响)"

fit_trend_desc = f"拟合指数{b_fit:.2f}" + ("为负, 方向上与'参数量↑→Loss↓'一致" if b_fit < 0 else "非负, 缩放规律不明显")

print(f"{'='*60}")
print(f"""
关键结论:
1. 架构对比 (步骤1, 15.3.3节)
   - DiT-S: {param_counts['dit_s']/1e6:.2f}M 参数 (hidden=64, depth=4)
   - DiT-M: {param_counts['dit_m']/1e6:.2f}M 参数 (hidden=96, depth=5)
   - DiT-B: {param_counts['dit_b']/1e6:.2f}M 参数 (hidden=128, depth=6)
   - 主要参数来自Transformer的全连接层和注意力层

2. 训练收敛 (步骤2, 15.4.2节)
   - DiT-S 最终Loss: {trained_histories['dit_s'][-1]:.6f}
   - DiT-M 最终Loss: {trained_histories['dit_m'][-1]:.6f}
   - DiT-B 最终Loss: {trained_histories['dit_b'][-1]:.6f}
   - {trend_desc}

3. 采样质量 (步骤3)
   - 样本图见图3, 供定性观察不同规模模型的生成效果

4. 缩放规律 (步骤4, 15.4.2节)
   - {fit_trend_desc}
   - 实际DiT论文在ImageNet上(6+规模)验证了更严格的幂律关系
   - 核心启示: DiT的缩放趋势可观察, 但3点欠定拟合不能作为严格统计结论
""")

print(f"\n{'='*60}")
print("实验15.4-2 完成!")
print(f"{'='*60}")