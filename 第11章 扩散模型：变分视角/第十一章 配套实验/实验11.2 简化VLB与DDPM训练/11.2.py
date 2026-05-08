# -*- coding: utf-8 -*-
"""
实验11.2 简化VLB与DDPM训练
对应知识点：
  - 11.4节 简化VLB与DDPM训练（L_simple、DDPM采样算法）
  - 11.3节 三种参数化（训练中对比ε-prediction vs x₀-prediction）

在MNIST上训练一个简化的DDPM模型（ResMLP架构），
验证简化VLB的训练效果，对比ε-prediction和x₀-prediction两种参数化。
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

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
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
# ========================================================

np.random.seed(42)
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')


# ============================================================
# 噪声调度
# ============================================================
T = 100  # 减少步数以适配MLP架构的学习能力
beta_min, beta_max = 1e-4, 0.02
betas = torch.linspace(beta_min, beta_max, T).to(device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
alpha_bars_prev = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)

# 后验方差 (用于采样)
posterior_var = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)
posterior_log_var = torch.log(posterior_var.clamp(min=1e-20))
sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)
beta_over_sqrt_1m_ab = betas / sqrt_one_minus_alpha_bars


# ============================================================
# 前向过程: q(x_t|x_0)
# ============================================================
def q_sample(x_0, t, noise=None):
    """x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε"""
    if noise is None:
        noise = torch.randn_like(x_0)
    return (
        sqrt_alpha_bars[t][:, None] * x_0 +
        sqrt_one_minus_alpha_bars[t][:, None] * noise
    )


# ============================================================
# 去噪网络: ResMLP（残差块 + 每层时间注入 + LayerNorm）
# ============================================================
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间嵌入（与02-ddpm.ipynb相同）"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResBlock(nn.Module):
    """残差块 + 时间条件注入 + LayerNorm"""
    def __init__(self, dim, time_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(time_dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.fc1(h)
        h = h + self.time_proj(F.silu(t_emb))  # 每层注入时间条件
        h = self.norm2(h)
        h = F.silu(h)
        h = self.fc2(h)
        return x + h  # 残差连接


class DenoisingMLP(nn.Module):
    """ResMLP去噪网络（ε-prediction或x₀-prediction）"""
    def __init__(self, input_dim=784, hidden_dim=512, time_dim=64, n_blocks=4, pred_type='epsilon'):
        super().__init__()
        self.pred_type = pred_type
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, time_dim) for _ in range(n_blocks)])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x_t, t):
        t_emb = self.time_mlp(t)
        h = self.input_proj(x_t)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_proj(h)


# ============================================================
# DDPM采样算法（11.4节）
# ============================================================
@torch.no_grad()
def ddpm_sample(model, shape, pred_type='epsilon'):
    """DDPM反向采样 x_T→x_0"""
    model.eval()
    x = torch.randn(shape, device=device)

    for t_idx in reversed(range(T)):
        t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)

        # 网络预测
        pred = model(x, t)

        if pred_type == 'epsilon':
            # ε-prediction: μ_θ = (1/√α_t)(x_t - β_t/√(1-ᾱ_t)·ε̂)
            model_mean = sqrt_recip_alphas[t_idx] * (
                x - beta_over_sqrt_1m_ab[t_idx] * pred
            )
        elif pred_type == 'x0':
            # x₀-prediction: μ_θ = √ᾱ_{t-1}·β_t/(1-ᾱ_t)·x̂₀ + √α_t(1-ᾱ_{t-1})/(1-ᾱ_t)·x_t
            ab = alpha_bars[t_idx]
            ab_prev = alpha_bars_prev[t_idx]
            a = alphas[t_idx]
            coeff_x0 = torch.sqrt(ab_prev) * betas[t_idx] / (1 - ab)
            coeff_xt = torch.sqrt(a) * (1 - ab_prev) / (1 - ab)
            model_mean = coeff_x0 * pred + coeff_xt * x

        if t_idx == 0:
            x = model_mean
        else:
            noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(posterior_var[t_idx]) * noise

    return x


# ============================================================
# 训练函数
# ============================================================
def train_ddpm(pred_type='epsilon', num_epochs=25):
    """训练DDPM模型"""
    model = DenoisingMLP(pred_type=pred_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    history = {'loss': []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for x, _ in train_loader:
            x = x.view(-1, 784).to(device)
            batch = x.shape[0]

            # 随机采样时间步
            t = torch.randint(0, T, (batch,), device=device)

            # 前向过程: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
            noise = torch.randn_like(x)
            x_t = q_sample(x, t, noise)

            # 网络预测
            pred = model(x_t, t)

            # 计算损失 (11.4节 L_simple)
            if pred_type == 'epsilon':
                # ε-prediction: ||ε - ε̂_θ(x_t, t)||²
                loss = F.mse_loss(pred, noise)
            elif pred_type == 'x0':
                # x₀-prediction: ||x_0 - x̂₀(x_t, t)||²
                loss = F.mse_loss(pred, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch

        avg_loss = total_loss / len(train_loader.dataset)
        history['loss'].append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs}  Loss={avg_loss:.6f}")

    return model, history


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
# 步骤1: 训练两种参数化
# ============================================================
num_epochs = 25

print(f"\n{'='*60}")
print(f"步骤1: 训练 ε-prediction DDPM (T={T}, epochs={num_epochs})")
print(f"{'='*60}")
model_eps, history_eps = train_ddpm('epsilon', num_epochs)

print(f"\n{'='*60}")
print(f"步骤1: 训练 x₀-prediction DDPM (T={T}, epochs={num_epochs})")
print(f"{'='*60}")
model_x0, history_x0 = train_ddpm('x0', num_epochs)


# ============================================================
# 可视化1: 训练曲线 + 加噪过程
# ============================================================
print("\n生成训练曲线与加噪过程图...")

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(range(1, num_epochs+1), history_eps['loss'], 'b-o', markersize=4, label='ε-prediction')
ax.plot(range(1, num_epochs+1), history_x0['loss'], 'r-s', markersize=4, label='x₀-prediction')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('L_simple损失', fontsize=12)
ax.set_title('(a) 两种参数化的训练曲线', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig_path1a = os.path.join(SAVE_DIR, '步骤1a_训练曲线.png')
plt.savefig(fig_path1a, dpi=150, bbox_inches='tight')
plt.close()
print(f"图1a已保存: {fig_path1a}")

# (b) 前向加噪过程可视化
test_imgs, _ = next(iter(test_loader))
t_show = [0, 5, 15, 30, 50, 99]

fig2, axes_sub = plt.subplots(2, 3, figsize=(14, 6))
for i, t_idx in enumerate(t_show):
    torch.manual_seed(42)
    x0_t = test_imgs[0:1].view(-1, 784).to(device)
    t_t = torch.tensor([t_idx], device=device)
    eps_t = torch.randn_like(x0_t)
    x_t = q_sample(x0_t, t_t, eps_t)
    row, col = i // 3, i % 3
    axes_sub[row, col].imshow(x_t.view(28, 28).cpu().numpy(), cmap='gray')
    if t_idx == 0:
        axes_sub[row, col].set_title('x₀ (原始)', fontsize=11)
    else:
        axes_sub[row, col].set_title(f't={t_idx}', fontsize=11)
    axes_sub[row, col].axis('off')
plt.suptitle('(b) 前向加噪过程', fontsize=13)
plt.tight_layout()
fig_path1b = os.path.join(SAVE_DIR, '步骤1b_前向加噪.png')
plt.savefig(fig_path1b, dpi=150, bbox_inches='tight')
plt.close()
print(f"图1b已保存: {fig_path1b}")


# ============================================================
# 步骤2: DDPM采样结果
# ============================================================
print("\n步骤2: DDPM采样...")

n_samples = 8

# ε-prediction采样
print("  ε-prediction采样中...")
samples_eps = ddpm_sample(model_eps, (n_samples, 784), 'epsilon')

# x₀-prediction采样
print("  x₀-prediction采样中...")
samples_x0 = ddpm_sample(model_x0, (n_samples, 784), 'x0')

fig, axes = plt.subplots(2, n_samples, figsize=(16, 4))

for i in range(n_samples):
    # ε-prediction
    axes[0, i].imshow(samples_eps[i].view(28, 28).cpu().numpy(), cmap='gray')
    axes[0, i].axis('off')
    if i == 0: axes[0, i].set_ylabel('ε-prediction', fontsize=12, rotation=0, labelpad=60)

    # x₀-prediction
    axes[1, i].imshow(samples_x0[i].view(28, 28).cpu().numpy(), cmap='gray')
    axes[1, i].axis('off')
    if i == 0: axes[1, i].set_ylabel('x₀-prediction', fontsize=12, rotation=0, labelpad=60)

plt.suptitle(f'DDPM采样 (T={T}, {num_epochs} epochs, ResMLP)', fontsize=14, y=1.02)
plt.tight_layout()
fig_path2 = os.path.join(SAVE_DIR, '步骤2_DDPM采样对比.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig_path2}")


# ============================================================
# ★ 步骤3: 原创设计 - 逐步去噪过程可视化
# ============================================================
print("\n步骤3: 逐步去噪过程可视化...")

model_eps.eval()
t_denoise = [T-1, T*3//4, T//2, T//4, T//8, T//16, 0]  # 从T到0

# 从纯噪声开始
torch.manual_seed(42)
x_T = torch.randn(1, 784, device=device)

# 完整采样并记录中间步骤
denoise_trajectory = []
with torch.no_grad():
    x = x_T.clone()
    for t_idx in reversed(range(T)):
        t = torch.tensor([t_idx], device=device)
        pred = model_eps(x, t)
        model_mean = sqrt_recip_alphas[t_idx] * (
            x - beta_over_sqrt_1m_ab[t_idx] * pred
        )
        if t_idx == 0:
            x = model_mean
        else:
            noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(posterior_var[t_idx]) * noise

        if t_idx in t_denoise:
            denoise_trajectory.append((t_idx, x.view(28, 28).cpu().numpy()))

fig, axes = plt.subplots(1, len(denoise_trajectory), figsize=(20, 3))
for i, (t_val, img) in enumerate(denoise_trajectory):
    axes[i].imshow(img, cmap='gray')
    if t_val == 0:
        axes[i].set_title('x₀ (去噪完成)', fontsize=11)
    else:
        axes[i].set_title(f't={t_val}', fontsize=11)
    axes[i].axis('off')

plt.suptitle('★ DDPM反向去噪过程: x_T → x₀', fontsize=14, y=1.05)
plt.tight_layout()
fig_path3 = os.path.join(SAVE_DIR, '步骤3_逐步去噪过程.png')
plt.savefig(fig_path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"图3已保存: {fig_path3}")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print("实验11.2 完成!")
print("=" * 60)
print(f"""
关键观察:
1. L_simple训练效果（11.4节）
   - ε-prediction和x₀-prediction都成功收敛
   - 简化目标 L_simple = E[||ε - ε̂_θ(x_t,t)||²] 忽略时间权重但训练稳定

2. ε-prediction vs x₀-prediction（11.3节）
   - 两者训练曲线收敛速度相近
   - ε-prediction是DDPM标准选择（11.4节）

3. DDPM采样算法（11.4节）
   - 从x_T~N(0,I)开始，逐步去噪: x_{{t-1}} = μ_θ(x_t,t) + σ_t·z
   - 去噪过程从纯噪声逐步恢复结构

4. 前向加噪=固定编码器（11.1节联系）
   - x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε 精确实现固定高斯编码器
   - 无需训练编码器，只需训练去噪网络
""")
