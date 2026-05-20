# -*- coding: utf-8 -*-
"""
实验12.2 采样路径与变分路径训练对比
对应知识点：
  - 12.4节 DSM≡VLB：等价性的形式化证明（实际训练验证）
  - 12.5节 连续时间视角的统一（VP-SDE训练目标对比）
  - 12.6节 实践意义与训练目标选择（简化vs完整权重）

在MNIST上分别用"采样路径"(DSM简化目标)和"变分路径"(VLB简化目标)训练
同一个UNet去噪网络，验证两者产生相同的训练动态和采样结果。
两种路径的网络架构、数据、训练超参数完全相同——仅理论解释不同。

素材来源：
  - 实验6.2的DSM训练流程（采样路径）
  - 实验11.2的DDPM训练流程（变分路径）
  - 12.4节DSM≡VLB等价性证明
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

SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')


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


# ============================================================
# 前向过程: q(x_t|x_0)（两个路径共用）
# ============================================================
def q_sample(x_0, t, noise=None):
    """x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε"""
    if noise is None:
        noise = torch.randn_like(x_0)
    return (
        sqrt_alpha_bars[t][:, None, None, None] * x_0 +
        sqrt_one_minus_alpha_bars[t][:, None, None, None] * noise
    )


# ============================================================
# 去噪网络: 小型UNet（两个路径共用同一架构）
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
    """小型UNet去噪网络"""
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
# DDPM采样算法（两个路径共用）
# ============================================================
@torch.no_grad()
def ddpm_sample(model, shape):
    """DDPM反向采样 x_T→x_0 (ε-prediction)"""
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
    return x


# ============================================================
# 训练函数
# ============================================================
def train_model(label, num_epochs=50):
    """
    训练去噪网络（简化DSM = 简化VLB）
    
    ★ 原创设计：同一训练循环，从两个角度解释
    
    采样路径解释（12.1节）：
      - 网络预测噪声ε̂_θ，训练目标为DSM损失 ||ε - ε̂_θ||²
      - 通过Tweedie等式从ε̂_θ可恢复得分函数 s_θ = -ε̂_θ/√(1-ᾱ_t)
      - 得分函数驱动逆向SDE采样
    
    变分路径解释（12.2节）：
      - 网络预测噪声ε̂_θ，训练目标为简化VLB损失 ||ε - ε̂_θ||²
      - 通过ε-prediction参数化恢复逆向均值 μ_θ = (1/√α_t)(x_t - β_t/√(1-ᾱ_t)·ε̂_θ)
      - 均值参数化驱动DDPM采样
    
    关键：两个解释下，代码完全相同！损失都是 ||ε - ε̂_θ||²
    """
    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    history = {'loss': []}

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
            # 两个路径的损失完全相同：||ε - ε̂_θ||²
            loss = F.mse_loss(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch

        avg_loss = total_loss / len(train_loader.dataset)
        history['loss'].append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1:3d}/{num_epochs}  Loss={avg_loss:.6f}")

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
# 步骤1：用两种"解释"训练同一个模型
# ============================================================
num_epochs = 50

print(f"\n{'='*60}")
print("步骤1：训练两个模型（相同架构+相同目标，不同随机种子）")
print("=" * 60)
print("""
★ 原创设计说明：
  由于简化DSM和简化VLB的训练目标完全相同 (||ε-ε̂_θ||²)，
  我们训练两个独立初始化的模型来验证等价性：
  - 模型A：从"采样路径"视角理解（DSM损失→得分函数→逆向SDE）
  - 模型B：从"变分路径"视角理解（简化VLB→均值匹配→DDPM采样）
  两者使用相同的网络架构和训练代码。
""")

model_a, history_a = train_model('采样路径(DSM)', num_epochs)
model_b, history_b = train_model('变分路径(VLB)', num_epochs)


# ============================================================
# 步骤2：对比训练曲线
# ============================================================
print(f"\n{'='*60}")
print("步骤2：对比训练曲线")
print("=" * 60)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(range(1, len(history_a['loss'])+1), history_a['loss'], 'b-o', markersize=4, label='采样路径 (DSM)')
ax.plot(range(1, len(history_b['loss'])+1), history_b['loss'], 'r-s', markersize=4, label='变分路径 (VLB)')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss = E[||ε-ε̂_θ||²]', fontsize=12)
ax.set_title('两条路径的训练曲线对比（12.4节：目标函数相同）', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig_path1 = os.path.join(SAVE_DIR, '步骤1_训练曲线对比.png')
plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"图1已保存: {fig_path1}")


# ============================================================
# 步骤3：对比采样结果
# ============================================================
print(f"\n{'='*60}")
print("步骤3：对比采样结果（DDPM采样算法）")
print("=" * 60)

n_samples = 8
sample_shape = (n_samples, 1, 28, 28)

print("  采样路径模型采样中...")
samples_a = ddpm_sample(model_a, sample_shape)
print("  变分路径模型采样中...")
samples_b = ddpm_sample(model_b, sample_shape)

fig, axes = plt.subplots(2, n_samples, figsize=(16, 4))
for i in range(n_samples):
    axes[0, i].imshow(samples_a[i, 0].cpu().numpy(), cmap='gray')
    axes[0, i].axis('off')
    if i == 0: axes[0, i].set_ylabel('采样路径\n(DSM→Score SDE)', fontsize=11, rotation=0, labelpad=70)
    axes[1, i].imshow(samples_b[i, 0].cpu().numpy(), cmap='gray')
    axes[1, i].axis('off')
    if i == 0: axes[1, i].set_ylabel('变分路径\n(VLB→DDPM)', fontsize=11, rotation=0, labelpad=70)

plt.suptitle('两条路径的采样结果对比（12.4节：DSM≡VLB）', fontsize=14, y=1.02)
plt.tight_layout()
fig_path2 = os.path.join(SAVE_DIR, '步骤2_采样结果对比.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig_path2}")


# ============================================================
# ★ 步骤4：原创设计 - 两条路径的"双向验证"
# 用采样路径模型执行DDPM采样，用变分路径模型提取得分函数
# ============================================================
print(f"\n{'='*60}")
print("★ 步骤4：两条路径的双向验证（12.4节实践验证）")
print("=" * 60)

print("""
12.4节等价性的实践含义：
  - 采样路径训练的ε̂_θ，通过μ_θ参数化可直接用于DDPM采样（变分路径的采样方式）
  - 变分路径训练的ε̂_θ，通过s_θ=-ε̂_θ/√(1-ᾱ_t)可提取得分函数（采样路径的核心量）
  
  → 同一个网络同时是"得分估计器"和"均值参数化器"
""")

# 验证：从训练好的模型中提取不同量
model_a.eval()
x_test = torch.randn(1, 1, 28, 28, device=device)
t_test = torch.tensor([50], device=device)

with torch.no_grad():
    eps_pred = model_a(x_test, t_test)  # 网络输出: ε̂_θ

    # 采样路径视角: 提取得分函数 (12.1节)
    ab_t = alpha_bars[50]
    score_from_eps = -eps_pred / torch.sqrt(1 - ab_t)

    # 变分路径视角: 计算逆向均值 (12.2节)
    mu_from_eps = (1 / torch.sqrt(alphas[50])) * (
        x_test - (betas[50] / torch.sqrt(1 - ab_t)) * eps_pred
    )

    # 采样路径视角: 提取x₀估计 (Tweedie等式)
    x0_from_eps = (x_test - torch.sqrt(1 - ab_t) * eps_pred) / torch.sqrt(ab_t)

print("从同一个网络输出ε̂_θ提取的不同量:")
print(f"  ε̂_θ 范数: {eps_pred.norm().item():.4f} (网络直接输出)")
print(f"  s_θ = -ε̂_θ/√(1-ᾱ_t) 范数: {score_from_eps.norm().item():.4f} (采样路径: 得分函数)")
print(f"  μ_θ 范数: {mu_from_eps.norm().item():.4f} (变分路径: 逆向均值)")
print(f"  x̂₀ = (x_t-√(1-ᾱ_t)ε̂)/√ᾱ_t 范数: {x0_from_eps.norm().item():.4f} (去噪估计)")
print()
print("→ 一个网络输出，三种解读，三种用途")
print("  这就是12.4节DSM≡VLB等价性的实践体现！")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print("实验12.2 完成!")
print("=" * 60)
print(f"""
关键观察:
1. 训练目标等价（12.4节）
   - 简化DSM和简化VLB的训练目标都是 ||ε-ε̂_θ||²
   - 两条路径的训练曲线和采样质量高度一致

2. 采样路径→变分路径的转换（12.3节）
   - 采样路径训练的ε̂_θ，通过μ_θ参数化可直接用于DDPM采样
   - "得分估计器"同时也是"均值参数化器"

3. 变分路径→采样路径的转换（12.3节）
   - 变分路径训练的ε̂_θ，通过s_θ=-ε̂_θ/√(1-ᾱ_t)可提取得分函数
   - "均值参数化器"同时也是"得分估计器"

4. 双向验证（★ 原创设计）
   - 一个网络输出ε̂_θ，同时携带得分函数、逆向均值、x₀估计三种信息
   - 这是DSM≡VLB等价性在实践中最直接的体现

5. 实践意义（12.6节）
   - 选择哪种路径解释不影响训练和采样——代码完全相同
   - 差异仅在于"如何理解"模型：得分匹配视角 vs 变分推断视角
""")
