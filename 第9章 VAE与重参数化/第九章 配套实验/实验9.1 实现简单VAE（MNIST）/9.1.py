# -*- coding: utf-8 -*-
"""
实验9.1 实现简单VAE（MNIST）
对应知识点：
  - 9.1节 编码器-解码器架构（识别模型q_φ(z|x)与生成模型p_θ(x|z)）
  - 9.2节 重参数化技巧（z = μ + σε）
  - 9.3节 ELBO训练与KL正则化（重建项 + KL项）
  - 9.5节 实践：实现与分析

代码参考9.5节嵌入的PyTorch实现，补充训练监控和可视化。
"""

import sys
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

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')


# ============================================================
# 模型定义（对应9.1节 VAE架构）
# ============================================================
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    """编码器（识别模型）: x → (μ, logσ²)
    对应9.1节: q_φ(z|x) = N(z | μ_φ(x), diag(σ_φ²(x)))
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """解码器（生成模型）: z → x̂
    对应9.1节: p_θ(x|z) = Bernoulli(x | f_θ(z))，使用sigmoid输出
    """
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(h))
        return x_recon


def reparameterize(mu, logvar):
    """重参数化技巧（对应9.2节）: z = μ + σε, ε ~ N(0,I)
    使梯度可通过μ和σ反传，避免梯度被随机采样阻断。
    """
    std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * logσ²)
    eps = torch.randn_like(std)     # ε ~ N(0, I)
    return mu + std * eps           # z = μ + σε


def loss_function(x, x_recon, mu, logvar, beta=1.0):
    """ELBO损失（对应9.3节）: -ELBO = BCE + β * KL
    BCE: 伯努利负对数似然（重建项）
    KL:  高斯KL散度闭式解（正则化项）
    """
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD, BCE, KLD


def compute_active_dims(mu, logvar, threshold=0.01):
    """活跃维度计算（对应9.3节）
    统计 KL(q(z_j|x) || p(z_j)) > threshold 的维度数
    """
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
    active = (kl_per_dim.mean(dim=0) > threshold).sum().item()
    return active, kl_per_dim.mean(dim=0).detach().cpu().numpy()


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
print(f"训练集: {len(train_dataset)} 样本, 测试集: {len(test_dataset)} 样本")


# ============================================================
# 训练
# ============================================================
encoder = Encoder().to(device)
decoder = Decoder().to(device)
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
)

num_epochs = 20
beta = 1.0

# 训练记录
history = {
    'loss': [], 'bce': [], 'kld': [], 'active_dims': []
}

print(f"\n开始训练 (epochs={num_epochs}, β={beta}, d_z=20)...")
print("-" * 70)

for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    total_loss, total_bce, total_kld = 0, 0, 0
    all_mu, all_logvar = [], []

    for x, _ in train_loader:
        x = x.view(-1, 784).to(device)

        # 前向传播（9.1节架构 + 9.2节重参数化）
        mu, logvar = encoder(x)
        z = reparameterize(mu, logvar)
        x_recon = decoder(z)

        # ELBO损失（9.3节）
        loss, bce, kld = loss_function(x, x_recon, mu, logvar, beta)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bce += bce.item()
        total_kld += kld.item()
        all_mu.append(mu.detach())
        all_logvar.append(logvar.detach())

    n = len(train_loader.dataset)
    avg_loss = total_loss / n
    avg_bce = total_bce / n
    avg_kld = total_kld / n

    # 活跃维度
    all_mu_cat = torch.cat(all_mu, dim=0)
    all_logvar_cat = torch.cat(all_logvar, dim=0)
    active, kl_per_dim = compute_active_dims(all_mu_cat, all_logvar_cat)

    history['loss'].append(avg_loss)
    history['bce'].append(avg_bce)
    history['kld'].append(avg_kld)
    history['active_dims'].append(active)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs}  "
              f"Loss={avg_loss:.2f}  BCE={avg_bce:.2f}  "
              f"KL={avg_kld:.2f}  Active={active}/20")

print("-" * 70)
print(f"训练完成！最终: Loss={history['loss'][-1]:.2f}, "
      f"BCE={history['bce'][-1]:.2f}, KL={history['kld'][-1]:.2f}, "
      f"Active={history['active_dims'][-1]}/20")


# ============================================================
# 可视化1: 训练曲线
# ============================================================
print("\n生成训练曲线图...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

epochs = range(1, num_epochs + 1)

# (a) 总损失
ax = axes[0, 0]
ax.plot(epochs, history['loss'], 'b-o', markersize=4)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('ELBO损失', fontsize=12)
ax.set_title('(a) 总损失 (BCE + β·KL)', fontsize=13)
ax.grid(True, alpha=0.3)

# (b) 重建损失
ax = axes[0, 1]
ax.plot(epochs, history['bce'], 'r-o', markersize=4, label='重建损失 (BCE)')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('BCE', fontsize=12)
ax.set_title('(b) 重建损失', fontsize=13)
ax.grid(True, alpha=0.3)

# (c) KL散度
ax = axes[1, 0]
ax.plot(epochs, history['kld'], 'g-o', markersize=4)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('KL散度', fontsize=12)
ax.set_title('(c) KL正则化项', fontsize=13)
ax.grid(True, alpha=0.3)
ax.annotate('KL从接近0逐步上升\n→编码器开始使用隐空间',
            xy=(0.4, 0.3), xycoords='axes fraction', fontsize=10, color='green',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3', alpha=0.8))

# (d) 活跃维度
ax = axes[1, 1]
ax.plot(epochs, history['active_dims'], 'm-o', markersize=4)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('活跃维度数', fontsize=12)
ax.set_title('(d) 活跃维度 (KL > 0.01)', fontsize=13)
ax.set_ylim(-0.5, 20.5)
ax.axhline(20, color='gray', linestyle='--', alpha=0.3, label='d_z=20')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤1_训练曲线.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图1已保存: {fig_path}")


# ============================================================
# 可视化2: 重建与先验采样
# ============================================================
print("生成重建与采样图...")

encoder.eval()
decoder.eval()

# 重建测试
with torch.no_grad():
    test_imgs, test_labels = next(iter(test_loader))
    test_flat = test_imgs.view(-1, 784).to(device)
    mu, logvar = encoder(test_flat)
    z = reparameterize(mu, logvar)
    recon = decoder(z)

n_show = 10
fig, axes = plt.subplots(3, n_show, figsize=(20, 6))

for i in range(n_show):
    # 原始图像
    axes[0, i].imshow(test_imgs[i, 0].numpy(), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('原始图像', fontsize=12)

    # 重建图像
    axes[1, i].imshow(recon[i].view(28, 28).cpu().numpy(), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('VAE重建', fontsize=12)

# 先验采样: z ~ N(0, I), x̂ = Decoder(z)
with torch.no_grad():
    z_prior = torch.randn(n_show, 20).to(device)
    samples = decoder(z_prior)

for i in range(n_show):
    axes[2, i].imshow(samples[i].view(28, 28).cpu().numpy(), cmap='gray')
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_title('先验采样', fontsize=12)

plt.suptitle('VAE重建与生成 (β=1, d_z=20, 20 epochs)', fontsize=14, y=1.02)
plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤2_重建与采样.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig_path}")


# ============================================================
# 额外统计
# ============================================================
print("\n" + "=" * 60)
print("训练统计摘要")
print("=" * 60)

# 测试集ELBO
encoder.eval()
decoder.eval()
test_loss, test_bce, test_kld = 0, 0, 0
with torch.no_grad():
    for x, _ in test_loader:
        x = x.view(-1, 784).to(device)
        mu, logvar = encoder(x)
        z = reparameterize(mu, logvar)
        x_recon = decoder(z)
        loss, bce, kld = loss_function(x, x_recon, mu, logvar, beta)
        test_loss += loss.item()
        test_bce += bce.item()
        test_kld += kld.item()

n_test = len(test_loader.dataset)
print(f"测试集: Loss={test_loss/n_test:.2f}, BCE={test_bce/n_test:.2f}, KL={test_kld/n_test:.2f}")

# 隐空间统计
all_mu_test, all_logvar_test, all_labels = [], [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.view(-1, 784).to(device)
        mu, logvar = encoder(x)
        all_mu_test.append(mu.cpu())
        all_logvar_test.append(logvar.cpu())
        all_labels.append(y)

all_mu_test = torch.cat(all_mu_test, dim=0)
all_logvar_test = torch.cat(all_logvar_test, dim=0)
all_labels = torch.cat(all_labels, dim=0)

active_test, kl_per_dim_test = compute_active_dims(all_mu_test, all_logvar_test)
print(f"测试集活跃维度: {active_test}/20")
print(f"各维度KL值: {[f'{v:.3f}' for v in kl_per_dim_test]}")

# μ的统计
print(f"编码器μ统计: 均值={all_mu_test.mean():.4f}, 标准差={all_mu_test.std():.4f}")
print(f"编码器logσ²统计: 均值={all_logvar_test.mean():.4f}, 标准差={all_logvar_test.std():.4f}")

print("\n" + "=" * 60)
print("实验9.1 完成！")
print("=" * 60)
print("""
关键观察：
1. 训练曲线：总损失持续下降，KL从0逐步上升（编码器逐渐使用隐空间）
2. 重建质量：VAE能较好地重建输入图像，但比AE略模糊（KL正则化的代价）
3. 先验采样：从N(0,I)采样解码能生成类似MNIST的数字
4. 活跃维度：并非所有20个维度都活跃，部分维度KL≈0（后验坍缩）
""")
