# -*- coding: utf-8 -*-
"""
实验9.3 VAE隐空间可视化与β-VAE
对应知识点：
  - 9.3节 ELBO训练与KL正则化（β-VAE、后验坍缩、活跃维度）
  - 9.5节 实践：实现与分析（t-SNE可视化、插值、β对比）

训练三个模型（β=0/1/4），对比隐空间结构、插值质量和活跃维度。
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
# 模型定义（同9.1，简化为函数式训练）
# ============================================================
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD, BCE, KLD


def compute_active_dims(mu, logvar, threshold=0.01):
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
    active = (kl_per_dim.mean(dim=0) > threshold).sum().item()
    return active, kl_per_dim.mean(dim=0).detach().cpu().numpy()


def train_vae(beta, train_loader, num_epochs=15, latent_dim=20):
    """训练一个VAE模型，返回encoder、decoder和训练历史"""
    encoder = Encoder(latent_dim=latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
    )

    history = {'loss': [], 'bce': [], 'kld': [], 'active': []}

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        total_loss, total_bce, total_kld = 0, 0, 0
        all_mu, all_logvar = [], []

        for x, _ in train_loader:
            x = x.view(-1, 784).to(device)
            mu, logvar = encoder(x)
            z = reparameterize(mu, logvar)
            x_recon = decoder(z)
            loss, bce, kld = vae_loss(x, x_recon, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()
            all_mu.append(mu.detach())
            all_logvar.append(logvar.detach())

        n = len(train_loader.dataset)
        all_mu_cat = torch.cat(all_mu, dim=0)
        all_logvar_cat = torch.cat(all_logvar, dim=0)
        active, _ = compute_active_dims(all_mu_cat, all_logvar_cat)

        history['loss'].append(total_loss / n)
        history['bce'].append(total_bce / n)
        history['kld'].append(total_kld / n)
        history['active'].append(active)

    return encoder, decoder, history


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


# ============================================================
# 训练三个模型: β=0 (AE), β=1 (VAE), β=4 (β-VAE)
# ============================================================
betas = [0, 1, 4]
beta_names = ['β=0 (AE)', 'β=1 (VAE)', 'β=4 (β-VAE)']
models = {}

for beta, name in zip(betas, beta_names):
    print(f"\n{'='*60}")
    print(f"训练 {name} ...")
    print(f"{'='*60}")
    encoder, decoder, history = train_vae(beta, train_loader, num_epochs=15)
    models[beta] = {'encoder': encoder, 'decoder': decoder, 'history': history}
    print(f"  最终: Loss={history['loss'][-1]:.2f}, "
          f"BCE={history['bce'][-1]:.2f}, "
          f"KL={history['kld'][-1]:.2f}, "
          f"Active={history['active'][-1]}/20")


# ============================================================
# 可视化1: t-SNE隐空间对比
# ============================================================
print("\n生成t-SNE隐空间可视化...")

try:
    from sklearn.manifold import TSNE
    has_tsne = True
except ImportError:
    print("警告: sklearn未安装，跳过t-SNE可视化")
    has_tsne = False

if has_tsne:
    # 编码测试集
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    for idx, (beta, name) in enumerate(zip(betas, beta_names)):
        encoder = models[beta]['encoder']
        encoder.eval()

        all_mu, all_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.view(-1, 784).to(device)
                mu, _ = encoder(x)
                all_mu.append(mu.cpu())
                all_labels.append(y)

        all_mu = torch.cat(all_mu, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        # t-SNE降维（使用前2000个样本加速）
        n_tsne = min(2000, len(all_mu))
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        mu_2d = tsne.fit_transform(all_mu[:n_tsne])

        ax = axes[idx]
        scatter = ax.scatter(mu_2d[:, 0], mu_2d[:, 1], c=all_labels[:n_tsne],
                           cmap='tab10', alpha=0.5, s=5)
        ax.set_title(name, fontsize=14)
        ax.set_xlabel('t-SNE 1', fontsize=11)
        ax.set_ylabel('t-SNE 2', fontsize=11)

        # 活跃维度标注
        active = models[beta]['history']['active'][-1]
        ax.text(0.02, 0.98, f'活跃维度: {active}/20',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.colorbar(scatter, ax=axes[-1], label='数字类别')
    plt.suptitle('隐空间t-SNE可视化：β对隐空间结构的影响', fontsize=15, y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, '步骤1_β对比_tSNE.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图1已保存: {fig_path}")


# ============================================================
# 可视化2: 插值对比 (AE vs VAE)
# ============================================================
print("生成插值对比图...")

encoder_ae = models[0]['encoder']
decoder_ae = models[0]['decoder']
encoder_vae = models[1]['encoder']
decoder_vae = models[1]['decoder']

# 选取两个不同数字
encoder_ae.eval()
decoder_ae.eval()
encoder_vae.eval()
decoder_vae.eval()

# 获取测试集中两个样本
test_iter = iter(test_loader)
imgs1, labels1 = next(test_iter)
imgs2, labels2 = next(test_iter)

# 找不同数字的样本
idx1, idx2 = 0, 0
for i in range(len(labels1)):
    for j in range(len(labels2)):
        if labels1[i] != labels2[j]:
            idx1, idx2 = i, j
            break
    if labels1[idx1] != labels2[idx2]:
        break

x1 = imgs1[idx1].view(-1, 784).to(device)
x2 = imgs2[idx2].view(-1, 784).to(device)
print(f"插值: 数字 {labels1[idx1].item()} → 数字 {labels2[idx2].item()}")

n_interp = 10
alphas = np.linspace(0, 1, n_interp)

fig, axes = plt.subplots(2, n_interp + 2, figsize=(24, 5))

for row, (enc, dec, name) in enumerate([
    (encoder_ae, decoder_ae, 'β=0 (AE)'),
    (encoder_vae, decoder_vae, 'β=1 (VAE)')
]):
    with torch.no_grad():
        mu1, _ = enc(x1)
        mu2, _ = enc(x2)

    # 端点
    axes[row, 0].imshow(imgs1[idx1, 0].numpy() if row == 0 else imgs1[idx1, 0].numpy(),
                        cmap='gray')
    axes[row, 0].axis('off')
    axes[row, 0].set_ylabel(name, fontsize=12, rotation=0, labelpad=50)

    # 插值
    for i, alpha in enumerate(alphas):
        z_interp = (1 - alpha) * mu1 + alpha * mu2
        with torch.no_grad():
            x_interp = dec(z_interp)
        axes[row, i + 1].imshow(x_interp.view(28, 28).cpu().numpy(), cmap='gray')
        axes[row, i + 1].axis('off')
        if row == 0:
            axes[row, i + 1].set_title(f'α={alpha:.1f}', fontsize=10)

    # 终点
    axes[row, -1].imshow(imgs2[idx2, 0].numpy(), cmap='gray')
    axes[row, -1].axis('off')

plt.suptitle('隐空间线性插值: AE vs VAE', fontsize=14, y=1.02)
plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤2_插值对比.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig_path}")


# ============================================================
# 可视化3: 活跃维度与生成质量对比
# ============================================================
print("生成活跃维度与生成质量对比图...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (a) 训练曲线对比
ax = axes[0, 0]
for beta, name in zip(betas, beta_names):
    h = models[beta]['history']
    ax.plot(h['loss'], label=name, marker='o', markersize=3)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('ELBO损失', fontsize=12)
ax.set_title('(a) 训练损失对比', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (b) KL散度对比
ax = axes[0, 1]
for beta, name in zip(betas, beta_names):
    h = models[beta]['history']
    ax.plot(h['kld'], label=name, marker='o', markersize=3)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('KL散度', fontsize=12)
ax.set_title('(b) KL散度对比', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.annotate('β=4: KL被强力压制\n→ 更少活跃维度',
            xy=(0.55, 0.7), xycoords='axes fraction', fontsize=10, color='green',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3', alpha=0.8))

# (c) 活跃维度对比
ax = axes[0, 2]
for beta, name in zip(betas, beta_names):
    h = models[beta]['history']
    ax.plot(h['active'], label=name, marker='o', markersize=3)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('活跃维度数', fontsize=12)
ax.set_title('(c) 活跃维度对比', fontsize=13)
ax.set_ylim(-0.5, 20.5)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (d)(e)(f) 各β值的KL per dimension
for idx, (beta, name) in enumerate(zip(betas, beta_names)):
    ax = axes[1, idx]
    encoder = models[beta]['encoder']
    encoder.eval()

    all_mu, all_logvar = [], []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.view(-1, 784).to(device)
            mu, logvar = encoder(x)
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())

    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)
    _, kl_per_dim = compute_active_dims(all_mu, all_logvar)

    colors = ['#e74c3c' if v > 0.01 else '#bdc3c7' for v in kl_per_dim]
    ax.bar(range(20), kl_per_dim, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(0.01, color='red', linestyle='--', alpha=0.5, label='阈值=0.01')
    active = sum(v > 0.01 for v in kl_per_dim)
    ax.set_xlabel('维度 j', fontsize=12)
    ax.set_ylabel('KL(q(z_j|x) || p(z_j))', fontsize=12)
    ax.set_title(f'({chr(100+idx)}) {name}: {active}/20 维活跃', fontsize=13)
    ax.legend(fontsize=9)

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤3_活跃维度与训练对比.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图3已保存: {fig_path}")


# ============================================================
# 额外: β-VAE先验采样对比
# ============================================================
print("生成先验采样对比图...")

n_gen = 10
fig, axes = plt.subplots(3, n_gen, figsize=(20, 6))

for row, (beta, name) in enumerate(zip(betas, beta_names)):
    decoder = models[beta]['decoder']
    decoder.eval()

    with torch.no_grad():
        z_prior = torch.randn(n_gen, 20).to(device)
        samples = decoder(z_prior)

    for i in range(n_gen):
        axes[row, i].imshow(samples[i].view(28, 28).cpu().numpy(), cmap='gray')
        axes[row, i].axis('off')
        if i == 0:
            axes[row, i].set_ylabel(name, fontsize=12, rotation=0, labelpad=50)

plt.suptitle('先验采样对比: z~N(0,I), x=Decoder(z)', fontsize=14, y=1.02)
plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤4_先验采样对比.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图4已保存: {fig_path}")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("实验9.3 完成！")
print("=" * 60)

for beta, name in zip(betas, beta_names):
    h = models[beta]['history']
    print(f"\n{name}:")
    print(f"  最终Loss={h['loss'][-1]:.2f}, BCE={h['bce'][-1]:.2f}, "
          f"KL={h['kld'][-1]:.2f}, Active={h['active'][-1]}/20")

print("""
关键结论：
1. β=0 (AE): 隐空间不规则，插值经过空洞产生模糊图像，活跃维度多但无结构
2. β=1 (VAE): 隐空间连续有结构，插值平滑过渡，部分维度后验坍缩
3. β=4 (β-VAE): KL被强力压制，活跃维度更少，生成更模糊但隐空间更解纠缠

β-VAE的权衡：
- β↑ → KL↓ → 隐空间更规则/解纠缠，但重建/生成质量下降
- β↓ → KL↑ → 重建更清晰，但隐空间可能不规则
- 后验坍缩：β过大时，q(z|x)≈p(z)，隐变量不携带信息
""")
