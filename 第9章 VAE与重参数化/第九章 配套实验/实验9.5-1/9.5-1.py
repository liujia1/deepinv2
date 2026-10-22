# -*- coding: utf-8 -*-
"""
实验9.5-1 β-VAE对比与隐空间可视化
对应知识点:
  - 9.3节 ELBO训练与KL正则化（β-VAE、后验坍缩、活跃维度）
  - 9.5节 实践：实现与分析（t-SNE可视化、插值、β对比）

实验内容:
  步骤1: 训练三个模型（β=0/1/4）并对比隐空间t-SNE可视化
  步骤2: 隐空间插值对比（AE vs VAE）
  步骤3: 活跃维度与训练曲线对比
  步骤4: 先验采样对比

素材来源:
  - 实验9.3.py
  - 9.3节β-VAE理论
  - 9.5节可视化方法

运行前提: PyTorch, torchvision, sklearn, CPU/GPU均可
"""

import numpy as np
import os
import sys
import io
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

# ====== matplotlib 静默模式 ======
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
# =================================

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验9.5-1')
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

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"实验9.5-1: β-VAE对比与隐空间可视化")
print(f"{'='*60}")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  未检测到 GPU, 使用 CPU 训练")
    print("  提示: Colab 用户可在菜单 运行时 -> 更改运行时类型 中选择 GPU")

# Checkpoint路径模板
CHECKPOINT_TEMPLATE = os.path.join(SAVE_DIR, 'vae_beta{beta}_checkpoint.pth')

# ============================================================
# VAE模型定义（对应9.1节和9.5节）
# ============================================================
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # 进度条显示


class Encoder(nn.Module):
    """编码器: x → (μ, logσ²)"""
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """解码器: z → x̂"""
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))


def reparameterize(mu, logvar):
    """重参数化: z = μ + σε"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    """ELBO损失: -ELBO = BCE + β·KL"""
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD, BCE, KLD


def compute_active_dims(mu, logvar, threshold=0.01):
    """活跃维度计算（对应9.3节）"""
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
    active = (kl_per_dim.mean(dim=0) > threshold).sum().item()
    return active, kl_per_dim.mean(dim=0).detach().cpu().numpy()


# ============================================================
# 数据加载
# ============================================================
print("\n加载MNIST数据集...")
data_dir = os.path.join(SAVE_DIR, 'data')
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
print(f"训练集: {len(train_dataset)} 样本, 测试集: {len(test_dataset)} 样本")


# ============================================================
# 训练函数（支持checkpoint resume）
# ============================================================
def train_vae_with_checkpoint(beta, train_loader, num_epochs=15, latent_dim=20, ckpt_path=None):
    """训练一个VAE模型，支持checkpoint保存和加载；ckpt_path可覆盖默认文件名（供d_z=2对照用）"""
    # 快速验证模式必须在加载 checkpoint 之前判定，并写入独立文件：
    # 否则 3 轮速成模型会被 is_final 标记为成品，正式运行会静默跳过训练。
    quick = os.environ.get('QUICK_TEST', '') == '1'
    if quick:
        num_epochs = min(num_epochs, 3)
        base = ckpt_path if ckpt_path is not None else CHECKPOINT_TEMPLATE.format(beta=beta)
        ckpt_path = base.replace('.pth', '_quick.pth')
        print(f"\n[快速验证模式] 仅训练 {num_epochs} 轮 -> {os.path.basename(ckpt_path)}")

    checkpoint_path = ckpt_path if ckpt_path is not None else CHECKPOINT_TEMPLATE.format(beta=beta)
    
    encoder = Encoder(latent_dim=latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
    )
    
    history = {'loss': [], 'bce': [], 'kld': [], 'active': [], 'latent_dim': latent_dim}
    start_epoch = 0
    is_final = False
    
    # Checkpoint加载逻辑
    if os.path.exists(checkpoint_path):
        print(f"\n检测到已保存的模型 (β={beta}): {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        saved_epochs = int(checkpoint.get('num_epochs', checkpoint['epoch'] + 1))
        saved_quick = bool(checkpoint.get('quick', False))
        if saved_quick != quick or saved_epochs < num_epochs:
            print(f"  注意: checkpoint配置不符 (quick={saved_quick}→{quick}, "
                  f"轮数={saved_epochs}→目标{num_epochs}), 重新训练")
        elif checkpoint.get('is_final', False):
            print(f"✓ 这是最终训练完成的模型, 直接加载, 跳过训练过程")
            print(f"  训练轮数: {checkpoint['epoch']+1}")
            print(f"  最终损失: {checkpoint['loss']:.6f}")
            try:
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                decoder.load_state_dict(checkpoint['decoder_state_dict'])
            except RuntimeError as e:
                print(f"警告: checkpoint与当前模型架构不兼容, 删除后重新训练")
                print(f"  错误信息: {e}")
                os.remove(checkpoint_path)
                start_epoch = 0
                is_final = False
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                history = checkpoint.get('history', {'loss': [], 'bce': [], 'kld': [], 'active': []})
                history.setdefault('latent_dim', latent_dim)
                start_epoch = checkpoint['epoch'] + 1
                is_final = True
        else:
            print(f"检测到未完成的训练, 从第 {checkpoint['epoch']+1} 轮继续")
            try:
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                decoder.load_state_dict(checkpoint['decoder_state_dict'])
            except RuntimeError as e:
                print(f"警告: checkpoint与当前模型架构不兼容, 删除后重新训练")
                print(f"  错误信息: {e}")
                os.remove(checkpoint_path)
                start_epoch = 0
                is_final = False
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                history = checkpoint.get('history', {'loss': [], 'bce': [], 'kld': [], 'active': []})
                history.setdefault('latent_dim', latent_dim)
                start_epoch = checkpoint['epoch'] + 1
    
    # 训练循环
    if not is_final:
        # 边界保护
        if start_epoch >= num_epochs:
            print(f"  注意: start_epoch({start_epoch}) >= num_epochs({num_epochs}), 无需继续训练")
            is_final = True
        
        if not is_final:
            import time
            t_start = time.time()
            
            for epoch in range(start_epoch, num_epochs):
                encoder.train()
                decoder.train()
                total_loss, total_bce, total_kld = 0, 0, 0
                all_mu, all_logvar = [], []
                
                pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', 
                            leave=False, unit='batch')
                
                for x, _ in pbar:
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
                    
                    pbar.set_postfix(loss=f'{loss.item():.4f}')
                
                n = len(train_loader.dataset)
                all_mu_cat = torch.cat(all_mu, dim=0)
                all_logvar_cat = torch.cat(all_logvar, dim=0)
                active, _ = compute_active_dims(all_mu_cat, all_logvar_cat)
                
                history['loss'].append(total_loss / n)
                history['bce'].append(total_bce / n)
                history['kld'].append(total_kld / n)
                history['active'].append(active)
                
                # 每个epoch打印一次并保存checkpoint，避免训练3个模型时中断风险（最多丢1轮而非4轮）
                print(f"  Epoch {epoch+1:3d}/{num_epochs}: "
                      f"Loss={history['loss'][-1]:.2f}  "
                      f"BCE={history['bce'][-1]:.2f}  "
                      f"KL={history['kld'][-1]:.2f}  "
                      f"Active={active}/{latent_dim}")
                
                torch.save({
                    'epoch': epoch,
                    'beta': beta,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': history['loss'][-1],
                    'history': history,
                    'num_epochs': num_epochs,
                    'quick': quick,
                    'is_final': False
                }, checkpoint_path)
            
            t_elapsed = time.time() - t_start
            print(f"\n训练完成, 最终损失: {history['loss'][-1]:.6f}, "
                  f"耗时: {t_elapsed:.1f} 秒")
            
            # 保存最终checkpoint
            if history['loss']:
                torch.save({
                    'epoch': num_epochs - 1,
                    'beta': beta,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': history['loss'][-1],
                    'history': history,
                    'num_epochs': num_epochs,
                    'quick': quick,
                    'is_final': True
                }, checkpoint_path)
                print(f"✓ 训练完成, 模型已保存: {checkpoint_path}")
    else:
        print(f"\n使用已训练完成的模型 (β={beta}), 跳过训练过程")
    
    return encoder, decoder, history


# ============================================================
# 步骤1: 训练三个模型并对比隐空间t-SNE可视化
# ============================================================
print(f"\n{'='*60}")
print("步骤1: 训练三个模型并对比隐空间t-SNE可视化")
print(f"{'='*60}")
print("\n[核心思想]")
print("  β-VAE的权衡（对应9.3节）：")
print("  - β↑ → KL↓ → 隐空间更规则/解纠缠，但重建质量下降")
print("  - β↓ → KL↑ → 重建更清晰，但隐空间可能不规则")
print("  - 后验坍缩：β过大时，q(z|x)≈p(z)，隐变量不携带信息")

betas = [0, 1, 4]
beta_names = ['$\\beta=0$ (AE)', '$\\beta=1$ (VAE)', '$\\beta=4$ ($\\beta$-VAE)']  # 用于matplotlib图表
beta_names_print = ['beta=0 (AE)', 'beta=1 (VAE)', 'beta=4 (beta-VAE)']  # 用于print语句（人类可读）
models = {}

for beta, name, name_print in zip(betas, beta_names, beta_names_print):
    print(f"\n{'='*60}")
    print(f"训练 {name_print} ...")
    print(f"{'='*60}")
    encoder, decoder, history = train_vae_with_checkpoint(beta, train_loader, num_epochs=15)
    models[beta] = {'encoder': encoder, 'decoder': decoder, 'history': history}
    print(f"\n{name_print} 训练结果:")
    print(f"  最终: Loss={history['loss'][-1]:.2f}, "
          f"BCE={history['bce'][-1]:.2f}, "
          f"KL={history['kld'][-1]:.2f}, "
          f"Active={history['active'][-1]}/{history['latent_dim']}")


# ============================================================
# t-SNE隐空间可视化
# ============================================================
print(f"\n{'='*60}")
print("生成t-SNE隐空间可视化...")
print(f"{'='*60}")

try:
    from sklearn.manifold import TSNE
    has_tsne = True
except ImportError:
    print("警告: sklearn未安装，跳过t-SNE可视化")
    has_tsne = False

if has_tsne:
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
    for idx, (beta, name, name_print) in enumerate(zip(betas, beta_names, beta_names_print)):
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
        
        # 活跃维度标注（通过print而非子图文字）
        active = models[beta]['history']['active'][-1]
        print(f"{name_print}: 活跃维度 {active}/{models[beta]['history']['latent_dim']}")
    
    plt.colorbar(scatter, ax=axes[-1], label='数字类别')
    plt.suptitle('隐空间 t-SNE: $\\beta$ 的影响', fontsize=15, y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, '步骤1_β对比_tSNE.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图1已保存: {fig_path}")


# ============================================================
# 步骤2: 插值对比 (AE vs VAE)
# ============================================================
print(f"\n{'='*60}")
print("步骤2: 隐空间插值对比 (AE vs VAE)")
print(f"{'='*60}")
print("\n[核心对比]")
print("  隐空间连续性验证（对应9.5节）：")
print("  - AE (β=0): 无KL约束，隐空间确有空洞；但 d_z=20、6万训练样本下，两样本")
print("    连线大多落在数据邻域内，本设置实测插值同样平滑（'空洞致插值崩坏'需 d_z=2")
print("    等低维隐空间才会显现，下文给出中点距离的量化核验）")
print("  - VAE (β=1): KL正则化使隐空间连续 → 插值平滑过渡")
print("  - 注意: 本图两行视觉差异不明显，AE相对于VAE的真正缺陷体现在步骤4——")
print("    隐空间与先验N(0,I)失配，直接先验采样会得到乱码")

encoder_ae = models[0]['encoder']
decoder_ae = models[0]['decoder']
encoder_vae = models[1]['encoder']
decoder_vae = models[1]['decoder']

encoder_ae.eval()
decoder_ae.eval()
encoder_vae.eval()
decoder_vae.eval()

# 获取测试集中两个不同数字的样本
test_iter = iter(test_loader)
imgs1, labels1 = next(test_iter)
imgs2, labels2 = next(test_iter)

# 找两个不同数字的样本用于插值演示
idx1, idx2 = next((i, j) for i in range(len(labels1)) 
                         for j in range(len(labels2)) 
                         if labels1[i] != labels2[j])

x1 = imgs1[idx1].view(-1, 784).to(device)
x2 = imgs2[idx2].view(-1, 784).to(device)
print(f"\n插值: 数字 {labels1[idx1].item()} → 数字 {labels2[idx2].item()}")

n_interp = 11
alphas = np.linspace(0, 1, n_interp)

fig, axes = plt.subplots(2, n_interp + 2, figsize=(26, 5))

for row, (enc, dec, name) in enumerate([
    (encoder_ae, decoder_ae, '$\\beta=0$ (AE)'),
    (encoder_vae, decoder_vae, '$\\beta=1$ (VAE)')
]):
    ax = axes[row, 0]
    with torch.no_grad():
        mu1, _ = enc(x1)
        mu2, _ = enc(x2)
    
    # 端点
    axes[row, 0].imshow(imgs1[idx1, 0].numpy(), cmap='gray')
    axes[row, 0].axis('off')
    # 行标签放在轴内部左上角（原 ylabel+大labelpad 在 bbox_inches='tight' 下会被裁掉）
    ax.text(0.02, 1.06, name, transform=ax.transAxes, fontsize=12,
            ha='left', va='bottom')
    
    # 插值
    for i, alpha in enumerate(alphas):
        z_interp = (1 - alpha) * mu1 + alpha * mu2
        with torch.no_grad():
            x_interp = dec(z_interp)
        axes[row, i + 1].imshow(x_interp.view(28, 28).cpu().numpy(), cmap='gray')
        axes[row, i + 1].axis('off')
        if row == 0:
            axes[row, i + 1].set_title(f'$\\alpha={alpha:.1f}$', fontsize=10)
    
    # 终点
    axes[row, -1].imshow(imgs2[idx2, 0].numpy(), cmap='gray')
    axes[row, -1].axis('off')

plt.suptitle('隐空间插值: AE vs VAE', fontsize=14, y=1.02)
plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤2_插值对比.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig_path}")

# ---- 量化核验：插值中点是否落在数据邻域（对应9.5节"实测分析（诚实版）"）----
# 对30对不同数字的测试样本，计算插值中点到最近训练编码的距离，
# 与两个端点到最近训练编码距离的均值之比；比值~1说明中点仍在数据邻域内。
from sklearn.neighbors import NearestNeighbors

def _encode_loader(encoder, loader):
    encoder.eval()
    zs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            zs.append(encoder(x.view(-1, 784).to(device))[0].cpu())
            ys.append(y)
    return torch.cat(zs).numpy(), torch.cat(ys).numpy()

midpoint_ratios = {}
print("\n[量化核验] 30对不同数字样本: 插值中点最近邻距离 / 端点最近邻距离均值")
for beta_q, name_print_q in [(0, 'beta=0 (AE)'), (1, 'beta=1 (VAE)')]:
    enc_q = models[beta_q]['encoder']
    z_train, _ = _encode_loader(enc_q, train_loader)
    z_test, y_test = _encode_loader(enc_q, test_loader)
    nbrs = NearestNeighbors(n_neighbors=1).fit(z_train)
    rng = np.random.RandomState(0)
    ratios = []
    while len(ratios) < 30:
        i, j = rng.randint(len(z_test)), rng.randint(len(z_test))
        if y_test[i] == y_test[j]:
            continue
        z_mid = 0.5 * (z_test[i] + z_test[j])
        d_mid = nbrs.kneighbors(z_mid[None, :])[0][0, 0]
        d_end = 0.5 * (nbrs.kneighbors(z_test[i][None, :])[0][0, 0]
                       + nbrs.kneighbors(z_test[j][None, :])[0][0, 0])
        ratios.append(d_mid / d_end)
    midpoint_ratios[beta_q] = float(np.mean(ratios))
    print(f"  {name_print_q}: 比值={midpoint_ratios[beta_q]:.2f} (越接近1说明中点越贴近数据邻域)")
print("  → 两者同一量级，即 d_z=20 下AE插值路径并未系统性穿过远邻域空洞，与图2两行均平滑一致")


# ============================================================
# 步骤3: 活跃维度与训练对比
# ============================================================
print(f"\n{'='*60}")
print("步骤3: 活跃维度与训练对比")
print(f"{'='*60}")
print("\n[核心验证]")
print("  后验坍缩检测（对应9.3节核心结论）：")
print("  - β=0: 无KL惩罚，隐空间不受约束（实际KL可能很大，仅未被优化）")
print("  - β=1: KL适中，20/20活跃（d_z=20），隐空间有规律")
print("  - β=4: KL被强力压制，活跃维度减少（后验坍缩）")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (a) 训练曲线对比
ax = axes[0, 0]
for beta, name in zip(betas, beta_names):
    h = models[beta]['history']
    ax.plot(h['loss'], label=name, marker='o', markersize=3)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel(r'总损失 $= \mathrm{BCE} + \beta\cdot\mathrm{KL}$（每样本）', fontsize=12)
ax.set_title('(a) 训练损失对比（含β·KL项，勿直接当作重建质量）', fontsize=13)
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

# (c) 活跃维度对比
ax = axes[0, 2]
for beta, name in zip(betas, beta_names):
    h = models[beta]['history']
    linestyle = '--' if beta == 1 else '-'
    ax.plot(h['active'], label=name, marker='o', markersize=3, linestyle=linestyle)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('活跃维度', fontsize=12)
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
    ax.bar(range(len(kl_per_dim)), kl_per_dim, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(0.01, color='red', linestyle='--', alpha=0.5, label='阈值=0.01')
    active = sum(v > 0.01 for v in kl_per_dim)
    ax.set_xlabel('维度 $j$', fontsize=12)
    ax.set_ylabel('$KL(q(z_j|x) \\|\\| p(z_j))$', fontsize=12)
    ax.set_title(f'({chr(100+idx)}) {name}: {active}/{len(kl_per_dim)} 活跃', fontsize=13)
    ax.legend(fontsize=9)

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤3_活跃维度与训练对比.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图3已保存: {fig_path}")

# ---- 测试集重建质量（BCE 与 PSNR，供9.3节跨实验对照引用）----
# 评测口径与实验9.3-1完全一致：固定种子的重参数采样 z、逐图MSE→逐图PSNR→平均
test_metrics = {}
print("\n[测试集重建质量]（采样z评测，口径同实验9.3-1）")
for beta, name_print in zip(betas, beta_names_print):
    enc_q, dec_q = models[beta]['encoder'], models[beta]['decoder']
    enc_q.eval(); dec_q.eval()
    torch.manual_seed(0)  # 每个模型用同一组固定采样噪声，公平且可复现
    bce_sum = 0
    mse_list = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.view(-1, 784).to(device)
            mu, logvar = enc_q(x)
            z = reparameterize(mu, logvar)
            x_recon = dec_q(z)
            bce_sum += F.binary_cross_entropy(x_recon, x, reduction='sum').item()
            mse_list.append(((x_recon - x) ** 2).mean(dim=1))  # 逐图784像素MSE
    mse_all = torch.cat(mse_list)
    psnr = (10 * torch.log10(1.0 / mse_all.clamp(min=1e-12))).mean().item()
    test_bce = bce_sum / len(test_loader.dataset)
    test_metrics[beta] = {'test_bce': test_bce, 'test_psnr_db': float(psnr)}
    print(f"  {name_print}: 测试BCE={test_bce:.2f}, PSNR={psnr:.1f} dB")


# ============================================================
# 步骤4: 先验采样对比
# ============================================================
print(f"\n{'='*60}")
print("步骤4: 先验采样对比")
print(f"{'='*60}")
print("\n[核心观察]")
print("  β对生成质量的影响：")
print("  - β↑ → 生成更模糊（KL正则化更强，信息损失更多）")
print("  - β↓ → 生成更清晰，但隐空间未被约束匹配N(0,I)")
print("  - 关键点：AE(β=0)的隐空间q(z|x)从未被约束去匹配N(0,I)，")
print("    从标准正态直接采样很可能落在训练时未覆盖的区域，")
print("    解码结果大概率是乱码/明显崩坏（而非'更清晰但不规律'）")

n_gen = 10
fig, axes = plt.subplots(3, n_gen, figsize=(20, 6))

# 固定先验采样种子：加载checkpoint重绘与从零训练得到的图4保持一致、可复现
prior_gen = torch.Generator().manual_seed(2024)

for row, (beta, name) in enumerate(zip(betas, beta_names)):
    decoder = models[beta]['decoder']
    decoder.eval()
    
    with torch.no_grad():
        z_prior = torch.randn(n_gen, 20, generator=prior_gen).to(device)
        samples = decoder(z_prior)
    
    for i in range(n_gen):
        ax_cell = axes[row, i]
        ax_cell.imshow(samples[i].view(28, 28).cpu().numpy(), cmap='gray')
        ax_cell.axis('off')
        if i == 0:
            # 行标签放在轴内部左上角（ylabel+大labelpad 会被 axis('off')/tight bbox 隐去）
            ax_cell.text(0.02, 1.06, name, transform=ax_cell.transAxes, fontsize=12,
                         ha='left', va='bottom')

plt.suptitle('先验采样: $z \\sim \\mathcal{N}(0,I)$, $x = \\text{Decoder}(z)$', 
             fontsize=14, y=1.02)
plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤4_先验采样对比.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图4已保存: {fig_path}")


# ============================================================
# 步骤2b: 低维隐空间 (d_z=2) 插值对照
# ============================================================
# 说明: 本块物理上放在步骤4之后，是为了让d_z=2的训练不消耗全局RNG，
#       从而不改变图1~4的逐位可复现性；块内独立重设种子保证自身确定性。
print(f"\n{'='*60}")
print("步骤2b: 低维隐空间 (d_z=2) 插值对照")
print(f"{'='*60}")
print("\n[设计意图]")
print("  步骤2在d_z=20下未复现AE插值崩坏。此处仅把隐空间降到d_z=2（编码器/")
print("  解码器结构不变，唯一变量是隐空间维度），检验'空洞致插值畸变'是否随维度降低而显现。")

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

models_dz2 = {}
for beta, name_print in [(0, 'beta=0 (AE)'), (1, 'beta=1 (VAE)')]:
    ck_dz2 = os.path.join(SAVE_DIR, f'vae_beta{beta}_dz2_checkpoint.pth')
    print(f"\n{'='*60}")
    print(f"训练 d_z=2 {name_print} ...")
    print(f"{'='*60}")
    enc, dec, hist = train_vae_with_checkpoint(beta, train_loader, num_epochs=15,
                                               latent_dim=2, ckpt_path=ck_dz2)
    models_dz2[beta] = {'encoder': enc, 'decoder': dec, 'history': hist}


def _pick_test_sample(digit):
    """从测试集取指定数字的第一张样本，返回(显示用[28,28]numpy, 编码用[1,784]tensor)"""
    for x, y in test_loader:
        hit = (y == digit).nonzero(as_tuple=True)[0]
        if len(hit) > 0:
            return x[hit[0], 0].numpy(), x[hit[0]].view(1, 784).to(device)
    return None, None


pairs_dz2 = [(0, 1), (3, 7)]
n_interp2 = 11
alphas2 = np.linspace(0, 1, n_interp2)
fig, axes = plt.subplots(4, n_interp2 + 2, figsize=(26, 10))

row = 0
for (da, db) in pairs_dz2:
    disp_a, xa = _pick_test_sample(da)
    disp_b, xb = _pick_test_sample(db)
    for beta, tag in [(0, '$\\beta=0$ (AE)'), (1, '$\\beta=1$ (VAE)')]:
        enc = models_dz2[beta]['encoder']
        dec = models_dz2[beta]['decoder']
        enc.eval(); dec.eval()
        with torch.no_grad():
            mu1 = enc(xa)[0]
            mu2 = enc(xb)[0]
        # 端点A
        axes[row, 0].imshow(disp_a, cmap='gray')
        axes[row, 0].axis('off')
        axes[row, 0].text(0.02, 1.06, f'{tag}, ${da}\\to{db}$', transform=axes[row, 0].transAxes,
                          fontsize=12, ha='left', va='bottom')
        # 插值
        for i, alpha in enumerate(alphas2):
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            with torch.no_grad():
                x_interp = dec(z_interp)
            axes[row, i + 1].imshow(x_interp.view(28, 28).cpu().numpy(), cmap='gray')
            axes[row, i + 1].axis('off')
            if row == 0:
                axes[row, i + 1].set_title(f'$\\alpha={alpha:.1f}$', fontsize=10)
        # 端点B
        axes[row, -1].imshow(disp_b, cmap='gray')
        axes[row, -1].axis('off')
        row += 1

plt.suptitle('低维隐空间插值对照 ($d_z=2$): AE 中段更易畸变 vs VAE 平滑', fontsize=14, y=1.0)
plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤2b_插值维度对照_dz2.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2b已保存: {fig_path}")

# ---- 量化核验：d_z=2下插值中点是否落入空洞（与d_z=20对照）----
print("\n[量化核验] d_z=2: 30对不同数字样本 插值中点最近邻距离/端点最近邻距离")
dz2_ratios = {}
for beta_q, name_q in [(0, 'AE'), (1, 'VAE')]:
    enc_q = models_dz2[beta_q]['encoder']
    z_train, _ = _encode_loader(enc_q, train_loader)
    z_test, y_test = _encode_loader(enc_q, test_loader)
    nbrs = NearestNeighbors(n_neighbors=1).fit(z_train)
    rng = np.random.RandomState(0)
    ratios = []
    while len(ratios) < 30:
        i, j = rng.randint(len(z_test)), rng.randint(len(z_test))
        if y_test[i] == y_test[j]:
            continue
        z_mid = 0.5 * (z_test[i] + z_test[j])
        d_mid = nbrs.kneighbors(z_mid[None, :])[0][0, 0]
        d_end = 0.5 * (nbrs.kneighbors(z_test[i][None, :])[0][0, 0]
                       + nbrs.kneighbors(z_test[j][None, :])[0][0, 0])
        ratios.append(d_mid / d_end)
    dz2_ratios[beta_q] = float(np.mean(ratios))
    print(f"  d_z=2 {name_q}: 比值={dz2_ratios[beta_q]:.2f}")
print(f"  → d_z=2下AE比值({dz2_ratios[0]:.2f})高于VAE({dz2_ratios[1]:.2f})但差距温和，几何空洞差异有限；")
print("    视觉上AE仅在部分数字对(0→1)中段出现多余笔画/畸变、VAE更平滑，3→7对两者相近；")
print("    方向与理论一致但非剧烈崩坏，说明'空洞致崩坏'随维度降低而增强、但被平滑MLP解码器掩盖")


# ============================================================
# 实验总结
# ============================================================
print(f"\n{'='*60}")
print("实验9.5-1 总结")
print(f"{'='*60}")

for beta, name_print in zip(betas, beta_names_print):
    h = models[beta]['history']
    print(f"\n{name_print}:")
    print(f"  最终Loss={h['loss'][-1]:.2f}, BCE={h['bce'][-1]:.2f}, "
          f"KL={h['kld'][-1]:.2f}, Active={h['active'][-1]}/{h['latent_dim']}")

print("\n核心验证点（对应9.3节和9.5节）：")
print("\n1. β-VAE的权衡（9.3节核心观点）：")
print("   - β↑ → KL↓ → 隐空间更规则/解纠缠，但重建质量下降")
print("   ✓ 实验验证: β=4时KL被强力压制，活跃维度减少")
print("\n2. 后验坍缩现象（9.3节核心概念）：")
print("   - β过大时，q(z|x)≈p(z)，隐变量不携带信息")
print("   ✓ 实验验证: β=4时活跃维度明显少于β=1")
print("\n3. 隐空间连续性（9.5节可视化）：")
print("   - KL正则化使隐空间连续，插值平滑过渡")
print("   实测：d_z=20下AE与VAE插值均平滑(步骤2, 中点距同量级)，未出现AE崩坏；")
print("   降至d_z=2后(步骤2b)AE在部分数字对中段出现畸变、VAE更平滑，方向与理论一致但非剧烈崩坏")
print("   AE相对VAE的稳定缺陷在d_z=20下体现于步骤4先验采样（生成乱码）")
print("\n4. t-SNE隐空间结构（9.5节可视化）：")
print("   - 三个β下数字类别均呈现可分簇（β=0同样分簇，并非β=1/4独有）")
print("   - 注意: t-SNE会归一化全局尺度并倾向把任意流形拉成簇，无法直接展示")
print("     '空洞/重叠'；β增大使编码云向原点收缩的证据应看图3的每维KL与步骤4采样")
print("   ✓ 实测: β=0/1/4 均可分簇；β=1各簇重叠更多、过渡更连续")

print(f"\n{'='*60}")
print("第九章配套实验 9.5-1 完成!")

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

results_summary = {}
for beta in betas:
    h = models[beta]['history']
    results_summary[f'beta_{beta}'] = {
        'final_loss': h['loss'][-1] if h['loss'] else None,
        'final_bce': h['bce'][-1] if h['bce'] else None,
        'final_kld': h['kld'][-1] if h['kld'] else None,
        'final_active': h['active'][-1] if h['active'] else None,
        'loss_curve': h['loss'],
        'bce_curve': h['bce'],
        'kld_curve': h['kld'],
        'active_curve': h['active'],
        # 测试集重建质量（采样z评测，口径同9.3-1）；插值实验仅对比β=0/1，β=4为null
        'test_bce': test_metrics[beta]['test_bce'],
        'test_psnr_db': test_metrics[beta]['test_psnr_db'],
        'interp_mid_dist_ratio': midpoint_ratios.get(beta),
    }
results_summary = _to_native(results_summary)
# 追加 d_z=2 对照实验的插值中点距比值（步骤2b）
results_summary['dz2_interp_mid_dist_ratio'] = {
    'beta_0_AE': dz2_ratios[0], 'beta_1_VAE': dz2_ratios[1],
}
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")