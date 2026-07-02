# -*- coding: utf-8 -*-
"""
实验10.1-1 层级VAE实现与ELBO验证
对应章节: 10.1 从VAE到层级VAE

知识点:
  - 马尔可夫推断链: x → z₁ → z₂
  - 马尔可夫生成链: z₂ → z₁ → x
  - 层级ELBO逐层KL分解: ELBO = 重建项 - KL₁ - KL₂
  - 第一层编码更多信息: KL₁ > KL₂

实验内容:
  步骤1: 层级VAE训练曲线对比
  步骤2: 重建与采样对比 (层级VAE vs 单层VAE)
  步骤3: 层级ELBO逐项验证

本实验需要GPU加速训练MNIST, CPU也可运行但较慢。
"""

import sys
import io
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import logging
import warnings

# 设置控制台输出为 UTF-8 (Windows下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# 静默matplotlib相关警告
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
    SAVE_DIR = os.path.join(_gdrive, '实验10.1-1')
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
    print("警告: chinese_font模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\n" + "="*60)
print("实验10.1-1: 层级VAE实现与ELBO验证")
print("="*60)
print("对应章节: 10.1 从VAE到层级VAE")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  未检测到GPU, 使用CPU训练")
    print("  提示: Colab用户可在菜单 运行时->更改运行时类型 中选择GPU")

# Checkpoint路径
HVAE_CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'hvae_mnist_checkpoint.pth')
SVAE_CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'svae_mnist_checkpoint.pth')

# 快速验证模式
_QUICK_TEST = os.environ.get('QUICK_TEST', '') == '1'

# ============================================================
# 层级VAE模型定义（对应10.1节）
# ============================================================
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class EncoderL1(nn.Module):
    """第一层编码器: x → (μ₁, logσ₁²)
    对应10.1节: q(z₁|x)
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)


class EncoderL2(nn.Module):
    """第二层编码器: z₁ → (μ₂, logσ₂²)
    对应10.1节: q(z₂|z₁)——马尔可夫推断链的第二步
    """
    def __init__(self, latent1_dim=20, hidden_dim=200, latent2_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(latent1_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent2_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent2_dim)

    def forward(self, z1):
        h = F.relu(self.fc1(z1))
        return self.fc_mu(h), self.fc_logvar(h)


class DecoderL1(nn.Module):
    """第一层解码器: z₂ → z₁̂
    对应10.1节: p(z₁|z₂)——马尔可夫生成链的第一步
    """
    def __init__(self, latent2_dim=10, hidden_dim=200, latent1_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(latent2_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent1_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent1_dim)

    def forward(self, z2):
        h = F.relu(self.fc1(z2))
        return self.fc_mu(h), self.fc_logvar(h)


class DecoderL0(nn.Module):
    """第零层解码器: z₁ → x̂
    对应10.1节: p(x|z₁)——伯努利似然
    """
    def __init__(self, latent1_dim=20, hidden_dim=400, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent1_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z1):
        h = F.relu(self.fc1(z1))
        return torch.sigmoid(self.fc2(h))


def reparameterize(mu, logvar):
    """重参数化技巧: z = μ + σε, ε ~ N(0,I)"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


def kl_divergence(mu, logvar):
    """高斯KL: D_KL(q(z|x) || N(0,I)) = 0.5 * Σ(μ² + σ² - logσ² - 1)"""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def hierarchical_elbo(x, mu1, logvar1, mu2, logvar2, x_recon, beta=1.0):
    """层级ELBO简化版（对应10.1节）:
    
    ⚠️ 注意：这是教学简化版，并非完整的层级VAE ELBO！
    
    严格层级ELBO应为:
      ELBO = E[log p(x|z₁)] - KL(q(z₁|x) || p(z₁|z₂)) - KL(q(z₂|z₁) || p(z₂))
    
    其中KL₁需要对生成分布p(z₁|z₂)计算（需积分z₂），实现复杂。
    本函数采用平均场近似，将KL₁简化为对标准正态先验N(0,I):
      KL₁ ≈ KL(q(z₁|x) || N(0,I))
      KL₂ = KL(q(z₂|z₁) || N(0,I))
    
    这与单层VAE的KL计算方式相同，但层级结构本身仍有优势：
    - 信息分散到两层瓶颈，每层KL约束更温和
    - 避免单层VAE的"过度压缩"问题
    """
    # 重建项: log p(x|z₁)
    recon = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL₁: 简化为 KL(q(z₁|x) || N(0,I))
    kl1 = kl_divergence(mu1, logvar1)
    
    # KL₂: KL(q(z₂|z₁) || N(0,I))
    kl2 = kl_divergence(mu2, logvar2)
    
    total = recon + beta * (kl1 + kl2)
    return total, recon, kl1, kl2


# ============================================================
# 训练函数（含checkpoint resume）
# ============================================================
def train_hierarchical_vae(checkpoint_path, train_loader, num_epochs=20, 
                           d1=20, d2=10, beta=1.0):
    """训练层级VAE，支持checkpoint resume"""
    
    enc1 = EncoderL1(latent_dim=d1).to(device)
    enc2 = EncoderL2(latent1_dim=d1, latent2_dim=d2).to(device)
    dec1 = DecoderL1(latent2_dim=d2, latent1_dim=d1).to(device)
    dec0 = DecoderL0(latent1_dim=d1).to(device)
    
    optimizer = torch.optim.Adam(
        list(enc1.parameters()) + list(enc2.parameters()) +
        list(dec1.parameters()) + list(dec0.parameters()),
        lr=1e-3
    )
    
    start_epoch = 0
    is_final = False
    history = {'loss': [], 'recon': [], 'kl1': [], 'kl2': [], 'kl_total': []}
    
    # Checkpoint加载逻辑
    if os.path.exists(checkpoint_path):
        print(f"\n检测到已保存的模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if checkpoint.get('is_final', False):
            print(f"✓ 这是最终训练完成的模型, 直接加载, 跳过训练过程")
            print(f"  训练轮数: {checkpoint['epoch']+1}")
            print(f"  最终损失: {checkpoint['loss']:.6f}")
            try:
                enc1.load_state_dict(checkpoint['enc1_state_dict'])
                enc2.load_state_dict(checkpoint['enc2_state_dict'])
                dec1.load_state_dict(checkpoint['dec1_state_dict'])
                dec0.load_state_dict(checkpoint['dec0_state_dict'])
            except RuntimeError as e:
                print(f"警告: checkpoint与当前模型架构不兼容, 删除后重新训练")
                print(f"  错误信息: {e}")
                os.remove(checkpoint_path)
                start_epoch = 0
                is_final = False
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                history = checkpoint.get('history', {'loss': [], 'recon': [], 'kl1': [], 'kl2': [], 'kl_total': []})
                start_epoch = checkpoint['epoch'] + 1
                is_final = True
        else:
            print(f"检测到未完成的训练, 从第 {checkpoint['epoch']+1} 轮继续")
            try:
                enc1.load_state_dict(checkpoint['enc1_state_dict'])
                enc2.load_state_dict(checkpoint['enc2_state_dict'])
                dec1.load_state_dict(checkpoint['dec1_state_dict'])
                dec0.load_state_dict(checkpoint['dec0_state_dict'])
            except RuntimeError as e:
                print(f"警告: checkpoint与当前模型架构不兼容, 删除后重新训练")
                print(f"  错误信息: {e}")
                os.remove(checkpoint_path)
                start_epoch = 0
                is_final = False
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                history = checkpoint.get('history', {'loss': [], 'recon': [], 'kl1': [], 'kl2': [], 'kl_total': []})
                start_epoch = checkpoint['epoch'] + 1
    
    # 训练循环
    if not is_final:
        # 快速验证模式
        if _QUICK_TEST:
            num_epochs = 3
            print(f"\n[快速验证模式] 仅训练 {num_epochs} 轮")
        
        print(f"\n开始训练层级VAE (L=2, d₁={d1}, d₂={d2}, epochs={num_epochs})...")
        print("-" * 75)
        
        # 边界保护
        if start_epoch >= num_epochs:
            print(f"  注意: start_epoch({start_epoch}) >= num_epochs({num_epochs}), 无需继续训练")
            is_final = True
        
        if not is_final:
            import time
            from tqdm import tqdm
            t_start = time.time()
            
            for epoch in range(start_epoch, num_epochs):
                enc1.train(); enc2.train(); dec1.train(); dec0.train()
                total_loss, total_recon, total_kl1, total_kl2 = 0, 0, 0, 0
                
                pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', 
                            leave=False, unit='batch')
                
                for x, _ in pbar:
                    x = x.view(-1, 784).to(device)
                    
                    # 编码链（马尔可夫推断链）: x → z₁ → z₂
                    mu1, logvar1 = enc1(x)
                    z1 = reparameterize(mu1, logvar1)
                    mu2, logvar2 = enc2(z1)
                    z2 = reparameterize(mu2, logvar2)
                    
                    # 生成链（马尔可夫生成链）: z₂ → z₁ → x
                    mu1_dec, logvar1_dec = dec1(z2)
                    z1_dec = reparameterize(mu1_dec, logvar1_dec)
                    x_recon = dec0(z1_dec)
                    
                    # 层级ELBO（简化版）
                    loss, recon, kl1, kl2 = hierarchical_elbo(
                        x, mu1, logvar1, mu2, logvar2, x_recon, beta
                    )
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    total_recon += recon.item()
                    total_kl1 += kl1.item()
                    total_kl2 += kl2.item()
                
                n = len(train_loader.dataset)
                avg_loss = total_loss / n
                avg_recon = total_recon / n
                avg_kl1 = total_kl1 / n
                avg_kl2 = total_kl2 / n
                
                history['loss'].append(avg_loss)
                history['recon'].append(avg_recon)
                history['kl1'].append(avg_kl1)
                history['kl2'].append(avg_kl2)
                history['kl_total'].append(avg_kl1 + avg_kl2)
                
                print(f"Epoch {epoch+1:3d}/{num_epochs}  "
                      f"Loss={avg_loss:.2f}  Recon={avg_recon:.2f}  "
                      f"KL₁={avg_kl1:.2f}  KL₂={avg_kl2:.2f}  "
                      f"KL_total={avg_kl1+avg_kl2:.2f}")
                
                # 每轮保存checkpoint
                torch.save({
                    'epoch': epoch,
                    'd1': d1, 'd2': d2, 'beta': beta,
                    'enc1_state_dict': enc1.state_dict(),
                    'enc2_state_dict': enc2.state_dict(),
                    'dec1_state_dict': dec1.state_dict(),
                    'dec0_state_dict': dec0.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'history': history,
                    'is_final': False
                }, checkpoint_path)
            
            t_elapsed = time.time() - t_start
            print("-" * 75)
            print(f"训练完成, 最终Loss={history['loss'][-1]:.2f}, 耗时: {t_elapsed:.1f}秒")
            
            # 保存最终checkpoint
            torch.save({
                'epoch': num_epochs - 1,
                'd1': d1, 'd2': d2, 'beta': beta,
                'enc1_state_dict': enc1.state_dict(),
                'enc2_state_dict': enc2.state_dict(),
                'dec1_state_dict': dec1.state_dict(),
                'dec0_state_dict': dec0.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': history['loss'][-1],
                'history': history,
                'is_final': True
            }, checkpoint_path)
            print(f"✓ 模型已保存: {checkpoint_path}")
    else:
        print(f"\n使用已训练完成的层级VAE模型, 跳过训练过程")
    
    return enc1, enc2, dec1, dec0, history


def train_single_vae(checkpoint_path, train_loader, num_epochs=20, d_z=20, beta=1.0):
    """训练单层VAE，支持checkpoint resume"""
    
    enc_single = EncoderL1(latent_dim=d_z).to(device)
    dec_single = DecoderL0(latent1_dim=d_z).to(device)
    
    optimizer = torch.optim.Adam(
        list(enc_single.parameters()) + list(dec_single.parameters()), lr=1e-3
    )
    
    start_epoch = 0
    is_final = False
    history_single = {'loss': [], 'recon': [], 'kl': []}
    
    # Checkpoint加载逻辑
    if os.path.exists(checkpoint_path):
        print(f"\n检测到已保存的模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if checkpoint.get('is_final', False):
            print(f"✓ 这是最终训练完成的模型, 直接加载, 跳过训练过程")
            print(f"  训练轮数: {checkpoint['epoch']+1}")
            print(f"  最终损失: {checkpoint['loss']:.6f}")
            try:
                enc_single.load_state_dict(checkpoint['encoder_state_dict'])
                dec_single.load_state_dict(checkpoint['decoder_state_dict'])
            except RuntimeError as e:
                print(f"警告: checkpoint与当前模型架构不兼容, 删除后重新训练")
                print(f"  错误信息: {e}")
                os.remove(checkpoint_path)
                start_epoch = 0
                is_final = False
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                history_single = checkpoint.get('history', {'loss': [], 'recon': [], 'kl': []})
                start_epoch = checkpoint['epoch'] + 1
                is_final = True
        else:
            print(f"检测到未完成的训练, 从第 {checkpoint['epoch']+1} 轮继续")
            try:
                enc_single.load_state_dict(checkpoint['encoder_state_dict'])
                dec_single.load_state_dict(checkpoint['decoder_state_dict'])
            except RuntimeError as e:
                print(f"警告: checkpoint与当前模型架构不兼容, 删除后重新训练")
                print(f"  错误信息: {e}")
                os.remove(checkpoint_path)
                start_epoch = 0
                is_final = False
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                history_single = checkpoint.get('history', {'loss': [], 'recon': [], 'kl': []})
                start_epoch = checkpoint['epoch'] + 1
    
    # 训练循环
    if not is_final:
        # 快速验证模式
        if _QUICK_TEST:
            num_epochs = 3
            print(f"\n[快速验证模式] 仅训练 {num_epochs} 轮")
        
        print(f"\n开始训练单层VAE (d_z={d_z}, epochs={num_epochs})...")
        print("-" * 60)
        
        # 边界保护
        if start_epoch >= num_epochs:
            print(f"  注意: start_epoch({start_epoch}) >= num_epochs({num_epochs}), 无需继续训练")
            is_final = True
        
        if not is_final:
            import time
            from tqdm import tqdm
            t_start = time.time()
            
            for epoch in range(start_epoch, num_epochs):
                enc_single.train(); dec_single.train()
                total_loss, total_recon, total_kl = 0, 0, 0
                
                pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', 
                            leave=False, unit='batch')
                
                for x, _ in pbar:
                    x = x.view(-1, 784).to(device)
                    mu, logvar = enc_single(x)
                    z = reparameterize(mu, logvar)
                    x_recon = dec_single(z)
                    
                    recon = F.binary_cross_entropy(x_recon, x, reduction='sum')
                    kl = kl_divergence(mu, logvar)
                    loss = recon + beta * kl
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    total_recon += recon.item()
                    total_kl += kl.item()
                
                n = len(train_loader.dataset)
                history_single['loss'].append(total_loss / n)
                history_single['recon'].append(total_recon / n)
                history_single['kl'].append(total_kl / n)
                
                print(f"Epoch {epoch+1:3d}/{num_epochs}  "
                      f"Loss={history_single['loss'][-1]:.2f}  "
                      f"Recon={history_single['recon'][-1]:.2f}  "
                      f"KL={history_single['kl'][-1]:.2f}")
                
                # 每轮保存checkpoint
                torch.save({
                    'epoch': epoch,
                    'd_z': d_z, 'beta': beta,
                    'encoder_state_dict': enc_single.state_dict(),
                    'decoder_state_dict': dec_single.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': history_single['loss'][-1],
                    'history': history_single,
                    'is_final': False
                }, checkpoint_path)
            
            t_elapsed = time.time() - t_start
            print("-" * 60)
            print(f"训练完成, 最终Loss={history_single['loss'][-1]:.2f}, 耗时: {t_elapsed:.1f}秒")
            
            # 保存最终checkpoint
            torch.save({
                'epoch': num_epochs - 1,
                'd_z': d_z, 'beta': beta,
                'encoder_state_dict': enc_single.state_dict(),
                'decoder_state_dict': dec_single.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': history_single['loss'][-1],
                'history': history_single,
                'is_final': True
            }, checkpoint_path)
            print(f"✓ 模型已保存: {checkpoint_path}")
    else:
        print(f"\n使用已训练完成的单层VAE模型, 跳过训练过程")
    
    return enc_single, dec_single, history_single


# ============================================================
# 步骤1: 层级VAE与单层VAE训练
# ============================================================
print("\n" + "="*60)
print("步骤1: 层级VAE与单层VAE训练")
print("="*60)
print("\n[核心思想]")
print("  层级VAE通过马尔可夫推断链扩展了单层VAE:")
print("  - 编码链: x → z₁ → z₂ (两层瓶颈逐级压缩)")
print("  - 生成链: z₂ → z₁ → x (两层解码逐级恢复)")
print("  - 层级ELBO: ELBO = 重建项 - KL₁ - KL₂")
print("  - KL₁ > KL₂: 第一层(靠近数据)编码更多信息")

# 加载MNIST数据集
print("\n加载MNIST数据集...")
data_dir = os.path.join(SAVE_DIR, 'data')
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
print(f"训练集: {len(train_dataset)} 样本, 测试集: {len(test_dataset)} 样本")

# 训练参数
d1, d2 = 20, 10  # 隐空间维度: z₁=20, z₂=10
beta = 1.0
num_epochs = 20

# 训练层级VAE
print("\n" + "="*60)
print("训练层级VAE (L=2)...")
print("="*60)
enc1, enc2, dec1, dec0, history = train_hierarchical_vae(
    HVAE_CHECKPOINT_PATH, train_loader, num_epochs, d1, d2, beta
)

# 训练单层VAE对比
print("\n" + "="*60)
print("训练单层VAE (L=1)...")
print("="*60)
enc_single, dec_single, history_single = train_single_vae(
    SVAE_CHECKPOINT_PATH, train_loader, num_epochs, d1, beta
)


# ============================================================
# 可视化1: 训练曲线对比 + KL分解
# ============================================================
print("\n生成训练曲线对比图...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
epochs_range = range(1, len(history['loss']) + 1)

# (a) 总损失对比
ax = axes[0, 0]
ax.plot(epochs_range, history['loss'], 'b-o', markersize=4, label='层级VAE (L=2)')
ax.plot(epochs_range, history_single['loss'], 'r-s', markersize=4, label='单层VAE (L=1)')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('ELBO损失', fontsize=12)
ax.set_title('(a) 总损失对比', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# (b) 重建损失对比
ax = axes[0, 1]
ax.plot(epochs_range, history['recon'], 'b-o', markersize=4, label='层级VAE')
ax.plot(epochs_range, history_single['recon'], 'r-s', markersize=4, label='单层VAE')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('重建损失 (BCE)', fontsize=12)
ax.set_title('(b) 重建损失对比', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# (c) KL分解（层级VAE特有）—— 使用LaTeX格式
ax = axes[1, 0]
ax.plot(epochs_range, history['kl1'], 'g-o', markersize=4, 
        label=r'$KL_1$: $q(z_1|x) \| p(z_1)$')
ax.plot(epochs_range, history['kl2'], 'm-s', markersize=4, 
        label=r'$KL_2$: $q(z_2|z_1) \| p(z_2)$')
ax.plot(epochs_range, history['kl_total'], 'k--^', markersize=4, 
        label=r'$KL_{total} = KL_1 + KL_2$')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('KL散度', fontsize=12)
ax.set_title('(c) 层级VAE的KL逐层分解', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (d) KL总量对比 —— 使用LaTeX格式
ax = axes[1, 1]
ax.plot(epochs_range, history['kl_total'], 'b-o', markersize=4, 
        label=r'层级VAE $KL_{total}$')
ax.plot(epochs_range, history_single['kl'], 'r-s', markersize=4, 
        label=r'单层VAE $KL$')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('KL散度', fontsize=12)
ax.set_title('(d) KL散度总量对比', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤1_训练曲线对比.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图1已保存: {fig_path}")


# ============================================================
# 可视化2: 重建与先验采样
# ============================================================
print("生成重建与采样对比图...")

enc1.eval(); enc2.eval(); dec1.eval(); dec0.eval()
enc_single.eval(); dec_single.eval()

n_show = 10

# 层级VAE重建（使用μ而非随机采样，展示确定性最佳重建）
with torch.no_grad():
    test_imgs, _ = next(iter(test_loader))
    x_flat = test_imgs.view(-1, 784).to(device)
    mu1, logvar1 = enc1(x_flat)
    z1 = mu1  # 使用均值
    mu2, logvar2 = enc2(z1)
    z2 = mu2  # 使用均值
    mu1_dec, logvar1_dec = dec1(z2)
    z1_dec = mu1_dec  # 使用均值
    recon_h = dec0(z1_dec)

# 单层VAE重建
with torch.no_grad():
    mu_s, logvar_s = enc_single(x_flat)
    z_s = mu_s  # 使用均值
    recon_s = dec_single(z_s)

# 层级VAE先验采样: z₂~N(0,I) → z₁ → x
with torch.no_grad():
    z2_prior = torch.randn(n_show, d2).to(device)
    mu1_p, logvar1_p = dec1(z2_prior)
    z1_prior = reparameterize(mu1_p, logvar1_p)
    samples_h = dec0(z1_prior)

# 单层VAE先验采样: z~N(0,I) → x
with torch.no_grad():
    z_prior = torch.randn(n_show, d1).to(device)
    samples_s = dec_single(z_prior)

fig, axes = plt.subplots(4, n_show, figsize=(20, 8))

for i in range(n_show):
    # 原始
    axes[0, i].imshow(test_imgs[i, 0].numpy(), cmap='gray')
    axes[0, i].axis('off')
    if i == 0: axes[0, i].set_title('原始', fontsize=12)

    # 层级VAE重建
    axes[1, i].imshow(recon_h[i].view(28, 28).cpu().numpy(), cmap='gray')
    axes[1, i].axis('off')
    if i == 0: axes[1, i].set_title('层级VAE重建', fontsize=12)

    # 单层VAE重建
    axes[2, i].imshow(recon_s[i].view(28, 28).cpu().numpy(), cmap='gray')
    axes[2, i].axis('off')
    if i == 0: axes[2, i].set_title('单层VAE重建', fontsize=12)

    # 层级VAE先验采样
    axes[3, i].imshow(samples_h[i].view(28, 28).cpu().numpy(), cmap='gray')
    axes[3, i].axis('off')
    if i == 0: axes[3, i].set_title('层级VAE采样', fontsize=12)

# 使用LaTeX格式的suptitle
plt.suptitle(r'层级VAE (L=2, $d_1$=20, $d_2$=10) vs 单层VAE ($d_z$=20)',
             fontsize=14, y=1.02)
plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤2_重建与采样对比.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig_path}")

print("\n[单层VAE局限 vs 层级VAE优势]")
print("  从重建与采样对比可观察到:")
print("  - 信息瓶颈: 单层VAE一步压缩所有信息, 高维数据难以保持细节")
print("              层级VAE分两步压缩(x→z₁→z₂), 每层只需处理'增量信息'")
print("  - KL两难:   强KL约束→重建模糊; 弱KL约束→先验采样无意义")
print("              层级结构提供了更灵活的先验建模能力")
print("  - 生成质量天花板: 单步解码跨距太大(简单z→复杂x)")
print("                    层级解码逐级恢复(z₂→z₁→x), 每步跨距更小")

# ============================================================
# 步骤3: 层级ELBO逐项验证
# ============================================================
print("\n" + "="*60)
print("步骤3: 层级ELBO逐项验证")
print("="*60)

enc1.eval(); enc2.eval(); dec1.eval(); dec0.eval()

# 对测试集计算各ELBO分量
total_recon, total_kl1, total_kl2 = 0, 0, 0
n_samples = 0

with torch.no_grad():
    for x, _ in test_loader:
        x = x.view(-1, 784).to(device)
        batch = x.shape[0]

        mu1, logvar1 = enc1(x)
        z1 = reparameterize(mu1, logvar1)
        mu2, logvar2 = enc2(z1)
        z2 = reparameterize(mu2, logvar2)

        mu1_dec, logvar1_dec = dec1(z2)
        z1_dec = reparameterize(mu1_dec, logvar1_dec)
        x_recon = dec0(z1_dec)

        recon = F.binary_cross_entropy(x_recon, x, reduction='sum')
        kl1 = kl_divergence(mu1, logvar1)
        kl2 = kl_divergence(mu2, logvar2)

        total_recon += recon.item()
        total_kl1 += kl1.item()
        total_kl2 += kl2.item()
        n_samples += batch

avg_recon = total_recon / n_samples
avg_kl1 = total_kl1 / n_samples
avg_kl2 = total_kl2 / n_samples
elbo = -(avg_recon + avg_kl1 + avg_kl2)

print(f"\n测试集层级ELBO逐项分解:")
print(f"  重建项  -E[log p(x|z₁)]    = {avg_recon:.2f}")
print(f"  KL₁:    KL(q(z₁|x)||p(z₁)) = {avg_kl1:.2f}")
print(f"  KL₂:    KL(q(z₂|z₁)||p(z₂)) = {avg_kl2:.2f}")
print(f"  ELBO = -(重建 + KL₁ + KL₂)  = {elbo:.2f}")
print(f"\n  KL₁/KL₂ = {avg_kl1/avg_kl2:.2f} → 第一层编码更多信息")

# 与单层VAE对比
enc_single.eval(); dec_single.eval()
total_recon_s, total_kl_s, n_samples_s = 0, 0, 0
with torch.no_grad():
    for x, _ in test_loader:
        x = x.view(-1, 784).to(device)
        batch = x.shape[0]
        mu, logvar = enc_single(x)
        z = reparameterize(mu, logvar)
        x_recon = dec_single(z)
        recon = F.binary_cross_entropy(x_recon, x, reduction='sum')
        kl = kl_divergence(mu, logvar)
        total_recon_s += recon.item()
        total_kl_s += kl.item()
        n_samples_s += batch

avg_recon_s = total_recon_s / n_samples_s
avg_kl_s = total_kl_s / n_samples_s
elbo_s = -(avg_recon_s + avg_kl_s)

print(f"\n对比:")
print(f"  {'':20s} {'层级VAE':>10s} {'单层VAE':>10s}")
print(f"  {'重建项':20s} {avg_recon:>10.2f} {avg_recon_s:>10.2f}")
print(f"  {'KL总量':20s} {avg_kl1+avg_kl2:>10.2f} {avg_kl_s:>10.2f}")
print(f"  {'ELBO':20s} {elbo:>10.2f} {elbo_s:>10.2f}")

print(f"\n[观察: 层级VAE的KL总量({avg_kl1+avg_kl2:.2f})小于单层VAE({avg_kl_s:.2f})]")
print("  可能原因:")
print("  - 层级结构分散了KL压力到两层瓶颈, 每层约束更温和")
print("  - 总维度差异(d₁+d₂=30 vs d_z=20)及多一层非线性变换")
print("  注: 本实验未控制总维度进行消融, 该观察与'层级缓解KL两难'的直觉一致,")
print("      但严格验证需进一步实验")


# ============================================================
# 总结
# ============================================================
print("\n" + "="*60)
print("实验10.1-1 完成!")
print("="*60)
print("""
关键观察:
1. 马尔可夫推断链 x→z₁→z₂ 的实现（10.1节）
   - 两层编码器分别输出 (μ₁,logσ₁²) 和 (μ₂,logσ₂²)
   - 重参数化在每个层级独立执行

2. 层级ELBO的逐层KL分解（10.1节核心公式）
   - ELBO = 重建项 - KL₁ - KL₂
   - KL₁ > KL₂: 第一层（靠近数据）编码更多信息
   - 第二层（更抽象）编码更高层特征

3. 层级VAE vs 单层VAE
   - 层级VAE的重建损失可能略高（信息需经过两层瓶颈）
   - 但层级结构提供了更灵活的先验建模能力
   - 当L→∞时，这就是扩散模型（10.4节）
""")