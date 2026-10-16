# -*- coding: utf-8 -*-
"""
实验10.1-1 层级VAE实现与ELBO验证
对应章节: 10.1 从VAE到层级VAE

知识点:
  - 马尔可夫推断链: x → z₁ → z₂
  - 马尔可夫生成链: z₂ → z₁ → x
  - 层级ELBO逐层KL分解: -ELBO = 重建项 + KL₁ + KL₂
    其中 KL₁ = KL(q(z₁|x)||p(z₁|z₂)), KL₂ = KL(q(z₂|z₁)||p(z₂))

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


def kl_divergence_gauss(mu_q, logvar_q, mu_p, logvar_p):
    """两个对角高斯之间的KL: D_KL(N(μ_q,σ_q²) || N(μ_p,σ_p²))
    = 0.5 * Σ(σ_q²/σ_p² + (μ_q-μ_p)²/σ_p² - 1 - log(σ_q²/σ_p²))
    """
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    return 0.5 * torch.sum(var_q / var_p + (mu_q - mu_p).pow(2) / var_p - 1.0
                           - logvar_q + logvar_p)


def hierarchical_elbo(x, mu1, logvar1, mu2, logvar2, x_recon,
                      mu1_dec, logvar1_dec, beta=1.0):
    """层级ELBO逐项分解（对应10.1节框式公式在L=2时的展开）:

      -ELBO = E_q[-log p(x|z₁)] + KL(q(z₁|x) || p(z₁|z₂)) + KL(q(z₂|z₁) || p(z₂))

    三个分量均可直接计算:
    - 重建项: x_recon 由 z₁ ~ q(z₁|x) 经 dec0 得到（伯努利负对数似然）
    - KL₁: q(z₁|x) 与生成分布 p(z₁|z₂)（dec1输出）均为对角高斯，解析计算；
      对 z₂ ~ q(z₂|z₁) 取期望即10.1节框式中的 E_{q(z₁,z₂|x)}[KL(...)]
    - KL₂: q(z₂|z₁) 对标准高斯先验 p(z₂)，解析计算

    注: 10.1"实验场景"小节把KL₁写成对边际聚合先验 p(z₁)=∫p(z₁|z₂)p(z₂)dz₂
    的形式，那是框式公式的另一种等价写法（Jensen后取边际）；本实现用
    条件形式 p(z₁|z₂)，与附录10A的框式主公式一致。

    ⚠️ 历史版本曾将KL₁简化为 KL(q(z₁|x)||N(0,I))并让重建走
    z₂→ẑ₁→x 路径，导致编码链 q(z₁|x) 与生成链 dec1 在损失中完全脱钩；
    后续版本虽改回条件形式，但 kl_divergence_gauss 的均值项漏除了 var_p，
    使得 σp²→0 时 KL₁ 被大幅低估，模型退化为"z₁无噪直通编码 + 先验
    采样模式崩溃"的作弊解（表现为 recon 虚低、报告KL₁≈1、KL₂≈0）。
    现已修正（均值项除以 var_p）。修正后信息必须经 z₂ 传递（付出KL₂），
    或如实计入 z₁ 通道的真实KL代价。
    """
    # 重建项: -E[log p(x|z₁)], z₁ ~ q(z₁|x)
    recon = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL₁: KL(q(z₁|x) || p(z₁|z₂)) —— 对角高斯解析式
    kl1 = kl_divergence_gauss(mu1, logvar1, mu1_dec, logvar1_dec)

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

                    # 生成分布 p(z₁|z₂): 用独立噪声重采一个 z₂'（保持无偏估计）
                    z2_prime = mu2 + torch.exp(0.5 * logvar2) * torch.randn_like(mu2)
                    mu1_dec, logvar1_dec = dec1(z2_prime)

                    # 重建路径: z₁ ~ q(z₁|x) → x̂ (对应 E[log p(x|z₁)])
                    x_recon = dec0(z1)

                    # 层级ELBO（严格形式）
                    loss, recon, kl1, kl2 = hierarchical_elbo(
                        x, mu1, logvar1, mu2, logvar2, x_recon,
                        mu1_dec, logvar1_dec, beta
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
print("  - 层级ELBO: -ELBO = 重建项 + KL₁ + KL₂")
print("  - 预期: KL₁ > KL₂ (第一层靠近数据, 编码更多信息) —— 由实测验证")

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
        label=r'$KL_1$: $q(z_1|x) \| p(z_1|z_2)$')
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
    recon_h = dec0(z1)

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

# 全测试集重建PSNR（均值路径，定量衡量重建质量）
with torch.no_grad():
    psnr_h_list, psnr_s_list = [], []
    for x_t, _ in test_loader:
        x_t = x_t.view(-1, 784).to(device)
        r_h = dec0(enc1(x_t)[0]).clamp(1e-6, 1 - 1e-6)
        r_s = dec_single(enc_single(x_t)[0]).clamp(1e-6, 1 - 1e-6)
        mse_h = ((r_h - x_t) ** 2).mean(dim=1)
        mse_s = ((r_s - x_t) ** 2).mean(dim=1)
        psnr_h_list.append((-10 * torch.log10(mse_h + 1e-12)).cpu().numpy())
        psnr_s_list.append((-10 * torch.log10(mse_s + 1e-12)).cpu().numpy())
    psnr_h = float(np.concatenate(psnr_h_list).mean())
    psnr_s = float(np.concatenate(psnr_s_list).mean())
print(f"全测试集重建PSNR: 层级VAE={psnr_h:.2f} dB, 单层VAE={psnr_s:.2f} dB")

# 先验采样多样性: 200个先验样本的平均成对L1距离/像素 (量化模式崩溃)
with torch.no_grad():
    n_div = 200
    z2d = torch.randn(n_div, d2).to(device)
    m1d, l1d = dec1(z2d)
    div_h = dec0(reparameterize(m1d, l1d))
    div_s = dec_single(torch.randn(n_div, d1).to(device))
    pd_h = torch.cdist(div_h, div_h, p=1).mean().item() / 784 / (n_div - 1) * n_div
    pd_s = torch.cdist(div_s, div_s, p=1).mean().item() / 784 / (n_div - 1) * n_div
print(f"先验采样多样性(平均成对L1/像素): 层级={pd_h:.5f}, 单层={pd_s:.5f}"
      + ("  ← 层级采样模式崩溃" if pd_h < 0.3 * pd_s else ""))

fig, axes = plt.subplots(5, n_show, figsize=(20, 10))

for i in range(n_show):
    # 原始
    axes[0, i].imshow(test_imgs[i, 0].numpy(), cmap='gray')
    axes[0, i].axis('off')
    if i == 0: axes[0, i].set_title('原始', fontsize=12)

    # 层级VAE重建
    axes[1, i].imshow(recon_h[i].view(28, 28).cpu().numpy(), cmap='gray')
    axes[1, i].axis('off')
    if i == 0: axes[1, i].set_title(f'层级VAE重建 (PSNR {psnr_h:.1f}dB)', fontsize=12)

    # 单层VAE重建
    axes[2, i].imshow(recon_s[i].view(28, 28).cpu().numpy(), cmap='gray')
    axes[2, i].axis('off')
    if i == 0: axes[2, i].set_title(f'单层VAE重建 (PSNR {psnr_s:.1f}dB)', fontsize=12)

    # 层级VAE先验采样
    axes[3, i].imshow(samples_h[i].view(28, 28).cpu().numpy(), cmap='gray')
    axes[3, i].axis('off')
    if i == 0: axes[3, i].set_title('层级VAE采样', fontsize=12)

    # 单层VAE先验采样
    axes[4, i].imshow(samples_s[i].view(28, 28).cpu().numpy(), cmap='gray')
    axes[4, i].axis('off')
    if i == 0: axes[4, i].set_title('单层VAE采样', fontsize=12)

# 使用LaTeX格式的suptitle
plt.suptitle(r'层级VAE (L=2, $d_1$=20, $d_2$=10) vs 单层VAE ($d_z$=20)',
             fontsize=14, y=1.02)
plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤2_重建与采样对比.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig_path}")

print(f"\n[单层VAE局限 vs 层级VAE优势]")
print("  理论上层级结构的优势（本小配置下需结合PSNR实测谨慎解读）:")
print("  - 信息瓶颈: 层级VAE分两步压缩(x→z₁→z₂), 每层只需处理'增量信息'")
print("  - KL两难:   强KL约束→重建模糊; 弱KL约束→先验采样无意义")
print("              层级结构(聚合先验 p(z₁)=∫p(z₁|z₂)p(z₂)dz₂)提供更灵活的先验建模")
print("  - 生成质量天花板: 层级解码逐级恢复(z₂→z₁→x), 每步跨距更小")
print("  注: 在MNIST+MLP这种低维流形任务上, L=2的层级未必能胜过单层;")
print("      层级优势需更大L(接近扩散)时才充分体现, 见10.4节")

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

        # 与训练同款: p(z₁|z₂)用独立噪声样本z₂'，重建走z₁→x̂
        z2_prime = mu2 + torch.exp(0.5 * logvar2) * torch.randn_like(mu2)
        mu1_dec, logvar1_dec = dec1(z2_prime)
        x_recon = dec0(z1)

        recon = F.binary_cross_entropy(x_recon, x, reduction='sum')
        kl1 = kl_divergence_gauss(mu1, logvar1, mu1_dec, logvar1_dec)
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
print(f"  重建项  -E[log p(x|z₁)]         = {avg_recon:.2f}")
print(f"  KL₁:    KL(q(z₁|x)||p(z₁|z₂))  = {avg_kl1:.2f}")
print(f"  KL₂:    KL(q(z₂|z₁)||p(z₂))    = {avg_kl2:.2f}")
print(f"  ELBO = -(重建 + KL₁ + KL₂)      = {elbo:.2f}")
if avg_kl2 > 0:
    verdict = '成立: 第一层编码更多信息' if avg_kl1 > avg_kl2 else '未成立: 本轮KL₁ ≤ KL₂'
    print(f"\n  KL₁/KL₂ = {avg_kl1/avg_kl2:.2f} → '预期KL₁ > KL₂'{verdict.split(':')[0]}")
    print(f"  实测: {verdict}")

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
print(f"  {'重建PSNR(dB)':20s} {psnr_h:>10.2f} {psnr_s:>10.2f}")

if avg_kl1 + avg_kl2 < avg_kl_s:
    print(f"\n[观察: 层级VAE的KL总量({avg_kl1+avg_kl2:.2f})小于单层VAE({avg_kl_s:.2f})]")
    print("  解读时需注意:")
    print("  - KL小不等于'缓解KL两难': 若z₂(10维)瓶颈限制了通过信息量,")
    print("    KL总量也会因'根本没编码信息'而偏小, 应结合重建PSNR判断")
    print("  - 两模型ELBO可直接比较(均为-log p(x)的下界), ELBO更高者拟合更好")
else:
    print(f"\n[观察: 层级VAE的KL总量({avg_kl1+avg_kl2:.2f})不小于单层VAE({avg_kl_s:.2f})]")
print(f"  ELBO对比: 层级({elbo:.2f}) vs 单层({elbo_s:.2f}) → "
      f"{'单层拟合更优' if elbo_s > elbo else '层级拟合更优'}")
print("  注: 本实验未控制总维度进行消融; 层级VAE的净收益来自"
      "\n      可学习的先验 p(z₁|z₂) 比固定 N(0,I) 更贴合聚合后验。")
# 第二层退化与采样多样性是否发生, 由本次实测判定(而非固有免责结论)
if avg_kl2 < 0.5:
    print(f"      风险已发生: KL₂={avg_kl2:.3f}≈0, 第二层退化, z₂几乎不携带信息")
else:
    print(f"      风险未发生: KL₂={avg_kl2:.2f}(未塌缩), 第二层实际携带了信息")
if pd_h < 0.7 * pd_s:
    print(f"      且先验采样多样性下降(层级{pd_h:.4f} vs 单层{pd_s:.4f}), 见采样行")
else:
    print(f"      先验采样多样性未下降(层级{pd_h:.4f} vs 单层{pd_s:.4f})")


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
   - -ELBO = 重建项 + KL(q(z₁|x)||p(z₁|z₂)) + KL(q(z₂|z₁)||p(z₂))
   - KL₁与KL₂的大小关系由实测决定, 见上方逐项验证输出

3. 层级VAE vs 单层VAE
   - ELBO是可直接比较的下界: 哪个模型ELBO更高, 哪个拟合更好
   - 学到的先验 p(z₁|z₂) 比固定 N(0,I) 拟合更好 → 层级ELBO更高
   - 风险: 若KL₂被压到≈0, 第二层先验退化为无条件单高斯,
     先验采样会模式崩溃(见采样行的多样性对比)
   - 当L→∞时，这就是扩散模型（10.4节）
""")

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

results_summary = {
    'config': {
        'd1': d1, 'd2': d2, 'd_z': d1, 'beta': beta,
        'num_epochs': num_epochs, 'batch_size': 128, 'lr': 1e-3,
        'objective': 'hierarchical ELBO with conditional KL1 vs p(z1|z2)',
    },
    'hierarchical_VAE': {
        'final_loss': history['loss'][-1] if history['loss'] else None,
        'final_recon': history['recon'][-1] if history['recon'] else None,
        'final_kl1': history['kl1'][-1] if history['kl1'] else None,
        'final_kl2': history['kl2'][-1] if history['kl2'] else None,
        'final_kl_total': history['kl_total'][-1] if history['kl_total'] else None,
        'recon_psnr_db_test': psnr_h,
        'prior_sample_diversity_pairwise_l1_per_pix': pd_h,
        'loss_curve': history['loss'],
        'kl1_curve': history['kl1'],
        'kl2_curve': history['kl2'],
    },
    'single_VAE': {
        'final_loss': history_single['loss'][-1] if history_single['loss'] else None,
        'final_recon': history_single['recon'][-1] if history_single['recon'] else None,
        'final_kl': history_single['kl'][-1] if history_single['kl'] else None,
        'recon_psnr_db_test': psnr_s,
        'prior_sample_diversity_pairwise_l1_per_pix': pd_s,
        'loss_curve': history_single['loss'],
        'kl_curve': history_single['kl'],
    },
    'ELBO_test': {
        'recon': avg_recon,
        'kl1_vs_p_z1_given_z2': avg_kl1,
        'kl2': avg_kl2,
        'kl_total': avg_kl1 + avg_kl2,
        'ELBO': elbo,
        'kl1_over_kl2': avg_kl1 / avg_kl2 if avg_kl2 != 0 else None,
    },
    'single_VAE_test': {
        'recon': avg_recon_s,
        'kl': avg_kl_s,
        'ELBO': elbo_s,
    },
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")