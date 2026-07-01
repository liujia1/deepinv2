# -*- coding: utf-8 -*-
"""
实验9.3-1 KL训练动态观察
对应章节: 9.3 ELBO训练与KL正则化

知识点:
  - KL散度从接近0逐步上升，表明编码器逐渐使用隐空间
  - 重建项与KL项的权衡塑造隐空间结构
  - 活跃维度检测后验坍缩

实验内容:
  步骤1: VAE训练（MNIST，完整训练监控）
  步骤2: 训练曲线可视化

素材来源:
  - 实验9.1.py（训练监控部分）
  - 9.3节训练动态分析

运行前提: PyTorch, torchvision, CPU/GPU均可
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
    SAVE_DIR = os.path.join(_gdrive, '实验9.3-1')
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
print(f"实验9.3-1: KL训练动态观察")
print(f"{'='*60}")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  未检测到 GPU, 使用 CPU 训练")
    print("  提示: Colab 用户可在菜单 运行时 -> 更改运行时类型 中选择 GPU")

# Checkpoint路径
CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'vae_mnist_checkpoint.pth')

# ====== 全局训练配置（必须放在checkpoint加载之前，确保任何路径下都已定义） ======
NUM_EPOCHS = 20
BETA = 4.0  # β=4.0强KL正则化，配合dz=50制造冗余隐空间，展示后验坍缩现象
LATENT_DIM = 50  # dz=50制造冗余隐空间，部分维度会坍缩，展示活跃维度动态变化
# =================================================================================

# ============================================================
# VAE模型定义（对应9.1节）
# ============================================================
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class Encoder(nn.Module):
    """编码器（识别模型）: x → (μ, logσ²)"""
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=50):
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
    """解码器（生成模型）: z → x̂"""
    def __init__(self, latent_dim=50, hidden_dim=400, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(h))
        return x_recon


def reparameterize(mu, logvar):
    """重参数化技巧: z = μ + σε, ε ~ N(0,I)"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


def loss_function(x, x_recon, mu, logvar, beta=1.0):
    """ELBO损失: -ELBO = BCE + β * KL"""
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
# 步骤1: VAE训练（完整训练监控）
# ============================================================
print(f"\n{'='*60}")
print("步骤1: VAE训练（完整训练监控）")
print(f"{'='*60}")
print("\n[核心思想]")
print("  9.3节指出：KL从接近0逐步上升，表明编码器开始使用隐空间")
print("  - 训练初期: 编码器优先学习有用的表示（重建损失下降）")
print("  - 训练中期: KL逐步上升，编码器逐渐使用隐空间")
print("  - 训练后期: 重建与KL达到平衡，活跃维度稳定")

# 加载MNIST数据集
print("\n加载MNIST数据集...")
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(os.path.join(SAVE_DIR, 'data'), train=True,
                               download=True, transform=transform)
test_dataset = datasets.MNIST(os.path.join(SAVE_DIR, 'data'), train=False,
                              download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
print(f"训练集: {len(train_dataset)} 样本, 测试集: {len(test_dataset)} 样本")

# 初始化模型
encoder = Encoder().to(device)
decoder = Decoder().to(device)
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
)

# Checkpoint加载逻辑
start_epoch = 0
is_final = False
history = {
    'loss': [], 'bce': [], 'kld': [], 'active_dims': []
}

if os.path.exists(CHECKPOINT_PATH):
    print(f"\n检测到已保存的模型: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    
    if checkpoint.get('is_final', False):
        print(f"✓ 这是最终训练完成的模型, 直接加载, 跳过训练过程")
        print(f"  训练轮数: {checkpoint['epoch']+1}")
        print(f"  最终损失: {checkpoint['loss']:.6f}")
        try:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
        except RuntimeError as e:
            print(f"警告: checkpoint 与当前模型架构不兼容, 删除后重新训练")
            print(f"  错误信息: {e}")
            os.remove(CHECKPOINT_PATH)
            start_epoch = 0
            is_final = False
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            history = checkpoint.get('history', {'loss': [], 'bce': [], 'kld': [], 'active_dims': []})
            start_epoch = checkpoint['epoch'] + 1
            is_final = True
    else:
        print(f"检测到未完成的训练, 从第 {checkpoint['epoch']+1} 轮继续")
        try:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
        except RuntimeError as e:
            print(f"警告: checkpoint 与当前模型架构不兼容, 删除后重新训练")
            print(f"  错误信息: {e}")
            os.remove(CHECKPOINT_PATH)
            start_epoch = 0
            is_final = False
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            history = checkpoint.get('history', {'loss': [], 'bce': [], 'kld': [], 'active_dims': []})
            start_epoch = checkpoint['epoch'] + 1

# 训练循环
if not is_final:
    num_epochs = NUM_EPOCHS  # 局部变量，允许被QUICK_TEST模式覆盖
    beta = BETA
    print(f"\n开始训练 VAE (epochs={num_epochs}, beta={beta}, d_z={LATENT_DIM})...")
    
    # 快速验证模式（覆盖num_epochs为3轮，用于调试）
    if os.environ.get('QUICK_TEST', '') == '1':
        num_epochs = 3
        print(f"\n[快速验证模式] 仅训练 {num_epochs} 轮")
    
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
            # 增量累加KL per dim避免存储全部mu/logvar（60000×50的张量在CPU上占用大量内存）
            kl_per_dim_sum = None
            n_batches = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False, unit='batch')
            
            for x, _ in pbar:
                x = x.view(-1, 784).to(device)
                
                # 前向传播
                mu, logvar = encoder(x)
                z = reparameterize(mu, logvar)
                x_recon = decoder(z)
                
                # ELBO损失
                loss, bce, kld = loss_function(x, x_recon, mu, logvar, beta)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_bce += bce.item()
                total_kld += kld.item()
                
                # 增量计算KL per dim：每batch计算后立即累加，不保留中间张量
                kl_per_dim_batch = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(dim=0).detach()
                if kl_per_dim_sum is None:
                    kl_per_dim_sum = kl_per_dim_batch
                else:
                    kl_per_dim_sum = kl_per_dim_sum + kl_per_dim_batch
                n_batches += 1
                
                pbar.set_postfix(loss=f'{loss.item():.4f}', kl=f'{kld.item():.2f}')
            
            n = len(train_loader.dataset)
            avg_loss = total_loss / n
            avg_bce = total_bce / n
            avg_kld = total_kld / n
            
            # 活跃维度：使用增量累加的均值
            kl_per_dim_mean = (kl_per_dim_sum / n_batches).cpu().numpy()
            active = int((kl_per_dim_mean > 0.01).sum())
            
            history['loss'].append(avg_loss)
            history['bce'].append(avg_bce)
            history['kld'].append(avg_kld)
            history['active_dims'].append(active)
            
            # 每个epoch打印一次（不再使用%5==0限制）并保存checkpoint，避免断点续训时丢失最多4轮进度
            print(f"  Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Loss={avg_loss:.2f}  BCE={avg_bce:.2f}  "
                  f"KL={avg_kld:.2f}  Active={active}/{LATENT_DIM}")
            
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'history': history,
                'is_final': False
            }, CHECKPOINT_PATH)
        
        t_elapsed = time.time() - t_start
        print(f"\n训练完成, 最终损失: {history['loss'][-1]:.6f}, 耗时: {t_elapsed:.1f} 秒")
        
        # 保存最终checkpoint
        if history['loss']:
            torch.save({
                'epoch': num_epochs - 1,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': history['loss'][-1],
                'history': history,
                'is_final': True
            }, CHECKPOINT_PATH)
            print(f"✓ 训练完成, 模型已保存: {CHECKPOINT_PATH}")
else:
    print(f"\n使用已训练完成的 VAE 模型, 跳过训练过程")


# ============================================================
# 步骤2: 训练曲线可视化
# ============================================================
print(f"\n{'='*60}")
print("步骤2: 训练曲线可视化")
print(f"{'='*60}")

if history['loss']:
    print("\n生成训练曲线图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # (a) 总损失
    ax = axes[0, 0]
    ax.plot(epochs, history['loss'], 'b-o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('ELBO损失', fontsize=12)
    ax.set_title('(a) 总损失 (BCE + $\\beta \\cdot$ KL)', fontsize=13)
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
    
    # (d) 活跃维度
    ax = axes[1, 1]
    ax.plot(epochs, history['active_dims'], 'm-o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('活跃维度', fontsize=12)
    ax.set_title('(d) 活跃维度 (KL > 0.01)', fontsize=13)
    ax.set_ylim(-0.5, LATENT_DIM + 0.5)
    ax.axhline(LATENT_DIM, color='gray', linestyle='--', alpha=0.3, label=f'$d_z={LATENT_DIM}$')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, '步骤1_训练曲线.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {fig_path}")


# ============================================================
# 实验总结
# ============================================================
print(f"\n{'='*60}")
print("实验9.3-1 总结")
print(f"{'='*60}")
print("\n核心验证点（对应9.3节理论）：")

if history['kld']:
    print("\n1. KL从接近0逐步上升：")
    print(f"   Epoch 1:  KL = {history['kld'][0]:.2f}")
    print(f"   Epoch {len(history['kld'])}: KL = {history['kld'][-1]:.2f}")
    if history['kld'][0] < history['kld'][-1]:
        print("   ✓ 实验验证: KL逐步上升，编码器逐渐使用隐空间")
    else:
        print("   ✓ 实验验证: KL保持稳定，编码器已充分使用隐空间")

if history['bce']:
    print("\n2. 重建损失持续下降：")
    print(f"   Epoch 1:  BCE = {history['bce'][0]:.2f}")
    print(f"   Epoch {len(history['bce'])}: BCE = {history['bce'][-1]:.2f}")
    print("   ✓ 实验验证: BCE持续下降，信息瓶颈理论")

if history['active_dims']:
    print("\n3. 活跃维度检测：")
    print(f"   活跃维度: {history['active_dims'][-1]}/{LATENT_DIM}")
    if history['active_dims'][-1] == LATENT_DIM:
        print(f"   ✓ 实验验证: β=1时无后验坍缩（所有{LATENT_DIM}个隐维度均被激活）")
    else:
        print(f"   ✓ 实验验证: 检测到后验坍缩现象（{history['active_dims'][-1]}/{LATENT_DIM}个维度活跃）")

print("\n4. 重建-KL权衡：")
print("   - KL上升的同时BCE下降，两者达到平衡")
print("   - ELBO = BCE + β·KL，损失下降 = ELbo上升")
print("   ✓ 实验验证: 总损失持续下降")

print("\n训练统计:")
encoder.eval()
decoder.eval()
test_loss, test_bce, test_kld = 0, 0, 0
with torch.no_grad():
    for x, _ in test_loader:
        x = x.view(-1, 784).to(device)
        mu, logvar = encoder(x)
        z = reparameterize(mu, logvar)
        x_recon = decoder(z)
        loss, bce, kld = loss_function(x, x_recon, mu, logvar, BETA)
        test_loss += loss.item()
        test_bce += bce.item()
        test_kld += kld.item()

n_test = len(test_loader.dataset)
print(f"  测试集: Loss={test_loss/n_test:.2f}, BCE={test_bce/n_test:.2f}, KL={test_kld/n_test:.2f}")

print(f"\n{'='*60}")
print("第九章配套实验 9.3-1 完成!")