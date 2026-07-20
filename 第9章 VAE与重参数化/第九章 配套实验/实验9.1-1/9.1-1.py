# -*- coding: utf-8 -*-
"""
实验9.1-1 VAE重建与生成能力验证（含AE对比）
对应章节: 9.1 VAE架构：编码器-解码器

知识点:
  - VAE建模的是分布而非确定性映射
  - VAE不仅能重建还能生成（对比AE只能重建）
  - KL正则化使隐空间连续有规律

实验内容:
  步骤1: VAE与AE训练（MNIST）
  步骤2: 重建与先验采样对比

素材来源:
  - 实验9.1.py
  - 9.5节PyTorch实现

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
    SAVE_DIR = os.path.join(_gdrive, '实验9.1-1')
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
print(f"实验9.1-1: VAE重建与生成能力验证（含AE对比）")
print(f"{'='*60}")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  未检测到 GPU, 使用 CPU 训练")
    print("  提示: Colab 用户可在菜单 运行时 -> 更改运行时类型 中选择 GPU")

# Checkpoint路径
VAE_CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'vae_mnist_checkpoint.pth')
AE_CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'ae_mnist_checkpoint.pth')

# ============================================================
# VAE/AE模型定义（对应9.1节）
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
    
    注意: beta=0 时即为普通Autoencoder的损失（仅重建项）
    """
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD, BCE, KLD


def train_model(beta, checkpoint_path, train_loader, num_epochs=20, model_name='VAE'):
    """训练模型（VAE或AE），支持checkpoint resume
    
    Args:
        beta: KL正则化系数（beta=0为AE，beta=1为VAE）
        checkpoint_path: checkpoint保存路径
        train_loader: 数据加载器
        num_epochs: 训练轮数
        model_name: 模型名称（用于打印）
    """
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
    )
    
    start_epoch = 0
    is_final = False
    train_losses = []
    
    # Checkpoint加载逻辑
    # weights_only=False用于加载optimizer state等非tensor对象
    # 在本地训练场景下安全，若checkpoint来自可信来源则风险可控
    if os.path.exists(checkpoint_path):
        print(f"\n检测到已保存的模型 ({model_name}): {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
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
                os.remove(checkpoint_path)
                start_epoch = 0
                is_final = False
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                train_losses = checkpoint.get('train_losses', [])
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
                os.remove(checkpoint_path)
                start_epoch = 0
                is_final = False
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                train_losses = checkpoint.get('train_losses', [])
                start_epoch = checkpoint['epoch'] + 1
    
    # 训练循环
    if not is_final:
        print(f"\n开始训练 {model_name} (epochs={num_epochs}, beta={beta}, d_z=20)...")
        
        # 快速验证模式
        import os as _os
        if _os.environ.get('QUICK_TEST', '') == '1':
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
                total_loss = 0
                
                from tqdm import tqdm
                pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', 
                            leave=False, unit='batch')
                
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
                    pbar.set_postfix(loss=f'{loss.item():.4f}')
                
                avg_loss = total_loss / len(train_loader.dataset)
                train_losses.append(avg_loss)
                
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"  Epoch {epoch+1:3d}/{num_epochs}: Loss = {avg_loss:.4f}")
                
                # 每个epoch保存checkpoint（而非每5轮），避免丢失进度
                torch.save({
                    'epoch': epoch,
                    'beta': beta,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'train_losses': train_losses,
                    'is_final': False
                }, checkpoint_path)
            
            t_elapsed = time.time() - t_start
            print(f"\n训练完成, 最终损失: {train_losses[-1]:.6f}, 耗时: {t_elapsed:.1f} 秒")
            
            # 保存最终checkpoint
            if train_losses:
                torch.save({
                    'epoch': num_epochs - 1,
                    'beta': beta,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_losses[-1],
                    'train_losses': train_losses,
                    'is_final': True
                }, checkpoint_path)
                print(f"✓ 训练完成, 模型已保存: {checkpoint_path}")
    else:
        print(f"\n使用已训练完成的 {model_name} 模型, 跳过训练过程")
    
    return encoder, decoder, train_losses


# ============================================================
# 步骤1: VAE与AE训练（MNIST）
# ============================================================
print(f"\n{'='*60}")
print("步骤1: VAE与AE训练（MNIST）")
print(f"{'='*60}")
print("\n[核心思想]")
print("  VAE建模的是分布而非确定性映射：")
print("  - 编码器输出分布参数 (μ, logσ²)，而非单点 z")
print("  - 解码器输出似然参数，而非确定性重建")
print("  - KL正则化使隐空间连续、有规律，支持生成")
print("\n[对比实验]")
print("  VAE (β=1) vs AE (β=0)：")
print("  - VAE: 有KL正则化，隐空间连续 → 可生成")
print("  - AE:  无KL约束，隐空间有空洞 → 无法生成")
print("\n[实现说明]")
print("  为复用同一套代码框架，AE在此实现为β=0的VAE退化形式")
print("  （Encoder仍输出μ和logσ²，但KL项为0），而非架构上移除")
print("  方差分支的纯确定性编码器。训练时logσ²会趋向很负值，")
print("  效果上逼近确定性编码，但理论上不是'架构上'的AE。")

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

# 训练VAE (β=1)
print(f"\n{'='*60}")
print("训练 VAE (β=1)...")
print(f"{'='*60}")
vae_encoder, vae_decoder, vae_losses = train_model(
    beta=1.0, 
    checkpoint_path=VAE_CHECKPOINT_PATH,
    train_loader=train_loader,
    num_epochs=20,
    model_name='VAE'
)

# 训练AE (β=0)
print(f"\n{'='*60}")
print("训练 AE (β=0)...")
print(f"{'='*60}")
ae_encoder, ae_decoder, ae_losses = train_model(
    beta=0.0,
    checkpoint_path=AE_CHECKPOINT_PATH,
    train_loader=train_loader,
    num_epochs=20,
    model_name='AE'
)


# ============================================================
# 步骤2: 重建与先验采样对比（VAE vs AE）
# ============================================================
print(f"\n{'='*60}")
print("步骤2: 重建与先验采样对比（VAE vs AE）")
print(f"{'='*60}")
print("\n[核心对比]")
print("  VAE vs AE 的本质区别：")
print("  - VAE: 编码分布经KL正则化，隐空间连续有规律 → 可生成")
print("  - AE:  隐空间无约束，存在空洞区域 → 无法生成")
print("\n[可视化说明]")
print("  重建使用 μ（后验均值）而非随机采样，展示确定性最佳重建")
print("  先验采样使用 z ~ N(0,I)，展示生成能力")

vae_encoder.eval()
vae_decoder.eval()
ae_encoder.eval()
ae_decoder.eval()

# 重建测试（使用μ而非随机采样）
print("\n生成重建与采样对比图...")
n_show = 10

fig, axes = plt.subplots(4, n_show, figsize=(20, 10))

with torch.no_grad():
    test_imgs, test_labels = next(iter(test_loader))
    test_flat = test_imgs.view(-1, 784).to(device)
    
    # VAE重建（使用μ而非reparameterize，展示确定性最佳重建）
    vae_mu, vae_logvar = vae_encoder(test_flat)
    vae_recon = vae_decoder(vae_mu)  # 用均值而非采样
    
    # AE重建（同样使用μ）
    ae_mu, ae_logvar = ae_encoder(test_flat)
    ae_recon = ae_decoder(ae_mu)
    
    # VAE先验采样: z ~ N(0, I)
    vae_z_prior = torch.randn(n_show, 20).to(device)
    vae_samples = vae_decoder(vae_z_prior)
    
    # AE先验采样: z ~ N(0, I)（理论上AE隐空间无此约束，采样结果无意义）
    ae_z_prior = torch.randn(n_show, 20).to(device)
    ae_samples = ae_decoder(ae_z_prior)

for i in range(n_show):
    # 第一行：原始图像
    axes[0, i].imshow(test_imgs[i, 0].numpy(), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('原始图像', fontsize=12)
    
    # 第二行：VAE重建
    axes[1, i].imshow(vae_recon[i].view(28, 28).cpu().numpy(), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('VAE重建', fontsize=12)
    
    # 第三行：AE重建
    axes[2, i].imshow(ae_recon[i].view(28, 28).cpu().numpy(), cmap='gray')
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_title('AE重建', fontsize=12)
    
    # 第四行：VAE先验采样
    axes[3, i].imshow(vae_samples[i].view(28, 28).cpu().numpy(), cmap='gray')
    axes[3, i].axis('off')
    if i == 0:
        axes[3, i].set_title('VAE先验采样', fontsize=12)

plt.suptitle('步骤2: VAE与AE重建与生成对比 ($\\beta_{VAE}=1$, $\\beta_{AE}=0$)', 
             fontsize=14, y=1.02)
plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤2_VAE与AE重建对比.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图表已保存: {fig_path}")

# AE先验采样单独展示（证明AE无法生成）
print("\n生成AE先验采样对比图...")
fig, axes = plt.subplots(2, n_show, figsize=(20, 6))

for i in range(n_show):
    # 第一行：VAE先验采样（可识别的数字）
    axes[0, i].imshow(vae_samples[i].view(28, 28).cpu().numpy(), cmap='gray')
    # 不使用axis('off')，因为会抑制ylabel显示；改为只关掉刻度
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])
    if i == 0:
        axes[0, i].set_ylabel('VAE先验采样\n($z \\sim \\mathcal{N}(0,I)$)', 
                              fontsize=12, rotation=0, labelpad=60)
    
    # 第二行：AE先验采样（无意义的图像）
    axes[1, i].imshow(ae_samples[i].view(28, 28).cpu().numpy(), cmap='gray')
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])
    if i == 0:
        axes[1, i].set_ylabel('AE先验采样\n($z \\sim \\mathcal{N}(0,I)$)', 
                              fontsize=12, rotation=0, labelpad=60)

plt.suptitle('先验采样对比: VAE可生成, AE不能', fontsize=14, y=1.02)
plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤3_AE无法生成对比.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图表已保存: {fig_path}")


# ============================================================
# 实验总结
# ============================================================
print(f"\n{'='*60}")
print("实验9.1-1 总结")
print(f"{'='*60}")
print("\n核心验证点（对应9.1节理论）：")
print("\n1. VAE建模的是分布而非确定性映射：")
print("   - 编码器输出 μ 和 logσ²（分布参数），而非单点 z")
print("   - 解码器输出 Bernoulli 分布参数，而非确定性重建")
print("   ✓ 实验验证: 编码器 forward() 返回 (mu, logvar)")
print("\n2. VAE能重建：")
print("   - 重建图像保留了原始数字的主要特征")
print("   - 重建略模糊，是 KL 正则化的代价（信息瓶颈）")
print("   ✓ 实验验证: VAE重建（第二行）质量良好，使用μ展示确定性重建")
print("\n3. VAE能生成：")
print("   - 从 N(0,I) 先验随机采样，经解码器生成合理图像")
print("   - KL 正则化使隐空间与先验一致，确保采样有效")
print("   ✓ 实验验证: VAE先验采样生成可识别的数字（第四行）")
print("\n4. AE重建能力强但无法生成：")
print("   - AE 隐空间无 KL 约束，编码自由度高 → 重建更清晰")
print("   - 但隐空间存在大量空洞，从 N(0,I) 采样解码无意义")
print("   ✓ 实验验证: AE重建清晰（第三行），但AE先验采样为噪声（见对比图）")
print("   [实现细节说明] 本实验AE为β=0的VAE退化形式（Encoder仍输出(μ, logσ²)），")
print("                 故AE先验采样图像呈现'有结构但无语义'的特征而非完全随机噪声；")
print("                 真正的纯AE（确定性编码，无方差分支）采样结果会更杂乱。")
print("                 此实现选择便于复用同一套代码框架，不影响VAE/AE核心差异的展示。")
print("\n5. VAE vs AE核心差异：")
print("   - VAE: KL正则化牺牲部分重建精度，换取隐空间规律性 → 可生成")
print("   - AE:  无KL约束，重建更清晰，但隐空间无保证 → 无法生成")
print("   ✓ 实验验证: 对比图直观展示VAE可生成、AE无法生成")

if vae_losses and ae_losses:
    print(f"\n训练统计:")
    print(f"  VAE (β=1): 最终损失 {vae_losses[-1]:.6f}")
    print(f"  AE  (β=0): 最终损失 {ae_losses[-1]:.6f}")
    print(f"  注意: AE损失更低（仅重建项），但无KL正则化")

print(f"\n{'='*60}")
print("第九章配套实验 9.1-1 完成!")

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
    'VAE_final_loss': vae_losses[-1] if vae_losses else None,
    'AE_final_loss': ae_losses[-1] if ae_losses else None,
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")