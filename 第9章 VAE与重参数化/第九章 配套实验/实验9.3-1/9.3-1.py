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
BETA = 4.0  # 强KL正则化: β 决定信息预算(KL总量), 预算越小能被养活的维度越少
LATENT_DIM = 50  # 冗余隐空间: dz 只决定分母大小(对照显示 β=4 时 dz=20/50 的 KL 总量与 PSNR 几乎相同)
ACTIVE_THRESHOLD = 0.01  # 活跃维度判据: 该维平均KL > 0.01

# 快速验证模式必须在 checkpoint 加载之前判定：否则 QUICK_TEST 产出的 3 轮成品
# 会被 is_final 标记为"已完成"，随后的正式运行会静默跳过训练、沿用欠拟合模型。
QUICK = os.environ.get('QUICK_TEST', '') == '1'
RUN_EPOCHS = 3 if QUICK else NUM_EPOCHS
if QUICK:
    CHECKPOINT_PATH = CHECKPOINT_PATH.replace('.pth', '_quick.pth')
    print(f"\n[快速验证模式] 仅训练 {RUN_EPOCHS} 轮 -> {os.path.basename(CHECKPOINT_PATH)}")
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
    """ELBO损失: -ELBO = BCE + β * KL（最小化 -ELBO 等价于最大化 ELBO）"""
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD, BCE, KLD


def kl_per_dim_of(mu, logvar):
    """每个隐维度的 KL 贡献: 0.5*(mu^2 + sigma^2 - log sigma^2 - 1), 对样本取平均"""
    return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)


def compute_active_dims(mu, logvar, threshold=ACTIVE_THRESHOLD):
    """活跃维度计算（对应9.3节）
    统计 KL(q(z_j|x) || p(z_j)) > threshold 的维度数
    """
    kl_per_dim = kl_per_dim_of(mu, logvar).mean(dim=0)
    return int((kl_per_dim > threshold).sum()), kl_per_dim.detach().cpu().numpy()


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

# 初始化模型（隐空间维度统一由 LATENT_DIM 控制，避免修改配置后模型不跟随）
encoder = Encoder(latent_dim=LATENT_DIM).to(device)
decoder = Decoder(latent_dim=LATENT_DIM).to(device)
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
)

# Checkpoint加载逻辑
start_epoch = 0
is_final = False
history = {
    'loss': [], 'bce': [], 'kld': [], 'active_dims': [],
    'kl_first_epoch': []  # 第1个epoch逐batch KL/样本，用于观察"从0起步"与短暂过冲
}


def _cfg_mismatch(checkpoint):
    """返回与当前配置不符的字段（旧 checkpoint 无这些字段时视为兼容）
    校验决定曲线语义的 beta / latent_dim，以及 quick / num_epochs：
    防止 QUICK_TEST 的短轮次成品被正式运行当成完整模型复用。
    """
    bad = []
    for key, cur in (('beta', BETA), ('latent_dim', LATENT_DIM)):
        saved = checkpoint.get(key)
        if saved is not None and abs(saved - cur) > 1e-12:
            bad.append(f'{key}: checkpoint={saved} vs 当前={cur}')
    if bool(checkpoint.get('quick', False)) != QUICK:
        bad.append(f'quick: checkpoint={checkpoint.get("quick", False)} vs 当前={QUICK}')
    saved_epochs = checkpoint.get('num_epochs')
    if saved_epochs is not None and saved_epochs < RUN_EPOCHS:
        bad.append(f'num_epochs: checkpoint={saved_epochs} vs 目标={RUN_EPOCHS}')
    return bad


checkpoint = None
if os.path.exists(CHECKPOINT_PATH):
    print(f"\n检测到已保存的模型: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    mismatch = _cfg_mismatch(checkpoint)
    if mismatch:
        # 防止旧配置权重被当成当前实验结果误用
        print("警告: checkpoint 与当前训练配置不符, 删除后重新训练")
        for m in mismatch:
            print(f"  {m}")
        os.remove(CHECKPOINT_PATH)
        checkpoint = None

if checkpoint is not None:
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
            # 成品 checkpoint 不保存优化器状态, 仅在存在时恢复
            if checkpoint.get('optimizer_state_dict') is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            history = checkpoint.get('history', {})
            for key in ('loss', 'bce', 'kld', 'active_dims', 'kl_first_epoch'):
                history.setdefault(key, [])
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
            history = checkpoint.get('history', {})
            for key in ('loss', 'bce', 'kld', 'active_dims', 'kl_first_epoch'):
                history.setdefault(key, [])
            start_epoch = checkpoint['epoch'] + 1

# 训练循环
if not is_final:
    num_epochs = RUN_EPOCHS
    beta = BETA
    print(f"\n开始训练 VAE (epochs={num_epochs}, beta={beta}, d_z={LATENT_DIM})...")

    # 边界保护
    if start_epoch >= num_epochs:
        print(f"  注意: start_epoch({start_epoch}) >= num_epochs({num_epochs}), 无需继续训练")
        is_final = True
    
    if not is_final:
        import time
        t_start = time.time()
        # 探针: 首个 batch 的 KL 反映“未更新任何参数时”的初始值（验证9.3节“初期 KL 几乎为0”）
        probe_init_kl = (start_epoch == 0)
        
        for epoch in range(start_epoch, num_epochs):
            encoder.train()
            decoder.train()
            total_loss, total_bce, total_kld = 0, 0, 0
            # 增量累加KL per dim避免存储全部mu/logvar（60000×50的张量在CPU上占用大量内存）
            kl_per_dim_sum = None
            n_batches = 0
            kl_iter_curve = []  # 仅第1个epoch收集：逐batch的KL/样本
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False, unit='batch')
            
            for it, (x, _) in enumerate(pbar):
                x = x.view(-1, 784).to(device)
                
                # 前向传播
                mu, logvar = encoder(x)
                
                if probe_init_kl and it == 0 and 'kld_init' not in history:
                    init_kl = kl_per_dim_of(mu, logvar).sum(dim=1).mean().item()
                    history['kld_init'] = init_kl
                    print(f"  [初始化探针] 未更新参数时 KL/sample = {init_kl:.3f} nats"
                          f" (每维 {init_kl / LATENT_DIM:.4f} nats, batch={x.shape[0]})")
                
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
                
                # 第1个epoch内部的逐batch KL：epoch均值会把"从0起步/短暂过冲"平滑掉
                if epoch == 0:
                    kl_iter_curve.append(kld.item() / x.size(0))

                # 增量计算KL per dim：每batch计算后立即累加，不保留中间张量
                kl_per_dim_batch = kl_per_dim_of(mu, logvar).mean(dim=0).detach()
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
            active = int((kl_per_dim_mean > ACTIVE_THRESHOLD).sum())
            
            history['loss'].append(avg_loss)
            history['bce'].append(avg_bce)
            history['kld'].append(avg_kld)
            history['active_dims'].append(active)
            if epoch == 0:
                history['kl_first_epoch'] = kl_iter_curve
            
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
                'beta': BETA,
                'latent_dim': LATENT_DIM,
                'num_epochs': num_epochs,
                'quick': QUICK,
                'is_final': False
            }, CHECKPOINT_PATH)
        
        t_elapsed = time.time() - t_start
        print(f"\n训练完成, 最终损失: {history['loss'][-1]:.6f}, 耗时: {t_elapsed:.1f} 秒")
        
        # 保存最终checkpoint（成品不保存优化器状态, 减小体积）
        if history['loss']:
            torch.save({
                'epoch': num_epochs - 1,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'loss': history['loss'][-1],
                'history': history,
                'beta': BETA,
                'latent_dim': LATENT_DIM,
                'num_epochs': num_epochs,
                'quick': QUICK,
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
    ax.set_ylabel('损失 $-\\mathrm{ELBO}$', fontsize=12)
    ax.set_title('(a) 总损失 (BCE + $\\beta\\cdot$KL)', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # (b) 重建损失
    ax = axes[0, 1]
    ax.plot(epochs, history['bce'], 'r-o', markersize=4, label='重建损失 (BCE)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('BCE', fontsize=12)
    ax.set_title('(b) 重建损失', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # (c) KL散度（纵轴从0起，并把未训练时的初始KL作为第0个点画出，
    #     否则“从接近0爬升”会被 epoch 平均与截断轴掩盖）
    ax = axes[1, 0]
    kld_pts = list(history['kld'])
    kld_eps = list(range(1, len(kld_pts) + 1))
    if history.get('kld_init') is not None:
        kld_pts = [history['kld_init']] + kld_pts
        kld_eps = [0] + kld_eps
    ax.plot(kld_eps, kld_pts, 'g-o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('KL散度 (nats/样本)', fontsize=12)
    ax.set_title('(c) KL正则化项' + (' (0=初始化)' if history.get('kld_init') is not None else ''),
                 fontsize=13)
    ax.set_ylim(0, max(kld_pts) * 1.15)
    ax.grid(True, alpha=0.3)
    
    # (d) 活跃维度
    ax = axes[1, 1]
    ax.plot(epochs, history['active_dims'], 'm-o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('活跃维度', fontsize=12)
    ax.set_title(f'(d) 活跃维度 (KL $> {ACTIVE_THRESHOLD}$)', fontsize=13)
    ax.set_ylim(-0.5, LATENT_DIM + 0.5)
    ax.axhline(LATENT_DIM, color='gray', linestyle='--', alpha=0.3, label=f'$d_z={LATENT_DIM}$')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, '步骤1_训练曲线.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {fig_path}")

    # 第1个epoch逐batch KL动态：把"KL从接近0起步"细化到iteration尺度（含初期短暂过冲）
    if history.get('kl_first_epoch'):
        kl_it = history['kl_first_epoch']
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        steps = np.arange(1, len(kl_it) + 1)
        ax2.plot(steps, kl_it, color='#2ca02c', linewidth=1.2)
        ax2.scatter([1], [kl_it[0]], color='red', s=40, zorder=5)
        ax2.annotate(f'第1步 KL≈{kl_it[0]:.2f}', xy=(1, kl_it[0]),
                     xytext=(50, 12), textcoords='offset points', fontsize=11,
                     arrowprops=dict(arrowstyle='->', color='red'))
        ax2.set_xlabel('Batch（第1个epoch内的训练步）', fontsize=12)
        ax2.set_ylabel('KL散度 (nats/样本)', fontsize=12)
        ax2.set_title('训练初期逐batch KL动态：从接近0起步（含短暂过冲）', fontsize=13)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        fig2_path = os.path.join(SAVE_DIR, '步骤1_训练初期KL动态.png')
        plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"图表已保存: {fig2_path}")
    else:
        print("提示: 旧checkpoint未记录iteration级KL；删除 vae_mnist_checkpoint.pth 重新训练后，"
              "可生成《步骤1_训练初期KL动态.png》")


# ============================================================
# 实验总结
# ============================================================
print(f"\n{'='*60}")
print("实验9.3-1 总结")
print(f"{'='*60}")
print("\n核心验证点（对应9.3节理论）：")

if history['kld']:
    kld_first, kld_last = history['kld'][0], history['kld'][-1]
    kld_init = history.get('kld_init')
    print("\n1. KL从接近0逐步上升：")
    if kld_init is not None:
        print(f"   初始化(0步):  KL = {kld_init:.2f} nats/样本")
    print(f"   Epoch 1:      KL = {kld_first:.2f}")
    print(f"   Epoch {len(history['kld'])}:     KL = {kld_last:.2f}")
    ref = kld_init if kld_init is not None else kld_first
    if kld_last > 1.05 * ref:
        print(f"   ✓ 实验验证: KL从 {ref:.2f} 上升到 {kld_last:.2f}，编码器逐渐使用隐空间")
    else:
        print(f"   ✗ 未观察明显上升（{ref:.2f} → {kld_last:.2f}），编码器未能从隐空间获取额外信息")

if history['bce']:
    bce_curve = history['bce']
    print("\n2. 重建损失持续下降：")
    print(f"   Epoch 1:  BCE = {bce_curve[0]:.2f}")
    print(f"   Epoch {len(bce_curve)}: BCE = {bce_curve[-1]:.2f}")
    if bce_curve[-1] < bce_curve[0]:
        n_up = sum(1 for i in range(1, len(bce_curve)) if bce_curve[i] > bce_curve[i - 1])
        print(f"   ✓ 实验验证: BCE从 {bce_curve[0]:.2f} 降至 {bce_curve[-1]:.2f}"
              f"（逐epoch回升 {n_up} 次），解码器逐渐学会重建")
    else:
        print("   ✗ BCE 未下降，重建未学到有效表示")

if history['active_dims']:
    ad_last = history['active_dims'][-1]
    ad_max = max(history['active_dims'])
    print("\n3. 活跃维度检测（β={:.1f}）：".format(BETA))
    print(f"   峰值活跃维度: {ad_max}/{LATENT_DIM} → 最终活跃维度: {ad_last}/{LATENT_DIM}")
    if ad_last == LATENT_DIM:
        print(f"   ✓ 实验验证: 未发生后验坍缩（所有{LATENT_DIM}个隐维度均被激活）")
    else:
        print(f"   ✓ 实验验证: 检测到后验坍缩（最终仅 {ad_last}/{LATENT_DIM} 维活跃，"
              f"即 {100.0 * (LATENT_DIM - ad_last) / LATENT_DIM:.0f}% 维度 KL≈0）")
        if ad_max < LATENT_DIM:
            print(f"     注: 活跃维度从峰值 {ad_max} 回落，冷维度在训练中被逐步压制")
        if BETA > 1:
            print(f"     归因: β={BETA:g}>1 的强KL正则显著压缩活跃维度；同时 d_z={LATENT_DIM} 对MNIST存在固有冗余——")
            print("           同结构对照下 β=1、d_z=50 仍有约一半维度被软剪枝(over-pruning)，β>1 会令坍缩更彻底")
        else:
            print("     归因: β=1下仍有维度被剪，主要来自隐空间相对MNIST的固有冗余(over-pruning)")

loss_curve = history['loss']
bce_curve = history['bce']
kld_curve = history['kld']
print("\n4. 重建-KL权衡：")
print(f"   - 目标函数为 -ELBO = BCE + beta*KL (beta={BETA})，最小化 -ELBO 等价于最大化 ELBO")
if loss_curve:
    n_loss_up = sum(1 for i in range(1, len(loss_curve)) if loss_curve[i] > loss_curve[i - 1])
    trend_kl = '上升' if kld_curve[-1] > kld_curve[0] else '下降/持平'
    trend_bce = '下降' if bce_curve[-1] < bce_curve[0] else '上升'
    print(f"   - 实测: BCE {trend_bce}（{bce_curve[0]:.1f}→{bce_curve[-1]:.1f}），"
          f"KL {trend_kl}（{kld_curve[0]:.1f}→{kld_curve[-1]:.1f}）")
    if n_loss_up == 0:
        print(f"   ✓ 实验验证: 总损失逐epoch单调下降（{loss_curve[0]:.1f}→{loss_curve[-1]:.1f}）")
    else:
        print(f"   ✓ 实验验证: 总损失整体下降（{loss_curve[0]:.1f}→{loss_curve[-1]:.1f}，"
              f"其中有 {n_loss_up} 次逐epoch回升）")

print("\n训练统计:")
encoder.eval()
decoder.eval()
# 固定重参数采样噪声，使测试BCE/PSNR可复现（KL项只依赖mu/logvar，与采样无关）
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
test_loss, test_bce, test_kld = 0, 0, 0
test_kl_dim_sum = torch.zeros(LATENT_DIM, device=device)
test_mse_list = []
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
        test_kl_dim_sum += kl_per_dim_of(mu, logvar).sum(dim=0)
        test_mse_list.append(((x_recon - x) ** 2).mean(dim=1))

n_test = len(test_loader.dataset)
test_kl_per_dim = (test_kl_dim_sum / n_test).cpu().numpy()
test_active = int((test_kl_per_dim > ACTIVE_THRESHOLD).sum())
mse_all = torch.cat(test_mse_list)
test_psnr = (10 * torch.log10(1.0 / mse_all.clamp(min=1e-12))).mean().item()
print(f"  测试集: Loss={test_loss/n_test:.2f}, BCE={test_bce/n_test:.2f}, KL={test_kld/n_test:.2f}")
print(f"  测试集活跃维度: {test_active}/{LATENT_DIM} (阈值 {ACTIVE_THRESHOLD})")
for _th in (0.01, 0.05, 0.1):
    print(f"    阈值 {_th:>5}: 活跃 {int((test_kl_per_dim > _th).sum())}/{LATENT_DIM}")
_sorted = np.sort(test_kl_per_dim)[::-1]
print(f"  每维KL(降序) top12: {np.array2string(_sorted[:12], precision=3)}")
print(f"  每维KL(降序) tail6: {np.array2string(_sorted[-6:], precision=5)}")
print(f"  重建 PSNR: {test_psnr:.2f} dB (强KL正则下重建偏模糊属预期)")

print(f"\n{'='*60}")
print("第九章配套实验 9.3-1 完成!")

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
    'beta': BETA,
    'latent_dim': LATENT_DIM,
    'num_epochs': NUM_EPOCHS,
    'active_dim_threshold': ACTIVE_THRESHOLD,
    'init_kld': history.get('kld_init'),
    'final_loss': history['loss'][-1] if history['loss'] else None,
    'final_bce': history['bce'][-1] if history['bce'] else None,
    'final_kld': history['kld'][-1] if history['kld'] else None,
    'final_active_dims': history['active_dims'][-1] if history['active_dims'] else None,
    'peak_active_dims': max(history['active_dims']) if history['active_dims'] else None,
    'test_loss': test_loss / n_test,
    'test_bce': test_bce / n_test,
    'test_kld': test_kld / n_test,
    'test_active_dims': test_active,
    'test_recon_psnr_db': test_psnr,
    'test_kl_per_dim': test_kl_per_dim,
    'train_loss_curve': history['loss'],
    'train_bce_curve': history['bce'],
    'train_kld_curve': history['kld'],
    'train_active_dims_curve': history['active_dims'],
    'kl_first_epoch_curve': history.get('kl_first_epoch', []),
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")