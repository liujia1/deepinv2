# -*- coding: utf-8 -*-
"""
实验12.6-1 实践意义与训练目标选择
对应章节: 12.6 实践意义与训练目标选择

知识点:
  - 简化DSM目标与简化VLB目标在代码层面完全相同
  - 采样路径(DSM)和变分路径(VLB)的训练目标都是 ||ε - ε̂_θ||²
  - 两条路径产生相同的训练动态和采样结果
  - 同一个网络输出 ε̂_θ 同时承载得分函数、逆向均值、x₀估计三种信息

实验内容:
  步骤1: 训练两个模型（相同架构+相同目标，不同随机种子）
  步骤2: 对比训练曲线
  步骤3: 对比采样结果
  步骤4: 双向验证——从 ε̂_θ 同时提取得分函数、逆向均值、x₀估计

运行前提: PyTorch + GPU推荐
"""

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys
import io
import time
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

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验12.6-1')
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

# 设置随机种子
np.random.seed(42)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"实验12.6-1: 实践意义与训练目标选择")
print(f"{'='*60}")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  未检测到 GPU, 使用 CPU 训练")
    print("  提示: Colab 用户可在菜单 运行时 -> 更改运行时类型 中选择 GPU")


# ============================================================
# 噪声调度: DDPM线性调度
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

print(f"\n噪声调度: T={T}, β_min={beta_min}, β_max={beta_max} (DDPM线性调度)")


# ============================================================
# 前向过程: q(x_t|x_0)
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
# 去噪网络: SmallUNet
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间嵌入"""
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
    """卷积块: Conv + GroupNorm + SiLU + 时间注入 + 残差连接"""
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
    """小型UNet去噪网络 (通道数 [1, 16, 32, 64], 时间嵌入维度64)"""
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
        h = self.bottleneck(h3, t_emb)           # bottleneck只调用一次
        h = F.interpolate(h, size=(14, 14), mode='nearest')
        h = self.up3(torch.cat([h, h2], dim=1), t_emb)
        h = F.interpolate(h, size=(28, 28), mode='nearest')
        h = self.up2(torch.cat([h, h1], dim=1), t_emb)
        h = self.up1(torch.cat([h, x_t], dim=1), t_emb)
        return self.out_conv(h)


# ============================================================
# DDPM采样算法
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
# 数据加载 (归一化到[-1,1])
# ============================================================
print("\n加载MNIST数据集...")
data_dir = os.path.join(SAVE_DIR, 'data')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)  # 归一化到[-1,1]
])
train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
print(f"训练集: {len(train_dataset)} 样本")


# ============================================================
# 步骤1：训练两个模型（相同架构+相同目标，不同随机种子）
# ============================================================
print(f"\n{'='*60}")
print("步骤1：训练两个模型（相同架构+相同目标，不同随机种子）")
print(f"{'='*60}")
print("""
核心设计:
  - 模型A (采样路径): 从DSM视角理解，训练目标为 ||ε - ε̂_θ||²
  - 模型B (变分路径): 从VLB视角理解，训练目标为 ||ε - ε̂_θ||²
  两个路径的训练目标完全相同，仅理论解释不同！

重要说明:
  本实验不再重复12.4-1的权重公式验证（那里已分别计算DSM权重wt_dsm和VLB权重
  w_t^vlb并验证等价性）。本实验展示的是——正因为12.3/12.4节已证明简化DSM≡简化VLB，
  所以在实现层面这两条路径**只需要写一份代码**。步骤1-3展示的是这份代码本身的
  训练/采样行为，而非重新验证等价性。两次运行仅随机种子不同，用于展示训练动态
  的稳定性，不等价于独立的等价性验证。
""")

num_epochs = 50


def train_model(label, seed, checkpoint_path):
    """训练去噪网络（简化DSM = 简化VLB）

    采样路径解释：网络预测噪声ε̂_θ，训练目标为DSM损失 ||ε - ε̂_θ||²
    变分路径解释：网络预测噪声ε̂_θ，训练目标为简化VLB损失 ||ε - ε̂_θ||²
    关键：两个解释下，代码完全相同！
    """
    # 设置不同的随机种子以产生独立初始化
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Checkpoint加载逻辑
    start_epoch = 0
    is_final = False
    train_losses = []

    if os.path.exists(checkpoint_path):
        print(f"\n检测到已保存的模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'train_losses' not in checkpoint:
            raise RuntimeError(
                f"检测到旧版本 checkpoint (缺少 'train_losses' 字段):\n"
                f"  {checkpoint_path}\n"
                f"请删除该文件后重新训练."
            )
        if checkpoint.get('is_final', False):
            print(f"✓ 这是最终训练完成的模型, 直接加载, 跳过训练过程")
            print(f"  训练轮数: {checkpoint['epoch']+1}")
            print(f"  最终损失: {checkpoint['loss']:.6f}")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            train_losses = checkpoint.get('train_losses', [])
            start_epoch = checkpoint['epoch'] + 1
            is_final = True
        else:
            print(f"检测到未完成的训练, 从第 {checkpoint['epoch']+1} 轮继续")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            train_losses = checkpoint.get('train_losses', [])
            start_epoch = checkpoint['epoch'] + 1

    # 训练循环
    if not is_final:
        # 快速验证模式
        _num_epochs = num_epochs
        if os.environ.get('QUICK_TEST', '') == '1':
            _num_epochs = 3
            print(f"  [快速验证模式] 仅训练 {_num_epochs} 轮")

        print(f"\n训练 [{label}] ({_num_epochs} 轮)...")
        t_start = time.time()

        # 边界保护
        if start_epoch >= _num_epochs:
            print(f"  注意: start_epoch({start_epoch}) >= num_epochs({_num_epochs}), 无需继续训练")
            if not train_losses:
                print(f"  警告: 无历史损失记录")
            is_final = True

        if not is_final:
            for epoch in range(start_epoch, _num_epochs):
                model.train()
                epoch_loss = 0.0
                pbar = tqdm(train_loader, desc=f'[{label}] Epoch {epoch+1}/{_num_epochs}',
                            leave=False, unit='batch')
                for x_batch, _ in pbar:
                    x_batch = x_batch.to(device)
                    batch = x_batch.shape[0]
                    t = torch.randint(0, T, (batch,), device=device)
                    noise = torch.randn_like(x_batch)
                    x_t = q_sample(x_batch, t, noise)
                    pred = model(x_t, t)
                    # 两个路径的损失完全相同：||ε - ε̂_θ||²
                    loss = F.mse_loss(pred, noise)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * batch
                    pbar.set_postfix(loss=f'{loss.item():.4f}')

                avg_loss = epoch_loss / len(train_loader.dataset)
                train_losses.append(avg_loss)

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"  [{label}] Epoch {epoch+1:3d}/{_num_epochs}  Loss={avg_loss:.6f}")

                # 每5轮保存中间checkpoint
                if (epoch + 1) % 5 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                        'train_losses': train_losses,
                        'is_final': False
                    }, checkpoint_path)

            t_elapsed = time.time() - t_start
            print(f"  [{label}] 训练完成, 最终损失: {train_losses[-1]:.6f}, 耗时: {t_elapsed:.1f} 秒")

        # 保存最终checkpoint
        if train_losses:
            torch.save({
                'epoch': _num_epochs - 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_losses[-1],
                'train_losses': train_losses,
                'is_final': True
            }, checkpoint_path)
            print(f"✓ [{label}] 模型已保存: {checkpoint_path}")
    else:
        print(f"\n使用已训练完成的 [{label}] 模型, 跳过训练过程")

    # 恢复全局随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    return model, train_losses


# 两个模型各自的checkpoint路径
CHECKPOINT_A = os.path.join(SAVE_DIR, 'dsm_model_checkpoint.pth')
CHECKPOINT_B = os.path.join(SAVE_DIR, 'vlb_model_checkpoint.pth')

# 训练模型A: 采样路径(DSM), 种子=42
model_a, losses_a = train_model('采样路径(DSM)', seed=42, checkpoint_path=CHECKPOINT_A)

# 训练模型B: 变分路径(VLB), 种子=123
model_b, losses_b = train_model('变分路径(VLB)', seed=123, checkpoint_path=CHECKPOINT_B)


# ============================================================
# 步骤2：对比训练曲线
# ============================================================
print(f"\n{'='*60}")
print("步骤2：对比训练曲线")
print(f"{'='*60}")
print("两条路径的训练目标都是 ||ε - ε̂_θ||², 总体收敛趋势相近（细节差异来自不同随机初始化）")

if losses_a and losses_b:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(range(1, len(losses_a)+1), losses_a, 'b-o', markersize=4,
            label='采样路径 (DSM)', alpha=0.8)
    ax.plot(range(1, len(losses_b)+1), losses_b, 'r-s', markersize=4,
            label='变分路径 (VLB)', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(r'Loss = $\mathbb{E}[\|\epsilon - \hat{\epsilon}_\theta\|^2]$', fontsize=12)
    ax.set_title('两条路径的训练曲线对比', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig_path1 = os.path.join(SAVE_DIR, '步骤1_训练曲线对比.png')
    plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: 步骤1_训练曲线对比.png")


# ============================================================
# 步骤3：对比采样结果
# ============================================================
print(f"\n{'='*60}")
print("步骤3：对比采样结果")
print(f"{'='*60}")
print("两个模型各采样8张, 上下排列对比")

n_samples = 8
sample_shape = (n_samples, 1, 28, 28)

print("  采样路径模型采样中...")
samples_a = ddpm_sample(model_a, sample_shape)
print("  变分路径模型采样中...")
samples_b = ddpm_sample(model_b, sample_shape)

# 反归一化: [-1,1] -> [0,1]
samples_a_img = (samples_a + 1) / 2
samples_a_img = samples_a_img.clamp(0, 1)
samples_b_img = (samples_b + 1) / 2
samples_b_img = samples_b_img.clamp(0, 1)

fig, axes = plt.subplots(2, n_samples, figsize=(16, 4))
fig.subplots_adjust(left=0.10, right=0.98, top=0.85, bottom=0.05)
for i in range(n_samples):
    axes[0, i].imshow(samples_a_img[i, 0].cpu().numpy(), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(samples_b_img[i, 0].cpu().numpy(), cmap='gray')
    axes[1, i].axis('off')

fig.text(0.02, 0.70, '采样路径\n(DSM→Score→SDE)', fontsize=10, ha='left', va='center', rotation=0)
fig.text(0.02, 0.30, '变分路径\n(VLB→μ_θ→DDPM)', fontsize=10, ha='left', va='center', rotation=0)

fig_path2 = os.path.join(SAVE_DIR, '步骤2_采样结果对比.png')
fig.suptitle('两条路径的采样结果对比', fontsize=14, y=0.98)
plt.savefig(fig_path2, dpi=150)
plt.close()
print(f"图表已保存: 步骤2_采样结果对比.png")


# ============================================================
# 步骤4：双向验证
# ============================================================
print(f"\n{'='*60}")
print("步骤4：双向验证")
print(f"{'='*60}")
print("""
12.4节等价性的实践含义:
  - 从同一个网络输出 ε̂_θ，可以同时提取三种信息:
    1. 得分函数 s_θ = -ε̂_θ / √(1-ᾱ_t)  (采样路径)
    2. 逆向均值 μ_θ = (1/√α_t)(x_t - β_t/√(1-ᾱ_t)·ε̂_θ)  (变分路径)
    3. x₀估计 x̂₀ = (x_t - √(1-ᾱ_t)·ε̂_θ) / √ᾱ_t  (Tweedie等式)
""")

# 使用采样路径模型进行验证
model_a.eval()
x_test = torch.randn(1, 1, 28, 28, device=device)
t_val = 50
t_test = torch.tensor([t_val], device=device)

with torch.no_grad():
    eps_pred = model_a(x_test, t_test)  # 网络输出: ε̂_θ

    # 采样路径视角: 提取得分函数
    ab_t = alpha_bars[t_val]
    score_from_eps = -eps_pred / torch.sqrt(1 - ab_t)

    # 变分路径视角: 计算逆向均值
    mu_from_eps = (1 / torch.sqrt(alphas[t_val])) * (
        x_test - (betas[t_val] / torch.sqrt(1 - ab_t)) * eps_pred
    )

    # Tweedie等式: 提取x₀估计
    x0_from_eps = (x_test - torch.sqrt(1 - ab_t) * eps_pred) / torch.sqrt(ab_t)

print("从同一个网络输出 ε̂_θ 提取的不同量:")
print(f"  ε̂_θ 范数: {eps_pred.norm().item():.4f}  (网络直接输出)")
print(f"  s_θ = -ε̂_θ/√(1-ᾱ_t) 范数: {score_from_eps.norm().item():.4f}  (采样路径: 得分函数)")
print(f"  μ_θ 范数: {mu_from_eps.norm().item():.4f}  (变分路径: 逆向均值)")
print(f"  x̂₀ = (x_t-√(1-ᾱ_t)ε̂)/√ᾱ_t 范数: {x0_from_eps.norm().item():.4f}  (去噪估计)")
print()
print("→ 一个网络输出，三种解读，三种用途")
print("  这就是12.4节 DSM≡VLB 等价性的实践体现！")


# ============================================================
# 实验总结
# ============================================================
print(f"\n{'='*60}")
print("实验12.6-1 总结")
print(f"{'='*60}")
print("""
1. 训练目标等价 (12.4节)
   - 简化DSM和简化VLB的训练目标都是 ||ε - ε̂_θ||²
   - 两条路径的训练曲线总体收敛趋势相近（细节差异来自不同随机初始化）

2. 采样结果一致
   - 两条路径训练出的模型，采样质量相当
   - 差异仅来自随机初始化不同

3. 双向验证 (12.4节实践验证)
   - 从同一个网络输出 ε̂_θ，同时提取得分函数、逆向均值、x₀估计
   - 采样路径训练的模型可直接用于DDPM采样（变分路径的采样方式）
   - 变分路径训练的模型可直接提取得分函数（采样路径的核心量）

4. 实践意义 (12.6节)
   - 选择哪种路径解释不影响训练和采样——代码完全相同
   - 差异仅在于"如何理解"模型：得分匹配视角 vs 变分推断视角
   - 在实践中，无需纠结选择DSM还是VLB，简化目标下两者完全等价
   - 关键是理解不同视角的互补性，而非在代码层面做选择
""")

print(f"{'='*60}")
print("第十二章配套实验12.6-1完成!")
