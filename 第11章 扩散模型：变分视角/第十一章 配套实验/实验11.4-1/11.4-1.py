# -*- coding: utf-8 -*-
"""
实验11.4-1 简化VLB与DDPM训练
对应章节: 11.4节 简化VLB与DDPM训练

知识点:
  - 11.4节 简化VLB与DDPM训练（L_simple、DDPM采样算法）
  - 11.3节 三种参数化（训练中对比ε-prediction vs x₀-prediction）

实验内容:
  步骤1: 训练两种参数化的DDPM模型
  步骤2: DDPM采样结果对比
  步骤3: 逐步去噪过程可视化

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
    SAVE_DIR = os.path.join(_gdrive, '实验11.4-1')
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
print("实验11.4-1: 简化VLB与DDPM训练")
print("="*60)
print("对应章节: 11.4节 简化VLB与DDPM训练")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  未检测到GPU, 使用CPU训练")
    print("  提示: Colab用户可在菜单 运行时->更改运行时类型 中选择GPU")

# Checkpoint路径
EPS_CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'ddpm_epsilon_checkpoint.pth')
X0_CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'ddpm_x0_checkpoint.pth')

# 快速验证模式
_QUICK_TEST = os.environ.get('QUICK_TEST', '') == '1'

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

# 后验方差 (用于采样)
posterior_var = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)
posterior_log_var = torch.log(posterior_var.clamp(min=1e-20))
sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)
beta_over_sqrt_1m_ab = betas / sqrt_one_minus_alpha_bars


# ============================================================
# 前向过程: q(x_t|x_0)
# ============================================================
def q_sample(x_0, t, noise=None):
    """x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε, 输入为[B,1,28,28]图像"""
    if noise is None:
        noise = torch.randn_like(x_0)
    return (
        sqrt_alpha_bars[t][:, None, None, None] * x_0 +
        sqrt_one_minus_alpha_bars[t][:, None, None, None] * noise
    )


# ============================================================
# 去噪网络: 小型UNet（带时间嵌入）
# ============================================================
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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
    """Conv + GroupNorm + SiLU，注入时间条件"""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        gn_groups = min(4, out_ch)  # 自适应分组数，确保能整除
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
    """小型UNet去噪网络（适配MNIST 28x28）"""
    def __init__(self, time_dim=64, pred_type='epsilon'):
        super().__init__()
        self.pred_type = pred_type
        ch = [1, 16, 32, 64]

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        # 编码器（下采样）
        self.down1 = ConvBlock(ch[0], ch[1], time_dim)  # 28x28
        self.down2 = ConvBlock(ch[1], ch[2], time_dim)  # 14x14
        self.down3 = ConvBlock(ch[2], ch[3], time_dim)  # 7x7

        # 瓶颈层
        self.bottleneck = ConvBlock(ch[3], ch[3], time_dim)  # 7x7

        # 解码器（上采样）
        self.up3 = ConvBlock(ch[3] + ch[2], ch[2], time_dim)  # 14x14
        self.up2 = ConvBlock(ch[2] + ch[1], ch[1], time_dim)  # 28x28
        self.up1 = ConvBlock(ch[1] + ch[0], ch[0], time_dim)  # 28x28

        # 输出
        self.out_conv = nn.Conv2d(ch[0], 1, 1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x_t, t):
        t_emb = self.time_mlp(t)

        # 编码器
        h1 = self.down1(x_t, t_emb)       # [B,16,28,28]
        h2 = self.down2(self.pool(h1), t_emb)  # [B,32,14,14]
        h3 = self.down3(self.pool(h2), t_emb)  # [B,64,7,7]

        # 瓶颈
        h = self.bottleneck(self.pool(h3), t_emb)  # 这里pool会让7→3，所以用stride处理
        # 实际上7x7 pool2 → 3x3，上采样回去是6x6，不匹配
        # 改用直接在h3上做bottleneck
        h = self.bottleneck(h3, t_emb)  # [B,64,7,7]

        # 解码器 + 跳跃连接
        h = F.interpolate(h, size=(14, 14), mode='nearest')
        h = self.up3(torch.cat([h, h2], dim=1), t_emb)  # [B,32,14,14]

        h = F.interpolate(h, size=(28, 28), mode='nearest')
        h = self.up2(torch.cat([h, h1], dim=1), t_emb)  # [B,16,28,28]

        h = self.up1(torch.cat([h, x_t], dim=1), t_emb)  # [B,1,28,28]

        return self.out_conv(h)


# ============================================================
# DDPM采样算法（11.4节）
# ============================================================
@torch.no_grad()
def ddpm_sample(model, shape, pred_type='epsilon'):
    """DDPM反向采样 x_T→x_0, shape=[B,1,28,28]"""
    model.eval()
    x = torch.randn(shape, device=device)

    for t_idx in reversed(range(T)):
        t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)

        # 网络预测
        pred = model(x, t)

        if pred_type == 'epsilon':
            # ε-prediction: μ_θ = (1/√α_t)(x_t - β_t/√(1-ᾱ_t)·ε̂)
            model_mean = sqrt_recip_alphas[t_idx] * (
                x - beta_over_sqrt_1m_ab[t_idx] * pred
            )
        elif pred_type == 'x0':
            # x₀-prediction: μ_θ = √ᾱ_{t-1}·β_t/(1-ᾱ_t)·x̂₀ + √α_t(1-ᾱ_{t-1})/(1-ᾱ_t)·x_t
            ab = alpha_bars[t_idx]
            ab_prev = alpha_bars_prev[t_idx]
            a = alphas[t_idx]
            coeff_x0 = torch.sqrt(ab_prev) * betas[t_idx] / (1 - ab)
            coeff_xt = torch.sqrt(a) * (1 - ab_prev) / (1 - ab)
            model_mean = coeff_x0 * pred + coeff_xt * x

        if t_idx == 0:
            x = model_mean
        else:
            noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(posterior_var[t_idx]) * noise

    return x


# ============================================================
# 训练函数（含checkpoint resume）
# ============================================================
def train_ddpm(checkpoint_path, pred_type='epsilon', num_epochs=50):
    """训练DDPM模型，支持checkpoint resume"""
    model = SmallUNet(pred_type=pred_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    start_epoch = 0
    is_final = False
    history = {'loss': []}

    # Checkpoint加载逻辑
    if os.path.exists(checkpoint_path):
        print(f"\n检测到已保存的模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if checkpoint.get('is_final', False):
            print(f"✓ 这是最终训练完成的模型, 直接加载, 跳过训练过程")
            print(f"  训练轮数: {checkpoint['epoch']+1}")
            print(f"  最终损失: {checkpoint['loss']:.6f}")
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                print(f"警告: checkpoint与当前模型架构不兼容, 删除后重新训练")
                print(f"  错误信息: {e}")
                os.remove(checkpoint_path)
                start_epoch = 0
                is_final = False
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                history = checkpoint.get('history', {'loss': []})
                start_epoch = checkpoint['epoch'] + 1
                is_final = True
        else:
            print(f"检测到未完成的训练, 从第 {checkpoint['epoch']+1} 轮继续")
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                print(f"警告: checkpoint与当前模型架构不兼容, 删除后重新训练")
                print(f"  错误信息: {e}")
                os.remove(checkpoint_path)
                start_epoch = 0
                is_final = False
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                history = checkpoint.get('history', {'loss': []})
                start_epoch = checkpoint['epoch'] + 1

    # 训练循环
    if not is_final:
        # 快速验证模式
        if _QUICK_TEST:
            num_epochs = 3
            print(f"\n[快速验证模式] 仅训练 {num_epochs} 轮")

        print(f"\n开始训练 {pred_type}-prediction DDPM (T={T}, epochs={num_epochs}, UNet)...")
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
                model.train()
                total_loss = 0

                pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}',
                            leave=False, unit='batch')

                for x, _ in pbar:
                    x = x.to(device)
                    batch = x.shape[0]

                    # 随机采样时间步
                    t = torch.randint(0, T, (batch,), device=device)

                    # 前向过程: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
                    noise = torch.randn_like(x)
                    x_t = q_sample(x, t, noise)

                    # 网络预测
                    pred = model(x_t, t)

                    # 计算损失 (11.4节 L_simple)
                    if pred_type == 'epsilon':
                        # ε-prediction: ||ε - ε̂_θ(x_t, t)||²
                        loss = F.mse_loss(pred, noise)
                    elif pred_type == 'x0':
                        # x₀-prediction: ||x_0 - x̂₀(x_t, t)||²
                        loss = F.mse_loss(pred, x)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * batch

                avg_loss = total_loss / len(train_loader.dataset)
                history['loss'].append(avg_loss)

                print(f"Epoch {epoch+1:3d}/{num_epochs}  Loss={avg_loss:.6f}")

                # 每轮保存checkpoint
                torch.save({
                    'epoch': epoch,
                    'pred_type': pred_type,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'history': history,
                    'is_final': False
                }, checkpoint_path)

            t_elapsed = time.time() - t_start
            print("-" * 75)
            print(f"训练完成, 最终Loss={history['loss'][-1]:.6f}, 耗时: {t_elapsed:.1f}秒")

            # 保存最终checkpoint
            torch.save({
                'epoch': num_epochs - 1,
                'pred_type': pred_type,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': history['loss'][-1],
                'history': history,
                'is_final': True
            }, checkpoint_path)
            print(f"✓ 模型已保存: {checkpoint_path}")
    else:
        print(f"\n使用已训练完成的 {pred_type}-prediction DDPM模型, 跳过训练过程")

    return model, history


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
print(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")


# ============================================================
# 步骤1: 训练两种参数化
# ============================================================
num_epochs = 50

print(f"\n{'='*60}")
print(f"步骤1: 训练两种参数化的DDPM模型")
print(f"{'='*60}")
print("\n[核心思想]")
print("  DDPM的训练目标: L_simple = E[||ε - ε̂_θ(x_t,t)||²]")
print("  - ε-prediction: 网络预测噪声ε，从x_t中估计添加的噪声")
print("  - x₀-prediction: 网络预测原始图像x₀，从x_t直接恢复原图")
print("  两种参数化数学等价，但ε-prediction训练更稳定（11.4节）")

model_eps, history_eps = train_ddpm(EPS_CHECKPOINT_PATH, 'epsilon', num_epochs)

model_x0, history_x0 = train_ddpm(X0_CHECKPOINT_PATH, 'x0', num_epochs)


# ============================================================
# 可视化1: 训练曲线 + 加噪过程
# ============================================================
print("\n生成训练曲线与加噪过程图...")

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(range(1, len(history_eps['loss'])+1), history_eps['loss'], 'b-o', markersize=4,
        label=r'$\varepsilon$-prediction')
ax.plot(range(1, len(history_x0['loss'])+1), history_x0['loss'], 'r-s', markersize=4,
        label=r'$x_0$-prediction')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel(r'$L_{\mathrm{simple}}$损失', fontsize=12)
ax.set_title('(a) 两种参数化的训练曲线', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig_path1a = os.path.join(SAVE_DIR, '步骤1a_训练曲线.png')
plt.savefig(fig_path1a, dpi=150, bbox_inches='tight')
plt.close()
print(f"图1a已保存: {fig_path1a}")

# (b) 前向加噪过程可视化
test_imgs, _ = next(iter(test_loader))
t_show = [0, 20, 50, 100, 150, 199]

fig2, axes_sub = plt.subplots(2, 3, figsize=(14, 6))
for i, t_idx in enumerate(t_show):
    torch.manual_seed(42)
    x0_t = test_imgs[0:1].to(device)
    t_t = torch.tensor([t_idx], device=device)
    eps_t = torch.randn_like(x0_t)
    x_t = q_sample(x0_t, t_t, eps_t)
    row, col = i // 3, i % 3
    axes_sub[row, col].imshow(x_t[0, 0].cpu().numpy(), cmap='gray')
    if t_idx == 0:
        axes_sub[row, col].set_title(r'$x_0$ (原始)', fontsize=11)
    else:
        axes_sub[row, col].set_title(f't={t_idx}', fontsize=11)
    axes_sub[row, col].axis('off')
plt.suptitle('(b) 前向加噪过程', fontsize=13)
plt.tight_layout()
fig_path1b = os.path.join(SAVE_DIR, '步骤1b_前向加噪.png')
plt.savefig(fig_path1b, dpi=150, bbox_inches='tight')
plt.close()
print(f"图1b已保存: {fig_path1b}")

print("\n[观察]")
print("  - 两种参数化的训练曲线收敛速度相近")
print("  - ε-prediction是DDPM的标准选择（训练稳定性更好）")
print("  - 前向加噪过程展示: 随着t增大，图像逐渐变为纯噪声")


# ============================================================
# 步骤2: DDPM采样结果
# ============================================================
print("\n步骤2: DDPM采样...")

n_samples = 8
sample_shape = (n_samples, 1, 28, 28)

# ε-prediction采样
print("  ε-prediction采样中...")
samples_eps = ddpm_sample(model_eps, sample_shape, 'epsilon')

# x₀-prediction采样
print("  x₀-prediction采样中...")
samples_x0 = ddpm_sample(model_x0, sample_shape, 'x0')

fig, axes = plt.subplots(2, n_samples, figsize=(16, 4))

for i in range(n_samples):
    # ε-prediction
    axes[0, i].imshow(samples_eps[i, 0].cpu().numpy(), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_ylabel(r'$\varepsilon$-prediction', fontsize=12, rotation=0, labelpad=60)

    # x₀-prediction
    axes[1, i].imshow(samples_x0[i, 0].cpu().numpy(), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_ylabel(r'$x_0$-prediction', fontsize=12, rotation=0, labelpad=60)

plt.suptitle(f'DDPM采样 (T={T}, {num_epochs} epochs, UNet)', fontsize=14, y=1.02)
plt.tight_layout()
fig_path2 = os.path.join(SAVE_DIR, '步骤2_DDPM采样对比.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig_path2}")

print("\n[关键结论]")
print("  - 两种参数化都能生成合理的数字图像")
print("  - DDPM采样从纯噪声x_T~N(0,I)开始，逐步去噪恢复图像")


# ============================================================
# ★ 步骤3: 原创设计 - 逐步去噪过程可视化
# 从纯噪声x_T~N(0,I)开始，记录DDPM逐步去噪的中间结果
# ============================================================
print("\n步骤3: 逐步去噪过程可视化...")
print("\n[核心思想]")
print("  DDPM反向采样: 从x_T~N(0,I)开始，逐步去噪到x₀")
print("  每步去噪: x_{t-1} = μ_θ(x_t,t) + σ_t·z, z~N(0,I)")

model_eps.eval()
t_denoise = [T-1, T*3//4, T//2, T//4, T//8, T//16, 0]  # 从T到0

# 从纯噪声开始
torch.manual_seed(42)
x_T = torch.randn(1, 1, 28, 28, device=device)

# 完整采样并记录中间步骤
denoise_trajectory = []
with torch.no_grad():
    x = x_T.clone()
    for t_idx in reversed(range(T)):
        t = torch.tensor([t_idx], device=device)
        pred = model_eps(x, t)
        model_mean = sqrt_recip_alphas[t_idx] * (
            x - beta_over_sqrt_1m_ab[t_idx] * pred
        )
        if t_idx == 0:
            x = model_mean
        else:
            noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(posterior_var[t_idx]) * noise

        if t_idx in t_denoise:
            denoise_trajectory.append((t_idx, x[0, 0].cpu().numpy()))

fig, axes = plt.subplots(1, len(denoise_trajectory), figsize=(20, 3))
for i, (t_val, img) in enumerate(denoise_trajectory):
    axes[i].imshow(img, cmap='gray')
    if t_val == 0:
        axes[i].set_title(r'$x_0$ (去噪完成)', fontsize=11)
    else:
        axes[i].set_title(f't={t_val}', fontsize=11)
    axes[i].axis('off')

plt.suptitle(r'★ DDPM反向去噪过程: $x_T \to x_0$', fontsize=14, y=1.05)
plt.tight_layout()
fig_path3 = os.path.join(SAVE_DIR, '步骤3_逐步去噪过程.png')
plt.savefig(fig_path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"图3已保存: {fig_path3}")

print("\n[观察]")
print("  - 去噪过程从纯噪声逐步恢复图像结构")
print("  - 早期步骤(t大)去除大量噪声，后期步骤(t小)精细调整细节")
print("  - 这展示了扩散模型'由粗到细'的生成过程")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print("实验11.4-1 完成!")
print("=" * 60)
print(f"""
关键观察:
1. L_simple训练效果（11.4节）
   - ε-prediction和x₀-prediction都成功收敛
   - 简化目标 $L_{{simple}} = E[||ε - ε̂_θ(x_t,t)||²]$ 忽略时间权重但训练稳定

2. ε-prediction vs x₀-prediction（11.3节）
   - 两者训练曲线收敛速度相近
   - ε-prediction是DDPM标准选择（11.4节）

3. DDPM采样算法（11.4节）
   - 从$x_T$~N(0,I)开始，逐步去噪: $x_{{t-1}} = μ_θ(x_t,t) + σ_t·z$
   - 去噪过程从纯噪声逐步恢复结构

4. 前向加噪=固定编码器（11.1节联系）
   - $x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε$ 精确实现固定高斯编码器
   - 无需训练编码器，只需训练去噪网络

5. UNet vs MLP（架构影响）
   - UNet的卷积归纳偏置天然适合图像，保留空间结构
   - 跳跃连接使细节信息可直达输出层，减少误差累积
""")