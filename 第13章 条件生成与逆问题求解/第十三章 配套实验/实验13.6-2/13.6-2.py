# -*- coding: utf-8 -*-
"""
实验13.3-3：DPS去模糊与Inpainting零样本迁移

实验内容：
  - DPS求解图像去模糊逆问题
  - DPS求解Inpainting逆问题（零样本迁移）
  - 展示"质量-多样性权衡"——同一观测y，不同zeta的采样结果

注意：本实验训练DDPM并执行多次DPS采样。
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
    SAVE_DIR = os.path.join(_gdrive, '实验13.3-3')
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
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

print("\n" + "=" * 60)
print("实验13.3-3: DPS去模糊与Inpainting零样本迁移")
print("=" * 60)
print("知识点: 零样本迁移, 质量-多样性权衡")


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
posterior_var = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)
sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)
beta_over_sqrt_1m_ab = betas / sqrt_one_minus_alpha_bars


def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    return (
        sqrt_alpha_bars[t][:, None, None, None] * x_0 +
        sqrt_one_minus_alpha_bars[t][:, None, None, None] * noise
    )


# ============================================================
# 去噪网络: 小型UNet
# ============================================================
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class SinusoidalTimeEmbedding(nn.Module):
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
        h = self.bottleneck(h3, t_emb)
        h = F.interpolate(h, size=(14, 14), mode='nearest')
        h = self.up3(torch.cat([h, h2], dim=1), t_emb)
        h = F.interpolate(h, size=(28, 28), mode='nearest')
        h = self.up2(torch.cat([h, h1], dim=1), t_emb)
        h = self.up1(torch.cat([h, x_t], dim=1), t_emb)
        return self.out_conv(h)


# ============================================================
# DPS采样算法
# ============================================================
def dps_sample(model, y, forward_op, shape, zeta=1.0, n_steps=None):
    """
    DPS 采样。
    注：本实现对似然梯度做单位范数归一化,只取其方向,
        残差强度统一由外部超参 zeta 控制,不依赖 sigma_y,
        故函数签名中未保留 sigma_y。
    """
    model.eval()
    if n_steps is None:
        n_steps = T
    x = torch.randn(shape, device=device)
    for t_idx in reversed(range(T)):
        t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        sqrt_ab_t = sqrt_alpha_bars[t_idx]
        sqrt_1mab_t = sqrt_one_minus_alpha_bars[t_idx]
        with torch.no_grad():
            eps_pred = model(x, t)
        x = x.detach().requires_grad_(True)
        eps_pred_grad = model(x, t)
        x0_hat = (x - sqrt_1mab_t * eps_pred_grad) / sqrt_ab_t
        Ax0_hat = forward_op(x0_hat)
        likelihood_loss = torch.sum((y - Ax0_hat) ** 2)
        likelihood_grad = torch.autograd.grad(likelihood_loss, x)[0]
        # 每个样本独立归一化梯度方向（DPS 论文推荐做法）
        batch_size = likelihood_grad.shape[0]
        grad_norms = likelihood_grad.view(batch_size, -1).norm(dim=1)  # shape=(batch_size,)
        grad_norms = grad_norms.clamp(min=1e-8)  # 防止零范数
        likelihood_grad = likelihood_grad / grad_norms.view(batch_size, 1, 1, 1)
        x = x.detach()
        eps_pred = eps_pred.detach()
        # 后验得分分解: ∇log p(x_t|y) = ∇log p(x_t) + ∇log p(y|x_t)
        # 网络预测 eps_pred ≈ -σ_t * ∇log p(x_t),σ_t = sqrt(1-ᾱ_t)
        # 似然(高斯): ∇log p(y|x_t) = -∇||y-Ax̂||²/(σ_y²) <-- 关键负号!
        # 后验噪声预测: ε̃ = -σ_t * ∇log p(x_t|y) = ε_θ + σ_t * (1/σ_y²) * ∇||y-Ax̂||²
        # 简化常数后: ε̃ = ε_θ + ζ * σ_t * ∇||y-Ax̂||² (加号,不是减号!)
        eps_corrected = eps_pred + zeta * sqrt_1mab_t * likelihood_grad
        with torch.no_grad():
            model_mean = sqrt_recip_alphas[t_idx] * (
                x - beta_over_sqrt_1m_ab[t_idx] * eps_corrected
            )
            if t_idx == 0:
                x = model_mean
            else:
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(posterior_var[t_idx]) * noise
    return x.clamp(-1, 1)


# ============================================================
# 正向算子定义
# ============================================================
class GaussianBlurOperator:
    """高斯模糊正向算子 A: x -> blur(x)"""
    def __init__(self, kernel_size=5, sigma=1.5, device='cpu'):
        self.kernel_size = kernel_size
        self.sigma = sigma
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel = torch.outer(g, g)
        kernel = kernel / kernel.sum()
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).to(device)
        self.pad = kernel_size // 2

    def __call__(self, x):
        return F.conv2d(x, self.kernel, padding=self.pad)


class InpaintingOperator:
    """遮挡算子 A: x -> mask * x"""
    def __init__(self, mask_ratio=0.3, device='cpu'):
        # 用局部 Generator 生成 mask,保证可复现的同时不污染全局 RNG 状态,
        # 避免影响脚本后续 y 的观测噪声、dps_sample 内部采样起点等随机操作。
        gen = torch.Generator(device=device).manual_seed(42)
        self.mask = (torch.rand(1, 1, 28, 28, device=device, generator=gen) > mask_ratio).float()

    def __call__(self, x):
        return self.mask * x


# ============================================================
# 训练函数（含checkpoint resume）
# ============================================================
def train_model(checkpoint_path, num_epochs=100):
    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    start_epoch = 0
    is_final = False

    if os.path.exists(checkpoint_path):
        print(f"\n检测到已保存的模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get('is_final', False):
            print(f"已检测到最终训练完成的模型, 直接加载, 跳过训练过程")
            print(f"  训练轮数: {checkpoint['epoch']+1}")
            print(f"  最终损失: {checkpoint['loss']:.6f}")
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                is_final = True
            except RuntimeError as e:
                print(f"警告: checkpoint与当前模型架构不兼容, 删除后重新训练")
                os.remove(checkpoint_path)
                is_final = False
        else:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"检测到未完成的训练, 从第 {start_epoch+1} 轮继续")
            except RuntimeError as e:
                print(f"警告: checkpoint与当前模型架构不兼容, 删除后重新训练")
                os.remove(checkpoint_path)
                start_epoch = 0

    if not is_final:
        if start_epoch >= num_epochs:
            print(f"  注意: start_epoch({start_epoch}) >= num_epochs({num_epochs}), 无需继续训练")
            is_final = True
        else:
            print(f"\n开始训练 epsilon-prediction DDPM (T={T}, epochs={num_epochs})...")
            print("-" * 75)
            for epoch in range(start_epoch, num_epochs):
                model.train()
                total_loss = 0
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ascii=True, leave=False)
                for x, _ in pbar:
                    x = x.to(device)
                    batch = x.shape[0]
                    t = torch.randint(0, T, (batch,), device=device)
                    noise = torch.randn_like(x)
                    x_t = q_sample(x, t, noise)
                    pred = model(x_t, t)
                    loss = F.mse_loss(pred, noise)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * batch
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                avg_loss = total_loss / len(train_loader.dataset)
                print(f"Epoch {epoch+1:3d}/{num_epochs}  Loss={avg_loss:.6f}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'is_final': False
                }, checkpoint_path)
            torch.save({
                'epoch': num_epochs - 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'is_final': True
            }, checkpoint_path)
            print(f"模型已保存: {checkpoint_path}")
    else:
        print(f"\n使用已训练完成的模型, 跳过训练过程")

    return model


# ============================================================
# 数据加载
# ============================================================
print("\n加载MNIST数据集...")
data_dir = os.path.join(SAVE_DIR, 'data')
os.makedirs(data_dir, exist_ok=True)
# MNIST数据归一化到[-1,1] (与11.4-1、13.3-2修复一致):
# 训练时网络看到的 x_T 分布(由 [-1,1] 数据前向扩散得到)与采样起点
# torch.randn(标准高斯) 在统计意义上更匹配,避免系统性均值偏移拖累重建质量。
# 采样函数末尾的 clamp 同步改为 [-1,1];PSNR 与可视化在转换回 [0,1] 空间后计算。
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1),  # [0,1] -> [-1,1]
])
train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
print(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")


# ============================================================
# 训练扩散模型
# ============================================================
print(f"\n{'='*60}")
print("训练UNet扩散模型...")
print("=" * 60)

CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'dps_inpaint_checkpoint.pth')
num_epochs = 100
model = train_model(CHECKPOINT_PATH, num_epochs=num_epochs)


# ============================================================
# DPS求解图像去模糊与Inpainting（方案A：质量-多样性权衡）
# ============================================================
print(f"\n{'='*60}")
print("DPS求解图像去模糊与Inpainting（质量-多样性权衡）")
print("=" * 60)

print("""
逆问题: y = A(x) + n, A=高斯模糊 或 A=遮挡算子
关键：模型从未见过模糊/遮挡图像——DPS的零样本迁移能力

质量-多样性权衡（对应13.4.3节）：
  zeta小 → 后验分布宽 → 高多样性（采样结果各异）
  zeta大 → 后验分布窄 → 高一致性（采样结果趋近观测）
""")

# zeta值对比：展示"质量-多样性权衡"的核心特性
# 增大ζ值以获得更好的重建质量
zeta_values = [1.0, 3.0, 5.0]
print(f"DPS引导权重对比: zeta = {zeta_values}")

test_images = next(iter(test_loader))[0][:4].to(device)

def compute_psnr(pred, target):
    """pred/target: 在 [-1,1] 空间,统一转换到 [0,1] 再用 MAX=1 计算 PSNR"""
    pred_01 = (pred + 1) / 2
    target_01 = (target + 1) / 2
    mse = torch.mean((pred_01 - target_01)**2).item()
    return 10 * np.log10(1.0 / (mse + 1e-10))

# ---- 去模糊 ----
blur_op = GaussianBlurOperator(kernel_size=7, sigma=2.0, device=device)
sigma_y_blur = 0.02
y_blur = blur_op(test_images) + torch.randn_like(test_images) * sigma_y_blur

print("\nDPS去模糊采样（不同zeta对比）...")
x_hat_blur_list = []
psnr_blur_list = []
for zeta_val in zeta_values:
    print(f"  zeta={zeta_val}...")
    x_hat = dps_sample(model, y_blur, blur_op, shape=test_images.shape, zeta=zeta_val)
    x_hat_blur_list.append(x_hat)
    psnr = compute_psnr(x_hat, test_images)
    psnr_blur_list.append(psnr)
    print(f"    PSNR: {psnr:.2f} dB")

psnr_blur_obs = compute_psnr(y_blur, test_images)
print(f"  模糊观测PSNR: {psnr_blur_obs:.2f} dB")

# ---- Inpainting ----
inpaint_op = InpaintingOperator(mask_ratio=0.3, device=device)
sigma_y_inpaint = 0.01
y_inpaint = inpaint_op(test_images) + torch.randn_like(test_images) * sigma_y_inpaint

print("\nDPS inpainting采样（不同zeta对比）...")
x_hat_inpaint_list = []
psnr_inpaint_list = []
for zeta_val in zeta_values:
    print(f"  zeta={zeta_val}...")
    x_hat = dps_sample(model, y_inpaint, inpaint_op, shape=test_images.shape, zeta=zeta_val)
    x_hat_inpaint_list.append(x_hat)
    psnr = compute_psnr(x_hat, test_images)
    psnr_inpaint_list.append(psnr)
    print(f"    PSNR: {psnr:.2f} dB")

psnr_inpaint_obs = compute_psnr(y_inpaint, test_images)
print(f"  遮挡观测PSNR: {psnr_inpaint_obs:.2f} dB")

# ============================================================
# 可视化：质量-多样性权衡
# ============================================================
# 数据在 [-1,1] 空间,imshow 之前统一转换到 [0,1]

# 图1: DPS去模糊 - 质量-多样性权衡（观测y + 不同zeta的采样）
n_rows = len(zeta_values) + 1
fig1, axes1 = plt.subplots(n_rows, 4, figsize=(16, 4*n_rows))
for i in range(4):
    # 第1行：观测y
    axes1[0, i].imshow(((y_blur[i, 0] + 1) / 2).clamp(0, 1).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes1[0, i].axis('off')
    if i == 0: axes1[0, i].set_ylabel('观测y\n(模糊)', fontsize=11, rotation=0, labelpad=50)
    
    # 第2-N行：不同zeta的采样结果
    for row_idx, (zeta_val, x_hat) in enumerate(zip(zeta_values, x_hat_blur_list)):
        axes1[row_idx+1, i].imshow(((x_hat[i, 0] + 1) / 2).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes1[row_idx+1, i].axis('off')
        if i == 0:
            axes1[row_idx+1, i].set_ylabel(f'ζ={zeta_val}\nPSNR={psnr_blur_list[row_idx]:.1f}dB', 
                                           fontsize=10, rotation=0, labelpad=50)

plt.suptitle('DPS去模糊：质量-多样性权衡（零样本迁移）', fontsize=14, y=1.02)
plt.tight_layout()
fig_path1 = os.path.join(SAVE_DIR, 'DPS去模糊.png')
plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图已保存: {fig_path1}")

# 图2: DPS Inpainting - 质量-多样性权衡（观测y + 不同zeta的采样）
fig2, axes2 = plt.subplots(n_rows, 4, figsize=(16, 4*n_rows))
for i in range(4):
    # 第1行：观测y
    axes2[0, i].imshow(((y_inpaint[i, 0] + 1) / 2).clamp(0, 1).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes2[0, i].axis('off')
    if i == 0: axes2[0, i].set_ylabel('观测y\n(遮挡)', fontsize=11, rotation=0, labelpad=50)
    
    # 第2-N行：不同zeta的采样结果
    for row_idx, (zeta_val, x_hat) in enumerate(zip(zeta_values, x_hat_inpaint_list)):
        axes2[row_idx+1, i].imshow(((x_hat[i, 0] + 1) / 2).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes2[row_idx+1, i].axis('off')
        if i == 0:
            axes2[row_idx+1, i].set_ylabel(f'ζ={zeta_val}\nPSNR={psnr_inpaint_list[row_idx]:.1f}dB', 
                                           fontsize=10, rotation=0, labelpad=50)

plt.suptitle('DPS Inpainting：质量-多样性权衡（零样本迁移）', fontsize=14, y=1.02)
plt.tight_layout()
fig_path2 = os.path.join(SAVE_DIR, 'DPS Inpainting.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图已保存: {fig_path2}")

print("\n" + "=" * 60)
print("实验13.3-3 完成!")
print("=" * 60)
print("""
关键结论:
1. 零样本迁移
   - 模型只训练了去噪（无条件DDPM），但能解决去模糊和inpainting
   - 无需针对特定逆问题重新训练——这是DPS的核心优势

2. 质量-多样性权衡（对应13.4.3节）
   - ζ小(0.1): 采样结果多样化，偏离观测（高多样性，低一致性）
   - ζ大(1.0): 采样结果趋近观测，但可能发散（低多样性，高一致性）
   - ζ=0.5是折中点，平衡质量与多样性

3. DPS vs DiffPIR对比
   - DPS: 修正SDE轨迹，理论清晰（后验得分分解）
   - DiffPIR: 交替去噪+投影，PnP思想的延伸
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
    "zeta_values": zeta_values,
    "sigma_y_blur": round(float(sigma_y_blur), 4),
    "sigma_y_inpaint": round(float(sigma_y_inpaint), 4),
    "PSNR_去模糊_各zeta": {f"zeta={z}": round(float(p), 2) for z, p in zip(zeta_values, psnr_blur_list)},
    "PSNR_去模糊_观测": round(float(psnr_blur_obs), 2),
    "PSNR_inpainting_各zeta": {f"zeta={z}": round(float(p), 2) for z, p in zip(zeta_values, psnr_inpaint_list)},
    "PSNR_inpainting_观测": round(float(psnr_inpaint_obs), 2),
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
