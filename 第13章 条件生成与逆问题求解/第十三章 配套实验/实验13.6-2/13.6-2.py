# -*- coding: utf-8 -*-
"""
实验13.6-2：DPS去模糊与Inpainting零样本迁移

实验内容：
  - DPS求解图像去模糊逆问题
  - DPS求解图像Inpainting逆问题（零样本迁移）
  - 质量对比：不同ζ使用CRN公共随机数（同一反向噪声路径），
    PSNR报告3条独立轨迹的均值±标准差，避免单轨迹数值的偶然性
  - 多样性直接测量：固定同一观测y，每个ζ取4个不同初始噪声的条件样本，
    用样本间MSE与逐像素std量化"质量-多样性权衡"，
    区分"后验多模态的有用多样性"与"引导不足的先验幻觉"
  - ζ网格外推至100，实证检验"大ζ过拟合回落"在本任务是否出现

注意：本实验训练DDPM并执行多次DPS采样（复用checkpoint时仅采样，CPU约数分钟）。
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
    SAVE_DIR = os.path.join(_gdrive, '实验13.6-2')
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
print("实验13.6-2: DPS去模糊与Inpainting零样本迁移")
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
def dps_sample(model, y, forward_op, shape, zeta=1.0, seed=None):
    """
    DPS 采样（单位范数归一化梯度参数化）。

    注：本实现对似然梯度做逐样本单位范数归一化,只取其方向,
        残差强度统一由外部超参 zeta 控制,不依赖 sigma_y
       （1/σ_y² 已被吸收进 ζ）,故函数签名中未保留 sigma_y。
        这与 Chung et al.(2022) 官方代码 ∇‖r‖ 的隐式归一化同类,
        但 ζ 的数值尺度与论文惯例(0.1~1.0)不可直接对比,本实现最优 ζ≈10 量级。

    seed:
      - 不同 ζ 传入同一种子 => 公共随机数(CRN),各 ζ 共享同一初始 x_T 与
        每一步反向噪声,ζ-PSNR 曲线只反映引导强度差异(质量对比用);
      - 同一 y 传入不同种子 => 同一后验的多个独立条件样本(多样性测量用)。

    实现说明:
      - 单次前向传播,带 grad 的 eps_pred 同时用于 x0_hat 估计与似然反传;
      - 后验得分分解给出 eps_corrected = eps_θ + ζ·σ_t·∇‖y-Ax̂₀‖²(加号):
        该加号与 Chung 官方代码 x_{t-1} -= scale·∇‖r‖ 方向严格一致
       （ε=-σ_t·score,似然得分是负梯度,负负得正);
      - 引导进入均值步进后 σ_t 恰相消,实际位移 ∝ ζ·β_t/√α_t·ĝ,
        末段 β_t→β_min 使引导相对注入噪声(∝√β_t)自动退场。
    """
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(shape, device=device)
    for t_idx in reversed(range(T)):
        t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        sqrt_ab_t = sqrt_alpha_bars[t_idx]
        sqrt_1mab_t = sqrt_one_minus_alpha_bars[t_idx]
        # 单次前向：输出同时服务于 x0_hat(Tweedie) 与梯度反传
        x = x.detach().requires_grad_(True)
        eps_pred = model(x, t)
        x0_hat = (x - sqrt_1mab_t * eps_pred) / sqrt_ab_t
        Ax0_hat = forward_op(x0_hat)
        likelihood_loss = torch.sum((y - Ax0_hat) ** 2)
        likelihood_grad = torch.autograd.grad(likelihood_loss, x)[0]
        # 每个样本独立归一化梯度方向（DPS 论文推荐做法）
        batch_size = likelihood_grad.shape[0]
        grad_norms = likelihood_grad.view(batch_size, -1).norm(dim=1)  # (batch_size,)
        grad_norms = grad_norms.clamp(min=1e-8)  # 防止零范数
        likelihood_grad = likelihood_grad / grad_norms.view(batch_size, 1, 1, 1)
        # 后验噪声预测修正（加号，见docstring推导）
        eps_corrected = eps_pred.detach() + zeta * sqrt_1mab_t * likelihood_grad
        with torch.no_grad():
            model_mean = sqrt_recip_alphas[t_idx] * (
                x.detach() - beta_over_sqrt_1m_ab[t_idx] * eps_corrected
            )
            if t_idx == 0:
                x = model_mean
            else:
                x = model_mean + torch.sqrt(posterior_var[t_idx]) * torch.randn_like(x)
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
        # replicate padding：零填充在 [-1,1] 空间等价于补“中灰”(0)，
        # 会把 MNIST 的黑色边界拉亮成光晕（实测边界偏差最大 0.27）。
        x = F.pad(x, (self.pad,) * 4, mode='replicate')
        return F.conv2d(x, self.kernel, padding=0)


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
# DPS求解图像去模糊与Inpainting（质量对比 + 多样性直接测量）
# ============================================================
print(f"\n{'='*60}")
print("DPS求解图像去模糊与Inpainting（质量-多样性权衡）")
print("=" * 60)

print("""
逆问题: y = A(x) + n, A=高斯模糊 或 A=遮挡算子
关键：模型从未见过模糊/遮挡图像——DPS的零样本迁移能力

实验设计（对应13.4.3节"质量-多样性权衡"）：
  质量：不同ζ使用CRN公共随机数(同一反向噪声路径), 3条独立轨迹报告均值±标准差
  多样性：固定同一观测y, 每个ζ用4个不同初始噪声采样,
          用样本间MSE与逐像素std直接测量后验宽窄
  ζ小 → 先验主导、引导不足 → 样本彼此分歧(但多为偏离真值的幻觉)
  ζ大 → 数据一致性主导 → 样本收敛、PSNR高
""")

# ζ网格：1(欠引导)→10/20(良好重建)→50/100(检验过拟合回落)
zeta_quality = [1.0, 10.0, 20.0, 50.0, 100.0]
zeta_div = [1.0, 10.0, 20.0, 50.0]
CRN_SEED = 42                    # 出图主轨迹种子；各ζ共用 => CRN
STAT_SEEDS = (42, 43, 44)        # PSNR统计：3条独立采样轨迹
DIV_SEEDS = (101, 102, 103, 104) # 多样性：同一y的4个独立条件样本
print(f"质量网格 zeta = {zeta_quality}（每ζ {len(STAT_SEEDS)} 个独立种子, 主轨迹CRN seed={CRN_SEED}）")
print(f"多样性网格 zeta = {zeta_div}（同一y, {len(DIV_SEEDS)} 个不同初始噪声）")

_test_batch = next(iter(test_loader))
test_images = _test_batch[0][:4].to(device)
test_labels = _test_batch[1][:4]

def compute_psnr(pred, target):
    """pred/target: 在 [-1,1] 空间,统一转换到 [0,1] 再用 MAX=1 计算 PSNR"""
    pred_01 = (pred + 1) / 2
    target_01 = (target + 1) / 2
    mse = torch.mean((pred_01 - target_01)**2).item()
    return 10 * np.log10(1.0 / (mse + 1e-10))

def run_quality(y, forward_op, gt):
    """质量对比：每ζ跑STAT_SEEDS多条独立轨迹；seed=CRN_SEED的主轨迹同时用于出图"""
    primary, primary_psnr, stats = {}, {}, {}
    for z in zeta_quality:
        ps_list = []
        for sd in STAT_SEEDS:
            x_hat = dps_sample(model, y, forward_op, shape=gt.shape, zeta=z, seed=sd)
            p = compute_psnr(x_hat, gt)
            ps_list.append(p)
            if sd == CRN_SEED:
                primary[z], primary_psnr[z] = x_hat, p
        stats[z] = (float(np.mean(ps_list)), float(np.std(ps_list)))
        print(f"  ζ={z:6.1f}: PSNR={stats[z][0]:6.2f} ± {stats[z][1]:4.2f} dB"
              f"  (3条轨迹 {[round(v, 2) for v in ps_list]})")
    return primary, primary_psnr, stats

def run_diversity(y1, gt1, forward_op):
    """多样性测量：固定同一观测y1, 每ζ用DIV_SEEDS取多个独立条件样本"""
    out = {}
    for z in zeta_div:
        xs = [dps_sample(model, y1, forward_op, shape=gt1.shape, zeta=z, seed=sd)
              for sd in DIV_SEEDS]
        X = torch.cat(xs, dim=0)  # (S,1,28,28)
        per_psnr = [compute_psnr(X[i:i+1], gt1) for i in range(len(DIV_SEEDS))]
        pair_mse = float(np.mean([
            torch.mean((X[i] - X[j]) ** 2).item()
            for i in range(len(X)) for j in range(i + 1, len(X))
        ]))
        std_map = X[:, 0].std(dim=0)  # 逐像素跨样本标准差
        consistency = float((forward_op(X) - y1).abs().mean().item())
        out[z] = dict(X=X, per_psnr=per_psnr, pair_mse=pair_mse,
                      std_map=std_map, std_mean=float(std_map.mean()),
                      consistency=consistency)
        print(f"  ζ={z:5.1f}: 每样本PSNR={[round(v, 1) for v in per_psnr]}  "
              f"样本间MSE={pair_mse:.4f}  逐像素std均值={std_map.mean():.4f}")
    return out

# ---- 去模糊观测 ----
blur_op = GaussianBlurOperator(kernel_size=7, sigma=2.0, device=device)
sigma_y_blur = 0.02
y_blur = blur_op(test_images) + torch.randn_like(test_images) * sigma_y_blur
psnr_blur_obs = compute_psnr(y_blur, test_images)

print("\n[去模糊] 质量对比（3条独立轨迹, CRN公平对比）...")
blur_primary, blur_primary_psnr, blur_stats = run_quality(y_blur, blur_op, test_images)
print(f"  模糊观测 PSNR: {psnr_blur_obs:.2f} dB")

print("\n[去模糊] 多样性（固定同一观测y, 4个独立条件样本）...")
blur_div = run_diversity(y_blur[0:1], test_images[0:1], blur_op)

# ---- Inpainting观测 ----
inpaint_op = InpaintingOperator(mask_ratio=0.3, device=device)
sigma_y_inpaint = 0.01
# 观测模型 y = M⊙(x+n)：噪声只加在已知像素上。
# 若写成 M⊙x + n，噪声会落到被遮挡区域，而 forward_op 在那里输出 0，
# 残差 y−M⊙x̂ 变成非零噪声，DPS 会把未知区域往噪声值上拉。
y_inpaint = inpaint_op(test_images + torch.randn_like(test_images) * sigma_y_inpaint)
psnr_inpaint_obs = compute_psnr(y_inpaint, test_images)

print("\n[Inpainting] 质量对比（3条独立轨迹, CRN公平对比）...")
inp_primary, inp_primary_psnr, inp_stats = run_quality(y_inpaint, inpaint_op, test_images)
print(f"  遮挡观测 PSNR: {psnr_inpaint_obs:.2f} dB（灰色像素=被遮挡的未知区域）")

print("\n[Inpainting] 多样性（固定同一观测y, 4个独立条件样本）...")
inp_div = run_diversity(y_inpaint[0:1], test_images[0:1], inpaint_op)

# ============================================================
# 可视化
# ============================================================
# 数据在 [-1,1] 空间,imshow 之前统一转换到 [0,1]

def plot_quality(task_cn, fname, y, primary, stats, obs_psnr, obs_cn, note=None):
    """质量图：第1行观测y(标PSNR), 其后各行=各ζ的CRN主轨迹(4个不同y)"""
    n_rows = 1 + len(zeta_quality)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    for i in range(4):
        axes[0, i].imshow(((y[i, 0] + 1) / 2).clamp(0, 1).cpu().numpy(),
                          cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].text(-0.13, 0.5, f'观测y\n({obs_cn})\nPSNR={obs_psnr:.1f}dB',
                            transform=axes[0, i].transAxes,
                            ha='right', va='center', fontsize=10)
        for row_idx, z in enumerate(zeta_quality):
            ax = axes[row_idx + 1, i]
            ax.imshow(((primary[z][i, 0] + 1) / 2).cpu().numpy(),
                      cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if i == 0:
                m, s = stats[z]
                ax.text(-0.13, 0.5, f'ζ={z:g}\n{m:.1f}±{s:.1f}dB',
                        transform=ax.transAxes, ha='right', va='center', fontsize=10)
    title = (f'DPS{task_cn}：质量对比（CRN公共随机数，零样本迁移）\n'
             f'每行图像为CRN主轨迹(seed={CRN_SEED})；行标为{len(STAT_SEEDS)}条独立轨迹的PSNR均值±标准差')
    if note:
        title += '\n' + note
    plt.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, fname)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图已保存: {fig_path}")

def plot_diversity(task_cn, fname, div, true_digit, obs_cn):
    """多样性图：每行固定同一观测y的4个独立条件样本 + 逐像素std热图"""
    n_rows = len(zeta_div)
    n_cols = len(DIV_SEEDS) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.4 * n_rows))
    for row_idx, z in enumerate(zeta_div):
        d = div[z]
        for i in range(len(DIV_SEEDS)):
            ax = axes[row_idx, i]
            ax.imshow(((d['X'][i, 0] + 1) / 2).cpu().numpy(),
                      cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'样本{i+1}，PSNR={d["per_psnr"][i]:.1f}', fontsize=10)
            ax.axis('off')
        ax_std = axes[row_idx, -1]
        im = ax_std.imshow(d['std_map'].cpu().numpy(), cmap='hot')
        ax_std.set_title(f'逐像素std\n均值={d["std_mean"]:.3f}', fontsize=10)
        ax_std.axis('off')
        axes[row_idx, 0].text(
            -0.30, 0.5,
            f'ζ={z:g}\n样本间\nMSE={d["pair_mse"]:.3f}',
            transform=axes[row_idx, 0].transAxes,
            ha='right', va='center', fontsize=10)
    plt.suptitle(
        f'DPS{task_cn}多样性：同一观测y（{obs_cn}）、{len(DIV_SEEDS)}个不同初始噪声的条件样本'
        f'（真值数字={true_digit}）',
        fontsize=13, y=1.01)
    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, fname)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图已保存: {fig_path}")

plot_quality('去模糊', 'DPS去模糊.png', y_blur, blur_primary, blur_stats,
             psnr_blur_obs, '模糊')
plot_quality('Inpainting', 'DPS Inpainting.png', y_inpaint, inp_primary, inp_stats,
             psnr_inpaint_obs, '遮挡',
             note='观测行中灰色像素=遮挡区（A置零，[-1,1]空间显示为0.5灰，不携带真值信息）')
plot_diversity('去模糊', 'DPS去模糊_多样性.png', blur_div,
               test_labels[0].item(), '模糊')
plot_diversity('Inpainting', 'DPS Inpainting_多样性.png', inp_div,
               test_labels[0].item(), '遮挡')

print("\n" + "=" * 60)
print("实验13.6-2 完成!")
print("=" * 60)
_b1, _w1 = blur_div[1.0], blur_div[50.0]
_i1, _w2 = inp_div[1.0], inp_div[50.0]
print(f"""
关键结论:
1. 零样本迁移
   - 模型只训练了去噪（无条件DDPM），却能解决去模糊和inpainting
   - 无需针对特定逆问题重新训练——这是DPS的核心优势
   - 质量: ζ=20时去模糊PSNR={blur_stats[20.0][0]:.1f}dB（观测{psnr_blur_obs:.1f}）,
           inpainting={inp_stats[20.0][0]:.1f}dB（观测{psnr_inpaint_obs:.1f}）

2. 质量-多样性权衡（本版用同一y的多个条件样本直接测量, 对应13.4.3节）
   - ζ=1引导不足: 去模糊样本间MSE={_b1['pair_mse']:.3f}、inpainting={_i1['pair_mse']:.3f},
     样本彼此分歧极大且PSNR全部低于观测——这是先验主导的"幻觉多样性",
     离散度高但质量低, 并非对后验多模态的有用刻画;
   - ζ增大到50: 去模糊样本间MSE={_w1['pair_mse']:.3f}、inpainting={_w2['pair_mse']:.3f},
     逐像素std降至{_w1['std_mean']:.3f}/{_w2['std_mean']:.3f},
     4个样本收敛为同一清晰数字, PSNR升至最高——后验收窄、一致性增强;
   - 归一化实现需较大ζ(与13.4-3一致, 最优ζ≈10量级), ζ≤1跑不赢含噪观测。

3. 大ζ是否过拟合回落（ζ网格已外推至100）
   - 两个任务在ζ=100处均未见显著回落(去模糊{blur_stats[100.0][0]:.1f}dB,
     inpainting{inp_stats[100.0][0]:.1f}dB), 原因有二:
     (a) 引导经eps修正进入均值步进后实际位移∝ζ·β_t/√α_t, 末段β_t→β_min,
         引导相对注入噪声(∝√β_t)结构性放缓退场;
     (b) 本任务观测噪声很小(σ_y=0.01/0.02), 过拟合噪声的空间有限;
     对照13.4-3(σ_y=0.3): 同一参数化在ζ=100时回落约4.3dB, 说明回落与噪声尺度相关。

4. 统计严谨性
   - ζ间对比采用CRN公共随机数; PSNR报告3条独立轨迹的均值±标准差
     (单轨迹PSNR的随机波动可达数dB, 只报单次数值不可靠)。

5. DPS vs DiffPIR
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

def _quality_table(primary_psnr, stats):
    """主轨迹(CRN seed=42)单次数值 + 3条独立轨迹的均值±标准差"""
    return {
        "各zeta_主轨迹CRN": {f"zeta={float(z)}": round(float(primary_psnr[z]), 2)
                            for z in zeta_quality},
        "各zeta_均值±标准差": {
            f"zeta={float(z)}": {"mean": round(stats[z][0], 2), "std": round(stats[z][1], 2)}
            for z in zeta_quality
        },
    }

def _diversity_table(div):
    """同一观测y、4个不同初始噪声条件样本的多样性指标（[-1,1]空间）"""
    return {
        f"zeta={float(z)}": {
            "每样本PSNR_dB": [round(float(v), 2) for v in div[z]["per_psnr"]],
            "样本间MSE": round(div[z]["pair_mse"], 5),
            "逐像素std均值": round(div[z]["std_mean"], 5),
            "数据一致性残差|A(xhat)-y|均值": round(div[z]["consistency"], 5),
        }
        for z in zeta_div
    }

results_summary = {
    "统计口径": (
        "质量: 4张MNIST测试图, 每ζ用3个独立采样种子(42/43/44), "
        "PSNR在[0,1]空间按MAX=1对4张图pooled计算, 报告均值±标准差; "
        "图中主轨迹seed=42, 不同ζ共享CRN(公共随机数, 同一反向噪声路径)。"
        "多样性: 固定第1张测试图的同一观测y, 每ζ用4个不同初始噪声种子"
        "(101-104)采样, 样本间MSE与逐像素std在[-1,1]空间计算。"
    ),
    "zeta_values_质量": zeta_quality,
    "zeta_values_多样性": zeta_div,
    "CRN_seed": CRN_SEED,
    "统计种子": list(STAT_SEEDS),
    "多样性种子": list(DIV_SEEDS),
    "sigma_y_blur": round(float(sigma_y_blur), 4),
    "sigma_y_inpaint": round(float(sigma_y_inpaint), 4),
    "PSNR_去模糊_观测": round(float(psnr_blur_obs), 2),
    "PSNR_inpainting_观测": round(float(psnr_inpaint_obs), 2),
    "PSNR_去模糊": _quality_table(blur_primary_psnr, blur_stats),
    "PSNR_inpainting": _quality_table(inp_primary_psnr, inp_stats),
    "多样性_去模糊": _diversity_table(blur_div),
    "多样性_inpainting": _diversity_table(inp_div),
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
