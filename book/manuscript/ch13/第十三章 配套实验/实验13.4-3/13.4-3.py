# -*- coding: utf-8 -*-
"""
实验13.4-3：引导权重zeta对PSNR的影响

★ 原创设计：固定测试图像，用不同zeta执行DPS去噪，对比PSNR和视觉效果；
  并加入 Chung et al. (2022) 原版步长的对照组，裁决 zeta-PSNR 曲线形态
  到底是"本实验单位归一化实现的固有属性"还是 DPS 算法的普适性质。

实验内容：
  - 训练UNet扩散模型（MNIST, epsilon-prediction）
  - 用不同zeta值执行DPS去噪（扩展扫描至zeta=100，覆盖上升段、峰值与下降段）
  - 绘制zeta-PSNR曲线（含Chung对照）

注意：本实验训练50轮DDPM并执行多次DPS采样。
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
    SAVE_DIR = os.path.join(_gdrive, '实验13.4-3')
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
print("实验13.4-3: 引导权重zeta对PSNR的影响")
print("=" * 60)
print("知识点: 引导权重zeta, PSNR权衡曲线形态, 引导步长参数化的尺度差异")


# ============================================================
# 噪声调度
# ============================================================
T = 1000  # 标准DDPM时间步
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
    DPS 采样。
    注: 本实现对似然梯度做单位范数归一化, 只取其方向, 残差强度统一由
        外部超参 zeta 控制, 不依赖 sigma_y. 故函数签名中未保留 sigma_y.
        这与 Chung et al. (2022) 原论文"步长∝1/‖y-Ax̂₀‖"的自适应方案不同,
        故本实验中 zeta=1 并非论文原版公式的"标准强度";
        实测本实现的最优 zeta≈10~15(10与15基本持平), 与 Chung 惯例(0.3-0.6)不可直接对比.

    修正 eps 的符号推导:
      Chung 原代码: x_{t-1} -= scale · g_chung, 其中
        g_chung = ∂‖y-Ax̂₀‖/∂x_t = -(1/√ᾱ)·A^T r / ‖r‖, r=y-Ax̂₀
      等价转换到 "通过 eps 改 mean" 形式, 对 g_loss=∂‖y-Ax̂₀‖²/∂x_t
        应当使用 + 号: eps_pred + ζ·√(1-ᾱ)·g_loss
      (其中 g_loss 方向同 g_chung, 即 x_t 增大→loss 减小, 与 Chung 一致)
      之前实现误用 - 号, 导致 ζ 越大 x_hat 离真值越远, 与 Chung 方向相反.

    seed: 若不为 None, 在采样开始前固定随机种子, 使不同 zeta 共享同一条
          反向过程噪声路径(公共随机数 CRN), 排除采样随机性对 zeta-PSNR
          曲线形状的干扰(与 13.4-2 做法一致)

    尺度提示: 单位归一化使修正幅度与残差 ‖r‖ 脱钩, zeta 的"有效尺度"
        比 Chung 原版(ζ'·∇‖r‖)大得多——实测本实现的最优 zeta≈10~15
        (见扫描结果, 10与15基本持平), 而非论文惯例的 0.3-0.6。

    退场机制(修订): eps 空间修正 ζ·√(1-ᾱ_t)·ĝ 乘以均值步进系数
        β_t/√(1-ᾱ_t) 后, √(1-ᾱ_t) 恰好完全相消, 引导对均值的实际位移
        = ζ·β_t/√α_t·ĝ。末段衰减的真实来源是 β_t→β_min(=1e-4), 而非
        "√(1-ᾱ_t)→0"; 相对注入噪声幅度√(posterior_var)≈√β_t, 引导强度
        按 √β_t 衰减。因此"大zeta过拟合噪声"的下降段在该实现中被显著
        放缓(实测ζ=100比峰值低约4.5dB), 而非完全消失。
    """
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(shape, device=device)
    for t_idx in reversed(range(T)):
        t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        sqrt_ab_t = sqrt_alpha_bars[t_idx]
        sqrt_1mab_t = sqrt_one_minus_alpha_bars[t_idx]
        # 单次前向传播: 带 grad 的输出同时用于 x0_hat 估计与噪声预测
        x = x.detach().requires_grad_(True)
        eps_pred = model(x, t)
        x0_hat = (x - sqrt_1mab_t * eps_pred) / sqrt_ab_t
        Ax0_hat = forward_op(x0_hat)
        likelihood_loss = torch.sum((y - Ax0_hat) ** 2)
        likelihood_grad = torch.autograd.grad(likelihood_loss, x)[0]
        grad_norm = likelihood_grad.norm()
        if grad_norm > 1e-8:
            likelihood_grad = likelihood_grad / grad_norm
        with torch.no_grad():
            # 关键: 这里是 + 号, 与 Chung 原论文方向一致
            eps_corrected = eps_pred.detach() + zeta * sqrt_1mab_t * likelihood_grad
            model_mean = sqrt_recip_alphas[t_idx] * (
                x - beta_over_sqrt_1m_ab[t_idx] * eps_corrected
            )
            if t_idx == 0:
                x = model_mean
            else:
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(posterior_var[t_idx]) * noise
    return x.clamp(-1, 1)


class IdentityOperator:
    def __call__(self, x):
        return x


def dps_sample_chung(model, y, forward_op, shape, zeta_prime=0.3, seed=None):
    """
    Chung et al. (2022) 原版 DPS 步长(对照组), 与论文/官方代码一致:
        论文 Algorithm 1:  x_{t-1} ← x'_{t-1} − ζ_i·∇_{x_t}‖y−A(x̂₀)‖²
        论文脚注: ζ_i = ζ'/‖y−A(x̂₀)‖  ⟹ 等价于 步长 = ζ'·∇‖r‖, r = y−Ax̂₀
        官方代码('ps'类): x_t -= norm_grad * scale  (norm_grad=∇‖r‖, scale=常数)
    即"常数系数 × 范数梯度": 系数不随 t 衰减, 无 σ̂_t²/σ_y² 尺度, 亦无额外
    1/‖r‖ 归一化(∇‖r‖ 自身已含 1/‖r‖ 因子)。引导加在含噪的 x_prev 上
    (噪声后), 与论文 Algorithm 1 的时机一致。
    论文附录: ζ' 最佳范围 [0.1, 1.0](常用 0.3-0.6), 并报告 ζ'>5 出现
    "放大噪声的饱和伪影"——本对照检验该行为在本任务上是否复现。
    注意: zeta_prime 的数值含义仅对原版参数化成立, 与主曲线(单位归一化
    实现)的 ζ 不可直接对比。同样用 CRN 保证公平对比。
    """
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(shape, device=device)
    for t_idx in reversed(range(T)):
        t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        sqrt_ab_t = sqrt_alpha_bars[t_idx]
        sqrt_1mab_t = sqrt_one_minus_alpha_bars[t_idx]
        x = x.detach().requires_grad_(True)
        eps_pred = model(x, t)
        x0_hat = (x - sqrt_1mab_t * eps_pred) / sqrt_ab_t
        r = y - forward_op(x0_hat)
        rnorm = torch.sqrt(torch.sum(r ** 2) + 1e-12)
        grad = torch.autograd.grad(rnorm, x)[0]  # ∇‖r‖ (自身已含 1/‖r‖ 因子)
        with torch.no_grad():
            model_mean = sqrt_recip_alphas[t_idx] * (
                x.detach() - beta_over_sqrt_1m_ab[t_idx] * eps_pred.detach()
            )
            if t_idx > 0:
                x_prev = model_mean + torch.sqrt(posterior_var[t_idx]) * torch.randn_like(x)
            else:
                x_prev = model_mean
            # 原版步长: 常数 ζ' × ∇‖r‖ (不随 t 衰减, 与官方代码 x_t -= norm_grad*scale 一致)
            x = x_prev - zeta_prime * grad
    return x.clamp(-1, 1)


# ============================================================
# 训练函数（含checkpoint resume）
# ============================================================
def train_model(checkpoint_path, num_epochs=50):
    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    start_epoch = 0
    is_final = False

    # 噪声调度超参指纹, 用于加载 checkpoint 时一致性校验
    # (模型架构不变但 T/beta 范围变化时, 旧 checkpoint 会被静默接受,
    # 引发训练-采样不一致, 这里存一份指纹以便在加载时拦截)
    schedule_fingerprint = {'T': T, 'beta_min': beta_min, 'beta_max': beta_max}

    if os.path.exists(checkpoint_path):
        print(f"\n检测到已保存的模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved_fp = checkpoint.get('schedule_fingerprint', None)
        fp_mismatch = (saved_fp is not None) and (saved_fp != schedule_fingerprint)
        if fp_mismatch:
            print(f"警告: checkpoint的噪声调度与当前脚本不一致, 删除后重新训练")
            print(f"  checkpoint: T={saved_fp['T']}, beta=[{saved_fp['beta_min']:.1e}, {saved_fp['beta_max']:.4f}]")
            print(f"  当前脚本:  T={schedule_fingerprint['T']}, beta=[{schedule_fingerprint['beta_min']:.1e}, {schedule_fingerprint['beta_max']:.4f}]")
            os.remove(checkpoint_path)
        else:
            if saved_fp is None:
                print(f"  提示: checkpoint未包含噪声调度指纹(旧版), 仅按架构加载")
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
                    'is_final': False,
                    'schedule_fingerprint': schedule_fingerprint,
                }, checkpoint_path)
            torch.save({
                'epoch': num_epochs - 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'is_final': True,
                'schedule_fingerprint': schedule_fingerprint,
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


# ============================================================
# 训练
# ============================================================
print(f"\n{'='*60}")
print("训练UNet扩散模型...")
print("=" * 60)

CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'zeta_psnr_checkpoint.pth')
num_epochs = 50
model = train_model(CHECKPOINT_PATH, num_epochs=num_epochs)


# ============================================================
# 步骤：引导权重zeta对重建质量的影响
# ============================================================
print(f"\n{'='*60}")
print("步骤：引导权重zeta对重建质量的影响（13.4.3节）")
print("=" * 60)

print("""
13.4.3节：zeta控制先验与似然的相对强度
  zeta=0: 纯先验（无条件采样，忽略观测）
  zeta增大: 数据一致性增强, PSNR上升, 但多样性下降
  理论预期: zeta过大时过拟合观测噪声, PSNR回落(倒U形)
  本实验同时对比两种引导步长参数化(单位归一化 vs Chung原版),
  裁决曲线形态是算法普适性质还是实现尺度差异所致
""")

zeta_values = [0.0, 0.3, 0.7, 1.0, 1.5, 3.0, 6.0, 10.0, 15.0, 30.0, 50.0, 100.0]
zeta_values_chung = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.3, 0.6, 1.0, 3.0, 5.0, 10.0, 30.0]  # 加密下边界以定位峰值；覆盖论文惯例[0.1,1.0]与"ζ'>5伪影"区间
psnr_results = []
psnr_results_chung = []

# 公共随机数(CRN)种子: 不同 zeta 共享同一条反向过程噪声路径,
# 排除采样随机性对 zeta-PSNR 曲线形状的干扰(与 13.4-2 做法一致)
CRN_SEED = 42

test_images = next(iter(test_loader))[0][:4].to(device)
single_img = test_images[:1]
sigma_y_denoise = 0.3
# 观测噪声用独立Generator固定: 避免"是否跳过训练"改变全局RNG状态
# 导致 y 不同、历史结果无法复现的缺陷
_gen_obs = torch.Generator().manual_seed(2026)
y_single = single_img + torch.randn(single_img.shape, generator=_gen_obs).to(device) * sigma_y_denoise

identity_op = IdentityOperator()

def compute_psnr(pred, target):
    """pred/target: 在 [-1,1] 空间,统一转换到 [0,1] 再用 MAX=1 计算 PSNR"""
    pred_01 = (pred + 1) / 2
    target_01 = (target + 1) / 2
    mse = torch.mean((pred_01 - target_01)**2).item()
    return 10 * np.log10(1.0 / (mse + 1e-10))

psnr_obs = compute_psnr(y_single, single_img)
print(f"\n参照: 含噪观测y自身的PSNR = {psnr_obs:.2f} dB (重建应高于此才有意义)")

# 一次性采样并缓存 x_hat, 供下方可视化复用
# 避免之前"PSNR 来自第一次采样, 图片来自第二次独立采样"导致图与数字不对应
x_hat_list = []
for zeta in zeta_values:
    x_hat = dps_sample(model, y_single, identity_op,
                        shape=single_img.shape, zeta=zeta, seed=CRN_SEED)
    psnr = compute_psnr(x_hat, single_img)
    psnr_results.append(psnr)
    x_hat_list.append(x_hat)
    print(f"  [归一化实现] zeta={zeta:5.1f}: PSNR={psnr:.2f} dB")

print()
for zeta_prime in zeta_values_chung:
    psnr_c = compute_psnr(
        dps_sample_chung(model, y_single, identity_op,
                          shape=single_img.shape, zeta_prime=zeta_prime,
                          seed=CRN_SEED),
        single_img)
    psnr_results_chung.append(psnr_c)
    print(f"  [Chung原版 ] zeta'={zeta_prime:5.1f}: PSNR={psnr_c:.2f} dB")

# 可视化: 复用上面已采样的 x_hat, 保证子图标题 PSNR 与图像严格对应
# 拆分为两个独立文件: 数字重建图(2x6) + zeta-PSNR 曲线图(单独)
fig1, axes1 = plt.subplots(2, 6, figsize=(21, 7.5))
for idx, zeta in enumerate(zeta_values):
    row, col = idx // 6, idx % 6
    x_hat = x_hat_list[idx]
    axes1[row, col].imshow(((x_hat[0, 0] + 1) / 2).cpu().numpy(),
                            cmap='gray', vmin=0, vmax=1)
    axes1[row, col].axis('off')
    label = "无条件" if zeta == 0 else r"$\zeta={}$".format(zeta)
    axes1[row, col].set_title(f'{label}\nPSNR={psnr_results[idx]:.1f}dB', fontsize=11)

fig1.tight_layout()
fig1_path = os.path.join(SAVE_DIR, '重建结果对比.png')
fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"\n图1(2x6数字重建对比)已保存: {fig1_path}")

# zeta-PSNR 曲线图: 单独一张, 两条曲线对比(归一化实现 vs Chung原版),
# 含含噪观测PSNR水平线, 如实展示"上升—峰值—平缓回落"形态
fig2, ax2 = plt.subplots(1, 1, figsize=(9, 6))
ax2.plot(zeta_values, psnr_results, 'ro-', markersize=8, lw=2,
         label='单位归一化实现(本实验主曲线)')
ax2.plot(zeta_values_chung, psnr_results_chung, 'bs--', markersize=8, lw=2,
         label=r"Chung et al. (2022) 原版步长 $\zeta'\!\cdot\!\nabla\Vert r\Vert$(对照)")
ax2.axhline(psnr_obs, color='gray', linestyle=':', lw=1.5)
# x用轴分数(贴右边缘)、y用数据坐标(贴着参照线上方):
# 必须用blended变换, 否则y=psnr_obs+0.15会被transAxes当成16.5倍轴高,
# bbox_inches='tight'保存时把画布纵向撑到上万像素
import matplotlib.transforms as mtransforms
_trans_obs = mtransforms.blended_transform_factory(ax2.transAxes, ax2.transData)
ax2.text(0.985, psnr_obs + 0.15, f'含噪观测 PSNR={psnr_obs:.1f}dB',
         transform=_trans_obs, ha='right', va='bottom', fontsize=9, color='dimgray')
ax2.set_xscale('symlog', linthresh=1.0)  # linthresh: matplotlib>=3.9已移除旧名linthreshx
ax2.set_xticks([0, 0.3, 0.7, 1.5, 3.0, 6.0, 10.0, 15.0, 30.0, 50.0, 100.0])
ax2.set_xticklabels(['0', '0.3', '0.7', '1.5', '3', '6', '10', '15', '30', '50', '100'])
ax2.set_xlabel(r'引导权重 $\zeta$', fontsize=12)
ax2.set_ylabel('PSNR (dB)', fontsize=12)
ax2.set_title(r'$\zeta$-重建质量权衡曲线（13.4.3节）', fontsize=13)
ax2.grid(alpha=0.3, which='both')
ax2.legend(loc='lower right', fontsize=9)

# 标注两条曲线各自的最优点(均在扫描内部取得)
best_idx = int(np.argmax(psnr_results))
best_zeta = zeta_values[best_idx]
best_psnr = psnr_results[best_idx]
ax2.plot(best_zeta, best_psnr, 'b*', markersize=16, zorder=5,
         markeredgecolor='navy', markeredgewidth=0.5)
ax2.annotate(rf'归一化实现最优 $\zeta$={best_zeta:g}' + '\n' + rf'PSNR={best_psnr:.2f}dB',
             xy=(best_zeta, best_psnr),
             xytext=(best_zeta * 0.5, best_psnr - 3.7),  # 移到红线下方，避开标题与曲线
             fontsize=9, ha='left', va='center',
             arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeaa7', alpha=0.85))
bc_idx = int(np.argmax(psnr_results_chung))
bc_zeta = zeta_values_chung[bc_idx]
bc_psnr = psnr_results_chung[bc_idx]
ax2.plot(bc_zeta, bc_psnr, 'g*', markersize=16, zorder=5,
         markeredgecolor='darkgreen', markeredgewidth=0.5)
ax2.annotate(rf"Chung对照最优 $\zeta'$={bc_zeta:g}" + '\n' + rf'PSNR={bc_psnr:.2f}dB',
             xy=(bc_zeta, bc_psnr),
             xytext=(0.65, bc_psnr - 3.75),  # 蓝线上方空档：高于下降段、低于绿星与标题
             fontsize=9, ha='left', va='center',
             arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#dff9e3', alpha=0.85))

fig2.tight_layout()
fig2_path = os.path.join(SAVE_DIR, '引导权重zeta与PSNR关系曲线.png')
fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"图2(zeta-PSNR曲线)已保存: {fig2_path}")

print("\n" + "=" * 60)
print("实验13.4-3 完成!")
print("=" * 60)
print(f"""
关键结论(基于实测数据, 归一化ζ∈[0,100] / 原版ζ'∈[0.01,30], CRN seed={CRN_SEED}):
1. zeta-重建质量权衡（13.4.3节）
   - zeta=0: 无条件采样, PSNR最低({psnr_results[0]:.1f}dB), 甚至低于含噪观测({psnr_obs:.1f}dB)
   - 归一化实现: PSNR随zeta先升后缓降, 最优zeta={best_zeta:g}(PSNR={best_psnr:.1f}dB),
     zeta=100时回落至{psnr_results[-1]:.1f}dB(较峰值约-{best_psnr - psnr_results[-1]:.1f}dB)——
     "倒U形"存在但下降段平缓, 且最优值远大于论文惯例的0.3-0.6:
     单位归一化使修正幅度与‖r‖脱钩, zeta的有效尺度被整体推大;
     引导对均值的位移=ζ·β_t/√α_t·ĝ(eps空间修正中的√(1-ᾱ_t)与步进系数相消),
     随β_t→β_min末段自动退场, "过拟合噪声"的下降段被结构性放缓(非完全消失)。
   - Chung原版步长(ζ'·∇‖r‖, 系数不随t衰减)对照: 最优ζ'={bc_zeta:g}
     (PSNR={bc_psnr:.1f}dB), 其数值尺度应与论文惯例[0.1,1.0]区间对照,
     而非与归一化实现的ζ互比; ζ'≳1后PSNR进入低平台并继续回落, 即复现论文记录的
     "饱和伪影/噪声放大"——峰值位置与下降段深浅由步长参数化方式决定。
2. 实践要点
   - zeta的"最优区间"依赖引导步长的参数化方式, 不可照搬论文惯例数值,
     必须在本实现自己的尺度上扫描定位峰值(先粗扫再细扫)
   - 小zeta区间的重建PSNR可低于含噪观测(引导不足以拉回先验漂移),
     含噪观测PSNR={psnr_obs:.1f}dB应作为必测参照线
   - 质量-多样性权衡的"下降段"本质在多样性维度, 单图PSNR只能部分反映;
     后验层面的权衡曲线见实验13.4-2(1D解析情形)
   - 不同zeta之间的PSNR对比采用公共随机数(CRN, seed={CRN_SEED})实现公平采样;
     观测噪声用独立Generator(seed=2026)固定, 与训练RNG历史解耦
   - 实验配置: T={T} 步DDPM, 训练 {num_epochs} 轮 MNIST UNet, σ_y={sigma_y_denoise}([-1,1]空间)。
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

def _chung_psnr(z):
    """按 ζ' 取值查 PSNR，避免加密网格后索引错位"""
    return psnr_results_chung[zeta_values_chung.index(z)]

results_summary = {
    "zeta_values": zeta_values,
    "PSNR_各zeta": {f"zeta={z}": round(float(p), 2) for z, p in zip(zeta_values, psnr_results)},
    "最优zeta_归一化实现": round(float(best_zeta), 2),
    "最优PSNR_归一化实现": round(float(best_psnr), 2),
    "对照_Chung原版": {
        "步长形式": "ζ'·∇‖r‖ (Algorithm 1 + 官方代码, 系数不随t衰减)",
        "zeta_prime_values": zeta_values_chung,
        "PSNR_各zeta_prime": {f"zeta'={z}": round(float(p), 2) for z, p in zip(zeta_values_chung, psnr_results_chung)},
        "最优zeta_prime": round(float(bc_zeta), 2),
        "最优PSNR": round(float(bc_psnr), 2),
    },
    "含噪观测PSNR参照": round(float(psnr_obs), 2),
    "sigma_y": sigma_y_denoise,
    "CRN_seed": CRN_SEED,
    "观测噪声seed": 2026,
    "曲线形态结论": (
        f"归一化实现呈平缓倒U, 最优ζ={best_zeta:g}(PSNR={best_psnr:.2f}, 10~15为平台), "
        f"ζ=100回落至{psnr_results[-1]:.2f}; "
        f"Chung原版对照(ζ'·∇‖r‖,系数不随t衰减)最优ζ'={bc_zeta:g}(PSNR={bc_psnr:.2f}), "
        f"位于论文惯例区间[0.1,1.0]下缘; ζ'≳1后PSNR进入低平台并继续回落(ζ'=1/3/5/10/30分别为"
        f"{_chung_psnr(1.0):.2f}/{_chung_psnr(3.0):.2f}/{_chung_psnr(5.0):.2f}/{_chung_psnr(10.0):.2f}/{_chung_psnr(30.0):.2f}dB), "
        "复现论文'ζ>5出现放大噪声的饱和伪影'记录; 两参数化最优权重相差约两个数量级, "
        "峰值位置与下降段深浅由步长参数化决定"
    ),
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")

