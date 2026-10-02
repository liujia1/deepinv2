# -*- coding: utf-8 -*-
"""
实验13.4-2：引导权重zeta对PSNR的影响

★ 原创设计：固定测试图像，用不同zeta执行DPS去噪，对比PSNR和视觉效果

实验内容：
  - 训练UNet扩散模型（MNIST, epsilon-prediction）
  - 用不同zeta值执行DPS去噪
  - 绘制zeta-PSNR曲线

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
    SAVE_DIR = os.path.join(_gdrive, '实验13.4-2')
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
print("实验13.4-2: 引导权重zeta对PSNR的影响")
print("=" * 60)
print("知识点: 引导权重zeta, PSNR倒U形曲线, 实际权重选择")


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
        故本实验中 zeta=1 仅代表本实验定义下的"标准强度", 而非论文原版公式.

    修正 eps 的符号推导:
      Chung 原代码: x_{t-1} -= scale · g_chung, 其中
        g_chung = ∂‖y-Ax̂₀‖/∂x_t = -(1/√ᾱ)·A^T r / ‖r‖, r=y-Ax̂₀
      等价转换到 "通过 eps 改 mean" 形式, 对 g_loss=∂‖y-Ax̂₀‖²/∂x_t
        应当使用 + 号: eps_pred + ζ·√(1-ᾱ)·g_loss
      (其中 g_loss 方向同 g_chung, 即 x_t 增大→loss 减小, 与 Chung 一致)
      之前实现误用 - 号, 导致 ζ 越大 x_hat 离真值越远, 与 Chung 方向相反.

    seed: 若不为 None, 在采样开始前固定随机种子, 使不同 zeta 共享同一条
          反向过程噪声路径(公共随机数 CRN), 排除采样随机性对 zeta-PSNR
          曲线形状的干扰(与 13.4-1 做法一致)
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
  zeta=1: 标准DPS（平衡先验与似然）
  zeta>1: 强数据一致性（类似MAP，多样性低）
""")

zeta_values = [0.0, 0.3, 0.7, 1.0, 1.5, 3.0]
psnr_results = []

# 公共随机数(CRN)种子: 不同 zeta 共享同一条反向过程噪声路径,
# 排除采样随机性对 zeta-PSNR 曲线形状的干扰(与 13.4-1 做法一致)
CRN_SEED = 42

test_images = next(iter(test_loader))[0][:4].to(device)
single_img = test_images[:1]
sigma_y_denoise = 0.3
y_single = single_img + torch.randn_like(single_img) * sigma_y_denoise

identity_op = IdentityOperator()

def compute_psnr(pred, target):
    """pred/target: 在 [-1,1] 空间,统一转换到 [0,1] 再用 MAX=1 计算 PSNR"""
    pred_01 = (pred + 1) / 2
    target_01 = (target + 1) / 2
    mse = torch.mean((pred_01 - target_01)**2).item()
    return 10 * np.log10(1.0 / (mse + 1e-10))

# 一次性采样并缓存 x_hat, 供下方可视化复用
# 避免之前"PSNR 来自第一次采样, 图片来自第二次独立采样"导致图与数字不对应
# 同时省掉重复采样的 6 次 dps_sample 算力
x_hat_list = []
for zeta in zeta_values:
    x_hat = dps_sample(model, y_single, identity_op,
                        shape=single_img.shape, zeta=zeta, seed=CRN_SEED)
    psnr = compute_psnr(x_hat, single_img)
    psnr_results.append(psnr)
    x_hat_list.append(x_hat)
    print(f"  zeta={zeta:4.1f}: PSNR={psnr:.2f} dB")

# 可视化: 复用上面已采样的 x_hat, 保证子图标题 PSNR 与图像严格对应
# 拆分为两个独立文件: 数字重建图(2x3) + zeta-PSNR 曲线图(单独)
fig1, axes1 = plt.subplots(2, 3, figsize=(12, 8))
for idx, zeta in enumerate(zeta_values):
    row, col = idx // 3, idx % 3
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
print(f"\n图1(2x3数字重建对比)已保存: {fig1_path}")

# zeta-PSNR 曲线图: 单独一张, 带最优点标注, 突出倒U形
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
ax2.plot(zeta_values, psnr_results, 'ro-', markersize=10, lw=2)
ax2.set_xlabel(r'引导权重 $\zeta$', fontsize=12)
ax2.set_ylabel('PSNR (dB)', fontsize=12)
ax2.set_title(r'$\zeta$-重建质量权衡曲线（13.4.3节）', fontsize=13)
ax2.grid(alpha=0.3)

# 标注倒U形峰值点(用竖线+散点+文本框三件套, 与13.4-1风格一致)
best_idx = int(np.argmax(psnr_results))
best_zeta = zeta_values[best_idx]
best_psnr = psnr_results[best_idx]
ax2.axvline(best_zeta, color='gray', linestyle='--', alpha=0.5, lw=1)
ax2.plot(best_zeta, best_psnr, 'b*', markersize=18, zorder=5,
         markeredgecolor='navy', markeredgewidth=0.5)
ax2.annotate(rf'最优 $\zeta$={best_zeta}' + '\n' + rf'PSNR={best_psnr:.2f}dB',
             xy=(best_zeta, best_psnr),
             xytext=(best_zeta + 0.4, best_psnr - 1.5),
             fontsize=10, ha='left',
             arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeaa7', alpha=0.8))

# 三段说明放在左下空白区
ax2.text(0.02, 0.05,
         r'$\zeta$=0: 无条件采样' + '\n' +
         r'$\zeta$=1: 本实验标准强度' + '\n' +
         r'$\zeta$>1: 强数据一致性',
         transform=ax2.transAxes, fontsize=9, va='bottom',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#dfe6e9', alpha=0.8))

fig2.tight_layout()
fig2_path = os.path.join(SAVE_DIR, '引导权重zeta与PSNR关系曲线.png')
fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"图2(zeta-PSNR曲线)已保存: {fig2_path}")

print("\n" + "=" * 60)
print("实验13.4-2 完成!")
print("=" * 60)
print(f"""
关键结论:
1. zeta-重建质量权衡（13.4.3节）
   - zeta=0: 无条件采样（忽略观测，PSNR低）
   - zeta=1: 本实验定义下的标准强度（平衡先验与似然，通常PSNR最高）
     注: 本实现对似然梯度做单位范数归一化, 故zeta直接控制修正量模长,
         与Chung et al. (2022)原论文"步长∝1/‖y-Ax̂₀‖"的自适应方案不同,
         此处zeta=1仅代表本实验的标定基准, 而非论文原版公式
   - zeta过大: 过度拟合观测噪声，PSNR下降
   - 最优zeta通常在0.5-1.5之间

2. 实践要点
   - 单一图像的最优zeta可能因图而异
   - 工程上常用 zeta=1 作为默认起点
   - zeta-PSNR曲线呈倒U形——这是质量-多样性权衡的具体表现
   - 不同zeta之间的PSNR对比采用公共随机数(CRN, seed=42)实现公平采样,
     排除采样随机性对曲线形状的干扰(与13.4-1做法一致)
   - 实验配置: T={T} 步DDPM, 训练 {num_epochs} 轮 MNIST UNet.
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
    "PSNR_各zeta": {f"zeta={z}": round(float(p), 2) for z, p in zip(zeta_values, psnr_results)},
    "最优zeta": round(float(best_zeta), 2),
    "最优PSNR": round(float(best_psnr), 2),
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")

