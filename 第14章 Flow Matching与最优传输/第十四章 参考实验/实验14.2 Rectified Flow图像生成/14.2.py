# -*- coding: utf-8 -*-
"""
实验14.2 Rectified Flow图像生成
对应知识点：
  - 14.3.4节 高斯条件路径（直线插值）
  - 14.4.1节 Rectified Flow训练
  - 14.4.3节 Reflow迭代变直
  - 14.3.6节 DDIM=FM with diffusion coupling

在MNIST上用小型UNet实现Rectified Flow，对比扩散模型(DDPM)和Flow Matching。

素材来源：
  - 实验11.2的SmallUNet架构和MNIST数据管道
  - 14.3/14.4节的理论内容
  - ★ 原创设计：DDPM vs Rectified Flow的少步采样质量对比
  - ★ 原创设计：1步/5步/10步/50步采样对比

运行前提：需要GPU

实验内容：
  步骤1：训练Rectified Flow（速度预测）
  步骤2：Flow ODE采样——少步采样对比（14.3.4节）
  步骤3：DDPM vs Rectified Flow少步采样对比（14.3.6节）
  步骤4：Reflow蒸馏为少步模型（14.4.3节）
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import warnings

# ====== 解决中文乱码的核心代码 ======
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
import platform
from matplotlib.font_manager import FontManager

def _find_chinese_font():
    candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong'] if platform.system() == 'Windows' else ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei']
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available:
            return font
    import os as _os, re
    for f in fm.ttflist:
        for pat in ['cjk', 'wqy', 'noto.*cjk', 'simhei']:
            if re.search(pat, f.name.lower()):
                return f.name
    return None

_cn_font = _find_chinese_font()
if _cn_font:
    plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['font.family'] = 'sans-serif'
    print(f"[Font] 已检测到中文字体: {_cn_font}")
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
# ========================================================

np.random.seed(42)
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    SAVE_DIR = os.path.join(_gdrive, '实验14_2_RectifiedFlow')
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"检测到 Google Drive，结果将保存至: {SAVE_DIR}")
else:
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    print(f"本地环境，结果将保存至: {SAVE_DIR}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')


# ============================================================
# 去噪网络: 小型UNet（与11.2/12.2一致，但预测速度v而非ε）
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
    """小型UNet——适配Rectified Flow的速度预测"""
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
        """预测速度场v_θ(x_t, t)，t∈[0,1]"""
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
# DDPM噪声调度（用于对比实验）
# ============================================================
T = 200
beta_min, beta_max = 1e-4, 0.02
betas = torch.linspace(beta_min, beta_max, T).to(device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
posterior_var = betas * (1 - torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])) / (1 - alpha_bars)
sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)
beta_over_sqrt_1m_ab = betas / sqrt_one_minus_alpha_bars


# ============================================================
# Flow ODE采样器（14.3.1节）
# ============================================================
@torch.no_grad()
def flow_ode_sample(model, shape, n_steps=50):
    """从Flow ODE采样: dx/dt = v_θ(x, t), t: 0→1
    
    14.3.1节: Flow ODE直接参数化向量场
    对比DDPM需要200步，FM理论上可以1步
    """
    model.eval()
    x = torch.randn(shape, device=device)  # z ~ N(0,I)
    dt = 1.0 / n_steps
    
    for step in range(n_steps):
        t_val = step / n_steps
        # ★ 修正：RF中t_val=0是噪声，DDPM中t_int=T-1是噪声
        # 反转映射使正弦嵌入语义对齐
        t_int = torch.full((shape[0],), int((1 - t_val) * (T - 1)), device=device, dtype=torch.long)
        v = model(x, t_int)
        x = x + v * dt
    
    return x.clamp(0, 1)


@torch.no_grad()
def ddpm_sample(model, shape):
    """标准DDPM采样（200步，对比基线）"""
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
    return x.clamp(0, 1)


# ============================================================
# 数据加载
# ============================================================
print("加载MNIST数据集...")
data_dir = os.path.join(SAVE_DIR, 'data')
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
print(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")


# ============================================================
# 步骤1：训练Rectified Flow（速度预测，14.4.1节）
# ============================================================
print(f"\n{'='*60}")
print("步骤1：训练Rectified Flow（速度预测，14.4.1节）")
print("=" * 60)

print("""
14.4.1节：Rectified Flow训练
  插值路径: x_t = (1-t)z + t*x_0, t∈[0,1]
  速度目标: v = x_0 - z
  训练损失: ||v_θ(x_t, t) - (x_0 - z)||²
  
  对比DDPM:
  - DDPM: ||ε̂_θ(x_t, t) - ε||², t∈{0,...,T-1}, 离散时间
  - RF:   ||v_θ(x_t, t) - (x_0-z)||², t∈[0,1], 连续时间
""")

num_epochs = 50
rf_model = SmallUNet().to(device)
optimizer = torch.optim.Adam(rf_model.parameters(), lr=2e-4)

for epoch in range(num_epochs):
    rf_model.train()
    total_loss = 0
    for x, _ in train_loader:
        x = x.to(device)
        batch = x.shape[0]
        
        # 采样噪声z ~ N(0,I)
        z = torch.randn_like(x)
        
        # 采样t ~ U[0,1]（映射到整数时间步）
        t_continuous = torch.rand(batch, device=device)
        # ★ 修正：RF中t_continuous=0是噪声，t_continuous=1是干净
        # DDPM中t_int=0是干净，t_int=T-1是噪声
        # 因此需要反转映射：t_int = (1 - t_continuous) * (T-1)
        # 这样RF的"噪声时间"对应DDPM的"噪声时间"，正弦嵌入语义对齐
        t_int = ((1 - t_continuous) * (T - 1)).long()
        
        # 线性插值: x_t = (1-t)z + t*x_0
        t_4d = t_continuous[:, None, None, None]
        x_t = (1 - t_4d) * z + t_4d * x
        
        # 速度目标: v = x_0 - z
        v_target = x - z
        
        # 网络预测
        v_pred = rf_model(x_t, t_int)
        
        # RF损失
        loss = F.mse_loss(v_pred, v_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"  Epoch {epoch+1:3d}/{num_epochs}  Loss={avg_loss:.6f}")

print("Rectified Flow训练完成！")


# ============================================================
# 步骤2：Flow ODE采样——少步采样对比（14.3.4节）
# ============================================================
print(f"\n{'='*60}")
print("步骤2：Flow ODE采样——少步采样对比（14.3.4节）")
print("=" * 60)

n_samples = 8
sample_shape = (n_samples, 1, 28, 28)

step_counts = [1, 5, 10, 50]
samples_by_steps = {}

for n_steps in step_counts:
    print(f"  {n_steps}步采样中...")
    samples = flow_ode_sample(rf_model, sample_shape, n_steps=n_steps)
    samples_by_steps[n_steps] = samples

# 可视化
fig, axes = plt.subplots(len(step_counts), n_samples, figsize=(16, 8))
for row, n_steps in enumerate(step_counts):
    for col in range(n_samples):
        axes[row, col].imshow(samples_by_steps[n_steps][col, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(f'{n_steps}步', fontsize=12, rotation=0, labelpad=40)

plt.suptitle('步骤2：Rectified Flow少步采样（14.3.4节）', fontsize=14, y=1.01)
plt.tight_layout()
fig_path1 = os.path.join(SAVE_DIR, '步骤1_少步采样.png')
plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"图1已保存: {fig_path1}")


# ============================================================
# ★ 步骤3：DDPM vs Rectified Flow少步采样对比（14.3.6节）
# ============================================================
print(f"\n{'='*60}")
print("★ 步骤3：DDPM vs Rectified Flow少步采样对比（14.3.6节）")
print("=" * 60)

print("""
14.3.6节：DDIM = FM with diffusion coupling
  - DDPM: SDE采样，路径弯曲，需~200步
  - DDIM: ODE采样，半直路径，~50步
  - Rectified Flow: ODE采样，更直的路径，更少步数
  
★ 原创设计：同一架构(UNet)，不同训练目标(ε-pred vs v-pred)，
  对比少步采样质量
""")

# 先训练一个DDPM（ε-prediction）作为对比基线
print("训练DDPM基线（ε-prediction）...")
ddpm_model = SmallUNet().to(device)
optimizer_ddpm = torch.optim.Adam(ddpm_model.parameters(), lr=2e-4)

for epoch in range(num_epochs):
    ddpm_model.train()
    total_loss = 0
    for x, _ in train_loader:
        x = x.to(device)
        batch = x.shape[0]
        t = torch.randint(0, T, (batch,), device=device)
        noise = torch.randn_like(x)
        x_t = sqrt_alpha_bars[t][:, None, None, None] * x + sqrt_one_minus_alpha_bars[t][:, None, None, None] * noise
        pred = ddpm_model(x_t, t)
        loss = F.mse_loss(pred, noise)
        optimizer_ddpm.zero_grad()
        loss.backward()
        optimizer_ddpm.step()
        total_loss += loss.item() * batch
    
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"  [DDPM] Epoch {epoch+1:3d}/{num_epochs}  Loss={avg_loss:.6f}")

# DDPM采样
print("DDPM采样中(200步)...")
ddpm_samples = ddpm_sample(ddpm_model, sample_shape)

# RF 50步采样
print("Rectified Flow采样中(50步)...")
rf_50_samples = flow_ode_sample(rf_model, sample_shape, n_steps=50)

# RF 10步采样
print("Rectified Flow采样中(10步)...")
rf_10_samples = flow_ode_sample(rf_model, sample_shape, n_steps=10)

# 可视化对比
fig, axes = plt.subplots(4, n_samples, figsize=(16, 10))
methods = [
    (ddpm_samples, 'DDPM (ε-pred, 200步)'),
    (rf_50_samples, 'Rectified Flow (v-pred, 50步)'),
    (rf_10_samples, 'Rectified Flow (v-pred, 10步)'),
    (samples_by_steps[1], 'Rectified Flow (v-pred, 1步)'),
]

for row, (samples, label) in enumerate(methods):
    for col in range(n_samples):
        axes[row, col].imshow(samples[col, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(label, fontsize=10, rotation=0, labelpad=80)

plt.suptitle('★ DDPM vs Rectified Flow 少步采样对比（14.3.6节）', fontsize=14, y=1.01)
plt.tight_layout()
fig_path2 = os.path.join(SAVE_DIR, '步骤2_DDPM_vs_RF.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig_path2}")


# ============================================================
# ★ 步骤4：Reflow蒸馏为少步模型（14.4.3节）
# ============================================================
print(f"\n{'='*60}")
print("★ 步骤4：Reflow蒸馏为少步模型（14.4.3节）")
print("=" * 60)

print("""
14.4.3节 Reflow：
  1. 用当前模型的Flow ODE生成端点对 (z, x̂_0)
  2. 用这些端点对重新训练模型
  3. 重复→轨迹逐步变直→少步采样质量提升

★ 原创设计：用1-RF模型生成端点对，训练2-RF模型，
  对比1-RF和2-RF在1步采样下的质量
""")

# 生成端点对
print("生成Reflow端点对...")
rf_model.eval()
reflow_pairs_z = []
reflow_pairs_x0 = []

with torch.no_grad():
    for x, _ in train_loader:
        x = x.to(device)
        z = torch.randn_like(x)
        # 运行Flow ODE得到终点
        x_t = z.clone()
        dt = 1.0 / 50
        for step in range(50):
            t_val = step / 50
            # ★ 修正：与flow_ode_sample一致的时间映射
            t_int = torch.full((x.shape[0],), int((1 - t_val) * (T - 1)), device=device, dtype=torch.long)
            v = rf_model(x_t, t_int)
            x_t = x_t + v * dt
        reflow_pairs_z.append(z.cpu())
        reflow_pairs_x0.append(x_t.cpu())

reflow_z = torch.cat(reflow_pairs_z, dim=0)
reflow_x0 = torch.cat(reflow_pairs_x0, dim=0)
print(f"  生成端点对: {len(reflow_z)} 对")

# 训练2-RF模型
print("训练 2-Rectified Flow...")
model_2rf = SmallUNet().to(device)
optimizer_2rf = torch.optim.Adam(model_2rf.parameters(), lr=2e-4)

reflow_dataset = torch.utils.data.TensorDataset(reflow_z, reflow_x0)
reflow_loader = DataLoader(reflow_dataset, batch_size=128, shuffle=True)

for epoch in range(num_epochs):
    model_2rf.train()
    total_loss = 0
    for z_batch, x0_batch in reflow_loader:
        z_batch = z_batch.to(device)
        x0_batch = x0_batch.to(device)
        batch = z_batch.shape[0]
        
        t_continuous = torch.rand(batch, device=device)
        # ★ 修正：与RF训练一致的反转映射
        t_int = ((1 - t_continuous) * (T - 1)).long()
        t_4d = t_continuous[:, None, None, None]
        
        x_t = (1 - t_4d) * z_batch + t_4d * x0_batch
        v_target = x0_batch - z_batch
        v_pred = model_2rf(x_t, t_int)
        
        loss = F.mse_loss(v_pred, v_target)
        optimizer_2rf.zero_grad()
        loss.backward()
        optimizer_2rf.step()
        total_loss += loss.item() * batch
    
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(reflow_dataset)
        print(f"  [2-RF] Epoch {epoch+1:3d}/{num_epochs}  Loss={avg_loss:.6f}")

# 对比1-RF和2-RF的1步采样
print("\n1步采样对比:")
rf_1step_1rf = flow_ode_sample(rf_model, sample_shape, n_steps=1)
rf_1step_2rf = flow_ode_sample(model_2rf, sample_shape, n_steps=1)

# 可视化
fig, axes = plt.subplots(3, n_samples, figsize=(16, 7))
for col in range(n_samples):
    # 1-RF 1步
    axes[0, col].imshow(rf_1step_1rf[col, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, col].axis('off')
    if col == 0: axes[0, col].set_ylabel('1-RF 1步', fontsize=12, rotation=0, labelpad=50)
    
    # 2-RF 1步
    axes[1, col].imshow(rf_1step_2rf[col, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1, col].axis('off')
    if col == 0: axes[1, col].set_ylabel('2-RF 1步', fontsize=12, rotation=0, labelpad=50)
    
    # 1-RF 50步（参考）
    axes[2, col].imshow(samples_by_steps[50][col, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[2, col].axis('off')
    if col == 0: axes[2, col].set_ylabel('1-RF 50步', fontsize=12, rotation=0, labelpad=50)

plt.suptitle('★ Reflow蒸馏：1步采样质量提升（14.4.3节）', fontsize=14, y=1.01)
plt.tight_layout()
fig_path3 = os.path.join(SAVE_DIR, '步骤3_Reflow蒸馏.png')
plt.savefig(fig_path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"图3已保存: {fig_path3}")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print("实验14.2 完成!")
print("=" * 60)
print("""
关键结论:
1. Rectified Flow训练（14.4.1节）
   - 训练目标: ||v_θ(x_t, t) - (x_0 - z)||²
   - 对比DDPM的 ||ε̂_θ(x_t, t) - ε||²
   - RF预测速度场v，DDPM预测噪声ε

2. 少步采样优势（14.3.4节）
   - RF用Flow ODE采样：1步/5步/10步/50步
   - DDPM需要200步（SDE采样）
   - RF的直线路径允许大幅减少采样步数

3. DDPM vs Rectified Flow对比（14.3.6节）★ 原创设计
   - 同一UNet架构，不同训练目标
   - DDPM 200步 ≈ RF 50步 ≈ RF 10步（质量接近）
   - RF 1步质量较差，但Reflow可以改善

4. Reflow蒸馏（14.4.3节）★ 原创设计
   - 用1-RF的ODE端点对训练2-RF
   - 2-RF的1步采样质量优于1-RF的1步
   - Reflow本质：蒸馏多步模型为少步模型
""")
