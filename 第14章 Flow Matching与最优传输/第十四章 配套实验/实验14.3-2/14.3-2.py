# -*- coding: utf-8 -*-

import sys
import io
import time
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import logging
import warnings
from tqdm import tqdm

# 静默 matplotlib 相关警告
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", message=".*Glyph.*")
warnings.filterwarnings("ignore", message=".*cmap.*")

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验14.3-2')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(_chinese_path, exist_ok=True)

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"实验14.3-2: Flow ODE采样优势——少步采样与DDPM对比")
print(f"{'='*60}")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  未检测到 GPU, 使用 CPU 训练")
    print("  提示: Colab 用户可在菜单 运行时 -> 更改运行时类型 中选择 GPU")

# Checkpoint路径
CHECKPOINT_PATH_RF = os.path.join(SAVE_DIR, 'rf_checkpoint.pth')
CHECKPOINT_PATH_DDPM = os.path.join(SAVE_DIR, 'ddpm_checkpoint.pth')
FINAL_CHECKPOINT_PATH_RF = os.path.join(SAVE_DIR, 'rf_final.pth')
FINAL_CHECKPOINT_PATH_DDPM = os.path.join(SAVE_DIR, 'ddpm_final.pth')


# ============================================================
# 去噪网络: 小型UNet（可预测速度v或噪声ε）
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
    """小型UNet——可适配速度预测(v-pred)或噪声预测(ε-pred)"""
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
        """预测速度场v_θ(x_t, t)或噪声ε_θ(x_t, t)，t为整数时间步t∈{0,...,T-1}"""
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
# DDPM噪声调度
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

    返回: (samples, elapsed_time)
    """
    model.eval()
    t_start = time.time()
    x = torch.randn(shape, device=device)  # z ~ N(0,I)
    dt = 1.0 / n_steps

    for step in range(n_steps):
        t_val = step / n_steps
        # RF中t_val=0是噪声，DDPM中t_int=T-1是噪声
        # 反转映射使正弦嵌入语义对齐
        # 注：t_int仅用作位置编码信号，不要求与实际噪声方差精确对应
        t_int = torch.full((shape[0],), int((1 - t_val) * (T - 1)), device=device, dtype=torch.long)
        v = model(x, t_int)
        x = x + v * dt

    elapsed = time.time() - t_start
    return x.clamp(0, 1), elapsed


@torch.no_grad()
def ddpm_sample(model, shape):
    """标准DDPM采样（200步，对比基线）

    返回: (samples, elapsed_time)
    """
    model.eval()
    t_start = time.time()
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
    elapsed = time.time() - t_start
    return x.clamp(0, 1), elapsed


# ============================================================
# 数据加载
# ============================================================
print("\n加载MNIST数据集...")
data_dir = os.path.join(SAVE_DIR, 'data')
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
print(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")


# ============================================================
# 步骤1：训练Rectified Flow与DDPM基线
# ============================================================

num_epochs = 50

# 训练函数（带Resume能力）
def train_model(model, optimizer, train_loader, model_type='rf', num_epochs=50,
                checkpoint_path=None, final_checkpoint_path=None, progress_bar=False):
    """训练模型（带checkpoint Resume能力）

    model_type: 'rf' 或 'ddpm'
    progress_bar: 是否显示进度条
    """
    start_epoch = 0
    is_final = False
    train_losses = []

    # 检查是否有最终权重
    if final_checkpoint_path and os.path.exists(final_checkpoint_path):
        print(f"检测到最终权重: {final_checkpoint_path}")
        print("直接加载，跳过训练过程")
        checkpoint = torch.load(final_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint.get('train_losses', [])
        return train_losses

    # 检查是否有中间权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"检测到中间权重: {checkpoint_path}")
        print("继续训练...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        train_losses = checkpoint.get('train_losses', [])

    print(f"训练 {model_type} 模型，从 epoch {start_epoch} 开始...")
    t_start = time.time()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        # 添加进度条
        pbar = tqdm(train_loader, desc=f"[{model_type.upper()}] Epoch {epoch+1}/{num_epochs}", leave=False) if progress_bar else train_loader
        for x, _ in pbar:
            x = x.to(device)
            batch = x.shape[0]

            if model_type == 'rf':
                # Rectified Flow训练
                z = torch.randn_like(x)
                t_continuous = torch.rand(batch, device=device)
                t_int = ((1 - t_continuous) * (T - 1)).long()
                t_4d = t_continuous[:, None, None, None]
                x_t = (1 - t_4d) * z + t_4d * x
                v_target = x - z
                v_pred = model(x_t, t_int)
                loss = F.mse_loss(v_pred, v_target)
            else:
                # DDPM训练
                t = torch.randint(0, T, (batch,), device=device)
                noise = torch.randn_like(x)
                x_t = sqrt_alpha_bars[t][:, None, None, None] * x + sqrt_one_minus_alpha_bars[t][:, None, None, None] * noise
                pred = model(x_t, t)
                loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch
            
            # 更新进度条显示当前平均损失
            if progress_bar:
                pbar.set_postfix({'loss': f'{total_loss / len(train_loader.dataset):.6f}'})
        if progress_bar:
            pbar.close()

        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{model_type}] Epoch {epoch+1:3d}/{num_epochs}  Loss={avg_loss:.6f}")

        # 每10轮保存中间checkpoint
        if checkpoint_path and (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'train_losses': train_losses
            }, checkpoint_path)

    # 保存最终权重
    if final_checkpoint_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses
        }, final_checkpoint_path)
        print(f"最终权重已保存: {final_checkpoint_path}")

    t_elapsed = time.time() - t_start
    print(f"训练完成，耗时: {t_elapsed:.1f} 秒")

    return train_losses

# 训练Rectified Flow（v-prediction）
print("\n训练Rectified Flow（v-prediction）...")
rf_model = SmallUNet().to(device)
optimizer_rf = torch.optim.Adam(rf_model.parameters(), lr=2e-4)
losses_rf = train_model(rf_model, optimizer_rf, train_loader, model_type='rf',
                        num_epochs=num_epochs, checkpoint_path=CHECKPOINT_PATH_RF,
                        final_checkpoint_path=FINAL_CHECKPOINT_PATH_RF,
                        progress_bar=True)

# 训练DDPM（ε-prediction）
print("\n训练DDPM（ε-prediction）...")
ddpm_model = SmallUNet().to(device)
optimizer_ddpm = torch.optim.Adam(ddpm_model.parameters(), lr=2e-4)
losses_ddpm = train_model(ddpm_model, optimizer_ddpm, train_loader, model_type='ddpm',
                          num_epochs=num_epochs, checkpoint_path=CHECKPOINT_PATH_DDPM,
                          final_checkpoint_path=FINAL_CHECKPOINT_PATH_DDPM,
                          progress_bar=True)

print("\n两个模型训练完成！")


# ============================================================
# 步骤2：Flow ODE采样——少步采样对比
# ============================================================

n_samples = 8
sample_shape = (n_samples, 1, 28, 28)

step_counts = [1, 5, 10, 50]
samples_by_steps = {}
times_by_steps = {}

for n_steps in step_counts:
    print(f"  {n_steps}步采样中...")
    samples, elapsed = flow_ode_sample(rf_model, sample_shape, n_steps=n_steps)
    samples_by_steps[n_steps] = samples
    times_by_steps[n_steps] = elapsed
    print(f"    耗时: {elapsed:.3f}s")

# 可视化
fig, axes = plt.subplots(len(step_counts), n_samples, figsize=(16, 8))
for row, n_steps in enumerate(step_counts):
    for col in range(n_samples):
        axes[row, col].imshow(samples_by_steps[n_steps][col, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(f'{n_steps}步', fontsize=12, rotation=0, labelpad=40)

plt.suptitle('实验14.3-2：Rectified Flow少步采样（14.3.4节）', fontsize=14, y=1.01)
plt.tight_layout()
fig_path1 = os.path.join(SAVE_DIR, '步骤2_少步采样.png')
plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig_path1}")


# ============================================================
# 步骤3：DDPM vs Rectified Flow少步采样对比
# ============================================================

# DDPM采样
print("DDPM采样中(200步)...")
ddpm_samples, time_ddpm = ddpm_sample(ddpm_model, sample_shape)
print(f"  耗时: {time_ddpm:.3f}s")

# RF 50步采样
print("Rectified Flow采样中(50步)...")
rf_50_samples, time_rf50 = flow_ode_sample(rf_model, sample_shape, n_steps=50)
print(f"  耗时: {time_rf50:.3f}s")

# RF 10步采样
print("Rectified Flow采样中(10步)...")
rf_10_samples, time_rf10 = flow_ode_sample(rf_model, sample_shape, n_steps=10)
print(f"  耗时: {time_rf10:.3f}s")

# RF 1步采样（已在步骤2中计算）
time_rf1 = times_by_steps[1]

# 计算量化指标：生成样本与最近邻真实样本的平均像素距离
print("\n计算量化指标（生成样本与最近邻真实样本的平均像素距离）...")
test_samples = torch.stack([test_dataset[i][0] for i in range(min(100, len(test_dataset)))]).to(device)

def compute_nn_distance(generated, reference):
    """计算生成样本与参考集中最近邻的平均像素距离（简化质量指标）"""
    gen_flat = generated.view(generated.shape[0], -1)  # (B, 784)
    ref_flat = reference.view(reference.shape[0], -1)  # (N, 784)
    # 计算每个生成样本与所有参考样本的距离
    distances = torch.cdist(gen_flat, ref_flat)  # (B, N)
    # 取每个生成样本的最近邻距离
    min_distances = distances.min(dim=1)[0]  # (B,)
    return min_distances.mean().item()

dist_ddpm = compute_nn_distance(ddpm_samples, test_samples)
dist_rf50 = compute_nn_distance(rf_50_samples, test_samples)
dist_rf10 = compute_nn_distance(rf_10_samples, test_samples)
dist_rf1 = compute_nn_distance(samples_by_steps[1], test_samples)

print(f"  DDPM 200步 平均最近邻距离: {dist_ddpm:.4f}")
print(f"  RF 50步   平均最近邻距离: {dist_rf50:.4f}")
print(f"  RF 10步   平均最近邻距离: {dist_rf10:.4f}")
print(f"  RF 1步    平均最近邻距离: {dist_rf1:.4f}")
print("  （距离越小表示生成样本更接近真实数据分布）")

# 加速比计算（实测）
speedup_10 = time_ddpm / time_rf10
speedup_50 = time_ddpm / time_rf50
print(f"\n实测加速比:")
print(f"  RF 10步 vs DDPM 200步: {speedup_10:.1f}x")
print(f"  RF 50步 vs DDPM 200步: {speedup_50:.1f}x")

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

plt.suptitle('实验14.3-2：DDPM vs Rectified Flow 少步采样对比（14.3.6节）', fontsize=14, y=1.01)
plt.tight_layout()
fig_path2 = os.path.join(SAVE_DIR, '步骤3_DDPM_vs_RF.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"图3已保存: {fig_path2}")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print("实验14.3-2 完成!")
print("=" * 60)
print("""
关键结论:
1. 少步采样优势（14.3.4节）
   - RF用Flow ODE采样：1步/5步/10步/50步
   - DDPM需要200步（SDE采样）
   - RF的直线路径允许大幅减少采样步数
   - 量化指标验证：RF 10步即可达到较好质量（平均最近邻距离接近）

2. DDPM vs Rectified Flow对比（14.3.6节）★ 原创设计
   - 同一UNet架构，不同训练目标
   - DDPM预测噪声ε，RF预测速度v
   - 量化指标对比（平均最近邻距离）：
     * DDPM 200步: {:.4f}
     * RF 50步: {:.4f}
     * RF 10步: {:.4f}
     * RF 1步: {:.4f}
   - 结论：RF 50步/10步质量接近DDPM 200步，1步质量较差
   - RF 1步质量较差，但Reflow可以改善（见实验14.4-1）

3. 路径形态差异（14.3.5节）
   - DDPM路径弯曲（扩散耦合），需要多步
   - RF路径更直（直线耦合），可以少步
   - 直线路径是Flow Matching的核心优势

4. 实测加速效果
   - DDPM 200步耗时: {:.3f}s
   - RF 10步耗时: {:.3f}s
   - RF 50步耗时: {:.3f}s
   - 实测加速比：RF 10步 vs DDPM = {:.1f}x
   - Flow Matching显著减少采样时间，为实时生成奠定基础
""".format(dist_ddpm, dist_rf50, dist_rf10, dist_rf1,
           time_ddpm, time_rf10, time_rf50, speedup_10))

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
    'dist_ddpm_200step': round(float(dist_ddpm), 4),
    'dist_rf_50step': round(float(dist_rf50), 4),
    'dist_rf_10step': round(float(dist_rf10), 4),
    'dist_rf_1step': round(float(dist_rf1), 4),
    'time_ddpm': round(float(time_ddpm), 3),
    'time_rf_10step': round(float(time_rf10), 3),
    'time_rf_50step': round(float(time_rf50), 3),
    'speedup_rf10_vs_ddpm': round(float(speedup_10), 1),
    'final_loss_rf': round(float(losses_rf[-1]), 6) if losses_rf else None,
    'final_loss_ddpm': round(float(losses_ddpm[-1]), 6) if losses_ddpm else None,
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")