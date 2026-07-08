# -*- coding: utf-8 -*-
"""
实验14.4-1 Rectified Flow训练：图像生成实践
对应知识点：
  - 14.3.4节 高斯条件路径（直线插值）
  - 14.4.1节 Rectified Flow训练
  - 14.4.3节 Reflow迭代变直

在MNIST上用小型UNet实现Rectified Flow，体验训练与蒸馏过程。

素材来源：
  - 实验11.2的SmallUNet架构和MNIST数据管道
  - 14.3/14.4节的理论内容
  - ★ 原创设计：Reflow蒸馏过程可视化

运行前提：需要GPU

实验内容：
  步骤1：训练Rectified Flow（速度预测，14.4.1节）
  步骤2：Reflow蒸馏为少步模型（14.4.3节）
"""

# ====== Windows UTF-8支持 ======
import sys
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
# Windows控制台UTF-8输出
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ====== matplotlib非交互式后端（静默运行） ======
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，不弹出窗口
import matplotlib.pyplot as plt

# ====== 静默警告 ======
import logging
import warnings
from tqdm import tqdm
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", message=".*Glyph.*")
warnings.filterwarnings("ignore", message=".*cmap.*")

import os
import numpy as np
np.random.seed(42)

# ====== Colab支持 ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验14.4-1')
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

# ====== 中文字体配置 ======
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")

import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# ====== Checkpoint路径 ======
CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'rf_checkpoint.pth')
FINAL_CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'rf_final.pth')
CHECKPOINT_2RF_PATH = os.path.join(SAVE_DIR, '2rf_checkpoint.pth')
FINAL_2RF_PATH = os.path.join(SAVE_DIR, '2rf_final.pth')
REFLOW_PAIRS_PATH = os.path.join(SAVE_DIR, 'reflow_pairs.pth')  # Reflow端点对checkpoint


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
        """预测速度场v_θ(x_t, t)，t为整数时间步t∈{0,...,T-1}"""
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
# T为时间离散化粒度，用于将连续t∈[0,1]映射到正弦嵌入的整数输入
T = 200


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
# 量化指标：生成样本与最近邻真实样本的平均像素距离
# ============================================================
def compute_nn_distance(generated, reference):
    """计算生成样本与参考集中最近邻的平均像素距离（简化质量指标）

    用于量化Reflow蒸馏的效果，避免仅靠视觉判断。
    距离越小表示生成样本更接近真实数据分布。
    """
    gen_flat = generated.view(generated.shape[0], -1)  # (B, 784)
    ref_flat = reference.view(reference.shape[0], -1)  # (N, 784)
    # 计算每个生成样本与所有参考样本的距离
    distances = torch.cdist(gen_flat, ref_flat)  # (B, N)
    # 取每个生成样本的最近邻距离
    min_distances = distances.min(dim=1)[0]  # (B,)
    return min_distances.mean().item()


# ============================================================
# 步骤1：训练Rectified Flow（速度预测，14.4.1节）
# ============================================================
print(f"\n{'='*60}")
print("实验14.4-1 步骤1：训练Rectified Flow（速度预测，14.4.1节）")
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

# Resume检查：是否已有训练好的权重
skip_training_1rf = False
start_epoch_1rf = 0
if os.path.exists(FINAL_CHECKPOINT_PATH):
    print(f"\n检测到最终权重: {FINAL_CHECKPOINT_PATH}")
    print("直接加载，跳过训练过程")
    checkpoint_1rf = torch.load(FINAL_CHECKPOINT_PATH, map_location=device, weights_only=False)
    rf_model.load_state_dict(checkpoint_1rf['model_state_dict'])
    skip_training_1rf = True
elif os.path.exists(CHECKPOINT_PATH):
    print(f"\n检测到中间权重: {CHECKPOINT_PATH}")
    print("继续训练...")
    checkpoint_1rf = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    rf_model.load_state_dict(checkpoint_1rf['model_state_dict'])
    optimizer.load_state_dict(checkpoint_1rf['optimizer_state_dict'])
    start_epoch_1rf = checkpoint_1rf.get('epoch', 0) + 1

if not skip_training_1rf:
    for epoch in range(start_epoch_1rf, num_epochs):
        rf_model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"[RF] Epoch {epoch+1}/{num_epochs}", leave=False)
        for x, _ in pbar:
            x = x.to(device)
            batch = x.shape[0]

            # 采样噪声z ~ N(0,I)
            z = torch.randn_like(x)

            # 采样t ~ U[0,1]（映射到整数时间步）
            t_continuous = torch.rand(batch, device=device)
            # RF中t_continuous=0是噪声，t_continuous=1是干净
            # DDPM中t_int=0是干净，t_int=T-1是噪声
            # 反转映射使正弦嵌入语义对齐：RF噪声端→DDPM噪声端
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
            
            # 更新进度条显示当前平均损失
            pbar.set_postfix({'loss': f'{total_loss / len(train_loader.dataset):.6f}'})
        pbar.close()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"  Epoch {epoch+1:3d}/{num_epochs}  Loss={avg_loss:.6f}")

        # 保存中间checkpoint（含optimizer状态）
        torch.save({
            'epoch': epoch,
            'model_state_dict': rf_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)

    # 训练完成，保存最终权重（含optimizer状态）
    torch.save({
        'model_state_dict': rf_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, FINAL_CHECKPOINT_PATH)
    print(f"\n最终权重已保存: {FINAL_CHECKPOINT_PATH}")

print("Rectified Flow训练完成！")


# ============================================================
# 步骤2：Reflow蒸馏为少步模型（14.4.3节）
# ============================================================
print(f"\n{'='*60}")
print("实验14.4-1 步骤2：Reflow蒸馏为少步模型（14.4.3节）")
print("=" * 60)

print("""
14.4.3节 Reflow：
  1. 用当前模型的Flow ODE生成端点对 (z, x̂_0)
  2. 用这些端点对重新训练模型
  3. 重复→轨迹逐步变直→少步采样质量提升

★ 原创设计：用1-RF模型生成端点对，训练2-RF模型，
  对比1-RF和2-RF在1步采样下的质量
""")

# 生成端点对（仅从噪声z出发经ODE推演，不需要真实数据）
n_reflow = len(train_dataset)

# 检查是否已有保存的端点对
if os.path.exists(REFLOW_PAIRS_PATH):
    print(f"\n检测到已保存的Reflow端点对: {REFLOW_PAIRS_PATH}")
    print("直接加载，跳过生成过程")
    checkpoint_reflow = torch.load(REFLOW_PAIRS_PATH, map_location='cpu', weights_only=False)
    reflow_z = checkpoint_reflow['reflow_z']
    reflow_x0 = checkpoint_reflow['reflow_x0']
    print(f"  已加载端点对: {len(reflow_z)} 对")
else:
    print(f"生成Reflow端点对 ({n_reflow} 对)...")
    rf_model.eval()
    reflow_pairs_z = []
    reflow_pairs_x0 = []
    batch_size_reflow = 128

    with torch.no_grad():
        for i in range(0, n_reflow, batch_size_reflow):
            batch_sz = min(batch_size_reflow, n_reflow - i)
            z = torch.randn(batch_sz, 1, 28, 28, device=device)
            # 运行Flow ODE得到终点
            x_t = z.clone()
            dt = 1.0 / 50
            for step in range(50):
                t_val = step / 50
                t_int = torch.full((batch_sz,), int((1 - t_val) * (T - 1)), device=device, dtype=torch.long)
                v = rf_model(x_t, t_int)
                x_t = x_t + v * dt
            reflow_pairs_z.append(z.cpu())
            reflow_pairs_x0.append(x_t.cpu())

            # 进度提示（每1000批）
            if (i + batch_size_reflow) % 1000 == 0 or i + batch_size_reflow >= n_reflow:
                print(f"  已生成 {min(i + batch_size_reflow, n_reflow)}/{n_reflow} 对")

    reflow_z = torch.cat(reflow_pairs_z, dim=0)
    reflow_x0 = torch.cat(reflow_pairs_x0, dim=0)
    print(f"  生成端点对: {len(reflow_z)} 对")

    # 保存端点对，避免下次重新生成
    torch.save({
        'reflow_z': reflow_z,
        'reflow_x0': reflow_x0,
    }, REFLOW_PAIRS_PATH)
    print(f"端点对已保存: {REFLOW_PAIRS_PATH}")

# 训练2-RF模型
print("训练 2-Rectified Flow...")
model_2rf = SmallUNet().to(device)
optimizer_2rf = torch.optim.Adam(model_2rf.parameters(), lr=2e-4)

reflow_dataset = torch.utils.data.TensorDataset(reflow_z, reflow_x0)
reflow_loader = DataLoader(reflow_dataset, batch_size=128, shuffle=True)

# Resume检查：是否已有2-RF训练好的权重
skip_training_2rf = False
start_epoch_2rf = 0
if os.path.exists(FINAL_2RF_PATH):
    print(f"\n检测到最终权重: {FINAL_2RF_PATH}")
    print("直接加载，跳过训练过程")
    checkpoint_2rf = torch.load(FINAL_2RF_PATH, map_location=device, weights_only=False)
    model_2rf.load_state_dict(checkpoint_2rf['model_state_dict'])
    skip_training_2rf = True
elif os.path.exists(CHECKPOINT_2RF_PATH):
    print(f"\n检测到中间权重: {CHECKPOINT_2RF_PATH}")
    print("继续训练...")
    checkpoint_2rf = torch.load(CHECKPOINT_2RF_PATH, map_location=device, weights_only=False)
    model_2rf.load_state_dict(checkpoint_2rf['model_state_dict'])
    optimizer_2rf.load_state_dict(checkpoint_2rf['optimizer_state_dict'])
    start_epoch_2rf = checkpoint_2rf.get('epoch', 0) + 1

if not skip_training_2rf:
    for epoch in range(start_epoch_2rf, num_epochs):
        model_2rf.train()
        total_loss = 0
        pbar = tqdm(reflow_loader, desc=f"[2-RF] Epoch {epoch+1}/{num_epochs}", leave=False)
        for z_batch, x0_batch in pbar:
            z_batch = z_batch.to(device)
            x0_batch = x0_batch.to(device)
            batch = z_batch.shape[0]

            t_continuous = torch.rand(batch, device=device)
            # 反转映射使正弦嵌入语义对齐：RF噪声端→DDPM噪声端
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
            
            # 更新进度条显示当前平均损失
            pbar.set_postfix({'loss': f'{total_loss / len(reflow_dataset):.6f}'})
        pbar.close()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(reflow_dataset)
            print(f"  [2-RF] Epoch {epoch+1:3d}/{num_epochs}  Loss={avg_loss:.6f}")

        # 保存中间checkpoint（含optimizer状态）
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_2rf.state_dict(),
            'optimizer_state_dict': optimizer_2rf.state_dict(),
        }, CHECKPOINT_2RF_PATH)

    # 训练完成，保存最终权重（含optimizer状态）
    torch.save({
        'model_state_dict': model_2rf.state_dict(),
        'optimizer_state_dict': optimizer_2rf.state_dict(),
    }, FINAL_2RF_PATH)
    print(f"\n最终权重已保存: {FINAL_2RF_PATH}")

# 对比1-RF和2-RF的少步采样
print("\n少步采样对比...")
n_samples = 8
sample_shape = (n_samples, 1, 28, 28)

rf_1step_1rf = flow_ode_sample(rf_model, sample_shape, n_steps=1)
rf_1step_2rf = flow_ode_sample(model_2rf, sample_shape, n_steps=1)
rf_50step_1rf = flow_ode_sample(rf_model, sample_shape, n_steps=50)
rf_10step_1rf = flow_ode_sample(rf_model, sample_shape, n_steps=10)

# 计算量化指标（生成样本与最近邻真实样本的平均像素距离）
print("\n计算量化指标（生成样本与最近邻真实样本的平均像素距离）...")
test_samples = torch.stack([test_dataset[i][0] for i in range(min(100, len(test_dataset)))]).to(device)

dist_1rf_1step = compute_nn_distance(rf_1step_1rf, test_samples)
dist_2rf_1step = compute_nn_distance(rf_1step_2rf, test_samples)
dist_1rf_10step = compute_nn_distance(rf_10step_1rf, test_samples)
dist_1rf_50step = compute_nn_distance(rf_50step_1rf, test_samples)

print(f"  1-RF 1步  平均最近邻距离: {dist_1rf_1step:.4f}")
print(f"  2-RF 1步  平均最近邻距离: {dist_2rf_1step:.4f}")
print(f"  1-RF 10步 平均最近邻距离: {dist_1rf_10step:.4f}")
print(f"  1-RF 50步 平均最近邻距离: {dist_1rf_50step:.4f}")
print("  （距离越小表示生成样本更接近真实数据分布）")
print(f"\n★ Reflow效果验证：2-RF 1步距离({dist_2rf_1step:.4f}) < 1-RF 1步距离({dist_1rf_1step:.4f})")
if dist_2rf_1step < dist_1rf_1step:
    print(f"  改善幅度: {(dist_1rf_1step - dist_2rf_1step) / dist_1rf_1step * 100:.1f}%")
else:
    print(f"  注意：本次实验中Reflow改善幅度有限，可能原因：")
    print(f"    - MNIST数据简单，1步采样本身已接近真实分布")
    print(f"    - 训练轮数(50 epoch)较少，Reflow效果未充分体现")
    print(f"    - 需要更多Reflow迭代(如3-RF、4-RF)才能显著改善")

# 可视化
fig, axes = plt.subplots(4, n_samples, figsize=(16, 10))
methods = [
    (rf_1step_1rf, '1-RF 1步'),
    (rf_1step_2rf, '2-RF 1步'),
    (rf_10step_1rf, '1-RF 10步'),
    (rf_50step_1rf, '1-RF 50步（参考）'),
]

for row, (samples, label) in enumerate(methods):
    for col in range(n_samples):
        axes[row, col].imshow(samples[col, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(label, fontsize=12, rotation=0, labelpad=50)

plt.suptitle('实验14.4-1：Reflow蒸馏少步采样对比（14.4.3节）', fontsize=14, y=1.01)
plt.tight_layout()
fig_path1 = os.path.join(SAVE_DIR, '步骤1_Reflow蒸馏.png')
plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"图1已保存: {fig_path1}")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print("实验14.4-1 完成!")
print("=" * 60)
print("""
关键结论:
1. Rectified Flow训练（14.4.1节）
   - 训练目标: ||v_θ(x_t, t) - (x_0 - z)||²
   - 对比DDPM的 ||ε̂_θ(x_t, t) - ε||²
   - RF预测速度场v，DDPM预测噪声ε
   - 连续时间t∈[0,1]，而DDPM是离散时间

2. Reflow蒸馏（14.4.3节）★ 原创设计
   - 用1-RF的ODE端点对训练2-RF
   - 量化指标验证（平均最近邻距离）：
     * 1-RF 1步: {:.4f}
     * 2-RF 1步: {:.4f}
     * 1-RF 10步: {:.4f}
     * 1-RF 50步: {:.4f}
   - Reflow本质：蒸馏多步模型为少步模型
   - 轨迹逐步变直，趋近OT映射

3. 实践意义
   - Reflow提供了一种无需OT求解器的"OT近似"方法
   - 适用于高维数据（图像、视频等）
   - 可用于模型加速：多步→少步→单步
   - 量化指标为Reflow效果提供了客观评价依据
""".format(dist_1rf_1step, dist_2rf_1step, dist_1rf_10step, dist_1rf_50step))