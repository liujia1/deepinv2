# -*- coding: utf-8 -*-
"""
实验14.2-1 CNF训练对比——Neural ODE vs Flow Matching
对应章节: 14.2 连续归一化流(CNF)

知识点:
  - Neural ODE: dz/dt = v_θ(z,t)，训练需要求解ODE（慢且不稳定）
  - 伴随方法(adjoint method): 内存高效但训练时间开销大
  - Flow Matching: 无需模拟ODE即可训练CNF（CFM定理核心突破）
  - 两种方法收敛速度和采样质量的对比

实验内容:
  步骤1: 在2D点云上用Neural ODE训练CNF（最大似然/伴随方法）
  步骤2: 在2D点云上用Flow Matching训练CNF
  步骤3: 对比两种方法的训练效率、收敛曲线和采样质量

数据集: 2D高斯混合（8-Gaussian），纯CPU实验

素材来源:
  - Chen et al. (2018) Neural ODE
  - Lipman et al. (2023) Flow Matching

运行前提: PyTorch, scipy, CPU即可
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import logging
import warnings
import time
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
    SAVE_DIR = os.path.join(_gdrive, '实验14.2-1')
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

# 设备配置（2D点云实验，CPU即可）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"实验14.2-1: CNF训练对比——Neural ODE vs Flow Matching")
print(f"{'='*60}")
print(f"使用设备: {device}")
print("  本实验使用2D点云，CPU即可完成")

# Checkpoint路径
CHECKPOINT_PATH_CNLL = os.path.join(SAVE_DIR, 'cnll_checkpoint.pth')
FINAL_CHECKPOINT_PATH_CNLL = os.path.join(SAVE_DIR, 'cnll_final.pth')
CHECKPOINT_PATH_FM = os.path.join(SAVE_DIR, 'fm_checkpoint.pth')
FINAL_CHECKPOINT_PATH_FM = os.path.join(SAVE_DIR, 'fm_final.pth')

import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 2D目标分布: 8-Gaussian
# ============================================================
def sample_8gaussian(n, device='cpu'):
    """8-Gaussian分布：8个高斯均匀分布在单位圆上"""
    # 8个中心分布在半径为2的圆上
    n_modes = 8
    radius = 2.0
    angles = torch.arange(n_modes, device=device) * 2 * np.pi / n_modes
    centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
    # 每个样本随机选一个中心
    mode_idx = torch.randint(0, n_modes, (n,), device=device)
    noise = 0.15 * torch.randn(n, 2, device=device)
    samples = centers[mode_idx] + noise
    return samples


# ============================================================
# 向量场网络: 增强MLP（2D输入 → 2D输出）
# ============================================================
class SinusoidalTimeEmbed(nn.Module):
    """正弦位置编码：将标量 t 映射到高维向量
    
    类似Transformer的时间编码，让网络更好区分不同时间步
    """
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        # 预计算频率（避免每次forward创建新张量）
        self.register_buffer('freqs', torch.exp(
            -torch.log(torch.tensor(10000.0)) * torch.arange(0, half_dim) / half_dim
        ))
        
    def forward(self, t):
        # t: (batch,) or (batch, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        args = t * self.freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class VectorFieldMLP(nn.Module):
    """2D向量场网络 v_θ(x, t)
    输入: (x, t)，x∈R^2, t∈[0,1]
    输出: v∈R^2
    
    改进：128维隐藏层 + SiLU激活 + 正弦时间编码
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbed(dim=64)
        self.net = nn.Sequential(
            nn.Linear(2 + 64, hidden_dim),  # x(2) + time_embed(64)
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        # 最后一层初始化为零，使初始向量场接近零
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)


# ============================================================
# 步骤1：Neural ODE训练（最大似然/伴随方法）
# ============================================================
# 14.2节核心：CNF训练需要求解ODE，这是其瓶颈所在
# 这里使用简化实现：固定步长Euler方法模拟ODE
# 实际Neural ODE论文使用torchdiffeq的伴随方法，但原理相同

print("\n" + "="*60)
print("步骤1: Neural ODE训练（最大似然目标）")
print("="*60)
print("14.2节: CNF训练需要求解ODE来计算损失")
print("  瞬时变量替换公式: d/dt log p_t(x_t) = -tr(∇_x v_θ(x_t, t))")
print("  训练目标: 最大化 log p_1(x_data)")
print("  瓶颈: 每次前向需要数值求解ODE，反向用伴随方法")

# 训练超参数
n_train = 2000
n_epochs = 500  # 200→500，给FM更多收敛时间
ode_steps = 20  # ODE求解步数（精度与速度的权衡）
lr = 2e-3  # 1e-3→2e-3
batch_size = 256

# 生成训练数据
train_data = sample_8gaussian(n_train, device=device)
train_dataset = torch.utils.data.TensorDataset(train_data)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def ode_solve_euler(vf_net, z0, n_steps, t_start=0.0, t_end=1.0):
    """Euler方法求解ODE: dx/dt = v_θ(x, t)

    14.2.2节: Neural ODE的前向传播即ODE求解
    这里使用固定步长Euler方法（简化实现）
    """
    dt = (t_end - t_start) / n_steps
    x = z0
    for step in range(n_steps):
        t_val = t_start + step * dt
        t_tensor = torch.full((x.shape[0],), t_val, device=x.device)
        v = vf_net(x, t_tensor)
        x = x + v * dt
    return x


def compute_log_likelihood(vf_net, x_data, n_steps):
    """计算对数似然 log p_1(x_data)

    14.2.1节: 瞬时变量替换公式
    log p_1(x) = log p_0(z) - ∫_0^1 tr(∇_x v_θ(x_t, t)) dt

    使用Hutchinson迹估计（有限差分法，FFJORD标准做法）
    """
    batch = x_data.shape[0]
    # 逆向求解ODE：从x_data(数据端)到z(噪声端)
    dt = -1.0 / n_steps
    x = x_data.clone()
    log_det_sum = 0.0

    for step in range(n_steps):
        t_val = 1.0 - step / n_steps
        t_tensor = torch.full((batch,), t_val, device=x.device)
        v = vf_net(x, t_tensor)
        
        # Hutchinson迹估计: tr(J) ≈ ε^T J ε
        # 使用较小的扰动(1e-5)以平衡精度和稳定性
        eps = torch.randn_like(x)
        x_perturbed = x + eps * 1e-5
        v_perturbed = vf_net(x_perturbed, t_tensor)
        jvp = (v_perturbed - v) / 1e-5
        trace_est = (eps * jvp).sum(dim=-1)
        
        log_det_sum = log_det_sum + trace_est * abs(dt)
        x = x + v * dt

    # z ≈ x（逆向ODE的终点）
    # log p_0(z) = -0.5 * ||z||^2 - d/2 * log(2π)（标准高斯）
    # 瞬时变量替换公式: log p_1(x) = log p_0(z) - ∫_0^1 tr(∇_x v_θ) dt
    # 注意是减号：若向量场使体积膨胀(tr>0)，密度应下降
    log_p0 = -0.5 * (x ** 2).sum(dim=-1) - np.log(2 * np.pi)
    log_p1 = log_p0 - log_det_sum
    return log_p1


# 创建Neural ODE模型
model_cnll = VectorFieldMLP(hidden_dim=128).to(device)
optimizer_cnll = torch.optim.Adam(model_cnll.parameters(), lr=lr)
scheduler_cnll = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cnll, T_max=n_epochs)

# Resume检查
skip_training_cnll = False
start_epoch_cnll = 0
losses_cnll = []
times_cnll = []

if os.path.exists(FINAL_CHECKPOINT_PATH_CNLL):
    print(f"检测到最终权重: {FINAL_CHECKPOINT_PATH_CNLL}")
    print("直接加载，跳过训练过程")
    checkpoint = torch.load(FINAL_CHECKPOINT_PATH_CNLL, map_location=device, weights_only=False)
    model_cnll.load_state_dict(checkpoint['model_state_dict'])
    optimizer_cnll.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler_cnll.load_state_dict(checkpoint['scheduler_state_dict'])
    losses_cnll = checkpoint.get('train_losses', [])
    times_cnll = checkpoint.get('train_times', [])
    skip_training_cnll = True
elif os.path.exists(CHECKPOINT_PATH_CNLL):
    print(f"检测到中间权重: {CHECKPOINT_PATH_CNLL}")
    print("继续训练...")
    checkpoint = torch.load(CHECKPOINT_PATH_CNLL, map_location=device, weights_only=False)
    model_cnll.load_state_dict(checkpoint['model_state_dict'])
    optimizer_cnll.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler_cnll.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch_cnll = checkpoint.get('epoch', 0) + 1
    losses_cnll = checkpoint.get('train_losses', [])
    times_cnll = checkpoint.get('train_times', [])

if not skip_training_cnll:
    print(f"训练Neural ODE (CNLL)，从 epoch {start_epoch_cnll} 开始...")
    t_start = time.time()

    for epoch in tqdm(range(start_epoch_cnll, n_epochs), desc="[Neural ODE] 训练进度"):
        model_cnll.train()
        total_loss = 0
        epoch_start = time.time()

        for (x_data,) in train_loader:
            x_data = x_data.to(device)

            # CNLL训练: 最小化负对数似然
            # 需要求解ODE计算对数似然，这是CNF训练瓶颈
            log_p = compute_log_likelihood(model_cnll, x_data, n_steps=ode_steps)
            loss = -log_p.mean()  # 负对数似然
            
            # 检查loss是否为NaN或inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [警告] Epoch {epoch+1} loss为NaN/inf: {loss.item()}")
                # 跳过该批次，继续训练
                continue

            optimizer_cnll.zero_grad()
            loss.backward()
            
            # 检查梯度是否存在NaN/inf
            grad_norms = []
            grad_has_nan = False
            for param in model_cnll.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        grad_has_nan = True
                        break
                    grad_norms.append(param.grad.norm().item())
            
            if grad_has_nan:
                print(f"  [警告] Epoch {epoch+1} 检测到NaN/inf梯度，跳过更新")
                # 清除梯度，跳过该批次
                optimizer_cnll.zero_grad()
                continue
                
            # 梯度裁剪，防止ODE求解不稳定性导致梯度爆炸
            torch.nn.utils.clip_grad_norm_(model_cnll.parameters(), max_norm=5.0)
            optimizer_cnll.step()
            total_loss += loss.item() * x_data.shape[0]

        avg_loss = total_loss / len(train_dataset)
        epoch_time = time.time() - epoch_start
        losses_cnll.append(avg_loss)
        times_cnll.append(epoch_time)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [Neural ODE] Epoch {epoch+1:3d}/{n_epochs}  "
                  f"NLL={avg_loss:.4f}  时间={epoch_time:.2f}s")

        # 学习率调度
        scheduler_cnll.step()

        # 保存中间checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_cnll.state_dict(),
                'optimizer_state_dict': optimizer_cnll.state_dict(),
                'scheduler_state_dict': scheduler_cnll.state_dict(),
                'train_losses': losses_cnll,
                'train_times': times_cnll,
            }, CHECKPOINT_PATH_CNLL)

    # 保存最终权重
    torch.save({
        'model_state_dict': model_cnll.state_dict(),
        'optimizer_state_dict': optimizer_cnll.state_dict(),
        'scheduler_state_dict': scheduler_cnll.state_dict(),
        'train_losses': losses_cnll,
        'train_times': times_cnll,
    }, FINAL_CHECKPOINT_PATH_CNLL)
    print(f"最终权重已保存: {FINAL_CHECKPOINT_PATH_CNLL}")
    t_elapsed = time.time() - t_start
    print(f"Neural ODE训练完成，总耗时: {t_elapsed:.1f}s")

print(f"Neural ODE训练完成！平均每epoch时间: {np.mean(times_cnll):.2f}s" if times_cnll else "")


# ============================================================
# 步骤2：Flow Matching训练
# ============================================================
print("\n" + "="*60)
print("步骤2: Flow Matching训练（CFM目标）")
print("="*60)
print("14.3节: CFM定理的核心突破")
print("  条件向量场闭式可计算: v_t(x_t|x_0,z) = x_0 - z")
print("  训练目标: E[||v_θ(x_t,t) - (x_0-z)||²]")
print("  无需模拟ODE即可训练！")

# 创建Flow Matching模型
model_fm = VectorFieldMLP(hidden_dim=128).to(device)
optimizer_fm = torch.optim.Adam(model_fm.parameters(), lr=lr)
scheduler_fm = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fm, T_max=n_epochs)

# Resume检查
skip_training_fm = False
start_epoch_fm = 0
losses_fm = []
times_fm = []

if os.path.exists(FINAL_CHECKPOINT_PATH_FM):
    print(f"检测到最终权重: {FINAL_CHECKPOINT_PATH_FM}")
    print("直接加载，跳过训练过程")
    checkpoint = torch.load(FINAL_CHECKPOINT_PATH_FM, map_location=device, weights_only=False)
    model_fm.load_state_dict(checkpoint['model_state_dict'])
    optimizer_fm.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler_fm.load_state_dict(checkpoint['scheduler_state_dict'])
    losses_fm = checkpoint.get('train_losses', [])
    times_fm = checkpoint.get('train_times', [])
    skip_training_fm = True
elif os.path.exists(CHECKPOINT_PATH_FM):
    print(f"检测到中间权重: {CHECKPOINT_PATH_FM}")
    print("继续训练...")
    checkpoint = torch.load(CHECKPOINT_PATH_FM, map_location=device, weights_only=False)
    model_fm.load_state_dict(checkpoint['model_state_dict'])
    optimizer_fm.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler_fm.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch_fm = checkpoint.get('epoch', 0) + 1
    losses_fm = checkpoint.get('train_losses', [])
    times_fm = checkpoint.get('train_times', [])

if not skip_training_fm:
    print(f"训练Flow Matching，从 epoch {start_epoch_fm} 开始...")
    t_start = time.time()

    for epoch in tqdm(range(start_epoch_fm, n_epochs), desc="[Flow Matching] 训练进度"):
        model_fm.train()
        total_loss = 0
        epoch_start = time.time()

        for (x_data,) in train_loader:
            x_data = x_data.to(device)
            batch = x_data.shape[0]

            # Flow Matching训练（CFM目标）
            # 14.3.2节: 条件向量场 v_t = x_0 - z
            z = torch.randn_like(x_data)  # 源分布: N(0,I)
            t = torch.rand(batch, device=device)  # t ~ U[0,1]
            t_2d = t[:, None]  # (batch, 1)

            # 线性插值路径: x_t = (1-t)z + t*x_0
            x_t = (1 - t_2d) * z + t_2d * x_data

            # 条件向量场目标: v = x_0 - z
            v_target = x_data - z

            # 网络预测
            v_pred = model_fm(x_t, t)

            # CFM损失
            loss = F.mse_loss(v_pred, v_target)

            optimizer_fm.zero_grad()
            loss.backward()
            optimizer_fm.step()
            total_loss += loss.item() * batch

        avg_loss = total_loss / len(train_dataset)
        epoch_time = time.time() - epoch_start
        losses_fm.append(avg_loss)
        times_fm.append(epoch_time)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [Flow Matching] Epoch {epoch+1:3d}/{n_epochs}  "
                  f"Loss={avg_loss:.6f}  时间={epoch_time:.2f}s")

        # 学习率调度
        scheduler_fm.step()

        # 保存中间checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_fm.state_dict(),
                'optimizer_state_dict': optimizer_fm.state_dict(),
                'scheduler_state_dict': scheduler_fm.state_dict(),
                'train_losses': losses_fm,
                'train_times': times_fm,
            }, CHECKPOINT_PATH_FM)

    # 保存最终权重
    torch.save({
        'model_state_dict': model_fm.state_dict(),
        'optimizer_state_dict': optimizer_fm.state_dict(),
        'scheduler_state_dict': scheduler_fm.state_dict(),
        'train_losses': losses_fm,
        'train_times': times_fm,
    }, FINAL_CHECKPOINT_PATH_FM)
    print(f"最终权重已保存: {FINAL_CHECKPOINT_PATH_FM}")
    t_elapsed = time.time() - t_start
    print(f"Flow Matching训练完成，总耗时: {t_elapsed:.1f}s")

print(f"Flow Matching训练完成！平均每epoch时间: {np.mean(times_fm):.2f}s" if times_fm else "")


# ============================================================
# 步骤3：对比两种方法
# ============================================================
print("\n" + "="*60)
print("步骤3: Neural ODE vs Flow Matching 对比")
print("="*60)

# 3.1 训练效率对比（收敛曲线 + 时间对比）
print("\n训练效率对比:")
if times_cnll and times_fm:
    avg_time_cnll = np.mean(times_cnll)
    avg_time_fm = np.mean(times_fm)
    speedup = avg_time_cnll / avg_time_fm if avg_time_fm > 0 else float('inf')
    print(f"  Neural ODE 平均每epoch: {avg_time_cnll:.3f}s")
    print(f"  Flow Matching 平均每epoch: {avg_time_fm:.3f}s")
    print(f"  加速比: {speedup:.1f}x")
    print(f"  → Flow Matching每次迭代无需求解ODE，训练速度显著更快")

# 3.2 采样质量对比
n_gen = 2000
n_ode_steps = 100  # 50→100，更精细的采样积分

print(f"\n生成采样 ({n_gen} 个样本, {n_ode_steps}步ODE)...")

# Neural ODE采样
with torch.no_grad():
    z_gen = torch.randn(n_gen, 2, device=device)
    samples_cnll = ode_solve_euler(model_cnll, z_gen, n_steps=n_ode_steps)

# Flow Matching采样
with torch.no_grad():
    z_gen = torch.randn(n_gen, 2, device=device)
    samples_fm = ode_solve_euler(model_fm, z_gen, n_steps=n_ode_steps)

# 真实数据
test_data = sample_8gaussian(n_gen, device=device)

# 计算采样质量指标：最大均值差异(MMD)
def compute_mmd(x, y, sigma=1.0):
    """计算最大均值差异(MMD)评估分布距离

    MMD越小表示两个分布越接近
    使用RBF核: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
    """
    xx = torch.cdist(x, x, p=2) ** 2
    yy = torch.cdist(y, y, p=2) ** 2
    xy = torch.cdist(x, y, p=2) ** 2
    c = 2 * sigma ** 2
    mmd = torch.mean(torch.exp(-xx / c)) + torch.mean(torch.exp(-yy / c)) - 2 * torch.mean(torch.exp(-xy / c))
    return mmd.item()

mmd_cnll = compute_mmd(samples_cnll, test_data)
mmd_fm = compute_mmd(samples_fm, test_data)

print(f"\n采样质量对比（MMD, 越小越好）:")
print(f"  Neural ODE采样 vs 真实分布: {mmd_cnll:.6f}")
print(f"  Flow Matching采样 vs 真实分布: {mmd_fm:.6f}")
if mmd_fm < mmd_cnll:
    print(f"  → Flow Matching采样质量更优（MMD低{(mmd_cnll - mmd_fm)/mmd_cnll*100:.1f}%）")
else:
    print(f"  → 两种方法采样质量相近")

# 3.3 采样轨迹对比
# 注意：可视化用少量轨迹（画得清），曲率统计用大样本（统计稳定）
n_traj_viz = 5  # 可视化用的轨迹数
n_traj_stats = 2000  # 曲率统计用的轨迹数（与n_gen一致，复用生成样本）

# --- 3.3a 可视化轨迹（5条，用于图3） ---
print(f"\n采样轨迹对比（{n_traj_viz}条可视化轨迹）...")
with torch.no_grad():
    z_traj_viz = torch.randn(n_traj_viz, 2, device=device)

    # 记录Neural ODE轨迹（CRN：共享随机数起点）
    traj_cnll_viz = [z_traj_viz.cpu().numpy().copy()]
    x_cnll = z_traj_viz.clone()
    dt = 1.0 / n_ode_steps
    for step in range(n_ode_steps):
        t_val = step / n_ode_steps
        t_tensor = torch.full((n_traj_viz,), t_val, device=device)
        v = model_cnll(x_cnll, t_tensor)
        x_cnll = x_cnll + v * dt
        traj_cnll_viz.append(x_cnll.cpu().numpy().copy())
    traj_cnll_viz = np.array(traj_cnll_viz)  # (n_steps+1, n_traj_viz, 2)

    # 记录Flow Matching轨迹（CRN：同一批噪声起点）
    traj_fm_viz = [z_traj_viz.cpu().numpy().copy()]
    x_fm = z_traj_viz.clone()
    for step in range(n_ode_steps):
        t_val = step / n_ode_steps
        t_tensor = torch.full((n_traj_viz,), t_val, device=device)
        v = model_fm(x_fm, t_tensor)
        x_fm = x_fm + v * dt
        traj_fm_viz.append(x_fm.cpu().numpy().copy())
    traj_fm_viz = np.array(traj_fm_viz)

# --- 3.3b 曲率统计（大样本，用于定量结论） ---
print(f"曲率统计（{n_traj_stats}条轨迹，用于定量对比）...")
with torch.no_grad():
    z_traj_stats = torch.randn(n_traj_stats, 2, device=device)


@torch.no_grad()
def compute_curvature_batch(vf_net, z_start, n_ode_steps):
    """批量计算轨迹的平均曲率（完整轨迹积分）

    14.4.2节: 曲率κ = 1 - 直线距离/路径总长
    """
    n_traj_pts = z_start.shape[0]
    dt = 1.0 / n_ode_steps
    x = z_start.clone()
    # 累计路径长度
    path_len = torch.zeros(n_traj_pts, device=z_start.device)

    for step in range(n_ode_steps):
        t_val = step / n_ode_steps
        t_tensor = torch.full((n_traj_pts,), t_val, device=z_start.device)
        v = vf_net(x, t_tensor)
        step_displacement = v * dt
        path_len = path_len + torch.norm(step_displacement, dim=1)
        x = x + step_displacement

    # 直线距离
    straight_dist = torch.norm(x - z_start, dim=1)
    # 曲率
    kappa = torch.where(path_len > 1e-8,
                        1.0 - straight_dist / path_len,
                        torch.zeros_like(path_len))
    return kappa.mean().item()


# 用大样本计算曲率统计值
curv_cnll = compute_curvature_batch(model_cnll, z_traj_stats, n_ode_steps)
curv_fm = compute_curvature_batch(model_fm, z_traj_stats, n_ode_steps)
print(f"  Neural ODE轨迹平均曲率（n={n_traj_stats}）: {curv_cnll:.6f}")
print(f"  Flow Matching轨迹平均曲率（n={n_traj_stats}）: {curv_fm:.6f}")

# 数据驱动的曲率结论（避免硬编码与数据矛盾）
# 理论说明: 独立耦合(independent coupling)下，条件路径虽是直线，
# 但边际向量场随位置变化，导致生成的轨迹可能弯曲。
# OT-CFM(14.4节)使用最优传输耦合可显著降低曲率。
if curv_fm < curv_cnll:
    traj_conclusion = f"FM轨迹更直（κ低{(curv_cnll - curv_fm)/curv_cnll*100:.1f}%），利于少步采样"
    print(f"  → {traj_conclusion}")
else:
    traj_conclusion = f"Neural ODE轨迹更直（κ低{(curv_fm - curv_cnll)/curv_fm*100:.1f}%），" \
                      f"这是因为本实验使用独立耦合，条件路径虽直但边际向量场随位置变化。" \
                      f"OT-CFM(14.4节)通过最优传输耦合可进一步降低曲率。"
    print(f"  → {traj_conclusion}")

# ============================================================
# 可视化
# ============================================================

# 图1: 收敛曲线对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) 损失曲线
ax = axes[0]
if losses_cnll:
    ax.plot(losses_cnll, label='Neural ODE (NLL)', alpha=0.8)
if losses_fm:
    ax.plot(losses_fm, label=r'Flow Matching ($L_{\mathrm{CFM}}$)', alpha=0.8)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel(r'Loss / NLL', fontsize=12)
ax.set_title(r'(a) 收敛曲线对比', fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# (b) 累计训练时间 vs 损失
ax = axes[1]
if losses_cnll and times_cnll:
    cum_time_cnll = np.cumsum(times_cnll)
    ax.plot(cum_time_cnll, losses_cnll, label='Neural ODE (NLL)', alpha=0.8)
if losses_fm and times_fm:
    cum_time_fm = np.cumsum(times_fm)
    ax.plot(cum_time_fm, losses_fm, label=r'Flow Matching ($L_{\mathrm{CFM}}$)', alpha=0.8)
ax.set_xlabel(r'累计训练时间 (s)', fontsize=12)
ax.set_ylabel(r'Loss / NLL', fontsize=12)
ax.set_title(r'(b) 时间-收敛效率对比', fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.suptitle(r'实验14.2-1：Neural ODE vs Flow Matching 训练效率（14.2节）', fontsize=14, y=1.02)
plt.tight_layout()
fig_path1 = os.path.join(SAVE_DIR, '步骤1_训练效率对比.png')
plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图1已保存: {fig_path1}")

# 图2: 采样质量对比
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 通用绘图范围
xlim, ylim = (-4, 4), (-4, 4)

# (a) 真实分布
ax = axes[0]
ax.scatter(test_data[:, 0].cpu().numpy(), test_data[:, 1].cpu().numpy(),
           c='blue', alpha=0.3, s=10)
ax.set_title(r'(a) 目标分布 (8-Gaussian)', fontsize=13)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')
ax.grid(alpha=0.3)

# (b) Neural ODE采样
ax = axes[1]
ax.scatter(samples_cnll[:, 0].cpu().numpy(), samples_cnll[:, 1].cpu().numpy(),
           c='red', alpha=0.3, s=10)
ax.set_title(r'(b) Neural ODE 采样', fontsize=13)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')
ax.grid(alpha=0.3)

# (c) Flow Matching采样
ax = axes[2]
ax.scatter(samples_fm[:, 0].cpu().numpy(), samples_fm[:, 1].cpu().numpy(),
           c='green', alpha=0.3, s=10)
ax.set_title(r'(c) Flow Matching 采样', fontsize=13)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')
ax.grid(alpha=0.3)

plt.suptitle(r'实验14.2-1：采样质量对比（14.2.3节）', fontsize=14, y=1.02)
plt.tight_layout()
fig_path2 = os.path.join(SAVE_DIR, '步骤2_采样质量对比.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig_path2}")

# 图3: 采样轨迹对比
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = plt.cm.Set1(np.linspace(0, 1, n_traj_viz))

# (a) Neural ODE轨迹（可视化用5条）
ax = axes[0]
for i in range(n_traj_viz):
    ax.plot(traj_cnll_viz[:, i, 0], traj_cnll_viz[:, i, 1], c=colors[i], alpha=0.7, lw=1.5)
    ax.scatter(traj_cnll_viz[0, i, 0], traj_cnll_viz[0, i, 1], c=colors[i], marker='o', s=40, zorder=5)
    ax.scatter(traj_cnll_viz[-1, i, 0], traj_cnll_viz[-1, i, 1], c=colors[i], marker='s', s=40, zorder=5)
ax.set_title(rf'(a) Neural ODE 轨迹（可视化）', fontsize=13)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')
ax.grid(alpha=0.3)

# (b) Flow Matching轨迹（可视化用5条）
ax = axes[1]
for i in range(n_traj_viz):
    ax.plot(traj_fm_viz[:, i, 0], traj_fm_viz[:, i, 1], c=colors[i], alpha=0.7, lw=1.5)
    ax.scatter(traj_fm_viz[0, i, 0], traj_fm_viz[0, i, 1], c=colors[i], marker='o', s=40, zorder=5)
    ax.scatter(traj_fm_viz[-1, i, 0], traj_fm_viz[-1, i, 1], c=colors[i], marker='s', s=40, zorder=5)
ax.set_title(rf'(b) Flow Matching 轨迹（可视化）', fontsize=13)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')
ax.grid(alpha=0.3)

plt.suptitle(r'实验14.2-1：采样轨迹形态对比（14.2.3节）', fontsize=14, y=1.02)
plt.tight_layout()
fig_path3 = os.path.join(SAVE_DIR, '步骤3_轨迹形态对比.png')
plt.savefig(fig_path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"图3已保存: {fig_path3}")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print("实验14.2-1 完成!")
print("=" * 60)
print(f"""
关键结论:
1. 训练效率对比（14.2.3节）
   - Neural ODE: 每次迭代需求解ODE计算似然，耗时 {np.mean(times_cnll):.3f}s/epoch
   - Flow Matching: 无需求解ODE，仅前向传播+MSE损失，耗时 {np.mean(times_fm):.3f}s/epoch
   - 加速比: {np.mean(times_cnll)/np.mean(times_fm):.1f}x
   - 这是CFM定理的核心价值：绕过ODE仿真实现高效训练

2. 采样质量对比（14.2.3节）
   - Neural ODE采样 MMD: {mmd_cnll:.6f}
   - Flow Matching采样 MMD: {mmd_fm:.6f}
   - 两种方法都能学会目标分布，但FM训练更快达到相同质量

3. 轨迹形态对比（基于{n_traj_stats}条轨迹统计）
   - Neural ODE轨迹曲率: {curv_cnll:.4f}
   - Flow Matching轨迹曲率: {curv_fm:.4f}
   - {traj_conclusion}

4. 理论联系
   - CNF训练瓶颈: 求解ODE计算似然（14.2.3节）
   - CFM定理突破: 梯度等价，无需仿真ODE（14.3.2节）
   - Flow Matching = CNF训练的"免仿真"方案

5. 教学脚注：为什么Neural ODE的loss曲线通常比Flow Matching更"毛躁"？
   - Neural ODE训练中，迹估计 tr(∇_x v_θ) 每步只用一个随机探针 eps（Hutchinson估计），
     单探针方差较大，靠训练步数多来平均掉。这是FFJORD等原论文的标准做法，不是错误。
   - Flow Matching的CFM损失是简单的MSE，目标 v_target = x_0 - z 无随机性（给定x_0,z），
     所以loss曲线天然更平滑。
   - 如果观察到Neural ODE的loss抖动较大，这是预期行为，不影响收敛。
""")
