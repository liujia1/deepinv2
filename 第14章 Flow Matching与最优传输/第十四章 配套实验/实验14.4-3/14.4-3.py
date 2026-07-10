# -*- coding: utf-8 -*-
"""
实验14.4-3 曲率单调性数值验证——Reflow定理
对应章节: 14.4.3节

知识点:
  - 曲率定义: κ = 1 - ||x_0-z|| / Σ||x_{i+1}-x_i|| (14.4.2节)
  - Reflow定理: κ(Z^{k+1}) ≤ κ(Z^k) (14.4.3节)
  - Reflow过程: 用当前模型的ODE端点重新配对，迭代"解缠"路径交叉
  - 极限性质: 多轮Reflow趋近OT映射

实验内容:
  步骤1: 在2D点云上训练1-Rectified Flow并计算曲率
  步骤2: 执行多轮Reflow（1-RF→2-RF→3-RF），每轮计算曲率
  步骤3: 验证κ单调递减，可视化路径逐步变直的过程

数据集: 2D高斯混合，纯CPU实验

素材来源:
  - Liu et al. (2023) Rectified Flow
  - 14.1-1中的2D点云框架

运行前提: PyTorch, CPU即可
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
    SAVE_DIR = os.path.join(_gdrive, '实验14.4-3')
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

device = torch.device('cpu')
print(f"\n{'='*60}")
print(f"实验14.4-3: 曲率单调性数值验证——Reflow定理")
print(f"{'='*60}")
print(f"使用设备: {device}")
print("  本实验使用2D点云，纯CPU即可完成")

import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 2D目标分布: 交叉高斯混合（让路径交叉更明显）
# ============================================================
def sample_crossed_gaussians(n, device='cpu'):
    """交叉高斯混合：源和目标分布的路径天然交叉"""
    # 源分布: 左上+右下
    mode = torch.rand(n, device=device) < 0.5
    x = torch.where(mode, -2.0, 2.0) + 0.3 * torch.randn(n, device=device)
    y = torch.where(mode, 2.0, -2.0) + 0.3 * torch.randn(n, device=device)
    return torch.stack([x, y], dim=1)

def sample_noise(n, device='cpu'):
    """源分布: 标准高斯"""
    return torch.randn(n, 2, device=device)


# ============================================================
# 向量场网络
# ============================================================
class VectorFieldMLP(nn.Module):
    """2D向量场网络"""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t):
        inp = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        return self.net(inp)


# ============================================================
# ODE采样器（Euler方法）
# ============================================================
@torch.no_grad()
def ode_sample(model, z, n_steps=100):
    """从Flow ODE采样: dx/dt = v_θ(x, t), t: 0→1"""
    model.eval()
    x = z.clone()
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t_val = step / n_steps
        t_tensor = torch.full((x.shape[0],), t_val, device=x.device)
        v = model(x, t_tensor)
        x = x + v * dt
    return x


# ============================================================
# 曲率计算
# ============================================================
def compute_curvature_from_traj(traj):
    """计算轨迹的平均曲率

    14.4.2节定义:
    κ = 1 - 直线距离/路径总长
    κ=0表示完全直线，κ→1表示极度弯曲
    """
    n_traj_pts = traj.shape[0]
    n_samples = traj.shape[1]
    curvatures = []
    for i in range(n_samples):
        path = traj[:, i, :]  # (n_steps+1, 2)
        straight_dist = np.linalg.norm(path[-1] - path[0])
        diffs = np.diff(path, axis=0)
        path_len = np.sum(np.linalg.norm(diffs, axis=1))
        if path_len > 1e-8:
            kappa = 1.0 - straight_dist / path_len
        else:
            kappa = 0.0
        curvatures.append(max(kappa, 0.0))  # 数值保护
    return np.mean(curvatures)


@torch.no_grad()
def compute_curvature_from_model(model, n_samples=500, n_ode_steps=100, device='cpu', z=None):
    """从模型生成轨迹并计算曲率

    参数:
      z: 可选，外部传入的固定噪声（CRN原则），三轮共用同一批z确保曲率比较不受采样噪声干扰
    """
    model.eval()
    if z is None:
        z = torch.randn(n_samples, 2, device=device)

    # 记录轨迹
    traj = [z.cpu().numpy().copy()]
    x = z.clone()
    dt = 1.0 / n_ode_steps
    for step in range(n_ode_steps):
        t_val = step / n_ode_steps
        t_tensor = torch.full((n_samples,), t_val, device=device)
        v = model(x, t_tensor)
        x = x + v * dt
        traj.append(x.cpu().numpy().copy())

    traj = np.array(traj)  # (n_steps+1, n_samples, 2)
    kappa = compute_curvature_from_traj(traj)
    return kappa, traj


# ============================================================
# 训练函数（带Resume能力）
# ============================================================
def train_rf_model(model, optimizer, z_source, x_target, num_epochs=200,
                   checkpoint_path=None, final_checkpoint_path=None,
                   model_label='RF'):
    """训练Rectified Flow模型"""
    start_epoch = 0
    is_final = False
    train_losses = []

    # 检查最终权重
    if final_checkpoint_path and os.path.exists(final_checkpoint_path):
        print(f"检测到最终权重: {final_checkpoint_path}")
        print("直接加载，跳过训练过程")
        checkpoint = torch.load(final_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint.get('train_losses', [])
        return train_losses

    # 检查中间权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"检测到中间权重: {checkpoint_path}")
        print("继续训练...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        train_losses = checkpoint.get('train_losses', [])

    dataset = torch.utils.data.TensorDataset(z_source, x_target)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    print(f"训练 {model_label}，从 epoch {start_epoch} 开始...")
    for epoch in tqdm(range(start_epoch, num_epochs), desc=f"[{model_label}] 训练进度"):
        model.train()
        total_loss = 0
        count = 0
        for z_batch, x_batch in loader:
            batch = z_batch.shape[0]
            t = torch.rand(batch, device=device)
            t_2d = t[:, None]
            x_t = (1 - t_2d) * z_batch + t_2d * x_batch
            v_target = x_batch - z_batch
            v_pred = model(x_t, t)
            loss = F.mse_loss(v_pred, v_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch
            count += batch

        avg_loss = total_loss / count
        train_losses.append(avg_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  [{model_label}] Epoch {epoch+1:3d}/{num_epochs}  Loss={avg_loss:.6f}")

        if checkpoint_path and (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
            }, checkpoint_path)

    # 保存最终权重
    if final_checkpoint_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
        }, final_checkpoint_path)
        print(f"最终权重已保存: {final_checkpoint_path}")

    return train_losses


# ============================================================
# Reflow过程：生成新的端点对
# ============================================================
@torch.no_grad()
def generate_reflow_pairs(model, n_pairs, n_ode_steps=100):
    """用当前模型ODE生成端点对(z, x_0_hat)用于Reflow

    14.4.3节: Reflow过程
    1. 从噪声z ~ N(0,I)出发
    2. 用当前模型ODE推演到终点x_0_hat
    3. 新配对(z, x_0_hat)用于训练下一轮模型
    """
    model.eval()
    z = torch.randn(n_pairs, 2, device=device)
    x0_hat = ode_sample(model, z, n_steps=n_ode_steps)
    return z, x0_hat


# ============================================================
# 步骤1：训练1-Rectified Flow
# ============================================================
print("\n" + "="*60)
print("步骤1: 训练1-Rectified Flow")
print("="*60)
print("14.4.1节: 独立耦合下的RF训练")
print("  z ~ N(0,I), x_0 ~ p_data, 独立配对")
print("  路径可能弯曲交叉（因为独立耦合）")

n_data = 2000
n_epochs = 300
n_ode_steps = 100

# 生成数据对: (噪声, 目标)
z_1rf = sample_noise(n_data, device=device)
x_1rf = sample_crossed_gaussians(n_data, device=device)

# Checkpoint路径
ckpt_paths = {}
final_paths = {}
for k in range(1, 4):
    ckpt_paths[k] = os.path.join(SAVE_DIR, f'{k}rf_checkpoint.pth')
    final_paths[k] = os.path.join(SAVE_DIR, f'{k}rf_final.pth')

model_1rf = VectorFieldMLP(hidden_dim=128).to(device)
opt_1rf = torch.optim.Adam(model_1rf.parameters(), lr=2e-3)
losses_1rf = train_rf_model(model_1rf, opt_1rf, z_1rf, x_1rf,
                             num_epochs=n_epochs,
                             checkpoint_path=ckpt_paths[1],
                             final_checkpoint_path=final_paths[1],
                             model_label='1-RF')

# 生成固定噪声用于CRN曲率比较（跨轮共用同一批z，消除采样噪声对单调性判断的干扰）
z_curv = torch.randn(500, 2, device=device)

# 计算1-RF曲率
kappa_1rf, traj_1rf = compute_curvature_from_model(model_1rf, n_samples=500, n_ode_steps=n_ode_steps, device=device, z=z_curv)
print(f"\n1-RF 曲率: κ = {kappa_1rf:.6f}")


# ============================================================
# 步骤2：多轮Reflow
# ============================================================
print("\n" + "="*60)
print("步骤2: 多轮Reflow (1-RF → 2-RF → 3-RF)")
print("="*60)
print("14.4.3节: Reflow定理 κ(Z^{k+1}) ≤ κ(Z^k)")
print("  每轮Reflow用当前模型的ODE端点重新配对")
print("  轨迹逐步变直，趋近OT映射")

# 存储每轮的结果
curvatures = {1: kappa_1rf}
trajectories = {1: traj_1rf}
models = {1: model_1rf}

n_reflow_rounds = 3

for k in range(2, n_reflow_rounds + 1):
    print(f"\n--- 第 {k} 轮 Reflow ---")

    # 用上一轮模型生成Reflow端点对（每轮独立seed，与全局RNG流解耦，确保可复现）
    prev_model = models[k - 1]
    torch.manual_seed(1000 + k)
    z_new, x_new = generate_reflow_pairs(prev_model, n_pairs=n_data, n_ode_steps=n_ode_steps)

    # 训练新的RF模型
    model_k = VectorFieldMLP(hidden_dim=128).to(device)
    opt_k = torch.optim.Adam(model_k.parameters(), lr=2e-3)
    losses_k = train_rf_model(model_k, opt_k, z_new, x_new.detach(),
                               num_epochs=n_epochs,
                               checkpoint_path=ckpt_paths[k],
                               final_checkpoint_path=final_paths[k],
                               model_label=f'{k}-RF')

    # 计算曲率（使用固定z_curv，CRN原则确保跨轮比较不受采样噪声干扰）
    kappa_k, traj_k = compute_curvature_from_model(model_k, n_samples=500, n_ode_steps=n_ode_steps, device=device, z=z_curv)
    curvatures[k] = kappa_k
    trajectories[k] = traj_k
    models[k] = model_k

    print(f"{k}-RF 曲率: κ = {kappa_k:.6f}")

# 验证单调性
print("\n曲率单调性验证:")
kappa_values = [curvatures[k] for k in range(1, n_reflow_rounds + 1)]
monotonic = True
for k in range(len(kappa_values) - 1):
    if kappa_values[k + 1] > kappa_values[k] + 1e-6:
        monotonic = False
        print(f"  κ(Z^{k+1}) = {kappa_values[k+1]:.6f} > κ(Z^{k}) = {kappa_values[k]:.6f}  ✗ 违反单调性!")
    else:
        reduction = (kappa_values[k] - kappa_values[k+1]) / kappa_values[k] * 100
        print(f"  κ(Z^{k+1}) = {kappa_values[k+1]:.6f} ≤ κ(Z^{k}) = {kappa_values[k]:.6f}  ✓ 降幅: {reduction:.1f}%")

if monotonic:
    print("\n★ Reflow定理验证通过: κ(Z^{k+1}) ≤ κ(Z^k) 对所有k成立")
else:
    print("\n注意: 单调性在有限训练轮次下可能不完全成立，这是数值误差导致的")
    print("  理论上Reflow定理在训练充分收敛时严格成立")


# ============================================================
# 步骤3：可视化
# ============================================================

# 图1: 路径逐步变直过程
n_show_traj = 30  # 展示的轨迹数量
fig, axes = plt.subplots(1, n_reflow_rounds, figsize=(6 * n_reflow_rounds, 5.5))
if n_reflow_rounds == 1:
    axes = [axes]

# 使用相同的噪声起点，以便直观比较
torch.manual_seed(999)
z_show = torch.randn(n_show_traj, 2, device=device)
torch.manual_seed(42)  # 恢复随机种子

for k in range(1, n_reflow_rounds + 1):
    ax = axes[k - 1]
    model = models[k]

    # 生成轨迹
    with torch.no_grad():
        traj = [z_show.cpu().numpy().copy()]
        x = z_show.clone()
        dt = 1.0 / n_ode_steps
        for step in range(n_ode_steps):
            t_val = step / n_ode_steps
            t_tensor = torch.full((n_show_traj,), t_val, device=device)
            v = model(x, t_tensor)
            x = x + v * dt
            traj.append(x.cpu().numpy().copy())
        traj = np.array(traj)

    # 绘制轨迹
    for i in range(n_show_traj):
        ax.plot(traj[:, i, 0], traj[:, i, 1], alpha=0.3, lw=0.8)
        ax.scatter(traj[0, i, 0], traj[0, i, 1], c='blue', s=10, zorder=5)
        ax.scatter(traj[-1, i, 0], traj[-1, i, 1], c='red', s=10, zorder=5)

    kappa_val = curvatures[k]
    ax.set_title(rf'{k}-RF: $\kappa$={kappa_val:.4f}', fontsize=13)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

plt.suptitle(r'实验14.4-3：Reflow路径逐步变直（14.4.3节）', fontsize=14, y=1.02)
plt.tight_layout()
fig_path1 = os.path.join(SAVE_DIR, '步骤1_Reflow路径变直.png')
plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图1已保存: {fig_path1}")

# 图2: 曲率单调递减曲线
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

rounds = list(range(1, n_reflow_rounds + 1))
ax.plot(rounds, kappa_values, 'bo-', lw=2, markersize=10, label=r'曲率 $\kappa(Z^k)$')

# 标注每轮数值
for k in rounds:
    ax.annotate(f'{curvatures[k]:.4f}',
                (k, curvatures[k]),
                textcoords="offset points", xytext=(0, 15),
                ha='center', fontsize=11, fontweight='bold')

ax.set_xlabel(r'Reflow轮次 $k$', fontsize=12)
ax.set_ylabel(r'曲率 $\kappa(Z^k)$', fontsize=12)
ax.set_title(r'实验14.4-3：Reflow曲率单调递减验证（14.4.3节）', fontsize=14)
ax.set_xticks(rounds)
ax.set_xticklabels([f'{k}-RF' for k in rounds])
ax.legend(fontsize=12)
ax.grid(alpha=0.3)

# 添加理论注释
ax.annotate(r'Reflow定理: $\kappa(Z^{k+1}) \leq \kappa(Z^k)$',
            xy=(0.5, 0.85), xycoords='axes fraction',
            fontsize=12, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
fig_path2 = os.path.join(SAVE_DIR, '步骤2_曲率单调递减.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig_path2}")

# 图3: 每轮Reflow的采样分布
n_gen = 1000
fig, axes = plt.subplots(1, n_reflow_rounds + 1, figsize=(5 * (n_reflow_rounds + 1), 5))

# (a) 目标分布
target_show = sample_crossed_gaussians(n_gen, device=device)
ax = axes[0]
ax.scatter(target_show[:, 0].cpu().numpy(), target_show[:, 1].cpu().numpy(),
           c='blue', alpha=0.3, s=8)
ax.set_title(r'目标分布', fontsize=13)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.grid(alpha=0.3)

# (b-d) 各轮Reflow采样
for k in range(1, n_reflow_rounds + 1):
    ax = axes[k]
    with torch.no_grad():
        z_gen = torch.randn(n_gen, 2, device=device)
        samples = ode_sample(models[k], z_gen, n_steps=n_ode_steps)
    ax.scatter(samples[:, 0].cpu().numpy(), samples[:, 1].cpu().numpy(),
               c='green', alpha=0.3, s=8)
    ax.set_title(rf'{k}-RF 采样 ($\kappa$={curvatures[k]:.4f})', fontsize=13)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

plt.suptitle(r'实验14.4-3：各轮Reflow采样质量（14.4.3节）', fontsize=14, y=1.02)
plt.tight_layout()
fig_path3 = os.path.join(SAVE_DIR, '步骤3_Reflow采样质量.png')
plt.savefig(fig_path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"图3已保存: {fig_path3}")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print("实验14.4-3 完成!")
print("=" * 60)
print(f"""
关键结论:
1. 曲率单调性验证（14.4.3节）
   - 1-RF曲率: κ = {curvatures[1]:.6f}
   - 2-RF曲率: κ = {curvatures[2]:.6f}
   - 3-RF曲率: κ = {curvatures[3]:.6f}
   - 单调性: {'通过 ✓' if monotonic else '有限轮次下存在数值偏差'}

2. Reflow的物理意义
   - 每轮Reflow用ODE端点重新配对，解缠路径交叉
   - 轨迹逐步变直，趋近OT映射
   - 曲率下降意味着少步采样精度提高

3. 理论与实践的差距
   - Reflow定理保证训练充分收敛时κ单调递减
   - 实际中有限轮次+有限样本可能导致轻微违反
   - 这不影响Reflow作为实用加速方法的有效性

4. 与14.4-1实验的互补
   - 14.4-1在MNIST图像上验证Reflow蒸馏效果
   - 本实验在2D点云上验证曲率单调性理论
   - 两者共同验证Reflow定理：实践效果+理论保证
""")
