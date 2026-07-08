# -*- coding: utf-8 -*-
"""
实验14.3-1 Flow Matching核心：CFM训练与路径对比
对应知识点：
  - 14.3节 Flow Matching（CFM定理、高斯条件路径、OT-CFM）
  - 14.3.3节 CFM定理（条件向量场可计算）
  - 14.3.4节 高斯条件路径
  - 14.3.5节 独立耦合 vs OT耦合
  - 14.3.6节 扩散路径 vs FM路径

本实验不需要GPU，通过2D点云的可视化直观理解Flow Matching的核心概念。

素材来源：
  - book_plan.md的实验14.1计划
  - 14.3节的理论内容
  - ★ 原创设计：扩散路径 vs FM路径 vs OT-CFM路径的对比

实验内容：
  步骤1：CFM训练——学习向量场（14.3.3节/14.3.4节）
  步骤2：扩散路径 vs FM路径 vs OT-CFM路径（14.3.5节/14.3.6节）
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
    SAVE_DIR = os.path.join(_gdrive, '实验14.3-1')
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

# Checkpoint路径
CHECKPOINT_PATH_IND = os.path.join(SAVE_DIR, 'cfm_independent_checkpoint.pth')
CHECKPOINT_PATH_OT = os.path.join(SAVE_DIR, 'cfm_ot_checkpoint.pth')
FINAL_CHECKPOINT_PATH_IND = os.path.join(SAVE_DIR, 'cfm_independent_final.pth')
FINAL_CHECKPOINT_PATH_OT = os.path.join(SAVE_DIR, 'cfm_ot_final.pth')


# ============================================================
# 2D点云分布定义
# ============================================================
def sample_source(n):
    """源分布：两个高斯的混合（左上+右下）"""
    mode = np.random.rand(n) < 0.5
    x = np.where(mode, -2, 2) + 0.3 * np.random.randn(n)
    y = np.where(mode, 2, -2) + 0.3 * np.random.randn(n)
    return np.stack([x, y], axis=1)

def sample_target(n):
    """目标分布：两个高斯的混合（右上+左下），与源交叉"""
    mode = np.random.rand(n) < 0.5
    x = np.where(mode, 2, -2) + 0.3 * np.random.randn(n)
    y = np.where(mode, 2, -2) + 0.3 * np.random.randn(n)
    return np.stack([x, y], axis=1)


# ============================================================
# 最优传输耦合（小规模：使用匈牙利算法）
# ============================================================
def ot_coupling(source, target):
    """计算最优传输配对（最小化总距离的配对）

    对于2D点云，使用匈牙利算法（O(n^3)）
    这对应14.1节Kantorovich问题的离散解
    """
    from scipy.optimize import linear_sum_assignment
    # 代价矩阵: C[i,j] = ||source[i] - target[j]||^2
    diff = source[:, None, :] - target[None, :, :]  # (n, n, 2)
    cost = np.sum(diff**2, axis=-1)  # (n, n)
    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind  # 返回source[i]应该配对的target索引

def independent_coupling(n):
    """独立耦合：随机配对（14.3.5节的默认CFM）"""
    return np.random.permutation(n)


# ============================================================
# 路径插值函数
# ============================================================
def linear_interp(z, x0, t):
    """线性插值路径（14.3.4节 OT路径/14.4.1节直线插值）
    x_t = (1-t)z + t*x_0,  v_t = x_0 - z
    """
    return (1 - t) * z + t * x0

def diffusion_interp(z, x0, t, beta_min=0.1, beta_max=20.0):
    """扩散路径（14.3.6节，VP-SDE条件路径）
    x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
    使用cosine schedule: ᾱ_t = cos²(π/2 * (1-t))，t∈[0,1]
    z作为噪声源ε
    """
    alpha_bar_t = np.cos(np.pi / 2 * (1 - t)) ** 2
    alpha_bar_t = max(alpha_bar_t, 1e-10)
    return np.sqrt(alpha_bar_t) * x0 + np.sqrt(1 - alpha_bar_t) * z


# ============================================================
# 步骤1：CFM训练——学习向量场（14.3.3节/14.3.4节）
# ============================================================
print("=" * 60)
print("实验14.3-1 步骤1：CFM训练——学习向量场（14.3.3节/14.3.4节）")
print("=" * 60)

print("""
14.3.3节 CFM定理：∇_θ L_FM = ∇_θ L_CFM
  - L_FM = E_{t,x_t} ||v_θ(x_t,t) - v_t(x_t)||²  （边际向量场，不可计算）
  - L_CFM = E_{t,x_0,z} ||v_θ(x_t,t) - (x_0-z)||²  （条件向量场，可计算）

  类比第6章DSM定理：边际得分不可计算→条件得分可计算
  CFM是向量场版本的"得分匹配"！

14.3.4节 高斯条件路径：
  x_t = (1-t)z + t*x_0  (OT路径, σ≡0)
  v_t(x_t|x_0,z) = x_0 - z  (条件向量场=常数)
""")

# 使用PyTorch实现简单的2D向量场网络
import torch.nn as nn

class VectorFieldNet(nn.Module):
    """2D向量场网络 v_θ(x_t, t)"""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),  # 输入: x_t(2) + t(1)
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),  # 输出: v(2)
        )

    def forward(self, x_t, t):
        """x_t: (B, 2), t: (B, 1)"""
        inp = torch.cat([x_t, t], dim=-1)
        return self.net(inp)

# 训练CFM（带Resume能力）
def train_cfm(n_epochs=2000, n_samples=256, coupling='independent', lr=1e-3, checkpoint_path=None, final_checkpoint_path=None):
    """训练Conditional Flow Matching（带checkpoint Resume能力）

    coupling: 'independent' 或 'ot'
    """
    model = VectorFieldNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    start_epoch = 0

    # 检查是否有最终权重
    if final_checkpoint_path and os.path.exists(final_checkpoint_path):
        print(f"检测到最终权重: {final_checkpoint_path}")
        print("直接加载，跳过训练过程")
        checkpoint = torch.load(final_checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        losses = checkpoint.get('losses', [])
        return model, losses

    # 检查是否有中间权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"检测到中间权重: {checkpoint_path}")
        print("继续训练...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        losses = checkpoint.get('losses', [])

    print(f"训练 {coupling} CFM，从 epoch {start_epoch} 开始...")

    for epoch in range(start_epoch, n_epochs):
        # 采样源和目标
        z = torch.FloatTensor(sample_source(n_samples))  # (B, 2)
        x0 = torch.FloatTensor(sample_target(n_samples))  # (B, 2)

        # 配对
        if coupling == 'ot':
            with torch.no_grad():
                z_np = z.numpy()
                x0_np = x0.numpy()
                idx = ot_coupling(z_np, x0_np)
                x0 = torch.FloatTensor(x0_np[idx])

        # 采样t ~ U[0,1]
        t = torch.rand(n_samples, 1)

        # 线性插值: x_t = (1-t)z + t*x_0
        x_t = (1 - t) * z + t * x0

        # 条件向量场: v_t = x_0 - z
        v_target = x0 - z

        # 网络预测
        v_pred = model(x_t, t)

        # CFM损失
        loss = nn.functional.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0 or epoch == 0:
            losses.append(loss.item())
            if (epoch + 1) % 500 == 0:
                print(f"  [{coupling}] Epoch {epoch+1}/{n_epochs} Loss={loss.item():.6f}")

        # 每200轮保存中间checkpoint
        if checkpoint_path and (epoch + 1) % 200 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'losses': losses
            }, checkpoint_path)

    # 保存最终权重
    if final_checkpoint_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses
        }, final_checkpoint_path)
        print(f"最终权重已保存: {final_checkpoint_path}")

    return model, losses

# 训练两个模型
print("\n训练独立耦合CFM...")
model_ind, losses_ind = train_cfm(n_epochs=2000, coupling='independent',
                                   checkpoint_path=CHECKPOINT_PATH_IND,
                                   final_checkpoint_path=FINAL_CHECKPOINT_PATH_IND)
print("\n训练OT耦合CFM (OT-CFM)...")
model_ot, losses_ot = train_cfm(n_epochs=2000, coupling='ot',
                                 checkpoint_path=CHECKPOINT_PATH_OT,
                                 final_checkpoint_path=FINAL_CHECKPOINT_PATH_OT)


# ============================================================
# Flow ODE采样
# ============================================================
@torch.no_grad()
def flow_sample(model, z_source, n_steps=100):
    """从Flow ODE采样: dx/dt = v_θ(x, t), t: 0→1"""
    model.eval()
    x = z_source.clone()
    dt = 1.0 / n_steps
    trajectory = [x.clone().numpy()]

    for step in range(n_steps):
        t_val = step / n_steps
        t = torch.full((x.shape[0], 1), t_val)
        v = model(x, t)
        x = x + v * dt  # Euler步进
        trajectory.append(x.clone().numpy())

    return np.array(trajectory)  # (n_steps+1, B, 2)

# 从源分布采样，用Flow ODE推到目标分布
n_test = 200
np.random.seed(42)
z_test = torch.FloatTensor(sample_source(n_test))

traj_ind = flow_sample(model_ind, z_test, n_steps=100)
traj_ot = flow_sample(model_ot, z_test, n_steps=100)

# 计算采样质量（与真实目标的Wasserstein-1距离近似）
from scipy.stats import wasserstein_distance_nd
target_test = sample_target(n_test)
wd_ind = wasserstein_distance_nd(traj_ind[-1], target_test)
wd_ot = wasserstein_distance_nd(traj_ot[-1], target_test)
print(f"\n独立耦合CFM采样 W1距离: {wd_ind:.4f}")
print(f"OT-CFM采样 W1距离:      {wd_ot:.4f}")


# ============================================================
# 步骤2：扩散路径 vs FM路径 vs OT-CFM路径（14.3.5节/14.3.6节）
# ============================================================
print(f"\n{'='*60}")
print("实验14.3-1 步骤2：扩散路径 vs FM路径 vs OT-CFM路径（14.3.5节/14.3.6节）")
print("=" * 60)

print("""
14.3.6节：DDIM = FM with diffusion coupling
  - 扩散模型(SDE): 弯曲路径，需要多步采样
  - DDIM(ODE, diffusion coupling): 半直路径
  - OT-CFM(ODE, OT coupling): 最直路径

直线路径的好处：ODE求解器步数少→采样快
  - 扩散模型: ~1000步
  - DDIM: ~50步
  - OT-CFM: ~10步（甚至1步！）
""")

# 为路径对比准备数据
n_points = 50
source = sample_source(n_points)
target = sample_target(n_points)

# OT耦合
np.random.seed(42)
ot_idx = ot_coupling(source, target)
target_ot = target[ot_idx]

# 独立耦合
np.random.seed(123)
ind_idx = independent_coupling(n_points)
target_ind = target[ind_idx]

# 可视化三种路径
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

# 只画10条轨迹以避免过于密集
n_show = 10
t_vals = np.linspace(0, 1, 100)

# (a) 扩散耦合条件路径（VP-SDE，14.3.6节）
ax = axes[0]
for i in range(n_show):
    path = np.array([diffusion_interp(source[i], target_ind[i], t) for t in t_vals])
    ax.plot(path[:, 0], path[:, 1], 'red', alpha=0.4, lw=1.2)
ax.scatter(source[:n_show, 0], source[:n_show, 1], c='blue', s=30, zorder=5)
ax.scatter(target_ind[:n_show, 0], target_ind[:n_show, 1], c='red', s=30, zorder=5)
ax.set_title('(a) 扩散耦合条件路径\n(VP-SDE, 弯曲)', fontsize=12)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(alpha=0.3)
ax.set_aspect('equal')

# (b) 独立耦合CFM路径（学习到的边际路径）
ax = axes[1]
for i in range(n_show):
    path = traj_ind[:101, i, :]  # 100步+1
    ax.plot(path[:, 0], path[:, 1], 'purple', alpha=0.4, lw=1.2)
ax.scatter(z_test.numpy()[:n_show, 0], z_test.numpy()[:n_show, 1], c='blue', s=30, zorder=5)
ax.scatter(traj_ind[-1, :n_show, 0], traj_ind[-1, :n_show, 1], c='red', s=30, zorder=5)
ax.set_title('(b) 独立耦合CFM边际路径\n(弯曲交叉)', fontsize=12)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(alpha=0.3)
ax.set_aspect('equal')

# (c) OT-CFM路径
ax = axes[2]
for i in range(n_show):
    path = traj_ot[:101, i, :]
    ax.plot(path[:, 0], path[:, 1], 'green', alpha=0.4, lw=1.2)
ax.scatter(z_test.numpy()[:n_show, 0], z_test.numpy()[:n_show, 1], c='blue', s=30, zorder=5)
ax.scatter(traj_ot[-1, :n_show, 0], traj_ot[-1, :n_show, 1], c='red', s=30, zorder=5)
ax.set_title('(c) OT-CFM边际路径\n(短且直)', fontsize=12)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(alpha=0.3)
ax.set_aspect('equal')

# (d) 曲率对比
ax = axes[3]
# 计算扩散耦合条件路径的曲率
diff_paths_curvature = []
for i in range(n_show):
    path = np.array([diffusion_interp(source[i], target_ind[i], t) for t in t_vals])
    direct = np.sqrt(np.sum((path[-1] - path[0])**2))
    diffs_p = np.diff(path, axis=0)
    actual = np.sum(np.sqrt(np.sum(diffs_p**2, axis=-1)))
    diff_paths_curvature.append(1.0 - direct / (actual + 1e-10))
kappa_diff = np.mean(diff_paths_curvature)

def compute_straightness(traj):
    """计算轨迹的曲率指标 κ = 1 - 直线距离/路径实际长度
    κ=0 表示完全直线，κ越大越弯曲
    traj: (T+1, B, 2) 形状的轨迹数组
    """
    curvatures = []
    for i in range(traj.shape[1]):
        path = traj[:, i, :]  # (T+1, 2)
        direct = np.sqrt(np.sum((path[-1] - path[0])**2))
        diffs = np.diff(path, axis=0)
        actual = np.sum(np.sqrt(np.sum(diffs**2, axis=-1)))
        curvatures.append(1.0 - direct / (actual + 1e-10))
    return np.mean(curvatures)

S_ind = compute_straightness(traj_ind)
S_ot = compute_straightness(traj_ot)

labels = ['扩散耦合\n(条件路径)', '独立耦合CFM\n(边际路径)', 'OT-CFM\n(边际路径)']
values = [kappa_diff, S_ind, S_ot]
colors = ['red', 'purple', 'green']
bars = ax.bar(labels, values, color=colors, alpha=0.7, width=0.5)
ax.set_ylabel(r'曲率 $\kappa$ (0=完全直线)', fontsize=12)
ax.set_title('(d) 路径曲率对比（14.4.2节）', fontsize=12)
ax.set_ylim(0, max(values) * 1.3 + 0.01)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{val:.4f}',
            ha='center', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.suptitle('实验14.3-1：扩散耦合 vs OT耦合路径对比（14.3.5/14.3.6节）', fontsize=14, y=1.01)
plt.tight_layout()
fig_path2 = os.path.join(SAVE_DIR, '步骤2_路径对比.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig_path2}")
print(f"  扩散耦合条件路径曲率: {kappa_diff:.4f}")
print(f"  独立耦合CFM边际路径曲率: {S_ind:.4f}")
print(f"  OT-CFM边际路径曲率:      {S_ot:.4f}")
print(f"  → 扩散耦合路径弯曲(κ大)，OT-CFM路径最直(κ小)")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print("实验14.3-1 完成!")
print("=" * 60)
print("""
关键结论:
1. CFM定理（14.3.3节）
   - 边际向量场不可直接计算，但条件向量场可计算
   - CFM损失与FM损失的梯度相同（类比第6章DSM定理）
   - 训练目标：||v_θ(x_t, t) - (x_0 - z)||²

2. 高斯条件路径（14.3.4节）
   - 线性插值：x_t = (1-t)z + t*x_0
   - 条件向量场：v_t = x_0 - z（常数向量场）
   - 简单且高效的训练框架

3. 路径形态对比（14.3.5/14.3.6节）
   - 扩散耦合：路径弯曲，需要多步采样（~1000步）
   - 独立耦合CFM：路径弯曲交叉，需多步ODE求解
   - OT-CFM：路径最直，少步即可采样（~10步）

4. 实践意义
   - OT耦合能够获得更直的传输路径
   - 直线路径显著减少ODE求解步数
   - OT-CFM是最优传输与Flow Matching的完美结合
""")