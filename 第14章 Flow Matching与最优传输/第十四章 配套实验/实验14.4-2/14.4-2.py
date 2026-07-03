# -*- coding: utf-8 -*-
"""
实验14.4-2 Reflow迭代变直：轨迹逐步变直
对应知识点：
  - 14.4节 Rectified Flow（直线插值、轨迹交叉、Reflow）
  - 14.4.1节 Rectified Flow训练
  - 14.4.2节 轨迹交叉问题
  - 14.4.3节 Reflow迭代变直

本实验不需要GPU，通过2D点云的可视化直观理解Reflow的核心概念。

素材来源：
  - book_plan.md的实验14.1计划
  - 14.4节的理论内容
  - ★ 原创设计：Reflow逐轮变直的轨迹可视化

实验内容：
  Rectified Flow与Reflow迭代变直（14.4节）
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
    SAVE_DIR = os.path.join(_gdrive, '实验14.4-2')
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
CHECKPOINT_PATH_1RF = os.path.join(SAVE_DIR, 'model_1rf_checkpoint.pth')
CHECKPOINT_PATH_2RF = os.path.join(SAVE_DIR, 'model_2rf_checkpoint.pth')
CHECKPOINT_PATH_3RF = os.path.join(SAVE_DIR, 'model_3rf_checkpoint.pth')
CHECKPOINT_PATH_OT = os.path.join(SAVE_DIR, 'model_ot_checkpoint.pth')
FINAL_CHECKPOINT_PATH_1RF = os.path.join(SAVE_DIR, 'model_1rf_final.pth')
FINAL_CHECKPOINT_PATH_2RF = os.path.join(SAVE_DIR, 'model_2rf_final.pth')
FINAL_CHECKPOINT_PATH_3RF = os.path.join(SAVE_DIR, 'model_3rf_final.pth')
FINAL_CHECKPOINT_PATH_OT = os.path.join(SAVE_DIR, 'model_ot_final.pth')


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
    # 代价矩阵: C[i,j] = ||source[i] - target[j||^2
    diff = source[:, None, :] - target[None, :, :]  # (n, n, 2)
    cost = np.sum(diff**2, axis=-1)  # (n, n)
    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind  # 返回source[i]应该配对的target索引


# ============================================================
# 2D向量场网络
# ============================================================
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


# ============================================================
# CFM训练函数（带Resume能力）
# ============================================================
def train_cfm(n_epochs=2000, n_samples=256, coupling='independent', lr=1e-3,
              checkpoint_path=None, final_checkpoint_path=None):
    """训练Conditional Flow Matching（带checkpoint Resume能力）

    coupling: 'independent' 或 'ot'
    """
    model = VectorFieldNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    start_epoch = 0
    skip_training = False

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


# ============================================================
# 曲率计算函数
# ============================================================
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


# ============================================================
# Rectified Flow与Reflow（14.4节）
# ============================================================
print("=" * 60)
print("实验14.4-2：Rectified Flow与Reflow（14.4节）")
print("=" * 60)

print("""
14.4.1节：Rectified Flow使用直线插值 x_t = (1-t)z + t*x_0
  训练目标: ||v_θ(x_t, t) - (x_0 - z)||²

14.4.2节：轨迹交叉导致边际路径弯曲
  - 条件路径是直的，但多条直线路径交叉→边际向量场需要"折中"→弯曲

14.4.3节：Reflow迭代变直
  1-RF: 用独立耦合训练
  2-RF: 用1-RF的ODE端点重新配对，再训练
  3-RF: 用2-RF的ODE端点重新配对，再训练
  ...
  Reflow极限→OT映射！

★ 原创设计：可视化1-RF→2-RF→3-RF的轨迹逐步变直
""")

# 训练初始模型（1-RF）
print("\n训练 1-Rectified Flow (独立耦合)...")
model_1rf, _ = train_cfm(n_epochs=2000, coupling='independent',
                         checkpoint_path=CHECKPOINT_PATH_1RF,
                         final_checkpoint_path=FINAL_CHECKPOINT_PATH_1RF)

# 训练OT-CFM作为对比基线
print("\n训练 OT-CFM (OT耦合)...")
model_ot, _ = train_cfm(n_epochs=2000, coupling='ot',
                        checkpoint_path=CHECKPOINT_PATH_OT,
                        final_checkpoint_path=FINAL_CHECKPOINT_PATH_OT)


# ============================================================
# Reflow过程（带Resume能力）
# ============================================================
def reflow_step(model, reflow_round, n_samples=256, n_epochs=1500, lr=1e-3,
                checkpoint_path=None, final_checkpoint_path=None):
    """执行一步Reflow：用当前模型的ODE端点重新配对（带checkpoint Resume能力）"""
    
    # 检查是否有最终权重
    if final_checkpoint_path and os.path.exists(final_checkpoint_path):
        print(f"检测到最终权重: {final_checkpoint_path}")
        print("直接加载，跳过Reflow训练过程")
        checkpoint = torch.load(final_checkpoint_path, map_location='cpu', weights_only=False)
        new_model = VectorFieldNet()
        new_model.load_state_dict(checkpoint['model_state_dict'])
        return new_model

    # 检查是否有中间权重
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"检测到中间权重: {checkpoint_path}")
        print("继续Reflow训练...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        new_model = VectorFieldNet()
        new_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(new_model.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        # 需要重新生成端点对
        model.eval()
        with torch.no_grad():
            z = torch.FloatTensor(sample_source(n_samples))
            x_endpoint = z.clone()
            dt = 1.0 / 50
            for step in range(50):
                t_val = step / 50
                t = torch.full((x_endpoint.shape[0], 1), t_val)
                v = model(x_endpoint, t)
                x_endpoint = x_endpoint + v * dt
    else:
        # 用当前模型生成端点对
        model.eval()
        with torch.no_grad():
            z = torch.FloatTensor(sample_source(n_samples))
            # 运行ODE得到终点
            x_endpoint = z.clone()
            dt = 1.0 / 50
            for step in range(50):
                t_val = step / 50
                t = torch.full((x_endpoint.shape[0], 1), t_val)
                v = model(x_endpoint, t)
                x_endpoint = x_endpoint + v * dt

        # 新的配对：(z, x_endpoint)作为训练数据
        # 重新训练模型
        new_model = VectorFieldNet()
        optimizer = torch.optim.Adam(new_model.parameters(), lr=lr)

    print(f"Reflow第{reflow_round}轮训练，从 epoch {start_epoch} 开始...")

    for epoch in range(start_epoch, n_epochs):
        # 从固定配对中采样mini-batch
        idx = np.random.choice(n_samples, min(256, n_samples), replace=False)
        z_batch = z[idx]
        x0_batch = x_endpoint.detach()[idx]

        t = torch.rand(len(idx), 1)
        x_t = (1 - t) * z_batch + t * x0_batch
        v_target = x0_batch - z_batch

        v_pred = new_model(x_t, t)
        loss = nn.functional.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"  [Reflow-{reflow_round}] Epoch {epoch+1}/{n_epochs} Loss={loss.item():.6f}")

        # 每200轮保存中间checkpoint
        if checkpoint_path and (epoch + 1) % 200 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': new_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }, checkpoint_path)

    # 保存最终权重
    if final_checkpoint_path:
        torch.save({
            'model_state_dict': new_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, final_checkpoint_path)
        print(f"最终权重已保存: {final_checkpoint_path}")

    return new_model

# 2-RF: Reflow一步
print("\n训练 2-Rectified Flow (1次Reflow)...")
model_2rf = reflow_step(model_1rf, reflow_round=1, n_samples=512, n_epochs=2000,
                        checkpoint_path=CHECKPOINT_PATH_2RF,
                        final_checkpoint_path=FINAL_CHECKPOINT_PATH_2RF)

# 3-RF: Reflow两步
print("\n训练 3-Rectified Flow (2次Reflow)...")
model_3rf = reflow_step(model_2rf, reflow_round=2, n_samples=512, n_epochs=2000,
                        checkpoint_path=CHECKPOINT_PATH_3RF,
                        final_checkpoint_path=FINAL_CHECKPOINT_PATH_3RF)


# ============================================================
# 对比Reflow效果
# ============================================================
from scipy.stats import wasserstein_distance_nd

n_test = 200
np.random.seed(42)
z_test_rf = torch.FloatTensor(sample_source(n_test))
target_test = sample_target(n_test)

traj_1rf = flow_sample(model_1rf, z_test_rf, n_steps=100)
traj_2rf = flow_sample(model_2rf, z_test_rf, n_steps=100)
traj_3rf = flow_sample(model_3rf, z_test_rf, n_steps=100)
traj_ot = flow_sample(model_ot, z_test_rf, n_steps=100)

S_1rf = compute_straightness(traj_1rf)
S_2rf = compute_straightness(traj_2rf)
S_3rf = compute_straightness(traj_3rf)
S_ot = compute_straightness(traj_ot)

wd_1rf = wasserstein_distance_nd(traj_1rf[-1], target_test)
wd_2rf = wasserstein_distance_nd(traj_2rf[-1], target_test)
wd_3rf = wasserstein_distance_nd(traj_3rf[-1], target_test)
wd_ot = wasserstein_distance_nd(traj_ot[-1], target_test)

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

n_show = 10
for col, (traj, label, s_val, wd_val) in enumerate([
    (traj_1rf, '1-RF (独立耦合)', S_1rf, wd_1rf),
    (traj_2rf, '2-RF (1次Reflow)', S_2rf, wd_2rf),
    (traj_3rf, '3-RF (2次Reflow)', S_3rf, wd_3rf),
]):
    # 上面一行：轨迹
    ax = axes[0, col]
    for i in range(n_show):
        path = traj[:101, i, :]
        ax.plot(path[:, 0], path[:, 1], alpha=0.4, lw=1.2, color=['purple', 'orange', 'green'][col])
    ax.scatter(z_test_rf.numpy()[:n_show, 0], z_test_rf.numpy()[:n_show, 1], c='blue', s=25, zorder=5)
    ax.scatter(traj[-1, :n_show, 0], traj[-1, :n_show, 1], c='red', s=25, zorder=5)
    ax.set_title(f'{label}\n曲率κ={s_val:.3f}', fontsize=12)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

# 下面一行左：直线度柱状图
ax = axes[1, 0]
labels = ['1-RF', '2-RF', '3-RF', 'OT-CFM']
vals = [S_1rf, S_2rf, S_3rf, S_ot]
colors = ['purple', 'orange', 'green', 'steelblue']
bars = ax.bar(labels, vals, color=colors, alpha=0.7)
ax.set_ylabel('曲率 κ (0=直线)', fontsize=12)
ax.set_title('(d) 曲率随Reflow降低', fontsize=12)
ax.set_ylim(0, max(vals) * 1.3 + 0.01)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}',
            ha='center', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# 下面行中：W1距离柱状图
ax = axes[1, 1]
labels = ['1-RF', '2-RF', '3-RF', 'OT-CFM']
vals_wd = [wd_1rf, wd_2rf, wd_3rf, wd_ot]
bars = ax.bar(labels, vals_wd, color=colors, alpha=0.7)
ax.set_ylabel('Wasserstein-1 距离', fontsize=12)
ax.set_title('(e) 采样质量随Reflow提升', fontsize=12)
for bar, val in zip(bars, vals_wd):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
            ha='center', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# 下面行右：少步采样对比
ax = axes[1, 2]
for label, model in [('1-RF', model_1rf), ('3-RF', model_3rf), ('OT-CFM', model_ot)]:
    psnrs = []
    for n_steps in [1, 5, 10, 20, 50, 100]:
        traj = flow_sample(model, z_test_rf, n_steps=n_steps)
        wd = wasserstein_distance_nd(traj[-1], target_test)
        psnrs.append(wd)
    ax.plot([1, 5, 10, 20, 50, 100], psnrs, '-o', markersize=5, label=label)
ax.set_xlabel('ODE求解步数', fontsize=12)
ax.set_ylabel('W1距离', fontsize=12)
ax.set_title('(f) 少步采样质量', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xscale('log')

plt.suptitle('实验14.4-2：Reflow逐步变直（14.4节）', fontsize=14, y=1.01)
plt.tight_layout()
fig_path3 = os.path.join(SAVE_DIR, '步骤3_Reflow.png')
plt.savefig(fig_path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"图3已保存: {fig_path3}")

print(f"\nReflow效果总结:")
print(f"  {'方法':<12s}  {'曲率κ':>8s}  {'W1距离':>8s}")
print(f"  {'-'*32}")
print(f"  {'1-RF':<12s}  {S_1rf:8.4f}  {wd_1rf:8.4f}")
print(f"  {'2-RF':<12s}  {S_2rf:8.4f}  {wd_2rf:8.4f}")
print(f"  {'3-RF':<12s}  {S_3rf:8.4f}  {wd_3rf:8.4f}")
print(f"  {'OT-CFM':<12s}  {S_ot:8.4f}  {wd_ot:8.4f}")
print(f"\n  → Reflow逐步降低曲率（轨迹变直），趋近OT映射")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print("实验14.4-2 完成!")
print("=" * 60)
print("""
关键结论:
1. Rectified Flow训练（14.4.1节）
   - 训练目标: ||v_θ(x_t, t) - (x_0 - z)||²
   - 使用直线插值路径：x_t = (1-t)z + t*x_0
   - 条件向量场是常数：v_t = x_0 - z

2. 轨迹交叉问题（14.4.2节）
   - 条件路径是直的，但多条路径交叉
   - 边际向量场需要"折中"→边际路径弯曲
   - 弯曲路径需要多步ODE求解

3. Reflow迭代变直（14.4.3节）★ 原创设计
   - Reflow迭代地用ODE端点重新配对
   - 直线度逐步提升：1-RF < 2-RF < 3-RF ≤ OT-CFM
   - Reflow极限 = OT映射（理论保证）
   - 实践意义：Reflow可蒸馏多步模型为少步/单步模型

4. 实践意义
   - Reflow提供了一种无需OT求解器的"OT近似"方法
   - 适用于高维数据（图像、视频等）
   - 可用于模型加速：多步→少步→单步
""")