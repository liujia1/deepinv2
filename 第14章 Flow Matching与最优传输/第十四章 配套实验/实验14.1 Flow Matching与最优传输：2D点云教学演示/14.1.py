# -*- coding: utf-8 -*-
"""
实验14.1 Flow Matching与最优传输：2D点云教学演示
对应知识点：
  - 14.1节 最优传输基础（Monge/Kantorovich问题、Wasserstein距离）
  - 14.3节 Flow Matching（CFM定理、高斯条件路径、OT-CFM）
  - 14.4节 Rectified Flow（直线插值、轨迹交叉、Reflow）

本实验不需要GPU，通过2D点云的可视化直观理解Flow Matching的核心概念。

素材来源：
  - book_plan.md的实验14.1计划
  - 14.3/14.4节的理论内容
  - ★ 原创设计：独立耦合 vs OT耦合的路径可视化
  - ★ 原创设计：Reflow逐轮变直的轨迹可视化
  - ★ 原创设计：扩散路径 vs FM路径 vs OT-CFM路径的对比

实验内容：
  步骤1：最优传输 vs 独立耦合——路径形态对比（14.1节/14.3.5节）
  步骤2：CFM训练——学习向量场（14.3.3节/14.3.4节）
  步骤3：扩散路径 vs FM路径 vs OT-CFM路径（14.3.5节/14.3.6节）
  步骤4：Rectified Flow与Reflow（14.4节）
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

SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()


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
    """★ 修正版：扩散路径（14.3.6节，VP-SDE条件路径）
    x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
    使用cosine schedule: ᾱ_t = cos²(π/2 * (1-t))，t∈[0,1]
    z作为噪声源ε
    """
    alpha_bar_t = np.cos(np.pi / 2 * (1 - t)) ** 2
    alpha_bar_t = max(alpha_bar_t, 1e-10)
    return np.sqrt(alpha_bar_t) * x0 + np.sqrt(1 - alpha_bar_t) * z


# ============================================================
# 步骤1：最优传输 vs 独立耦合——路径形态对比（14.1节/14.3.5节）
# ============================================================
print("=" * 60)
print("步骤1：最优传输 vs 独立耦合——路径形态对比（14.1节/14.3.5节）")
print("=" * 60)

print("""
14.1节核心：Monge问题寻找确定性传输映射T，使得T_#p_0 = p_1
14.3.5节核心：独立耦合→弯曲路径，OT耦合→直线路径

McCann插值（14.1.4节）：x_t = (1-t)x_0 + t*T^*(x_0)
  - OT映射T^*下的路径是直线（Wasserstein空间的测地线）
  - 独立配对下的路径是弯曲的（轨迹交叉导致向量场需要"折中"）
""")

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

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# (a) 源和目标分布
ax = axes[0]
ax.scatter(source[:, 0], source[:, 1], c='blue', alpha=0.6, s=30, label='源 p_0')
ax.scatter(target[:, 0], target[:, 1], c='red', alpha=0.6, s=30, label='目标 p_1')
ax.set_title('(a) 源分布 vs 目标分布', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(alpha=0.3)
ax.set_aspect('equal')

# (b) 独立耦合路径
ax = axes[1]
t_vals = np.linspace(0, 1, 20)
for i in range(n_points):
    path = np.array([linear_interp(source[i], target_ind[i], t) for t in t_vals])
    ax.plot(path[:, 0], path[:, 1], 'purple', alpha=0.2, lw=1)
ax.scatter(source[:, 0], source[:, 1], c='blue', alpha=0.6, s=20)
ax.scatter(target_ind[:, 0], target_ind[:, 1], c='red', alpha=0.6, s=20)
ax.set_title('(b) 独立耦合：路径弯曲交叉', fontsize=13)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(alpha=0.3)
ax.set_aspect('equal')

# (c) OT耦合路径
ax = axes[2]
for i in range(n_points):
    path = np.array([linear_interp(source[i], target_ot[i], t) for t in t_vals])
    ax.plot(path[:, 0], path[:, 1], 'green', alpha=0.2, lw=1)
ax.scatter(source[:, 0], source[:, 1], c='blue', alpha=0.6, s=20)
ax.scatter(target_ot[:, 0], target_ot[:, 1], c='red', alpha=0.6, s=20)
ax.set_title('(c) OT耦合：路径短且直', fontsize=13)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(alpha=0.3)
ax.set_aspect('equal')

plt.suptitle('步骤1：独立耦合 vs OT耦合（14.1节/14.3.5节）', fontsize=14, y=1.01)
plt.tight_layout()
fig_path1 = os.path.join(SAVE_DIR, '步骤1_耦合对比.png')
plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"图1已保存: {fig_path1}")

# 计算传输代价
cost_indep = np.mean(np.sum((source - target_ind)**2, axis=1))
cost_ot = np.mean(np.sum((source - target_ot)**2, axis=1))
print(f"  独立耦合平均传输代价: {cost_indep:.4f}")
print(f"  OT耦合平均传输代价:   {cost_ot:.4f}")
print(f"  OT代价 / 独立代价:    {cost_ot/cost_indep:.4f}")
print(f"  → OT耦合传输代价更低（Wasserstein距离更短）")


# ============================================================
# 步骤2：CFM训练——学习向量场（14.3.3节/14.3.4节）
# ============================================================
print(f"\n{'='*60}")
print("步骤2：CFM训练——学习向量场（14.3.3节/14.3.4节）")
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

# 训练CFM
def train_cfm(n_epochs=2000, n_samples=256, coupling='independent', lr=1e-3):
    """训练Conditional Flow Matching
    
    coupling: 'independent' 或 'ot'
    """
    model = VectorFieldNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    for epoch in range(n_epochs):
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
    
    return model, losses

# 训练两个模型
print("\n训练独立耦合CFM...")
model_ind, losses_ind = train_cfm(n_epochs=2000, coupling='independent')
print("\n训练OT耦合CFM (OT-CFM)...")
model_ot, losses_ot = train_cfm(n_epochs=2000, coupling='ot')


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
# 步骤3：扩散路径 vs FM路径 vs OT-CFM路径（14.3.5节/14.3.6节）
# ============================================================
print(f"\n{'='*60}")
print("步骤3：扩散路径 vs FM路径 vs OT-CFM路径（14.3.5节/14.3.6节）")
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

# 可视化三种路径（★ 修正：增加真正的扩散耦合路径）
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

# 只画10条轨迹以避免过于密集
n_show = 10
t_vals = np.linspace(0, 1, 100)

# (a) ★ 扩散耦合条件路径（VP-SDE，14.3.6节）
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
ax.set_ylabel('曲率 κ (0=完全直线)', fontsize=12)
ax.set_title('(d) 路径曲率对比（14.4.2节）', fontsize=12)
ax.set_ylim(0, max(values) * 1.3 + 0.01)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{val:.4f}', 
            ha='center', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.suptitle('步骤3：扩散耦合 vs OT耦合路径对比（14.3.5/14.3.6节）', fontsize=14, y=1.01)
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
# ★ 步骤4：Rectified Flow与Reflow（14.4节）
# ============================================================
print(f"\n{'='*60}")
print("★ 步骤4：Rectified Flow与Reflow（14.4节）")
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

# Reflow过程
def reflow_step(model, n_samples=256, n_epochs=1500, lr=1e-3):
    """执行一步Reflow：用当前模型的ODE端点重新配对"""
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
    
    for epoch in range(n_epochs):
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
    
    return new_model

# 1-RF: 初始训练（独立耦合）
print("  训练 1-Rectified Flow (独立耦合)...")
model_1rf = model_ind  # 已训练好的独立耦合模型

# 2-RF: Reflow一步
print("  训练 2-Rectified Flow (1次Reflow)...")
model_2rf = reflow_step(model_1rf, n_samples=512, n_epochs=2000)

# 3-RF: Reflow两步
print("  训练 3-Rectified Flow (2次Reflow)...")
model_3rf = reflow_step(model_2rf, n_samples=512, n_epochs=2000)

# 对比Reflow效果
np.random.seed(42)
z_test_rf = torch.FloatTensor(sample_source(n_test))

traj_1rf = flow_sample(model_1rf, z_test_rf, n_steps=100)
traj_2rf = flow_sample(model_2rf, z_test_rf, n_steps=100)
traj_3rf = flow_sample(model_3rf, z_test_rf, n_steps=100)

S_1rf = compute_straightness(traj_1rf)
S_2rf = compute_straightness(traj_2rf)
S_3rf = compute_straightness(traj_3rf)

wd_1rf = wasserstein_distance_nd(traj_1rf[-1], target_test)
wd_2rf = wasserstein_distance_nd(traj_2rf[-1], target_test)
wd_3rf = wasserstein_distance_nd(traj_3rf[-1], target_test)

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

plt.suptitle('★ 步骤4：Reflow逐步变直（14.4节）', fontsize=14, y=1.01)
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
print("实验14.1 完成!")
print("=" * 60)
print("""
关键结论:
1. 最优传输 vs 独立耦合（14.1/14.3.5节）
   - 独立耦合：路径弯曲交叉，传输代价高
   - OT耦合：路径短且直，传输代价低（Wasserstein距离）

2. CFM定理（14.3.3节）
   - 边际向量场不可直接计算，但条件向量场可计算
   - CFM损失与FM损失的梯度相同（类比第6章DSM定理）

3. 路径形态对比（14.3.5/14.3.6节）
   - 独立耦合CFM：路径弯曲，需多步ODE求解
   - OT-CFM：路径更直，少步即可采样

4. Rectified Flow与Reflow（14.4节）★ 原创设计
   - Reflow迭代地用ODE端点重新配对
   - 直线度逐步提升：1-RF < 2-RF < 3-RF ≤ OT-CFM
   - Reflow极限 = OT映射（14.4.3节理论保证）
   - 实践意义：Reflow可蒸馏多步模型为少步/单步模型
""")
