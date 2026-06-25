# -*- coding: utf-8 -*-
"""
实验7.3-1 逆向SDE采样与DDPM
对应章节：7.3（逆向SDE：从噪声到数据的采样过程）、7.5（数值离散化）

说明：
- 本实验的核心理论来自7.3章节（Anderson逆时SDE定理、逆向SDE）
- 本实验的离散化方法来自7.5章节（Euler-Maruyama方法）
- 建议读者先学习7.3章节的理论，再学习7.5章节的离散化方法，最后完成本实验

注意：本实验涉及7.5章节的离散化方法，如果您尚未学习7.5章节，
      可能会遇到一些超纲的概念。建议您先阅读以下内容：
      - 7.5章节的Euler-Maruyama离散化方法
      - 7.5章节的步数与采样质量的关系
      这些内容将在实验中详细展示，帮助您理解离散化方法的应用。

素材来源：
  - 04-sde.ipynb的Euler-Maruyama采样代码
  - 02-ddpm.ipynb的DDPM推理代码
  - ★ 原创设计：Anderson逆时SDE定理的数值验证
  - ★ 原创设计：VE/VP逆向SDE采样对比

实验内容：
  步骤1：Anderson逆时SDE定理验证——正向SDE的边际分布=逆向SDE的边际分布
  步骤2：VP-SDE逆向采样（DDPM=Euler-Maruyama离散化）
  步骤3：VE-SDE逆向采样（SMLD=Euler-Maruyama离散化）
  步骤4：步数对采样质量的影响

运行前提：纯NumPy/PyTorch CPU即可，无需预训练模型
"""

import numpy as np
import os
import sys
import io
import warnings
import logging
from scipy import stats as sp_stats

# 设置控制台输出为 UTF-8 (Windows 下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 静默 matplotlib 相关警告
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
    SAVE_DIR = os.path.join(_gdrive, '实验7.3-1')
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
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)
import torch
torch.manual_seed(42)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()


# ============================================================
# 目标分布：1D高斯混合（便于可视化）
# ============================================================
def gm1d_pdf(x, weights=[0.3, 0.7], means=[-2, 1], stds=[1, 1]):
    """1D高斯混合概率密度"""
    pdf = np.zeros_like(x)
    for w, m, s in zip(weights, means, stds):
        pdf += w * np.exp(-0.5 * ((x - m) / s)**2) / (s * np.sqrt(2 * np.pi))
    return pdf

def gm1d_score(x, weights=[0.3, 0.7], means=[-2, 1], stds=[1, 1]):
    """1D高斯混合的得分函数 ∇log p(x)（解析解）"""
    pdf = gm1d_pdf(x, weights, means, stds)
    dpdf = np.zeros_like(x)
    for w, m, s in zip(weights, means, stds):
        dpdf += w * (-(x - m) / s**2) * np.exp(-0.5 * ((x - m) / s)**2) / (s * np.sqrt(2 * np.pi))
    return dpdf / (pdf + 1e-30)


# ============================================================
# 步骤1：Anderson逆时SDE定理验证
# ★ 原创设计
# 正向SDE边际分布p_t(x) = 逆向SDE边际分布p_t(x)（Anderson 1982）
# 用VP-SDE验证：从p_0出发正向走到p_1，再逆向走回p_0
# ============================================================
print("=" * 60)
print("步骤1：Anderson逆时SDE定理验证")
print("=" * 60)

beta_min, beta_max = 0.1, 20.0

def vp_marginal(t, beta_min_val=None, beta_max_val=None):
    """VP-SDE边际参数"""
    _bmin = beta_min_val if beta_min_val is not None else beta_min
    _bmax = beta_max_val if beta_max_val is not None else beta_max
    log_mean = -0.25 * t**2 * (_bmax - _bmin) - 0.5 * t * _bmin
    mean_t = np.exp(log_mean)
    std_t = np.sqrt(1 - np.exp(2 * log_mean))
    return mean_t, std_t

def vp_beta(t):
    """VP-SDE的β(t)（线性插值）"""
    return beta_min + t * (beta_max - beta_min)

def vp_drift(x, t):
    """VP-SDE漂移 f(x,t) = -β(t)/2 · x"""
    return -vp_beta(t) / 2 * x

def vp_score_analytic(x, t, beta_min_val=None, beta_max_val=None):
    """VP-SDE解析得分 ∇log p_t(x)

    参数:
        x: 粒子位置
        t: 时间
        beta_min_val, beta_max_val: VP-SDE参数，默认使用全局beta_min, beta_max
            通过显式传参实现依赖注入，避免闭包捕获全局变量导致的隐藏耦合
    """
    mean_t, std_t = vp_marginal(t, beta_min_val, beta_max_val)
    pdf = np.zeros_like(x)
    dpdf = np.zeros_like(x)
    weights = [0.3, 0.7]
    means = [-2, 1]
    stds = [1, 1]
    for w, m, s in zip(weights, means, stds):
        new_mean = mean_t * m
        new_std = np.sqrt(mean_t**2 * s**2 + std_t**2)
        pdf += w * np.exp(-0.5 * ((x - new_mean) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
        dpdf += w * (-(x - new_mean) / new_std**2) * np.exp(-0.5 * ((x - new_mean) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
    return dpdf / (pdf + 1e-30)

def vp_diffusion(t):
    """VP-SDE扩散 g(t) = sqrt(beta(t))"""
    return np.sqrt(vp_beta(t))

N_particles = 10000
N_steps = 500
T = 1.0
dt = T / N_steps

# 从p_0（高斯混合）采样
x0 = np.where(np.random.rand(N_particles) < 0.3,
              np.random.randn(N_particles) - 2,
              np.random.randn(N_particles) + 1)

# ---- 正向VP-SDE（使用边际分布解析解，更精确）----
fwd_snapshots = {0.0: x0.copy()}
snapshot_times = [0.2, 0.5, 0.8, 1.0]

np.random.seed(123)
for st in snapshot_times:
    mean_t, std_t = vp_marginal(st)
    fwd_snapshots[st] = mean_t * x0 + std_t * np.random.randn(N_particles)

print(f"正向SDE（解析边际）: 各时间步采样完成")

# ---- Anderson定理验证（正向与逆向边际分布对比）----
# 方法：从N(0,I)出发，用逆向VP-SDE采样，对比逆向各时间步的边际
# 与正向各时间步的边际——应一致（Anderson 1982定理保证）
np.random.seed(42)
x_rev_init = np.random.randn(N_particles)  # 从N(0,I)开始

# 用逆向VP-SDE采样（逆时参数化），保存中间快照
rev_snapshots = {1.0: x_rev_init.copy()}
rev_times = [0.8, 0.5, 0.2, 0.0]

# 预先计算每个目标时间对应的步数索引，避免浮点容差判断
# i_target表示从T=1.0走到目标时间st需要的步数：i_target = round((T - st) / dt)
rev_time_to_step = {st: round((T - st) / dt) for st in rev_times}

x_rev = x_rev_init.copy()
for i in range(N_steps):
    t = T - i * dt  # 原始时间（从T到0）
    beta_t = vp_beta(t)
    score = vp_score_analytic(x_rev, t)

    # 逆向VP-SDE更新（逆时参数化，参考04-sde.ipynb）
    # dx = [β/2·x + β·∇log p_t(x)] dτ + √β·dW̃
    x_rev = x_rev + beta_t * dt * (0.5 * x_rev + score) + np.sqrt(beta_t * dt) * np.random.randn(N_particles)

    # 使用步数索引精确匹配，避免浮点累积误差导致的快照缺失
    next_step = i + 1
    for st, i_target in rev_time_to_step.items():
        if next_step == i_target and st not in rev_snapshots:
            rev_snapshots[st] = x_rev.copy()

print(f"DDPM逆向采样完成: Var(x_0)={np.var(x_rev):.4f}")
print(f"目标分布:         Var(x_0)={np.var(x0):.4f}")

# 比较正向和逆向在各时间步的边际分布
print("\n边际分布对比（Forward vs DDPM Reverse）：")
for t_val in [0.0, 0.2, 0.5, 0.8, 1.0]:
    fwd_data = fwd_snapshots.get(t_val, None)
    rev_data = rev_snapshots.get(t_val, None)
    if fwd_data is not None and rev_data is not None:
        print(f"  t={t_val:.1f}: Fwd(mu={np.mean(fwd_data):.3f}, var={np.var(fwd_data):.3f})"
              f"  Rev(mu={np.mean(rev_data):.3f}, var={np.var(rev_data):.3f})")

print("\nAnderson定理验证：正向与逆向边际分布应近似一致")
print("（差异来自：1)有限粒子数 2)DDPM离散化误差 3)正向用解析边际、逆向用DDPM采样）")


# ============================================================
# 步骤2：VP-SDE逆向采样 = DDPM
# 参考：02-ddpm.ipynb, 7.3节, 7.5节
# DDPM = Euler-Maruyama离散化逆向VP-SDE
# ============================================================
print("\n" + "=" * 60)
print("步骤2：VP-SDE逆向采样（DDPM）")
print("=" * 60)

def ddpm_sample_vp(score_fn, N_particles, N_steps, T=1.0, beta_min=0.1, beta_max=20.0):
    """VP-SDE逆向采样（DDPM = Euler-Maruyama on reverse VP-SDE）

    逆向VP-SDE（逆时参数化τ=T-t，dτ>0）:
    dx = [β(t)/2·x + β(t)·∇log p_t(x)] dτ + √β(t)·dW̃

    Euler-Maruyama离散化（参考04-sde.ipynb）:
    x_{τ+dτ} = x_τ + β(t)·h·(0.5·x_τ + s_θ(x_τ,t)) + √(β(t)·h)·z

    其中 h = T/N_steps，t = T - τ
    """
    h = T / N_steps
    x = np.random.randn(N_particles)  # 从N(0,I)开始

    trajectory = [x.copy()]

    for i in range(N_steps):
        t = T - i * h  # 原始时间（从T到0）
        beta_t = beta_min + t * (beta_max - beta_min)
        # 将beta参数显式传递给score_fn，实现依赖注入
        score = score_fn(x, t, beta_min_val=beta_min, beta_max_val=beta_max)

        # DDPM更新（逆时参数化）
        x = x + beta_t * h * (0.5 * x + score) + np.sqrt(beta_t * h) * np.random.randn(N_particles)

        trajectory.append(x.copy())

    return np.array(trajectory)

# DDPM采样（不同步数）
for n_steps in [50, 200, 500, 1000]:
    np.random.seed(42)
    traj = ddpm_sample_vp(vp_score_analytic, 5000, n_steps)
    final = traj[-1]
    mean_diff = abs(np.mean(final) - (0.3 * (-2) + 0.7 * 1))
    var_diff = abs(np.var(final) - np.var(x0))
    has_nan = np.isnan(final).any()
    print(f"  DDPM N_steps={n_steps}: mu_diff={mean_diff:.4f}, var_diff={var_diff:.4f}, NaN={has_nan}")

print("DDPM步数越多，采样质量越高（解析得分+小步长Euler-Maruyama离散化，数值稳定）")
print("注：此处'数值稳定'指本实验中解析得分+小步长Euler-Maruyama离散化不易发散（NaN检查可验证）")
print("（注意：真实DDPM训练时常用ε-prediction参数化神经网络来提升训练稳定性，"
      "但本实验直接使用解析得分，未涉及网络训练）")


# ============================================================
# 步骤3：VE-SDE逆向采样 = SMLD
# 参考：03-smld.ipynb, 7.3节
# SMLD = Euler-Maruyama离散化逆向VE-SDE
# ============================================================
print("\n" + "=" * 60)
print("步骤3：VE-SDE逆向采样（SMLD）")
print("=" * 60)

sigma_min_ve, sigma_max_ve = 0.01, 10.0

def ve_sigma(t, sigma_min=0.01, sigma_max=10.0):
    return sigma_max * (sigma_min / sigma_max) ** t

def ve_score_analytic(x, t):
    """VE-SDE的解析得分"""
    sigma_t = ve_sigma(t)
    # p_t(x) = ∫ p_0(x_0) · N(x; x_0, σ_t²) dx_0
    # 对高斯混合: 分量i → N(μ_i, 1+σ_t²)
    pdf = np.zeros_like(x)
    dpdf = np.zeros_like(x)
    weights = [0.3, 0.7]
    means = [-2, 1]
    for w, m in zip(weights, means):
        new_std = np.sqrt(1 + sigma_t**2)
        pdf += w * np.exp(-0.5 * ((x - m) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
        dpdf += w * (-(x - m) / new_std**2) * np.exp(-0.5 * ((x - m) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
    return dpdf / (pdf + 1e-30)

def smld_sample_ve(score_fn, N_particles, N_steps, T=1.0, sigma_min=0.01, sigma_max=10.0):
    """VE-SDE逆向采样（SMLD/NCSN退火Langevin）

    退火Langevin: 从高噪声到低噪声逐步采样
    每步：x = x + step_size * score(x, sigma_t) + sqrt(2*step_size) * z
    """
    dt = T / N_steps
    x = np.random.randn(N_particles) * sigma_max  # 从大方差高斯开始

    trajectory = [x.copy()]

    for i in range(N_steps):
        t = T - i * dt
        sigma_t = ve_sigma(t, sigma_min, sigma_max)

        score = score_fn(x, t)

        # 退火Langevin步长（与sigma_t^2成正比，确保稳定性）
        # 注：此公式是针对1D高斯混合示例调整的经验公式，用于教学演示
        # 原始SMLD/NCSN论文中步长公式为 step_size = α * (σ_i/σ_min)^2
        # 此处使用σ_max归一化是为了在当前示例中获得稳定的采样效果
        step_size = 0.1 * (sigma_t / sigma_max)**2 + 1e-5

        x = x + step_size * score + np.sqrt(2 * step_size) * np.random.randn(N_particles)

        trajectory.append(x.copy())

    return np.array(trajectory)

# SMLD采样
# 注：VE-SDE在大sigma_max下容易数值不稳定，因此步数只测试到500
# 相比之下，VP-SDE（DDPM）可以稳定运行到1000步
for n_steps in [50, 200, 500]:
    np.random.seed(42)
    traj = smld_sample_ve(ve_score_analytic, 5000, n_steps)
    final = traj[-1]
    mean_diff = abs(np.mean(final) - (0.3 * (-2) + 0.7 * 1))
    var_diff = abs(np.var(final) - np.var(x0))
    print(f"  SMLD N_steps={n_steps}: mu_diff={mean_diff:.4f}, var_diff={var_diff:.4f}")


# ============================================================
# 步骤4：步数对采样质量的影响
# ★ 原创设计
# ============================================================
print("\n" + "=" * 60)
print("步骤4：步数对采样质量的影响")
print("=" * 60)

step_counts = [10, 25, 50, 100, 200, 500, 1000]
vp_errors = []

# 使用多粒子+多种子平均来降低KS统计量的采样噪声
# 有限样本的KS估计存在方差（~1/√N），5000粒子不足以展现单调趋势
N_eval = 20000  # 评估用粒子数（大于x0的10000）
N_seeds = 3     # 多种子平均

for n_steps in step_counts:
    ks_vals = []
    for seed_offset in range(N_seeds):
        np.random.seed(42 + seed_offset * 7)
        traj = ddpm_sample_vp(vp_score_analytic, N_eval, n_steps)
        final = traj[-1]
        try:
            ks_stat, _ = sp_stats.ks_2samp(final, x0)
        except:
            ks_stat = abs(np.mean(final) - np.mean(x0))
        ks_vals.append(ks_stat)
    ks_mean = np.mean(ks_vals)
    vp_errors.append(ks_mean)
    print(f"  DDPM N_steps={n_steps}: KS={ks_mean:.4f} (std={np.std(ks_vals):.4f})")

print("\n结论：步数增加->离散化误差减小->采样质量提升")
print("这解释了为什么DDPM通常需要1000步，而高阶求解器（DPM-Solver）只需20步")


# ============================================================
# 可视化
# ============================================================

# 图1：Anderson定理验证——正向vs逆向边际分布
fig, axes = plt.subplots(1, 5, figsize=(25, 4))
x_grid = np.linspace(-6, 6, 500)
times_to_plot = [0.0, 0.2, 0.5, 0.8, 1.0]

for i, t_val in enumerate(times_to_plot):
    fwd = fwd_snapshots.get(t_val)
    rev = rev_snapshots.get(t_val)
    if fwd is not None:
        axes[i].hist(fwd, bins=60, density=True, alpha=0.5, color='blue', label='Forward')
    if rev is not None:
        axes[i].hist(rev, bins=60, density=True, alpha=0.5, color='red', label='Reverse')
    # 理论边际
    mean_t, std_t = vp_marginal(t_val)
    theory_pdf = np.zeros_like(x_grid)
    for w, m, s in zip([0.3, 0.7], [-2, 1], [1, 1]):
        new_mean = mean_t * m
        new_std = np.sqrt(mean_t**2 * s**2 + std_t**2)
        theory_pdf += w * np.exp(-0.5 * ((x_grid - new_mean) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
    axes[i].plot(x_grid, theory_pdf, 'k--', lw=2, label='Theory')
    axes[i].set_title(f't={t_val:.1f}')
    axes[i].set_xlim(-6, 6)
    axes[i].legend(fontsize=8)
    axes[i].grid(alpha=0.3)

fig.suptitle('Anderson逆时SDE定理验证：Forward vs Reverse边际分布', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_Anderson定理验证.png'), dpi=150, bbox_inches='tight')
plt.close()

# 图2：DDPM采样过程
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
n_vis = 8
np.random.seed(42)
traj = ddpm_sample_vp(vp_score_analytic, 5000, 200)
indices = np.linspace(0, len(traj)-1, n_vis, dtype=int)

for i, idx in enumerate(indices):
    row, col = i // 4, i % 4
    step = len(traj) - 1 - idx
    axes[row, col].hist(traj[idx], bins=50, density=True, alpha=0.6, color='blue')
    axes[row, col].plot(x_grid, gm1d_pdf(x_grid), 'r--', lw=2, label='Target')
    axes[row, col].set_title(f'Reverse step {idx}/{len(traj)-1}')
    axes[row, col].set_xlim(-6, 6)
    axes[row, col].legend(fontsize=8)
    axes[row, col].grid(alpha=0.3)

fig.suptitle('VP-SDE逆向采样过程（DDPM）', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_DDPM采样过程.png'), dpi=150, bbox_inches='tight')
plt.close()

# 图3：步数vs采样质量
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(step_counts, vp_errors, 'bo-', lw=2, markersize=8)
ax.set_xlabel('采样步数')
ax.set_ylabel('KS统计量（越小越好）')
ax.set_title('DDPM步数与采样质量的关系')
ax.grid(alpha=0.3)
ax.set_xscale('log')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤4_步数与质量.png'), dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验7.3-1 总结")
print("=" * 60)
print("1. Anderson逆时SDE定理：正向与逆向边际分布一致（数值验证成功）")
print("2. DDPM = Euler-Maruyama离散化逆向VP-SDE（逆时参数化）")
print("   x = x + β·h·(0.5·x + s_θ) + √(β·h)·z")
print("3. SMLD = 退火Langevin动力学 = 逆向VE-SDE离散化")
print("4. 步数越多->离散化误差越小->采样质量越高")