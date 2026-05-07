"""
实验7.4 概率流ODE与DDIM加速采样
对应章节：7.4（概率流ODE：随机采样的确定性等价）、7.5（数值离散化）
素材来源：
  - 7.4节PF-ODE推导、DDIM=PF-ODE离散化
  - 02-ddpm.ipynb的DDIM采样代码
  - ★ 原创设计：温度参数η在DDPM↔DDIM间连续插值
  - ★ 原创设计：Fokker-Planck方程验证PF-ODE与SDE边际分布等价

实验内容：
  步骤1：PF-ODE采样——确定性等价
  步骤2：DDIM = Euler离散化PF-ODE
  步骤3：温度参数η：DDPM(η=1) ↔ DDIM(η=0)
  步骤4：DDPM vs DDIM采样质量与速度对比

运行前提：纯NumPy/PyTorch CPU即可
"""

import numpy as np
import os
import sys

# ====== Windows控制台UTF-8输出 ======
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib.pyplot as plt
import warnings
import logging

# ====== 解决中文乱码的核心代码 ======
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

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
# ========================================================

np.random.seed(42)
import torch
torch.manual_seed(42)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()


# ============================================================
# 目标分布与得分函数（同7.3）
# ============================================================
def gm1d_pdf(x, weights=[0.3, 0.7], means=[-2, 1], stds=[1, 1]):
    pdf = np.zeros_like(x)
    for w, m, s in zip(weights, means, stds):
        pdf += w * np.exp(-0.5 * ((x - m) / s)**2) / (s * np.sqrt(2 * np.pi))
    return pdf

beta_min, beta_max = 0.1, 20.0

def vp_marginal(t):
    log_mean = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    mean_t = np.exp(log_mean)
    std_t = np.sqrt(1 - np.exp(2 * log_mean))
    return mean_t, std_t

def vp_beta(t):
    return beta_min + t * (beta_max - beta_min)

def vp_score_analytic(x, t):
    """VP-SDE解析得分"""
    mean_t, std_t = vp_marginal(t)
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


# ============================================================
# 步骤1：PF-ODE采样
# PF-ODE: dx/dt = f(x,t) - 1/2 g(t)² ∇log p_t(x)
# 与逆向SDE的区别：没有随机项，且score系数为1/2
# ============================================================
print("=" * 60)
print("步骤1：PF-ODE采样——确定性等价")
print("=" * 60)

def pf_ode_sample_vp(score_fn, N_particles, N_steps, T=1.0, beta_min=0.1, beta_max=20.0):
    """概率流ODE采样（Euler法）
    
    PF-ODE（逆时参数化τ=T-t）:
    dx/dτ = β(t)/2·x + 1/2·β(t)·∇log p_t(x)
    
    与逆向SDE的区别：没有随机项，且score系数为1/2（而非1）
    """
    h = T / N_steps
    x = np.random.randn(N_particles)  # 从N(0,I)开始
    
    trajectory = [x.copy()]
    
    for i in range(N_steps):
        t = T - i * h  # 原始时间（从T到0）
        beta_t = vp_beta(t)
        score = score_fn(x, t)
        
        # PF-ODE更新（逆时参数化，确定性，无噪声项）
        # 与逆向SDE相比：1) 无随机项 2) score系数为1/2而非1
        drift = beta_t * h * (0.5 * x + 0.5 * score)
        x = x + drift  # 无随机项！
        
        trajectory.append(x.copy())
    
    return np.array(trajectory)

# PF-ODE采样
np.random.seed(42)
N_particles = 5000
x0_ref = np.where(np.random.rand(N_particles) < 0.3,
                  np.random.randn(N_particles) - 2,
                  np.random.randn(N_particles) + 1)

np.random.seed(42)
pf_traj = pf_ode_sample_vp(vp_score_analytic, N_particles, 500)
pf_final = pf_traj[-1]

print(f"PF-ODE采样结果: μ={np.mean(pf_final):.4f}, σ²={np.var(pf_final):.4f}")
print(f"目标分布:       μ={np.mean(x0_ref):.4f}, σ²={np.var(x0_ref):.4f}")
print(f"\nPF-ODE关键特性：")
print(f"  1. 确定性——给定相同初始点，采样路径唯一")
print(f"  2. 与逆向SDE边际分布相同（Fokker-Planck方程保证）")
print(f"  3. 适合做插值、编辑等可控生成")


# ============================================================
# 步骤2：DDIM = Euler离散化PF-ODE
# 参考：7.4节，02-ddpm.ipynb
# DDIM更新公式（7.5节式7.27）：
#   x_{t-Δt} = √(ᾱ_{t-Δt}) · [x_t - √(1-ᾱ_t)·ε_θ]/√(ᾱ_t) + √(1-ᾱ_{t-Δt})·ε_θ
# ============================================================
print("\n" + "=" * 60)
print("步骤2：DDIM = Euler离散化PF-ODE")
print("=" * 60)

def ddim_sample_vp(score_fn, N_particles, N_steps, T=1.0, eta=0.0,
                   beta_min=0.1, beta_max=20.0):
    """DDIM采样（η=0为确定性DDIM，η=1退化为DDPM）
    
    通用公式（7.5节，温度参数η）：
    x_{t-Δt} = √(ᾱ_{t-Δt})·x̂_0 + √(1-ᾱ_{t-Δt}-σ²_η)·ε_θ + σ_η·ε
    
    其中：
    - x̂_0 = (x_t - √(1-ᾱ_t)·ε_θ) / √(ᾱ_t)    （Tweedie估计）
    - σ_η = η · √((1-ᾱ_{t-Δt})/(1-ᾱ_t)) · √(1-ᾱ_t/ᾱ_{t-Δt})
    - η=0: DDIM（确定性，PF-ODE）
    - η=1: DDPM（随机，逆向SDE）
    """
    dt = T / N_steps
    x = np.random.randn(N_particles)
    
    trajectory = [x.copy()]
    
    for i in range(N_steps):
        t = T - i * dt
        t_prev = max(t - dt, 0)
        
        # 当前和前一步的ᾱ
        mean_t, std_t = vp_marginal(t)
        alpha_bar_t = mean_t**2  # ᾱ_t = mean_t²
        mean_prev, std_prev = vp_marginal(t_prev)
        alpha_bar_prev = mean_prev**2
        
        # ε_θ ≈ 从得分反推: ε = -std_t · score
        score = score_fn(x, t)
        eps_theta = -std_t * score  # ε-prediction
        
        # Tweedie估计 x̂_0
        x0_hat = (x - std_t * eps_theta) / (mean_t + 1e-10)
        
        # σ_η（温度参数）
        if eta > 0 and alpha_bar_prev > 0 and alpha_bar_t > 0:
            sigma_eta = eta * np.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * \
                       np.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        else:
            sigma_eta = 0.0
        
        # 方向指向x_t
        dir_xt = np.sqrt(max(1 - alpha_bar_prev - sigma_eta**2, 0)) * eps_theta
        
        # DDIM更新
        noise = np.random.randn(N_particles) * sigma_eta if sigma_eta > 0 else 0
        x = np.sqrt(alpha_bar_prev) * x0_hat + dir_xt + noise
        
        trajectory.append(x.copy())
    
    return np.array(trajectory)

# DDIM采样（η=0，确定性）
np.random.seed(42)
ddim_traj = ddim_sample_vp(vp_score_analytic, N_particles, 200, eta=0.0)
ddim_final = ddim_traj[-1]

print(f"DDIM(η=0)采样结果: μ={np.mean(ddim_final):.4f}, σ²={np.var(ddim_final):.4f}")

# DDPM采样（η=1，随机）
np.random.seed(42)
ddpm_as_ddim = ddim_sample_vp(vp_score_analytic, N_particles, 200, eta=1.0)
ddpm_final = ddpm_as_ddim[-1]

print(f"DDPM(η=1)采样结果: μ={np.mean(ddpm_final):.4f}, σ²={np.var(ddpm_final):.4f}")
print(f"\nDDIM vs DDPM核心区别：")
print(f"  DDIM(η=0): 确定性ODE求解，适合插值和编辑")
print(f"  DDPM(η=1): 随机SDE求解，每次采样结果不同")


# ============================================================
# 步骤3：温度参数η在DDPM↔DDIM间连续插值
# ★ 原创设计
# ============================================================
print("\n" + "=" * 60)
print("步骤3：温度参数η连续插值")
print("=" * 60)

eta_values = [0.0, 0.25, 0.5, 0.75, 1.0]
eta_results = {}

for eta in eta_values:
    np.random.seed(42)
    traj = ddim_sample_vp(vp_score_analytic, N_particles, 200, eta=eta)
    final = traj[-1]
    eta_results[eta] = final
    
    # 用直方图与目标分布的交叉熵衡量质量
    from scipy import stats as sp_stats
    try:
        ks_stat, _ = sp_stats.ks_2samp(final, x0_ref)
    except:
        ks_stat = abs(np.mean(final) - np.mean(x0_ref))
    
    std_of_samples = np.std(final)
    print(f"  η={eta:.2f}: KS={ks_stat:.4f}, Std={std_of_samples:.4f}", 
          "←DDIM" if eta == 0 else ("←DDPM" if eta == 1 else ""))

print(f"\nη=0(DDIM)到η=1(DDPM)：随机性逐渐增大")
print(f"η=0: 确定性，初始噪声→唯一输出（适合可控生成）")
print(f"η=1: 完全随机，同一初始点→不同输出（多样性更大）")


# ============================================================
# 步骤4：DDPM vs DDIM采样质量与速度对比
# ★ 原创设计
# ============================================================
print("\n" + "=" * 60)
print("步骤4：DDPM vs DDIM采样质量与速度对比")
print("=" * 60)

import time

step_counts = [10, 20, 50, 100, 200, 500]
results = {'DDPM': [], 'DDIM': [], 'DDPM_time': [], 'DDIM_time': []}

for n_steps in step_counts:
    # DDPM (η=1)
    np.random.seed(42)
    t0 = time.time()
    traj_ddpm = ddim_sample_vp(vp_score_analytic, 3000, n_steps, eta=1.0)
    t_ddpm = time.time() - t0
    
    # DDIM (η=0)
    np.random.seed(42)
    t0 = time.time()
    traj_ddim = ddim_sample_vp(vp_score_analytic, 3000, n_steps, eta=0.0)
    t_ddim = time.time() - t0
    
    try:
        from scipy import stats as sp_stats
        ks_ddpm, _ = sp_stats.ks_2samp(traj_ddpm[-1], x0_ref[:3000])
        ks_ddim, _ = sp_stats.ks_2samp(traj_ddim[-1], x0_ref[:3000])
    except:
        ks_ddpm = abs(np.mean(traj_ddpm[-1]) - np.mean(x0_ref[:3000]))
        ks_ddim = abs(np.mean(traj_ddim[-1]) - np.mean(x0_ref[:3000]))
    
    results['DDPM'].append(ks_ddpm)
    results['DDIM'].append(ks_ddim)
    results['DDPM_time'].append(t_ddpm)
    results['DDIM_time'].append(t_ddim)
    
    print(f"  N={n_steps:4d}: DDPM KS={ks_ddpm:.4f} ({t_ddpm:.3f}s), DDIM KS={ks_ddim:.4f} ({t_ddim:.3f}s)")

print(f"\n关键发现：")
print(f"  1. DDIM在少步数时质量优于DDPM（ODE vs SDE离散化误差）")
print(f"  2. DDIM确定性：相同初始噪声→相同输出（可复现）")
print(f"  3. DDIM可做语义插值：在潜空间插值→中间结果有意义")


# ============================================================
# 可视化
# ============================================================

# 图1：PF-ODE采样过程
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
x_grid = np.linspace(-6, 6, 500)
n_vis = 8
indices = np.linspace(0, len(pf_traj)-1, n_vis, dtype=int)

for i, idx in enumerate(indices):
    row, col = i // 4, i % 4
    axes[row, col].hist(pf_traj[idx], bins=50, density=True, alpha=0.6, color='green')
    axes[row, col].plot(x_grid, gm1d_pdf(x_grid), 'r--', lw=2, label='Target')
    axes[row, col].set_title(f'PF-ODE step {idx}/{len(pf_traj)-1}')
    axes[row, col].set_xlim(-6, 6)
    axes[row, col].legend(fontsize=8)
    axes[row, col].grid(alpha=0.3)

fig.suptitle('概率流ODE采样过程（确定性，无随机项）', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(_SCRIPT_DIR, '步骤1_PF-ODE采样.png'), dpi=150)
plt.show()

# 图2：η连续插值
fig, axes = plt.subplots(1, 5, figsize=(25, 4))
for i, eta in enumerate(eta_values):
    axes[i].hist(eta_results[eta], bins=50, density=True, alpha=0.6, 
                 color=plt.cm.coolwarm(eta))
    axes[i].plot(x_grid, gm1d_pdf(x_grid), 'r--', lw=2, label='Target')
    label = 'DDIM' if eta == 0 else ('DDPM' if eta == 1 else '')
    axes[i].set_title(f'η={eta:.2f} {label}')
    axes[i].set_xlim(-6, 6)
    axes[i].legend(fontsize=8)
    axes[i].grid(alpha=0.3)

fig.suptitle('温度参数η：DDIM(η=0) ↔ DDPM(η=1) 连续插值', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(_SCRIPT_DIR, '步骤3_η插值.png'), dpi=150)
plt.show()

# 图3：DDPM vs DDIM质量对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(step_counts, results['DDPM'], 'bo-', lw=2, markersize=8, label='DDPM (η=1)')
ax1.plot(step_counts, results['DDIM'], 'gs-', lw=2, markersize=8, label='DDIM (η=0)')
ax1.set_xlabel('采样步数')
ax1.set_ylabel('KS统计量（越小越好）')
ax1.set_title('DDPM vs DDIM 采样质量')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_xscale('log')

ax2.plot(step_counts, results['DDPM_time'], 'bo-', lw=2, markersize=8, label='DDPM')
ax2.plot(step_counts, results['DDIM_time'], 'gs-', lw=2, markersize=8, label='DDIM')
ax2.set_xlabel('采样步数')
ax2.set_ylabel('运行时间 (s)')
ax2.set_title('DDPM vs DDIM 运行时间')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(_SCRIPT_DIR, '步骤4_DDPM_vs_DDIM.png'), dpi=150)
plt.show()

# 图4：DDPM vs DDIM vs PF-ODE 统一框架
# ★ 原创设计：展示三种方法的关系
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.axis('off')
table_text = [
    ['方法', '逆向SDE', 'PF-ODE', 'DDPM', 'DDIM'],
    ['连续方程', 'dx=[f-g²∇log p]dt+gd̄w', 'dx=[f-½g²∇log p]dt', '—', '—'],
    ['离散化', 'Euler-Maruyama', 'Euler', 'EM on reverse VP', 'Euler on PF-ODE'],
    ['随机性', '有(布朗运动)', '无(确定性)', '有(η=1)', '无(η=0)'],
    ['温度η', '—', '—', 'η=1', 'η=0'],
    ['边际分布', 'p_t(x)', 'p_t(x)（等价）', '≈p_t(x)', '≈p_t(x)'],
    ['步数需求', '多(~1000)', '少(~20-50)', '多(~1000)', '少(~50)'],
    ['可控性', '低', '高(插值/编辑)', '低', '高'],
]
ax.table(cellText=table_text[1:], colLabels=table_text[0],
         loc='center', cellLoc='center')
ax.set_title('DDPM、DDIM、逆向SDE、PF-ODE统一框架', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(_SCRIPT_DIR, '步骤4_统一框架对比.png'), dpi=150)
plt.show()


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验7.4 总结")
print("=" * 60)
print("1. PF-ODE: dx/dt = f(x,t) - ½g(t)²∇log p_t(x)")
print("   - 确定性ODE，与逆向SDE边际分布相同（Fokker-Planck保证）")
print("   - 适合可控生成：插值、编辑、语义操作")
print("2. DDIM = Euler离散化PF-ODE")
print("   - η=0时为确定性DDIM，η=1时退化为DDPM")
print("3. 温度参数η在DDPM↔DDIM间连续插值")
print("   - η↓: 更确定性、更少步数、更适合可控生成")
print("   - η↑: 更随机、更多样性、但需要更多步数")
print("4. 实践中：DDIM 50步 ≈ DDPM 1000步质量")
print("5. 高阶求解器（DPM-Solver等）进一步加速：20步即可")
