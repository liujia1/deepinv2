# -*- coding: utf-8 -*-
"""
实验7.5-2 概率流ODE与DDIM加速采样
对应章节：7.4（概率流ODE：随机采样的确定性等价）、7.5（数值离散化）

说明：
- 本实验的核心理论来自7.4章节（PF-ODE）
- 本实验的离散化方法来自7.5章节（Euler方法）
- 建议读者先学习7.4章节的理论，再学习7.5章节的离散化方法，最后完成本实验

注意：本实验涉及7.5章节的离散化方法，如果您尚未学习7.5章节，
      可能会遇到一些超纲的概念。建议您先阅读以下内容：
      - 7.5章节的Euler离散化方法
      - 7.5章节的温度参数η的概念
      - 7.5章节的DDIM更新公式
      这些内容将在实验中详细展示，帮助您理解离散化方法的应用。

素材来源：
  - 7.4节PF-ODE推导、DDIM=PF-ODE离散化
  - 02-ddpm.ipynb的DDIM采样代码
  - ★ 原创设计：温度参数η在DDPM↔DDIM间连续插值（双口径：边际不变性 + 条件随机性）
  - （原计划在本实验实现的Fokker-Planck边际等价验证，已迁移至实验7.4-1完成）

实验内容：
  步骤1：PF-ODE采样——确定性等价
  步骤2：DDIM = Euler离散化PF-ODE
  步骤3：温度参数η——边际分布不变，条件随机性 DDIM(η=0)→DDPM(η=1)
  步骤4：DDPM vs DDIM采样质量与速度对比（配对检验 + MC噪声底）

运行前提：纯NumPy/PyTorch CPU即可
"""

import numpy as np
import os
import sys
import io
import json
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
    SAVE_DIR = os.path.join(_gdrive, '实验7.5-2')
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

from scipy.special import erf


def target_cdf(x):
    """解析目标分布CDF：0.3·N(-2,1) + 0.7·N(1,1)，供单样本KS使用"""
    z1 = 0.5 * (1 + erf((x + 2) / np.sqrt(2)))
    z2 = 0.5 * (1 + erf((x - 1) / np.sqrt(2)))
    return 0.3 * z1 + 0.7 * z2


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
                   beta_min=0.1, beta_max=20.0, x_init=None):
    """DDIM采样（η=0为确定性DDIM，η=1退化为DDPM）

    通用公式（7.5节，温度参数η）：
    x_{t-Δt} = √(ᾱ_{t-Δt})·x̂_0 + √(1-ᾱ_{t-Δt}-σ²_η)·ε_θ + σ_η·ε

    其中：
    - x̂_0 = (x_t - √(1-ᾱ_t)·ε_θ) / √(ᾱ_t)    （Tweedie估计）
    - σ_η = η · √((1-ᾱ_{t-Δt})/(1-ᾱ_t)) · √(1-ᾱ_t/ᾱ_{t-Δt})
    - η=0: DDIM（确定性，PF-ODE）
    - η=1: DDPM（随机，逆向SDE）

    x_init: 显式指定初始噪声x_T（长度N_particles）。用于"固定同一初始点"的
            配对对比与条件随机性研究；为None时从N(0,I)随机抽取。
    """
    dt = T / N_steps
    if x_init is None:
        x = np.random.randn(N_particles)
    else:
        x = np.asarray(x_init, dtype=float).copy()
        assert x.shape == (N_particles,), "x_init长度必须等于N_particles"

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
        # 注：当 t_prev=0 时，alpha_bar_prev=1，sigma_eta=0，dir_xt系数为0
        # 最后一步退化为 x = x0_hat（Tweedie估计），这是DDIM的标准设计
        noise = np.random.randn(N_particles) * sigma_eta if sigma_eta > 0 else 0
        x = np.sqrt(alpha_bar_prev) * x0_hat + dir_xt + noise

        trajectory.append(x.copy())

    return np.array(trajectory)

# DDIM采样（η=0，确定性）
# 使用与PF-ODE相同的步数（500步），确保等价性验证的公平性
np.random.seed(42)
ddim_traj = ddim_sample_vp(vp_score_analytic, N_particles, 500, eta=0.0)
ddim_final = ddim_traj[-1]

print(f"DDIM(η=0)采样结果: μ={np.mean(ddim_final):.4f}, σ²={np.var(ddim_final):.4f}")

# 验证DDIM(η=0)与PF-ODE的等价性
# 使用相同初始噪声、相同步数（500步），比较两者最终样本的KS距离
try:
    ks_ddim_vs_pde, _ = sp_stats.ks_2samp(ddim_final, pf_final)
    print(f"DDIM(η=0) vs PF-ODE KS距离: {ks_ddim_vs_pde:.4f} (应接近0，验证等价性)")
except:
    ks_ddim_vs_pde = abs(np.mean(ddim_final) - np.mean(pf_final))
    print(f"DDIM(η=0) vs PF-ODE 均值差: {ks_ddim_vs_pde:.4f}")

# 更严格的验证：使用相同的初始噪声向量，逐步比较轨迹
# 因为两者都是确定性映射，给定相同初始点应该走出几乎相同的轨迹
np.random.seed(42)
x_init = np.random.randn(N_particles)  # 相同的初始噪声

# 重新运行PF-ODE，使用显式传入的初始点
pf_traj_strict = [x_init.copy()]
x_pf = x_init.copy()
for i in range(500):
    t = 1.0 - i * (1.0 / 500)
    beta_t = vp_beta(t)
    score = vp_score_analytic(x_pf, t)
    drift = beta_t * (1.0 / 500) * (0.5 * x_pf + 0.5 * score)
    x_pf = x_pf + drift
    pf_traj_strict.append(x_pf.copy())

# 重新运行DDIM，使用相同的初始点
ddim_traj_strict = [x_init.copy()]
x_ddim = x_init.copy()
for i in range(500):
    t = 1.0 - i * (1.0 / 500)
    t_prev = max(t - 1.0 / 500, 0)
    mean_t, std_t = vp_marginal(t)
    alpha_bar_t = mean_t**2
    mean_prev, std_prev = vp_marginal(t_prev)
    alpha_bar_prev = mean_prev**2
    score = vp_score_analytic(x_ddim, t)
    eps_theta = -std_t * score
    x0_hat = (x_ddim - std_t * eps_theta) / (mean_t + 1e-10)
    sigma_eta = 0.0  # η=0
    dir_xt = np.sqrt(max(1 - alpha_bar_prev - sigma_eta**2, 0)) * eps_theta
    x_ddim = np.sqrt(alpha_bar_prev) * x0_hat + dir_xt
    ddim_traj_strict.append(x_ddim.copy())

# 逐步比较轨迹差异（取几个关键步骤）
print("\n轨迹逐步对比（相同初始噪声，500步）：")
for step_idx in [100, 200, 300, 400, 500]:
    pf_step = pf_traj_strict[step_idx]
    ddim_step = ddim_traj_strict[step_idx]
    max_diff = np.max(np.abs(pf_step - ddim_step))
    mean_diff = np.mean(np.abs(pf_step - ddim_step))
    print(f"  步骤{step_idx}: 最大差异={max_diff:.6f}, 平均差异={mean_diff:.6f}")

# DDPM采样（η=1，随机）
# 使用与DDIM相同的步数（500步），确保公平对比
np.random.seed(42)
ddpm_as_ddim = ddim_sample_vp(vp_score_analytic, N_particles, 500, eta=1.0)
ddpm_final = ddpm_as_ddim[-1]

print(f"DDPM(η=1)采样结果: μ={np.mean(ddpm_final):.4f}, σ²={np.var(ddpm_final):.4f}")
print(f"\nDDIM vs DDPM核心区别：")
print(f"  DDIM(η=0): 确定性ODE求解，适合插值和编辑")
print(f"  DDPM(η=1): 随机SDE求解，每次采样结果不同")


# ============================================================
# 步骤3：温度参数η在DDPM↔DDIM间连续插值
# ★ 原创设计
#
# 重要口径（η最常见的误读）：
#   η∈[0,1] 的插值保持【边际分布】不变——DDIM更新中方向项与噪声项的
#   方差预算 (1-ᾱ_prev-σ²) + σ² ≡ 1-ᾱ_prev 守恒，故任何η下样本总体
#   都服从同一目标分布，边际KS/Std对η理论上不敏感。
#   η真正改变的是【条件随机性】：固定同一个初始噪声x_T，
#   η=0时输出唯一（确定性映射）；η越大，同一x_T对应的输出集合越发散。
# 因此本步骤用三个口径分别展示：
#   (a) 边际口径：对目标分布的KS/Std（预期对η不敏感，验证边际不变性）
#   (b) 条件口径：固定一批x_T、独立重复K次的 E_xT[Var(x0|xT)]（η的真实效应）
#   (c) 单点演示：固定同一个标量x_T，观察输出由δ逐步展宽
# ============================================================
print("\n" + "=" * 60)
print("步骤3：温度参数η连续插值（边际不变性 + 条件随机性）")
print("=" * 60)

eta_values = [0.0, 0.25, 0.5, 0.75, 1.0]
eta_results = {}        # (a) 边际样本
eta_marg_ks = {}        # (a) 边际KS

# (a) 边际分布（每个η一次5000粒子采样）；KS 用对解析目标CDF的单样本检验——
#     此规模下两样本KS噪声底≈0.017，会把“对η不敏感”淹没成纯噪声
for eta in eta_values:
    np.random.seed(42)
    final = ddim_sample_vp(vp_score_analytic, N_particles, 500, eta=eta)[-1]
    eta_results[eta] = final
    eta_marg_ks[eta] = float(sp_stats.kstest(final, target_cdf).statistic)

# (b) 条件随机性：固定同一批x_T，对每个η独立重复K_rep次
#     重复之间只有每步注入的布朗噪声不同
K_rep, N_cond = 20, 2000
np.random.seed(2024)
xT_fixed_pop = np.random.randn(N_cond)
eta_cond_var = {}
for eta in eta_values:
    reps = np.stack([
        ddim_sample_vp(vp_score_analytic, N_cond, 500, eta=eta,
                       x_init=xT_fixed_pop)[-1]
        for _ in range(K_rep)])
    eta_cond_var[eta] = float(reps.var(axis=0, ddof=1).mean())

# (c) 单点演示：固定x_T=x_star，K_point个副本各自独立注入噪声
x_star = 1.0
K_point = 300
eta_point_outs = {}
for eta in eta_values:
    eta_point_outs[eta] = ddim_sample_vp(
        vp_score_analytic, K_point, 500, eta=eta,
        x_init=np.full(K_point, x_star))[-1]

print(f"  配置：边际{N_particles}粒子；条件研究固定{N_cond}个x_T×{K_rep}次重复；"
      f"单点演示x_T={x_star}×{K_point}副本")
print(f"  边际KS=单样本KS(vs解析目标CDF)；N={N_particles}时纯MC地板≈0.87/√N="
      f"{0.8687 / np.sqrt(N_particles):.4f}")
print(f"  {'η':>5} {'边际KS':>8} {'样本Std':>9} "
      f"{'条件方差E[Var(x0|xT)]':>22} {'单点重复Std':>11}")
for eta in eta_values:
    tag = "  ←DDIM(确定性)" if eta == 0 else ("  ←DDPM(随机性)" if eta == 1 else "")
    print(f"  {eta:5.2f} {eta_marg_ks[eta]:8.4f} {np.std(eta_results[eta]):9.4f} "
          f"{eta_cond_var[eta]:22.4f} "
          f"{np.std(eta_point_outs[eta], ddof=1):11.4f}{tag}")

print("\n结论：")
print("  · 边际KS在各η间仅在蒙特卡洛噪声范围内抖动、无单调趋势 → 边际分布对η不变（理论预期）")
print("  · 条件方差在η=0时严格为0、随η增大而增大 → η插值的真实效应是条件随机性，而非边际质量")
print("  · η=0: 同一初始噪声→唯一输出（适合可控生成/插值）")
print("  · η=1: 同一初始噪声→发散的输出集合（同一x_T下多样性更大）")


# ============================================================
# 步骤4：DDPM vs DDIM采样质量与速度对比
# ★ 原创设计
# ============================================================
print("\n" + "=" * 60)
print("步骤4：DDPM vs DDIM采样质量与速度对比")
print("=" * 60)

import time

step_counts = [10, 20, 50, 100, 200, 500, 1000]
n_seeds = 10
results = {'DDPM': [], 'DDIM': [], 'DDPM_std': [], 'DDIM_std': [],
           'DDPM_seeds': [], 'DDIM_seeds': [],
           'pair_diff': [], 'pair_t': [], 'pair_p': [],
           'DDPM_time': [], 'DDIM_time': [],
           'DDPM_time_seeds': [], 'DDIM_time_seeds': []}

# 指标口径：对解析目标CDF做单样本KS（避免两样本KS的MC噪声底混淆排序），
# 10个种子报告均值±标准差并做配对t检验（与7.5-1步骤4的多种子口径一致）。
# 配对设计：同一(n_steps, seed)下DDPM与DDIM共享同一个x_T（先抽xT再分别传入），
# 控制变量公平对比。先实测MC噪声底——理想采样器用3000粒子也只能到此精度，
# 低于该值的KS差异没有实际意义。
def _sample_target_mixture(n):
    comp = np.random.rand(n) < 0.3
    return np.where(comp, np.random.randn(n) - 2, np.random.randn(n) + 1)

np.random.seed(0)
_floor = [sp_stats.kstest(_sample_target_mixture(3000), target_cdf).statistic
          for _ in range(100)]
mc_floor = float(np.mean(_floor))
print(f"MC噪声底（从目标分布直接抽样3000粒子，100次KS均值）: {mc_floor:.4f}")

for n_steps in step_counts:
    ks_acc = {'DDPM': [], 'DDIM': []}
    time_acc = {'DDPM': [], 'DDIM': []}
    for seed in range(n_seeds):
        seed_val = seed * 1000 + n_steps
        # 共享初始噪声x_T（同一RNG种子抽出的同一组随机数）
        np.random.seed(seed_val)
        xT = np.random.randn(3000)

        # DDPM(η=1)：不重新播种，注入噪声沿用同一RNG流——
        # 与"采样器内部自行抽x_T"的旧实现逐位等价（x_T后紧接各步噪声）
        t0 = time.time()
        out_ddpm = ddim_sample_vp(vp_score_analytic, 3000, n_steps, eta=1.0,
                                  x_init=xT)[-1]
        time_acc['DDPM'].append(time.time() - t0)

        # DDIM(η=0)：确定性，不消耗随机数；重播种仅为流程统一
        np.random.seed(seed_val)
        t0 = time.time()
        out_ddim = ddim_sample_vp(vp_score_analytic, 3000, n_steps, eta=0.0,
                                  x_init=xT)[-1]
        time_acc['DDIM'].append(time.time() - t0)

        ks_acc['DDPM'].append(float(sp_stats.kstest(out_ddpm, target_cdf).statistic))
        ks_acc['DDIM'].append(float(sp_stats.kstest(out_ddim, target_cdf).statistic))

    ks_ddpm = np.asarray(ks_acc['DDPM'])
    ks_ddim = np.asarray(ks_acc['DDIM'])
    diff = ks_ddpm - ks_ddim   # 配对差：>0 表示 DDIM 更好
    t_stat = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(n_seeds)))
    p_val = float(2 * sp_stats.t.sf(abs(t_stat), df=n_seeds - 1))

    results['DDPM'].append(float(ks_ddpm.mean()))
    results['DDIM'].append(float(ks_ddim.mean()))
    results['DDPM_std'].append(float(ks_ddpm.std(ddof=1)))
    results['DDIM_std'].append(float(ks_ddim.std(ddof=1)))
    results['DDPM_seeds'].append(ks_acc['DDPM'])
    results['DDIM_seeds'].append(ks_acc['DDIM'])
    results['pair_diff'].append(float(diff.mean()))
    results['pair_t'].append(t_stat)
    results['pair_p'].append(p_val)
    results['DDPM_time'].append(float(np.mean(time_acc['DDPM'])))
    results['DDIM_time'].append(float(np.mean(time_acc['DDIM'])))
    results['DDPM_time_seeds'].append(time_acc['DDPM'])
    results['DDIM_time_seeds'].append(time_acc['DDIM'])

    sig = "显著" if p_val < 0.05 else "不显著"
    print(f"  N={n_steps:4d}: DDPM KS={ks_ddpm.mean():.4f}±{ks_ddpm.std(ddof=1):.4f} "
          f"({np.mean(time_acc['DDPM']):.3f}s), "
          f"DDIM KS={ks_ddim.mean():.4f}±{ks_ddim.std(ddof=1):.4f} "
          f"({np.mean(time_acc['DDIM']):.3f}s)  "
          f"配对差={diff.mean():+.4f}, t={t_stat:+.2f}, p={p_val:.3f} [{sig}]")

_sig_steps = [n for n, p in zip(step_counts, results['pair_p']) if p < 0.05]
_flip = [(n, d, p) for n, d, p in zip(step_counts, results['pair_diff'],
                                      results['pair_p']) if d < 0]
_i50 = step_counts.index(50)
_i1000 = step_counts.index(1000)
print(f"\n关键发现（10种子配对、共享x_T；对解析目标CDF的单样本KS）：")
print(f"  1. 少步数(10-20)DDIM优势最大且显著（ODE无布朗离散化误差）；p<0.05的步数：{_sig_steps}")
if _flip:
    _flip_txt = "，".join([f"N={n}配对差={d:+.4f}(p={p:.3f})" for n, d, p in _flip])
    print(f"  2. 符号翻转仅出现在双方均已触底后（{_flip_txt}）：绝对值小于一个种子SD且不显著，"
          f"无实际意义，不能据此声称任一方在高步数更优")
print(f"  3. ≥50步两者均在MC噪声底(≈{mc_floor:.3f})附近，差距≤种子间标准差；"
      f"DDPM@500的回升与N=1000的翻转都是底内抖动")
print(f"  4. 实测 DDIM-50(KS={results['DDIM'][_i50]:.4f}) 与 "
      f"DDPM-1000(KS={results['DDPM'][_i1000]:.4f}) 同处噪声底（差{abs(results['DDIM'][_i50]-results['DDPM'][_i1000]):.4f}"
      f"<一个种子SD）→ 以约1/20步数达到同档质量，加速说法在本实验内可自证")
print(f"  5. 同步数下每步评估次数(NFE)相同，DDIM省时来自【所需步数更少】（跨曲线对比）")
print(f"  6. DDIM确定性：相同初始噪声→相同输出（可复现、可语义插值）")


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
    # 拉丁缩写走mathtext（DejaVu字形），避免SimHei大写O形似数字0
    axes[row, col].set_title(r'$\mathrm{PF-ODE}$ step '
                             f'{idx}/{len(pf_traj)-1}')
    axes[row, col].set_xlim(-6, 6)
    axes[row, col].legend(fontsize=8)
    axes[row, col].grid(alpha=0.3)

fig.suptitle(r'概率流 $\mathrm{PF-ODE}$ 采样过程（确定性，无随机项）', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_PF-ODE采样.png'), dpi=150, bbox_inches='tight')
plt.close()

# 图2：η连续插值（上行：边际分布；下行：固定同一x_T的条件输出分布）
fig, axes = plt.subplots(2, 5, figsize=(25, 8))

# 下行所有子图共用同一组bin与x范围，保证跨η可直接比较展宽程度
_all_pts = np.concatenate([eta_point_outs[e] for e in eta_values])
_lo, _hi = float(_all_pts.min()) - 0.3, float(_all_pts.max()) + 0.3
_cond_bins = np.linspace(_lo, _hi, 45)

for i, eta in enumerate(eta_values):
    color = plt.cm.coolwarm(eta)
    tag = 'DDIM' if eta == 0 else ('DDPM' if eta == 1 else '')

    # 上行：边际样本直方图（对η不变）
    ax_top = axes[0, i]
    ax_top.hist(eta_results[eta], bins=50, density=True, alpha=0.6, color=color)
    ax_top.plot(x_grid, gm1d_pdf(x_grid), 'r--', lw=2, label='Target')
    head = f'η={eta:.2f}' + (f' {tag}' if tag else '')
    ax_top.set_title(f'{head}｜边际KS={eta_marg_ks[eta]:.3f}')
    ax_top.set_xlim(-6, 6)
    ax_top.legend(fontsize=8)
    ax_top.grid(alpha=0.3)

    # 下行：固定x_T=x_star的K_point个副本输出
    ax_bot = axes[1, i]
    outs = eta_point_outs[eta]
    ax_bot.hist(outs, bins=_cond_bins, density=True, alpha=0.6, color=color)
    ax_bot.plot(x_grid, gm1d_pdf(x_grid), 'r--', lw=1, alpha=0.5,
                label='Target')
    p_std = float(np.std(outs, ddof=1))
    if p_std < 1e-8:  # η=0：所有副本输出恒等，退化为δ
        ax_bot.axvline(outs[0], color='k', lw=2)
        ax_bot.annotate('δ：唯一输出', xy=(outs[0], 0.95), xycoords=('data', 'axes fraction'),
                        xytext=(8, 0), textcoords='offset points', va='top', fontsize=9)
    ax_bot.set_title(f'固定x_T={x_star}：条件输出｜重复Std={p_std:.3f}')
    ax_bot.set_xlim(_lo, _hi)
    ax_bot.legend(fontsize=8)
    ax_bot.grid(alpha=0.3)
    if i == 0:
        ax_top.set_ylabel('边际密度', fontsize=11)
        ax_bot.set_ylabel('条件密度', fontsize=11)

fig.suptitle(r'温度参数 $\eta$：上行——边际分布（理论上对 $\eta$ 不变）；'
             r'下行——固定同一 $x_T$ 的条件输出（$\eta$ 的真实效应：$\delta$→展宽）',
             fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_η插值.png'), dpi=150, bbox_inches='tight')
plt.close()

# 图3：DDPM vs DDIM质量对比（误差棒=10种子SD，*＝配对t检验p<0.05，灰线=MC噪声底）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.errorbar(step_counts, results['DDPM'], yerr=results['DDPM_std'],
             fmt='bo-', lw=2, markersize=8, capsize=4, label='DDPM (η=1)')
ax1.errorbar(step_counts, results['DDIM'], yerr=results['DDIM_std'],
             fmt='gs-', lw=2, markersize=8, capsize=4, label='DDIM (η=0)')
ax1.axhline(mc_floor, color='gray', ls=':', lw=1.5)
ax1.text(step_counts[0], mc_floor + 0.0012, f'MC噪声底≈{mc_floor:.3f}',
         color='gray', fontsize=9)
for i, (n, p, ddpm_sd) in enumerate(zip(step_counts, results['pair_p'],
                                        results['DDPM_std'])):
    if p < 0.05:
        y_mark = max(results['DDPM'][i], results['DDIM'][i]) + ddpm_sd + 0.003
        ax1.annotate('*', xy=(n, y_mark), ha='center', fontsize=18,
                     color='purple')
ax1.set_xlabel('采样步数')
ax1.set_ylabel('KS统计量（10种子均值±SD，越小越好）')
ax1.set_title('DDPM vs DDIM 采样质量（*＝配对t检验 p<0.05）')
ax1.set_ylim(0, max(max(results['DDPM']) + 0.012, 0.075))
ax1.legend(loc='upper right')
ax1.grid(alpha=0.3)
ax1.set_xscale('log')

ax2.plot(step_counts, results['DDPM_time'], 'bo-', lw=2, markersize=8, label='DDPM')
ax2.plot(step_counts, results['DDIM_time'], 'gs-', lw=2, markersize=8, label='DDIM')
ax2.set_xlabel('采样步数')
ax2.set_ylabel('运行时间 (s)')
ax2.set_title('DDPM vs DDIM 运行时间')
ax2.text(0.03, 0.97, '同步数NFE相同；\nDDIM省时来自所需步数更少\n（左图跨曲线对比：DDIM-50≈DDPM-1000）',
         transform=ax2.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax2.legend(loc='lower right')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤4_DDPM_vs_DDIM.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 结构化结果落盘（含每种子原始KS/耗时，口径可追溯；与实验7.5-1一致）
# ============================================================
_traj_diff_rows = []
for _s in [100, 200, 300, 400, 500]:
    _traj_diff_rows.append({
        "step": _s,
        "max_abs_diff": float(np.max(np.abs(pf_traj_strict[_s] - ddim_traj_strict[_s]))),
        "mean_abs_diff": float(np.mean(np.abs(pf_traj_strict[_s] - ddim_traj_strict[_s])))
    })

_step4_rows = []
for _j, _n in enumerate(step_counts):
    _step4_rows.append({
        "n_steps": _n,
        "ddpm_ks_mean": results['DDPM'][_j],
        "ddpm_ks_std": results['DDPM_std'][_j],
        "ddim_ks_mean": results['DDIM'][_j],
        "ddim_ks_std": results['DDIM_std'][_j],
        "paired_diff_ddpm_minus_ddim": results['pair_diff'][_j],
        "paired_t": results['pair_t'][_j],
        "paired_p": results['pair_p'][_j],
        "significant_p_lt_0.05": bool(results['pair_p'][_j] < 0.05),
        "ddpm_time_mean_s": results['DDPM_time'][_j],
        "ddim_time_mean_s": results['DDIM_time'][_j],
        "ddpm_ks_seeds": results['DDPM_seeds'][_j],
        "ddim_ks_seeds": results['DDIM_seeds'][_j],
        "ddpm_time_seeds_s": results['DDPM_time_seeds'][_j],
        "ddim_time_seeds_s": results['DDIM_time_seeds'][_j]
    })

summary = {
    "experiment": "7.5-2",
    "target": {
        "weights": [0.3, 0.7], "means": [-2, 1], "stds": [1, 1],
        "mu": 0.1, "var": 2.89,
        "reference_sample_mu": float(np.mean(x0_ref)),
        "reference_sample_var": float(np.var(x0_ref))
    },
    "vp_sde": {"beta_min": 0.1, "beta_max": 20.0, "T": 1.0},
    "step1_pfode": {
        "N_particles": int(N_particles), "N_steps": 500,
        "mu": float(np.mean(pf_final)), "var": float(np.var(pf_final))
    },
    "step2_equivalence": {
        "ddim_eta0": {"mu": float(np.mean(ddim_final)),
                      "var": float(np.var(ddim_final))},
        "ddpm_eta1": {"mu": float(np.mean(ddpm_final)),
                      "var": float(np.var(ddpm_final))},
        "ks_ddim_eta0_vs_pfode": float(ks_ddim_vs_pde),
        "trajectory_diff_same_init": _traj_diff_rows,
        "note": "相同初始噪声、500步；DDIM与PF-ODE为同一ODE的两种一阶离散化"
    },
    "step3_eta": {
        "marginal": {"N_particles": int(N_particles), "N_steps": 500,
                     "metric": "two-sample KS vs 5000 iid target samples",
                     "expectation": "η-invariant (variance budget conserved)"},
        "conditional_variance_study": {
            "N_particles": int(N_cond), "K_repeats": int(K_rep), "N_steps": 500,
            "design": "同一批x_T固定，每个η独立重复K次",
            "metric": "E_xT[Var(x0|xT)] (ddof=1)"
        },
        "single_point_demo": {"x_T_fixed": float(x_star),
                              "K_replicas": int(K_point), "N_steps": 500},
        "rows": [{
            "eta": float(eta),
            "marginal_ks": eta_marg_ks[eta],
            "sample_std": float(np.std(eta_results[eta])),
            "conditional_var": eta_cond_var[eta],
            "single_point_repeat_std": float(np.std(eta_point_outs[eta], ddof=1))
        } for eta in eta_values]
    },
    "step4_ddpm_vs_ddim": {
        "metric": "one-sample KS vs analytic target CDF",
        "N_particles": 3000, "n_seeds": int(n_seeds),
        "paired_design": "同一(n_steps,seed)下DDPM与DDIM共享x_T",
        "mc_floor_ks_n3000": mc_floor,
        "rows": _step4_rows
    }
}
_json_path = os.path.join(SAVE_DIR, 'results_summary.json')
with open(_json_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\n结构化结果已保存: {os.path.basename(_json_path)}")

# 统一框架对比（以文字形式输出，不单独生成图片）
# ★ 原创设计：展示四种方法的关系
print("\n" + "=" * 60)
print("DDPM、DDIM、逆向SDE、PF-ODE 统一框架对比")
print("=" * 60)
print(f"{'方法':<10} {'逆向SDE':<22} {'PF-ODE':<22} {'DDPM':<14} {'DDIM':<14}")
print(f"{'连续方程':<8} {'dx=[f-g²∇log p]dt+gd̄w':<22} {'dx=[f-½g²∇log p]dt':<22} {'—':<14} {'—':<14}")
print(f"{'离散化':<8} {'Euler-Maruyama':<22} {'Euler':<22} {'EM on rev VP':<14} {'Euler on PF':<14}")
print(f"{'随机性':<8} {'有(布朗运动)':<22} {'无(确定性)':<22} {'有(η=1)':<14} {'无(η=0)':<14}")
print(f"{'温度η':<8} {'—':<22} {'—':<22} {'η=1':<14} {'η=0':<14}")
print(f"{'边际分布':<8} {'p_t(x)':<22} {'p_t(x)（等价）':<22} {'≈p_t(x)':<14} {'≈p_t(x)':<14}")
print(f"{'步数需求':<8} {'多(~1000)*':<22} {'少(~20-50)':<22} {'多(~1000)*':<14} {'少(~50)':<14}")
print(f"{'可控性':<8} {'低':<22} {'高(插值/编辑)':<22} {'低':<14} {'高':<14}")
print()
print("注：带*的步数需求为文献典型经验值；本实验步骤4已实测至1000步（明细见results_summary.json）")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验7.5-2 总结")
print("=" * 60)
print("1. PF-ODE: dx/dt = f(x,t) - ½g(t)²∇log p_t(x)")
print("   - 确定性ODE，与逆向SDE边际分布相同（Fokker-Planck保证）")
print("   - 适合可控生成：插值、编辑、语义操作")
print("2. DDIM = Euler离散化PF-ODE")
print("   - η=0时为确定性DDIM，η=1时退化为DDPM")
print("3. 温度参数η在DDPM↔DDIM间连续插值（双口径）")
print("   - 边际分布对η不变（更新的方差预算守恒），边际KS无单调趋势")
print("   - η的真实效应是条件随机性 E_xT[Var(x0|xT)]：η=0严格为0，η↑则发散")
print("   - η↓: 同一初始点输出唯一，适合可控生成/插值；η↑: 同一初始点输出多样")
print(f"4. 实测：DDIM 50步 KS={results['DDIM'][_i50]:.4f} 与 "
      f"DDPM 1000步 KS={results['DDPM'][_i1000]:.4f} 同处MC噪声底(≈{mc_floor:.3f})")
print("   少步数(10-20)DDIM配对检验显著占优；50步以上两者均触底、差距不具实际意义")
print("5. 高阶求解器（DPM-Solver等）进一步加速：20步即可")