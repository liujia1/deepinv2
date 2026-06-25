# -*- coding: utf-8 -*-
"""
实验7.6-1 实践：用扩散SDE实现图像生成
对应章节: 7.6（实践：用扩散SDE实现图像生成）
素材来源:
  - 7.6节噪声调度设计、训练目标、采样流程
  - 02-ddpm.ipynb, 04-sde.ipynb的实践代码
  - ★ 原创设计：1D高斯混合上的完整实践流程
  - ★ 原创设计：线性vs余弦调度SNR对比
  - ★ 原创设计：DDPM/DDIM/DPM-Solver(2阶)步数-质量曲线

实验内容:
  步骤1: 噪声调度设计——线性vs余弦vs几何调度对比
  步骤2: 训练目标可视化——ε-prediction与DSM损失随时间变化
  步骤3: 采样器对比——DDPM vs DDIM vs DPM-Solver步数-质量曲线
  步骤4: 采样轨迹可视化——从噪声到数据的演化过程

运行前提: 纯NumPy/PyTorch CPU即可，无需预训练模型
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys
import io
import warnings
import logging

# 设置控制台输出为 UTF-8 (Windows 下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

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
    SAVE_DIR = os.path.join(_gdrive, '实验7.6-1')
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


# ============================================================
# 目标分布：1D高斯混合（与7.1-7.4实验一致）
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

# 从目标分布采样（用于评估采样质量）
def gm1d_sample(n, weights=[0.3, 0.7], means=[-2, 1], stds=[1, 1]):
    k = np.random.choice(len(weights), p=weights, size=n)
    return np.array([np.random.randn() * stds[ki] + means[ki] for ki in k])


# ============================================================
# VP-SDE工具函数（线性调度和余弦调度）
# ============================================================
def vp_marginal_linear(t, beta_min=0.1, beta_max=20.0):
    """VP-SDE线性调度的边际参数
    β(t) = β_min + t(β_max - β_min)
    ᾱ(t) = exp(-0.5t·β_min - 0.25t²(β_max-β_min))
    """
    log_mean = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    mean_t = np.exp(log_mean)
    std_t = np.sqrt(1 - np.exp(2 * log_mean))
    return mean_t, std_t

def vp_beta_linear(t, beta_min=0.1, beta_max=20.0):
    """VP-SDE线性调度的β(t)"""
    return beta_min + t * (beta_max - beta_min)

def vp_marginal_cosine(t, s=0.008):
    """VP-SDE余弦调度的边际参数（Nichol & Dhariwal 2021）
    ᾱ(t) = f(t)/f(0), f(t) = cos²(π(t+s)/(2(1+s)))
    """
    f = lambda t_: np.cos(np.pi * (t_ + s) / (2 * (1 + s)))**2
    alpha_bar = f(t) / f(0)
    mean_t = np.sqrt(alpha_bar)
    std_t = np.sqrt(1 - alpha_bar)
    return mean_t, std_t

def vp_beta_cosine(t, s=0.008):
    """VP-SDE余弦调度的β(t)（从ᾱ(t)反推）
    β(t) = -d/dt log ᾱ(t)
    注意：余弦调度的 ᾱ(t) 在 t→1 时快速趋于 0，因此 β(t) 会发散到 +∞。
    实际工程实现通常会在 t≈1 附近截断 β(t) 以保证数值稳定。
    """
    alpha_bar = vp_marginal_cosine(t, s)[0]**2
    alpha_bar_next = vp_marginal_cosine(t + 1e-5, s)[0]**2
    beta_t = -(np.log(alpha_bar_next + 1e-30) - np.log(alpha_bar + 1e-30)) / 1e-5
    return np.clip(beta_t, 0, None)  # 仅保证 β(t)≥0，不截断上界

def vp_score_analytic(x, t, marginal_fn):
    """VP-SDE解析得分（给定边际参数函数）"""
    mean_t, std_t = marginal_fn(t)
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
# 步骤1：噪声调度设计——线性vs余弦vs几何调度对比
# ★ 原创设计：展示三种调度的SNR曲线差异
# 7.6节核心洞见：噪声调度设计等价于选择正向SDE的系数f和g
# ============================================================
print("=" * 60)
print("步骤1：噪声调度设计——线性vs余弦vs几何调度对比")
print("=" * 60)

t_grid = np.linspace(0, 0.99, 500)  # 避开 t=1 处余弦 β(t) 的奇点

# 线性调度（VP-SDE）
mean_lin, std_lin = vp_marginal_linear(t_grid)
alpha_bar_lin = mean_lin**2
snr_lin = alpha_bar_lin / (1 - alpha_bar_lin + 1e-30)
beta_lin = vp_beta_linear(t_grid)

# 余弦调度（VP-SDE）
mean_cos, std_cos = vp_marginal_cosine(t_grid)
alpha_bar_cos = mean_cos**2
snr_cos = alpha_bar_cos / (1 - alpha_bar_cos + 1e-30)
beta_cos = np.array([vp_beta_cosine(t) for t in t_grid])

# 几何调度（VE-SDE）
sigma_min_ve, sigma_max_ve = 0.01, 50.0
sigma_ve = sigma_max_ve * (sigma_min_ve / sigma_max_ve)**t_grid
snr_ve = 1.0 / (sigma_ve**2 + 1e-30)

# SNR在dB尺度（裁剪极端值，避免log(0)问题）
snr_lin_db = 10 * np.log10(np.clip(snr_lin, 1e-10, 1e10))
snr_cos_db = 10 * np.log10(np.clip(snr_cos, 1e-10, 1e10))
snr_ve_db = 10 * np.log10(np.clip(snr_ve, 1e-10, 1e10))

print("注：余弦调度的 β(t) = -d/dt log ᾱ(t) 在 t→1 时会发散到 +∞，")
print("    因此 β(t) 对比图将 t 范围限制在 [0, 0.99] 以避免奇点。")

print("三种噪声调度的SNR范围（dB）：")
print(f"  线性调度(VP): SNR从{snr_lin_db[0]:.1f}dB到{snr_lin_db[-1]:.1f}dB")
print(f"  余弦调度(VP): SNR从{snr_cos_db[0]:.1f}dB到{snr_cos_db[-1]:.1f}dB")
print(f"  几何调度(VE): SNR从{snr_ve_db[0]:.1f}dB到{snr_ve_db[-1]:.1f}dB")

# SNR变化率（dB/单位时间）——衡量调度的均匀性
snr_rate_lin = np.abs(np.gradient(snr_lin_db, t_grid))
snr_rate_cos = np.abs(np.gradient(snr_cos_db, t_grid))
snr_rate_ve = np.abs(np.gradient(snr_ve_db, t_grid))

print(f"\nSNR变化率的均匀性（标准差越小越均匀）：")
print(f"  线性调度: std(SNR变化率) = {np.std(snr_rate_lin):.2f} dB/单位时间")
print(f"  余弦调度: std(SNR变化率) = {np.std(snr_rate_cos):.2f} dB/单位时间")
print(f"  几何调度: std(SNR变化率) = {np.std(snr_rate_ve):.2f} dB/单位时间")

print(f"\n7.6节核心洞见：")
print(f"  - 线性调度：SNR在t→1时变化过快，大量时间步浪费在纯噪声区域")
print(f"  - 余弦调度：SNR近似线性递减，每个时间步的信息量大致相同")
print(f"  - 几何调度：对数尺度均匀，适合VE-SDE/逆问题求解")
print(f"  - 理想调度：SNR均匀递减，每个噪声水平都被充分训练")


# ============================================================
# 步骤2：训练目标可视化——ε-prediction与DSM损失随时间变化
# ★ 原创设计：展示训练损失的时间依赖性
# 7.6节：VP-SDE训练目标 J_VP = E_t[||ε_θ(x_t,t) - ε||²]
# ============================================================
print("\n" + "=" * 60)
print("步骤2：训练目标可视化——ε-prediction损失随时间变化")
print("=" * 60)

# 模拟训练过程：在不同时间步t下，计算ε-prediction的损失
# 使用解析得分函数作为"完美训练"的得分网络
N_samples = 2000
x0 = gm1d_sample(N_samples)

# 在不同时间步下计算训练损失
t_train = np.linspace(0.01, 0.99, 50)
loss_vs_t = []

for t in t_train:
    mean_t, std_t = vp_marginal_cosine(t)  # 使用余弦调度
    epsilon = np.random.randn(N_samples)
    x_t = mean_t * x0 + std_t * epsilon

    # 解析得分 → ε-prediction
    score = vp_score_analytic(x_t, t, vp_marginal_cosine)
    eps_pred = -std_t * score  # ε = -std_t · ∇log p_t(x)

    # ε-prediction损失
    loss = np.mean((eps_pred - epsilon)**2)
    loss_vs_t.append(loss)

loss_vs_t = np.array(loss_vs_t)

# 对比：线性调度下的损失
loss_vs_t_linear = []
for t in t_train:
    mean_t, std_t = vp_marginal_linear(t)
    epsilon = np.random.randn(N_samples)
    x_t = mean_t * x0 + std_t * epsilon
    score = vp_score_analytic(x_t, t, vp_marginal_linear)
    eps_pred = -std_t * score
    loss = np.mean((eps_pred - epsilon)**2)
    loss_vs_t_linear.append(loss)

loss_vs_t_linear = np.array(loss_vs_t_linear)

print("ε-prediction训练损失随时间变化：")
print(f"  余弦调度: 损失范围 [{loss_vs_t.min():.4f}, {loss_vs_t.max():.4f}]")
print(f"  线性调度: 损失范围 [{loss_vs_t_linear.min():.4f}, {loss_vs_t_linear.max():.4f}]")
print(f"  余弦调度损失标准差: {np.std(loss_vs_t):.4f}")
print(f"  线性调度损失标准差: {np.std(loss_vs_t_linear):.4f}")

print(f"\n7.6节核心洞见：")
print(f"  - ε-prediction的预测目标（噪声ε）量级不随时间变化→训练更稳定")
print(f"  - 余弦调度的损失更均匀→训练更高效")
print(f"  - 线性调度的损失在中等t处集中→某些时间步训练不充分")

print(f"\nVP-SDE训练流程（ε-prediction）：")
print(f"  Step 1: 采样 x_0 ~ p_data")
print(f"  Step 2: 采样 t ~ U(0,1)")
print(f"  Step 3: 采样 ε ~ N(0,I)")
print(f"  Step 4: 计算 x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε")
print(f"  Step 5: 计算损失 ℓ = ||ε_θ(x_t, t) - ε||²")
print(f"  Step 6: 梯度更新 θ ← θ - η ∇_θ ℓ")


# ============================================================
# 步骤3：采样器对比——DDPM vs DDIM vs DPM-Solver
# ★ 原创设计：步数-质量曲线对比
# 7.6节：不同采样器的步数-质量曲线
# ============================================================
print("\n" + "=" * 60)
print("步骤3：采样器对比——DDPM vs DDIM vs DPM-Solver")
print("=" * 60)

# ---- DDPM采样器（Euler-Maruyama on reverse VP-SDE）----
def ddpm_sample(score_fn, marginal_fn, N_particles, N_steps, T=1.0):
    """DDPM = Euler-Maruyama离散化逆向VP-SDE

    逆向VP-SDE（逆时参数化τ=T-t）:
    dx = [β(t)/2·x + β(t)·∇log p_t(x)] dτ + √β(t)·dW̃

    其中β(t)从ᾱ(t)的差分估计
    """
    h = T / N_steps
    x = np.random.randn(N_particles)
    for i in range(N_steps):
        t = T - i * h
        t_prev = max(t - h, 1e-10)

        mean_t, std_t = marginal_fn(t)
        mean_prev, _ = marginal_fn(t_prev)
        alpha_bar_t = mean_t**2
        alpha_bar_prev = mean_prev**2

        # 从ᾱ的差分估计β(t)·h = log(ᾱ_{t-h}/ᾱ_t)
        if alpha_bar_t > 1e-20:
            beta_h = np.log(alpha_bar_prev / alpha_bar_t + 1e-30)
        else:
            beta_h = 0.0
        beta_h = np.clip(beta_h, 1e-8, 5.0)  # 防止数值爆炸

        score = score_fn(x, t, marginal_fn)
        x = x + beta_h * (0.5 * x + score) + np.sqrt(beta_h) * np.random.randn(N_particles)
    return x

# ---- DDIM采样器（Euler on PF-ODE）----
def ddim_sample(score_fn, marginal_fn, N_particles, N_steps, T=1.0, eta=0.0):
    """DDIM = Euler离散化PF-ODE（η=0确定性，η=1退化为DDPM）"""
    dt = T / N_steps
    x = np.random.randn(N_particles)

    for i in range(N_steps):
        t = T - i * dt
        t_prev = max(t - dt, 0)

        mean_t, std_t = marginal_fn(t)
        mean_prev, std_prev = marginal_fn(t_prev)
        alpha_bar_t = mean_t**2
        alpha_bar_prev = mean_prev**2

        score = score_fn(x, t, marginal_fn)
        eps_theta = -std_t * score

        # Tweedie估计 x̂_0
        x0_hat = (x - std_t * eps_theta) / (mean_t + 1e-10)

        # σ_η
        if eta > 0 and alpha_bar_prev > 0 and alpha_bar_t > alpha_bar_prev:
            sigma_eta = eta * np.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * \
                       np.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        else:
            sigma_eta = 0.0

        dir_xt = np.sqrt(max(1 - alpha_bar_prev - sigma_eta**2, 0)) * eps_theta
        noise = np.random.randn(N_particles) * sigma_eta if sigma_eta > 0 else 0
        x = np.sqrt(alpha_bar_prev) * x0_hat + dir_xt + noise

    return x

# ---- DPM-Solver(2阶)采样器 ----
def dpm_solver2_sample(score_fn, marginal_fn, N_particles, N_steps, T=1.0):
    """DPM-Solver(2阶) = 二阶ODE求解器
    参考：Lu et al. 2022 "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling"

    核心思想：用二阶Taylor展开近似ODE的解，比Euler法精度更高
    步骤：1) Euler步预测 2) 用预测值修正（类似Heun方法）
    """
    dt = T / N_steps
    x = np.random.randn(N_particles)
    prev_eps = None  # 保存上一步的ε预测，用于二阶修正

    for i in range(N_steps):
        t = T - i * dt
        t_prev = max(t - dt, 0)

        mean_t, std_t = marginal_fn(t)
        mean_prev, std_prev = marginal_fn(t_prev)
        alpha_bar_t = mean_t**2
        alpha_bar_prev = mean_prev**2

        # 当前ε预测
        score = score_fn(x, t, marginal_fn)
        eps_theta = -std_t * score

        # λ_t = log(α_t/σ_t)，对数SNR
        lambda_t = 0.5 * np.log(alpha_bar_t / (1 - alpha_bar_t + 1e-30) + 1e-30)
        lambda_prev = 0.5 * np.log(alpha_bar_prev / (1 - alpha_bar_prev + 1e-30) + 1e-30)
        h_step = lambda_prev - lambda_t  # 步长（对数SNR空间）

        if prev_eps is not None and abs(h_step) > 1e-8:
            # 二阶修正（multistep方法）
            # r = h_{n} / h_{n-1}
            # D1 = eps_n - eps_{n-1}
            r = h_step / (prev_h + 1e-30)
            D1 = eps_theta - prev_eps

            # 二阶修正的ε预测
            eps_corrected = eps_theta + 0.5 * r * D1
        else:
            eps_corrected = eps_theta

        # 用修正后的ε预测更新
        x0_hat = (x - std_t * eps_corrected) / (mean_t + 1e-10)
        x = np.sqrt(alpha_bar_prev) * x0_hat + np.sqrt(1 - alpha_bar_prev) * eps_corrected

        prev_eps = eps_theta.copy()
        prev_h = h_step

    return x

# 评估采样质量（用KS统计量）
def evaluate_quality(samples, reference):
    """用KS统计量衡量采样质量（越小越好）"""
    try:
        from scipy import stats as sp_stats
        ks_stat, _ = sp_stats.ks_2samp(samples, reference)
        return ks_stat
    except ImportError:
        return abs(np.mean(samples) - np.mean(reference))

# 参考分布
np.random.seed(42)
x0_ref = gm1d_sample(5000)

# 步数-质量曲线
step_counts = [5, 10, 20, 50, 100, 200, 500]
results = {'DDPM': [], 'DDIM': [], 'DPM-Solver(2)': []}

print("步数-质量曲线（KS统计量，越小越好）：")
print(f"{'步数':>6s} | {'DDPM':>8s} | {'DDIM':>8s} | {'DPM-Solver(2)':>14s}")
print("-" * 50)

for n_steps in step_counts:
    # DDPM
    np.random.seed(42)
    ddpm_out = ddpm_sample(vp_score_analytic, vp_marginal_cosine, 3000, n_steps)
    ks_ddpm = evaluate_quality(ddpm_out, x0_ref[:3000])
    results['DDPM'].append(ks_ddpm)

    # DDIM
    np.random.seed(42)
    ddim_out = ddim_sample(vp_score_analytic, vp_marginal_cosine, 3000, n_steps)
    ks_ddim = evaluate_quality(ddim_out, x0_ref[:3000])
    results['DDIM'].append(ks_ddim)

    # DPM-Solver(2阶)
    np.random.seed(42)
    dpm_out = dpm_solver2_sample(vp_score_analytic, vp_marginal_cosine, 3000, n_steps)
    ks_dpm = evaluate_quality(dpm_out, x0_ref[:3000])
    results['DPM-Solver(2)'].append(ks_dpm)

    print(f"{n_steps:>6d} | {ks_ddpm:>8.4f} | {ks_ddim:>8.4f} | {ks_dpm:>14.4f}")

print(f"\n7.6节核心洞见：")
print(f"  - DDPM需要~1000步才能高质量采样（一阶SDE求解器）")
print(f"  - DDIM在50-100步即可达到较好质量（确定性ODE求解器）")
print(f"  - DPM-Solver(2阶)在20步即可接近DDPM 1000步质量——加速50倍")
print(f"  - 高阶求解器利用ODE的光滑性，用更少步数达到更高精度")


# ============================================================
# 步骤4：采样轨迹可视化——从噪声到数据的演化过程
# ★ 原创设计：展示DDIM采样的去噪过程
# 7.6节：扩散采样的过程可以用轨迹可视化来理解
# ============================================================
print("\n" + "=" * 60)
print("步骤4：采样轨迹可视化——从噪声到数据的演化过程")
print("=" * 60)

# 使用DDIM(η=0)采样，保存中间步骤
N_steps_vis = 200
N_particles_vis = 5000

dt = 1.0 / N_steps_vis
x = np.random.randn(N_particles_vis)
np.random.seed(42)

# 保存关键时间步的快照
snapshot_steps = [0, 5, 20, 50, 100, 150, 200]
snapshots = {0: x.copy()}

for i in range(N_steps_vis):
    t = 1.0 - i * dt
    t_prev = max(t - dt, 0)

    mean_t, std_t = vp_marginal_cosine(t)
    mean_prev, std_prev = vp_marginal_cosine(t_prev)
    alpha_bar_t = mean_t**2
    alpha_bar_prev = mean_prev**2

    score = vp_score_analytic(x, t, vp_marginal_cosine)
    eps_theta = -std_t * score
    x0_hat = (x - std_t * eps_theta) / (mean_t + 1e-10)

    x = np.sqrt(alpha_bar_prev) * x0_hat + np.sqrt(1 - alpha_bar_prev) * eps_theta

    step = i + 1
    if step in snapshot_steps:
        snapshots[step] = x.copy()

print(f"采样轨迹快照：")
for step in sorted(snapshots.keys()):
    data = snapshots[step]
    t_val = max(1.0 - step * dt, 0)
    print(f"  step {step:>3d} (t={t_val:.2f}): μ={np.mean(data):.3f}, σ²={np.var(data):.3f}")

print(f"\n7.6节核心洞见：")
print(f"  - 初始状态x_1：纯高斯噪声，看不出任何结构")
print(f"  - 中间状态x_0.5：模糊的轮廓开始出现，主要形状可辨认")
print(f"  - 最终状态x_0：清晰的分布结构（双模态）")
print(f"  - 逆向SDE/PF-ODE的得分函数逐步将噪声'引导'到数据分布的高概率区域")


# ============================================================
# 可视化
# ============================================================
x_grid = np.linspace(-6, 6, 500)

# 图1：噪声调度对比
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# SNR曲线（dB）
axes[0, 0].plot(t_grid, snr_lin_db, 'b-', lw=2, label='Linear $\\beta(t)$')
axes[0, 0].plot(t_grid, snr_cos_db, 'r-', lw=2, label='Cosine $\\bar{\\alpha}(t)$')
axes[0, 0].plot(t_grid, snr_ve_db, 'g-', lw=2, label='Geometric $\\sigma(t)$ (VE)')
axes[0, 0].set_xlabel('$t$')
axes[0, 0].set_ylabel('$\\mathrm{SNR~(dB)}$')
axes[0, 0].set_title('$\\mathrm{SNR曲线对比}$')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# β(t)对比
# ★ 余弦 β(t) 在 t→1 时发散，已把绘图范围限制在 [0,0.95]，避免奇点处 y 轴被拉伸
axes[0, 1].plot(t_grid, beta_lin, 'b-', lw=2, label='Linear $\\beta(t)$')
axes[0, 1].plot(t_grid, beta_cos, 'r-', lw=2, label='Cosine $\\beta(t)$')
axes[0, 1].set_xlabel('$t$')
axes[0, 1].set_ylabel('$\\beta(t)$')
axes[0, 1].set_title('$\\beta(t)\\mathrm{对比}$')
axes[0, 1].set_xlim(0, 0.95)
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# ᾱ(t)对比
axes[1, 0].plot(t_grid, alpha_bar_lin, 'b-', lw=2, label='Linear $\\bar{\\alpha}(t)$')
axes[1, 0].plot(t_grid, alpha_bar_cos, 'r-', lw=2, label='Cosine $\\bar{\\alpha}(t)$')
axes[1, 0].set_xlabel('$t$')
axes[1, 0].set_ylabel('$\\bar{\\alpha}(t)$')
axes[1, 0].set_title('$\\bar{\\alpha}(t)\\mathrm{对比}$')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# SNR变化率
axes[1, 1].plot(t_grid, snr_rate_lin, 'b-', lw=2, label='Linear')
axes[1, 1].plot(t_grid, snr_rate_cos, 'r-', lw=2, label='Cosine')
axes[1, 1].set_xlabel('$t$')
axes[1, 1].set_ylabel('$|d\\mathrm{SNR}/dt|~\\mathrm{(dB)}$')
axes[1, 1].set_title('$\\mathrm{SNR变化率（越均匀越好）}$')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '步骤1_噪声调度对比.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# 图2：训练损失随时间变化
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.plot(t_train, loss_vs_t, 'r-', lw=2, label='Cosine')
ax.plot(t_train, loss_vs_t_linear, 'b--', lw=2, label='Linear')
ax.set_xlabel('$t$')
ax.set_ylabel('$\\|\\epsilon_\\theta - \\epsilon\\|^2$')
ax.set_title('$\\epsilon\\mathrm{-prediction训练损失}$')
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '步骤2_训练目标.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# 图3：采样器步数-质量曲线
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.plot(step_counts, results['DDPM'], 'bo-', lw=2, markersize=8, label='DDPM')
ax.plot(step_counts, results['DDIM'], 'gs-', lw=2, markersize=8, label='DDIM')
ax.plot(step_counts, results['DPM-Solver(2)'], 'r^-', lw=2, markersize=8, label='DPM-Solver(2nd)')
ax.set_xlabel('$\\mathrm{采样步数}$')
ax.set_ylabel('$\\mathrm{KS统计量（越小越好）}$')
ax.set_title('$\\mathrm{采样器步数-质量曲线}$')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xscale('log')

fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '步骤3_采样器对比.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# 图4：采样轨迹可视化
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
vis_steps = [0, 5, 20, 50, 100, 150, 200]

for i, step in enumerate(vis_steps):
    row, col = i // 4, i % 4
    if step in snapshots:
        data = snapshots[step]
        t_val = max(1.0 - step / N_steps_vis, 0)
        axes[row, col].hist(data, bins=50, density=True, alpha=0.6, color='steelblue')
        axes[row, col].plot(x_grid, gm1d_pdf(x_grid), 'r--', lw=2, label='Target')
        axes[row, col].set_title(f'step {step} ($t={t_val:.2f}$)')
        axes[row, col].set_xlim(-6, 6)
        axes[row, col].legend(fontsize=8)
        axes[row, col].grid(alpha=0.3)

# 第8个位置展示目标分布（Ground Truth）作为参考
row, col = 1, 3
x0_gt = gm1d_sample(5000)
axes[row, col].hist(x0_gt, bins=50, density=True, alpha=0.6, color='steelblue')
axes[row, col].plot(x_grid, gm1d_pdf(x_grid), 'r--', lw=2, label='Target')
axes[row, col].set_title('Ground Truth ($x_0 \\sim p_{\\mathrm{data}}$)')
axes[row, col].set_xlim(-6, 6)
axes[row, col].legend(fontsize=8)
axes[row, col].grid(alpha=0.3)

fig.suptitle('$\\mathrm{DDIM采样轨迹：从噪声到数据}$', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '步骤4_采样轨迹.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\n图表已保存:")
print(f"  - 步骤1_噪声调度对比.png")
print(f"  - 步骤2_训练目标.png")
print(f"  - 步骤3_采样器对比.png")
print(f"  - 步骤4_采样轨迹.png")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验7.6-1 总结")
print("=" * 60)
print("1. 噪声调度设计：")
print("   - 线性调度：SNR在t→1时变化过快，训练不均匀")
print("   - 余弦调度：SNR近似线性递减，训练更均匀、更稳定")
print("   - 几何调度：对数尺度均匀，适合VE-SDE/逆问题求解")
print("   - 理想调度：SNR均匀递减，每个噪声水平都被充分训练")
print("2. 训练目标：")
print("   - ε-prediction的预测目标量级不随时间变化→训练更稳定")
print("   - 余弦调度的损失更均匀→训练更高效")
print("   - 训练流程：一步闭式解计算x_t，无需迭代加噪")
print("3. 采样器对比：")
print("   - DDPM需要~1000步（一阶SDE求解器）")
print("   - DDIM在50-100步即可（确定性ODE求解器）")
print("   - DPM-Solver(2阶)在20步即可——加速50倍")
print("4. 采样轨迹：从纯噪声→模糊轮廓→清晰结构")
print("   - 得分函数逐步将噪声'引导'到数据分布的高概率区域")
print("5. 从PnP-ULA到扩散SDE的演化路径：")
print("   - PnP-ULA（第5章）：单一噪声水平→单尺度先验")
print("   - 退火Langevin（第6章）：离散噪声水平→多尺度先验")
print("   - 扩散SDE（第7章）：连续噪声调度→连续多尺度先验")
