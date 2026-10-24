# -*- coding: utf-8 -*-
"""
实验7.6-1 实践：用扩散SDE实现图像生成
对应章节: 7.6（实践：用扩散SDE实现图像生成）
素材来源:
  - 7.6节噪声调度设计、训练目标、采样流程
  - 02-ddpm.ipynb, 04-sde.ipynb的实践代码
  - ★ 原创设计：1D高斯混合上的完整实践流程
  - ★ 原创设计：线性vs余弦调度SNR对比
  - ★ 原创设计：DDPM/DDIM/AB2(2阶多步)步数-质量曲线

实验内容:
  步骤1: 噪声调度设计——线性vs余弦vs几何调度对比
  步骤2: 训练目标可视化——ε-prediction与DSM损失随时间变化
  步骤3: 采样器对比——DDPM vs DDIM vs AB2(2阶多步)步数-质量曲线
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
from scipy.special import erf

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

# ====== 可选诊断：数值稳定性检查（默认关闭，避免干扰教学节奏）======
# 设为 True 可查看多seed统计检验、基线噪声诊断、系统排查清单等深入分析。
DIAGNOSE = os.environ.get("DIAGNOSE", "0") == "1"


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

# 从目标分布采样（用于评估采样质量）——向量化实现
def gm1d_sample(n, weights=[0.3, 0.7], means=[-2, 1], stds=[1, 1]):
    k = np.random.choice(len(weights), p=weights, size=n)
    means_arr = np.array(means)[k]
    stds_arr = np.array(stds)[k]
    return np.random.randn(n) * stds_arr + means_arr

def gm1d_cdf(x, weights=[0.3, 0.7], means=[-2, 1], stds=[1, 1]):
    """1D高斯混合的解析CDF（单样本KS检验用）"""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for w, m, sd in zip(weights, means, stds):
        out += w * 0.5 * (1.0 + erf((x - m) / (sd * np.sqrt(2.0))))
    return out


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
# 注意：VE-SDE的原始SMLD实现是从高噪声退火到低噪声（σ_max→σ_min），
# 但为了与VP-SDE公平对比（t=0干净数据→t=1纯噪声），
# 这里将时间方向统一为σ_min→σ_max，使SNR单调下降
sigma_min_ve, sigma_max_ve = 0.01, 50.0
sigma_ve = sigma_min_ve * (sigma_max_ve / sigma_min_ve)**t_grid
snr_ve = 1.0 / (sigma_ve**2 + 1e-30)

# SNR在dB尺度（裁剪极端值，避免log(0)问题）
snr_lin_db = 10 * np.log10(np.clip(snr_lin, 1e-10, 1e10))
snr_cos_db = 10 * np.log10(np.clip(snr_cos, 1e-10, 1e10))
snr_ve_db = 10 * np.log10(np.clip(snr_ve, 1e-10, 1e10))

# SNR变化率（dB/单位时间）——衡量调度的均匀性
snr_rate_lin = np.abs(np.gradient(snr_lin_db, t_grid))
snr_rate_cos = np.abs(np.gradient(snr_cos_db, t_grid))
snr_rate_ve = np.abs(np.gradient(snr_ve_db, t_grid))

print("三种噪声调度的SNR范围（dB）：")
print(f"  线性调度(VP): SNR从{snr_lin_db[0]:.1f}dB到{snr_lin_db[-1]:.1f}dB")
print(f"  余弦调度(VP): SNR从{snr_cos_db[0]:.1f}dB到{snr_cos_db[-1]:.1f}dB")
print(f"  几何调度(VE): SNR从{snr_ve_db[0]:.1f}dB到{snr_ve_db[-1]:.1f}dB")
# 内部区间（剔除端点；VP调度SNR在t→0/t→1发散，全区间dB变化率被裁剪主导，无参考意义）
_interior = (t_grid > 0.05) & (t_grid < 0.98)
print(f"\nSNR变化率的均匀性（内部区间 t∈(0.05,0.98)，标准差越小越均匀）：")
print(f"  线性调度: std={np.std(snr_rate_lin[_interior]):.2f} dB/单位时间")
print(f"  余弦调度: std={np.std(snr_rate_cos[_interior]):.2f} dB/单位时间")
print(f"  几何调度: std={np.std(snr_rate_ve[_interior]):.2f} dB/单位时间 (精确为0，对数尺度严格均匀)")
print(f"  注：几何调度dB尺度严格均匀是其定义使然；余弦调度的设计目标是ᾱ空间两端平滑")
print(f"      （避免t→0/t→1处信噪比剧变），并非dB-SNR变化率均匀，故其内部区间波动反而更大；")
print(f"      全区间std（线性{np.std(snr_rate_lin):.0f}/余弦{np.std(snr_rate_cos):.0f}）"
      f"由端点裁剪(1e-10~1e10)主导，不作为结论依据")


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
loss_vs_t_linear = []

# 多次平均以减少采样噪声，更准确地体现调度本身的性质
n_repeats = 5

for t in t_train:
    # 余弦调度损失
    losses_cos = []
    for _ in range(n_repeats):
        mean_t, std_t = vp_marginal_cosine(t)
        epsilon = np.random.randn(N_samples)
        x_t = mean_t * x0 + std_t * epsilon
        score = vp_score_analytic(x_t, t, vp_marginal_cosine)
        eps_pred = -std_t * score
        loss = np.mean((eps_pred - epsilon)**2)
        losses_cos.append(loss)
    loss_vs_t.append(np.mean(losses_cos))

    # 线性调度损失
    losses_lin = []
    for _ in range(n_repeats):
        mean_t_lin, std_t_lin = vp_marginal_linear(t)
        epsilon = np.random.randn(N_samples)
        x_t_lin = mean_t_lin * x0 + std_t_lin * epsilon
        score_lin = vp_score_analytic(x_t_lin, t, vp_marginal_linear)
        eps_pred_lin = -std_t_lin * score_lin
        loss_lin = np.mean((eps_pred_lin - epsilon)**2)
        losses_lin.append(loss_lin)
    loss_vs_t_linear.append(np.mean(losses_lin))

loss_vs_t = np.array(loss_vs_t)
loss_vs_t_linear = np.array(loss_vs_t_linear)

print("ε-prediction训练损失随时间变化：")
print(f"  余弦调度: 损失范围 [{loss_vs_t.min():.4f}, {loss_vs_t.max():.4f}], std={np.std(loss_vs_t):.4f}")
print(f"  线性调度: 损失范围 [{loss_vs_t_linear.min():.4f}, {loss_vs_t_linear.max():.4f}], std={np.std(loss_vs_t_linear):.4f}")
print(f"  注：ε的分布固定为N(0,I)（量级不随t变化），loss随t变化反映Var[ε|x_t]的差异")


# ============================================================
# 步骤3：采样器对比——DDPM vs DDIM vs AB2(2阶多步)
# ★ 原创设计：步数-质量曲线对比
# 7.6节：不同采样器的步数-质量曲线
# ============================================================
print("\n" + "=" * 60)
print("步骤3：采样器对比——DDPM vs DDIM vs AB2(2阶多步)")
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
            beta_h = 0.0  # 余弦调度下 α̅(1)=cos²(π/2)=0，此步退化为 no-op（数学合理：t=1时分布为纯噪声）
        beta_h = np.clip(beta_h, 1e-8, 20.0)  # 防止数值爆炸（上限宽松，仅防NaN/Inf）

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
        # DDIM不需要 t_prev=1e-10 边界修复：t_prev=0 时余弦调度 alpha_bar_prev=1，
        # 但DDIM不计算 lambda=log(alpha_bar/(1-alpha_bar))，无对数SNR运算。
        # 最终 x = sqrt(1)*x0_hat + sqrt(max(1-1,0))*eps = x0_hat，数值稳定。
        # AB2需要修复是因为它在对数SNR空间做二阶展开，1-alpha_bar_prev=0 会触发1e-30兜底膨胀。
        t_prev = max(t - dt, 0)

        mean_t, std_t = marginal_fn(t)
        mean_prev, std_prev = marginal_fn(t_prev)
        alpha_bar_t = mean_t**2
        alpha_bar_prev = mean_prev**2

        score = score_fn(x, t, marginal_fn)
        eps_theta = -std_t * score

        # Tweedie估计 x̂_0
        x0_hat = (x - std_t * eps_theta) / (mean_t + 1e-10)

        # σ_η（随机性控制）
        # ★ Bug修复：逆向采样中alpha_bar_t < alpha_bar_prev，条件判断应为alpha_bar_prev > alpha_bar_t
        if eta > 0 and alpha_bar_prev > 0 and alpha_bar_prev > alpha_bar_t:
            sigma_eta = eta * np.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * \
                       np.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        else:
            sigma_eta = 0.0

        dir_xt = np.sqrt(max(1 - alpha_bar_prev - sigma_eta**2, 0)) * eps_theta
        noise = np.random.randn(N_particles) * sigma_eta if sigma_eta > 0 else 0
        x = np.sqrt(alpha_bar_prev) * x0_hat + dir_xt + noise

    return x

# ---- AB2(2阶多步)采样器（非论文 DPM-Solver++(2M)） ----
def ab2_sample(score_fn, marginal_fn, N_particles, N_steps, T=1.0):
    """AB2（二阶 Adams–Bashforth 多步）——本实验实际使用的二阶 ODE 求解器

    名称说明：这不是 Lu et al. (2022) 的 DPM-Solver++(2M)，而是 AB2 风格的多步外推：
    - 利用前一步和当前步的 ε 预测做线性外推修正
    - 第一步没有历史信息，退化为 Euler 一阶（multistep 的固有特性）
    - N_steps 步中实际只有 N_steps-1 步是二阶修正

    与论文 2M 的 D(x) = (1+1/(2r))D_n - (1/(2r))D_{n-1} 不同；本实验在 t 上均匀取步，
    λ 步长并不均匀，0.5*r 修正也不是严格的变步长 AB2 系数。因此标签不应写成
    "DPM-Solver(2阶)"。
    """
    dt = T / N_steps
    x = np.random.randn(N_particles)
    prev_eps = None  # 保存上一步的ε预测，用于二阶修正
    prev_h = None    # 保存上一步的步长

    for i in range(N_steps):
        t = T - i * dt
        # ★ Bug修复：t_prev 下界用 1e-10 而非 0。
        # 原因：t_prev=0 时余弦调度 alpha_bar_prev=1，使 1-alpha_bar_prev=0，
        # 被 1e-30 兜底项撑成 1e-30 → lambda_prev=0.5*log(1/1e-30)≈34.5（虚假奇点），
        # 导致最后一步 h_step、r 跳变 30~60x，eps_corrected 被严重污染。
        # 改用 1e-10 后 alpha_bar_prev 严格 <1，lambda_prev 保持有限合理值。
        t_prev = max(t - dt, 1e-10)

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
            # 二阶修正（multistep Adams-Bashforth风格）
            # r = h_{n} / h_{n-1}
            # D1 = eps_n - eps_{n-1}
            r = h_step / (prev_h + 1e-30)
            D1 = eps_theta - prev_eps

            # ★ 边界保护：t→0 时余弦调度的 alpha_bar_prev 极度接近 1，
            # 1-alpha_bar_prev 极小，导致 lambda_prev 被兜底项人为撑大（~13），
            # 使最后一步 r 异常巨大（r≈15~17，正常步 r≈0.02~1.6）。
            # 直接跳过二阶修正会丢失方向信息（最后一步跨度大，Euler一阶误差也大），
            # 因此对 r 做 clip：保留 D1 的修正方向，仅限制幅度避免污染。
            r = np.clip(r, -2.0, 2.0)
            # 二阶修正的ε预测
            eps_corrected = eps_theta + 0.5 * r * D1
        else:
            # 第一步没有历史信息，退化为Euler一阶
            eps_corrected = eps_theta

        # 用修正后的ε预测更新
        x0_hat = (x - std_t * eps_corrected) / (mean_t + 1e-10)
        x = np.sqrt(alpha_bar_prev) * x0_hat + np.sqrt(1 - alpha_bar_prev) * eps_corrected

        prev_eps = eps_theta.copy()
        prev_h = h_step

    return x

# 评估采样质量（单样本KS统计量：样本ECDF vs 解析p0的CDF）
def evaluate_quality(samples):
    """单样本KS统计量，越小越好；零分布量级约 0.87/sqrt(N)。"""
    x = np.sort(np.asarray(samples, dtype=float))
    n = x.size
    cdf = gm1d_cdf(x)
    ecdf_hi = np.arange(1, n + 1) / n
    ecdf_lo = np.arange(0, n) / n
    return float(max(np.max(ecdf_hi - cdf), np.max(cdf - ecdf_lo)))

# 参考分布
np.random.seed(42)
x0_ref = gm1d_sample(5000)

# 步数-质量曲线
step_counts = [5, 10, 20, 100, 500]
results = {'DDPM': [], 'DDIM': [], 'DDIM(η=0.1)': [], 'AB2(2阶多步)': []}

print("\n步数-质量曲线（KS统计量，越小越好）：")
print(f"{'步数':>6s} | {'DDPM':>8s} | {'DDIM':>8s} | {'DDIM(η=0.1)':>12s} | {'AB2(2阶多步)':>16s}")
print("-" * 76)

for n_steps in step_counts:
    # DDPM
    np.random.seed(42)
    ddpm_out = ddpm_sample(vp_score_analytic, vp_marginal_cosine, 3000, n_steps)
    ks_ddpm = evaluate_quality(ddpm_out)
    results['DDPM'].append(ks_ddpm)

    # DDIM
    np.random.seed(42)
    ddim_out = ddim_sample(vp_score_analytic, vp_marginal_cosine, 3000, n_steps)
    ks_ddim = evaluate_quality(ddim_out)
    results['DDIM'].append(ks_ddim)

    # DDIM(η=0.1) - 对照实验：验证随机性是否有帮助
    np.random.seed(42)
    ddim_eta_out = ddim_sample(vp_score_analytic, vp_marginal_cosine, 3000, n_steps, eta=0.1)
    ks_ddim_eta = evaluate_quality(ddim_eta_out)
    results['DDIM(η=0.1)'].append(ks_ddim_eta)

    # AB2(2阶多步)
    np.random.seed(42)
    dpm_out = ab2_sample(vp_score_analytic, vp_marginal_cosine, 3000, n_steps)
    ks_dpm = evaluate_quality(dpm_out)
    results['AB2(2阶多步)'].append(ks_dpm)

    print(f"{n_steps:>6d} | {ks_ddpm:>8.4f} | {ks_ddim:>8.4f} | {ks_ddim_eta:>12.4f} | {ks_dpm:>14.4f}")

print(f"\n注：η=0.1 与 η=0 的差异见上表；是否系统性有益需看 DIAGNOSE=True 的多 seed 检验。")

print(f"\n理论背景：")
print(f"  - DDIM/DPM-Solver作为确定性ODE求解器，在图像任务中通常比DDPM（随机SDE求解器）更省步数")
print(f"  - DPM-Solver原论文(Lu et al. 2022)在图像生成任务中实现了20步高质量采样")
print(f"  - 本实验使用解析得分函数（完美训练），是验证数值方法的理想场景")

# 单seed趋势性结论（与7.6节实测KS表格对应）
print(f"\n观察（单seed=42，趋势性结论；多seed严格检验见DIAGNOSE=True）：")
print(f"  - 小步数(5-10步)：DDPM最优（但仍高于单样本KS基线≈0.016），DDIM(一阶确定性)显著最差，")
print(f"    AB2(二阶多步)明显优于DDIM但不敌DDPM")
print(f"  - 中大步数：20步时DDIM仍显著偏大；≥100步各方法均压到基线附近(0.012~0.020)，差异不显著")
print(f"  - 本实验用解析得分（完美训练），DDPM每步注入的噪声恰好补偿一阶离散化偏差")
print(f"    （DDIM 存在系统性标准差收缩，见 DIAGNOSE）；这是'低维1D+极少步数'场景的特性，")
print(f"    并非'ODE求解器只有在学习得分下才有优势'——精确得分反而是ODE求解器的理想情形。")
print(f"    图像实验中DDIM/DPM-Solver的省步优势来自高阶收敛及与过离散化(千步)DDPM的对比")

if DIAGNOSE:
    # ====== 基线噪声诊断：KS统计量的固有噪声水平 ======
    print("\n" + "=" * 60)
    print("基线噪声诊断：KS统计量的固有噪声水平")
    print("=" * 60)

    print("\n问题：即使样本完全取自目标分布 p0，单样本KS统计量也不会是0")
    print("      而是有一个由有限样本数决定的基线噪声")
    print("      需要先确认这个噪声有多大，才能判断表格中的差异是否是真实信号")

    print("\n基线噪声测试：每个独立的gm1d_sample(3000)与解析CDF做单样本KS检验")
    print(f"{'测试次数':>8s} | {'KS统计量':>10s}")
    print("-" * 30)

    baseline_ks_values = []
    n_baseline_tests = 10

    for test_idx in range(n_baseline_tests):
        np.random.seed(1000 + test_idx)
        sample1 = gm1d_sample(3000)
        ks_baseline = evaluate_quality(sample1)
        baseline_ks_values.append(ks_baseline)
        print(f"{test_idx + 1:>8d} | {ks_baseline:>10.4f}")

    baseline_mean = np.mean(baseline_ks_values)
    baseline_std = np.std(baseline_ks_values)
    print("-" * 30)
    print(f"{'均值':>8s} | {baseline_mean:>10.4f}")
    print(f"{'标准差':>8s} | {baseline_std:>10.4f}")
    print(f"{'范围':>8s} | [{np.min(baseline_ks_values):.4f}, {np.max(baseline_ks_values):.4f}]")
    print(f"\n基线噪声水平：{baseline_mean:.4f} ± {baseline_std:.4f}（单样本KS零分布基线，理论量级约0.87/sqrt(N)）")

    # ====== 同口径多seed统计检验（替代单seed"3σ"判断）======
    from scipy.stats import mannwhitneyu
    n_seeds_stat = int(os.environ.get("N_SEEDS_STAT", "20"))
    print(f"\n★ 同口径多seed统计检验（5/10/20/500步, N=3000, {n_seeds_stat}个seed）")
    print(f"  目的：用严格统计检验覆盖小步数到大步数全区间，替代单seed判断")
    print(f"  方法：每个采样器×每个步数用多个seed重复采样，Mann-Whitney U检验")
    print(f"  注：默认 n_seeds_stat=20，可用环境变量 N_SEEDS_STAT 调整。")

    def _sig(p):
        return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

    ddpm_ks_multi = []; ddim_ks_multi = []; dpm_ks_multi = []; base_ks_multi = []
    ref_std = np.std(x0_ref[:3000])

    for n_steps_test in [5, 10, 20, 500]:
        ddpm_ks = []; ddim_ks = []; dpm_ks = []; base_ks = []
        ddpm_std = []; ddim_std = []; dpm_std = []
        for seed in range(42, 42 + n_seeds_stat):
            np.random.seed(seed)
            out = ddpm_sample(vp_score_analytic, vp_marginal_cosine, 3000, n_steps_test)
            ddpm_ks.append(evaluate_quality(out)); ddpm_std.append(np.std(out))
            np.random.seed(seed)
            out = ddim_sample(vp_score_analytic, vp_marginal_cosine, 3000, n_steps_test)
            ddim_ks.append(evaluate_quality(out)); ddim_std.append(np.std(out))
            np.random.seed(seed)
            out = ab2_sample(vp_score_analytic, vp_marginal_cosine, 3000, n_steps_test)
            dpm_ks.append(evaluate_quality(out)); dpm_std.append(np.std(out))
            np.random.seed(2000 + seed)
            base_ks.append(evaluate_quality(gm1d_sample(3000)))

        p_ddpm_b = mannwhitneyu(ddpm_ks, base_ks, alternative='greater')[1]
        p_ddim_b = mannwhitneyu(ddim_ks, base_ks, alternative='greater')[1]
        p_dpm_b = mannwhitneyu(dpm_ks, base_ks, alternative='greater')[1]
        p_ddim_ddpm = mannwhitneyu(ddim_ks, ddpm_ks, alternative='greater')[1]
        p_dpm_ddpm = mannwhitneyu(dpm_ks, ddpm_ks, alternative='greater')[1]
        p_ddim_dpm = mannwhitneyu(ddim_ks, dpm_ks, alternative='greater')[1]

        if n_steps_test == 500:
            ddpm_ks_multi = ddpm_ks; ddim_ks_multi = ddim_ks
            dpm_ks_multi = dpm_ks; base_ks_multi = base_ks

        print(f"\n--- {n_steps_test}步 ---")
        print(f"  {'采样器':>12s} | {'KS均值':>8s} | {'样本std':>8s} | {'vs基线p':>10s} | {'vs DDPM p':>10s}")
        print(f"  {'基线':>12s} | {np.mean(base_ks):>8.4f} | {ref_std:>8.4f} | {'—':>10s} | {'—':>10s}")
        print(f"  {'DDPM':>12s} | {np.mean(ddpm_ks):>8.4f} | {np.mean(ddpm_std):>8.4f} | {p_ddpm_b:.4f}{_sig(p_ddpm_b):>5s} | {'—':>10s}")
        print(f"  {'DDIM':>12s} | {np.mean(ddim_ks):>8.4f} | {np.mean(ddim_std):>8.4f} | {p_ddim_b:.4f}{_sig(p_ddim_b):>5s} | {p_ddim_ddpm:.4f}{_sig(p_ddim_ddpm):>5s}")
        print(f"  {'AB2(2阶)':>12s} | {np.mean(dpm_ks):>8.4f} | {np.mean(dpm_std):>8.4f} | {p_dpm_b:.4f}{_sig(p_dpm_b):>5s} | {p_dpm_ddpm:.4f}{_sig(p_dpm_ddpm):>5s}")
        print(f"  DDIM vs AB2 (H1: DDIM更差): p={p_ddim_dpm:.4f}{_sig(p_ddim_dpm)}")
        if n_steps_test <= 20:
            print(f"  ★ 标准差收缩：DDIM std={np.mean(ddim_std):.3f} vs 参考{ref_std:.3f} "
                  f"(收缩{100*(1-np.mean(ddim_std)/ref_std):.1f}%)，"
                  f"DDPM std={np.mean(ddpm_std):.3f} (收缩{100*(1-np.mean(ddpm_std)/ref_std):.1f}%)")

    print()
    print(f"★ 500步时三者 KS 的离散度都在 0.005 附近"
          f"（DDPM {np.std(ddpm_ks_multi):.4f} / DDIM {np.std(ddim_ks_multi):.4f} / AB2 {np.std(dpm_ks_multi):.4f}），统计上已不可区分")
    print(f"    机制上：DDPM 每步重新注入噪声，分布不断向理论边际回平衡；")
    print(f"    DDIM/AB2是确定性映射，初始噪声的小偏差被整体带入，但到500步已与DDPM的波动同量级。")

    # ====== η 对照：随机性是否有系统收益（同 seed 配对检验）======
    from scipy.stats import wilcoxon
    print()
    print(f"★ eta 对照（{n_seeds_stat}个seed，同seed配对，Wilcoxon符号秩检验）")
    print(f"  {'步数':>6s} | {'eta=0 KS':>9s} | {'eta=0.1 KS':>11s} | {'配对p':>8s}")
    for n_steps_eta in [5, 20]:
        ks_eta0 = []; ks_eta01 = []
        for seed in range(42, 42 + n_seeds_stat):
            np.random.seed(seed)
            ks_eta0.append(evaluate_quality(
                ddim_sample(vp_score_analytic, vp_marginal_cosine, 3000, n_steps_eta, eta=0.0)))
            np.random.seed(seed)
            ks_eta01.append(evaluate_quality(
                ddim_sample(vp_score_analytic, vp_marginal_cosine, 3000, n_steps_eta, eta=0.1)))
        p_eta = wilcoxon(ks_eta01, ks_eta0).pvalue
        print(f"  {n_steps_eta:>6d} | {np.mean(ks_eta0):>9.4f} | {np.mean(ks_eta01):>11.4f} | {p_eta:.4f}{_sig(p_eta)}")
    print(f"  注：p 小表示两种 eta 的 KS 分布存在系统性差异（不分方向）；未配对则不可比。")

    print(f"\n★ 系统排查（结论以本表为准，不再粘贴开发期的固定 20-seed 数字）：")
    print(f"  已排查：score边界精度、DDPM beta_h clip、KS指标口径、DDIM公式正确性、Tweedie反推")
    print(f"  → 未发现公式错误；DDIM 少步数劣势来自一阶截断误差（表现为样本标准差收缩）。")
    print(f"  AB2 的 t_prev=1e-10 + clip r 用于避免 λ 端点奇点，其影响见上表 p 值。")

    print(f"\n结论（跨步数分层，趋势性结论；具体p值以当前动态表格为准）：")
    print(f"  ★ 小步数(5-10步)：DDIM(一阶ODE)显著劣于DDPM(一阶SDE)，体现一阶截断误差的影响")
    print(f"    AB2(二阶多步)显著优于DDIM(一阶)，体现二阶方法的真实加速优势")
    print(f"  ★ 中步数(20步)：DDPM 与 AB2 均已与'完美采样'不可区分（p>0.08），DDIM 仍显著偏大")
    print(f"  ★ 大步数(500步)：三者互相比较均无显著差异（p>0.4），单seed表中的细微排序属抽样噪声")
    print(f"  ★ 核心机制：确定性一阶方法(DDIM)在少步数下存在系统性标准差收缩；")
    print(f"    DDPM的随机噪声项恰好补偿截断误差；AB2(二阶多步)用高精度避免标准差收缩")
    print(f"  ★ 核心洞察：分界线不在'确定性 vs 随机'，而在能否控制截断误差")
    print(f"")
    print(f"    ┌────────────────────┬────────────┬───────────────────────────┐")
    print(f"    │ 方法               │ 少步数表现 │ 原因                      │")
    print(f"    ├────────────────────┼────────────┼───────────────────────────┤")
    print(f"    │ DDIM（一阶确定性） │ 最差       │ 截断误差暴露，std 收缩    │")
    print(f"    │ DDPM（一阶随机）   │ 最好       │ 随机噪声补偿截断误差      │")
    print(f"    │ AB2（二阶多步）    │ 居中       │ 二阶精度，避免收缩        │")
    print(f"    └────────────────────┴────────────┴───────────────────────────┘")
    print(f"")
    print(f"    教学意义：本实验（解析得分理想设定）中确定性ODE求解器在少步数下并无优势，")
    print(f"    'DDIM/DPM-Solver省步数'依赖真实神经网络得分的场景；在步数受限的部署中，")
    print(f"    控制截断误差（二阶方法优于一阶确定性）比'确定性 vs 随机'本身更重要")
    print("=" * 60)


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
np.random.seed(42)  # 确保初始噪声可复现（seed必须在randn之前）
x = np.random.randn(N_particles_vis)

# 保存关键时间步的快照
snapshot_steps = [0, 50, 150, 200]
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


# ============================================================
# 可视化
# ============================================================
x_grid = np.linspace(-6, 6, 500)

# 图1：噪声调度对比（1×2：SNR曲线 + SNR变化率）
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# SNR曲线（dB）
axes[0].plot(t_grid, snr_lin_db, 'b-', lw=2, label='线性调度 (VP)')
axes[0].plot(t_grid, snr_cos_db, 'r-', lw=2, label='余弦调度 (VP)')
axes[0].plot(t_grid, snr_ve_db, 'g-', lw=2, label='几何调度 (VE)')
axes[0].set_xlabel('$t$')
axes[0].set_ylabel('$\\mathrm{SNR~(dB)}$')
axes[0].set_title('SNR曲线对比')
axes[0].legend()
axes[0].grid(alpha=0.3)

# SNR变化率（仅绘内部区间；两端SNR发散产生裁剪伪影，全区间图无参考价值）
axes[1].plot(t_grid[_interior], snr_rate_lin[_interior], 'b-', lw=2, label='线性调度')
axes[1].plot(t_grid[_interior], snr_rate_cos[_interior], 'r-', lw=2, label='余弦调度')
axes[1].plot(t_grid[_interior], snr_rate_ve[_interior], 'g-', lw=2, label='几何调度')
axes[1].set_xlabel('$t$')
axes[1].set_ylabel('$|d\\mathrm{SNR}/dt|~\\mathrm{(dB)}$')
axes[1].set_title('SNR变化率（内部区间，越均匀越好）')
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '步骤1_噪声调度对比.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# 图2：训练损失随时间变化
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.plot(t_train, loss_vs_t, 'r-', lw=2, label='余弦调度')
ax.plot(t_train, loss_vs_t_linear, 'b--', lw=2, label='线性调度')
ax.set_xlabel('$t$')
ax.set_ylabel('$\\|\\epsilon_\\theta - \\epsilon\\|^2$')
ax.set_title('$\\epsilon$-prediction训练损失')
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '步骤2_训练目标.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# 图3：采样器步数-质量曲线
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.plot(step_counts, results['DDPM'], 'bo-', lw=2, markersize=8, label='DDPM')
ax.plot(step_counts, results['DDIM'], 'gs-', lw=2, markersize=8, label='DDIM')
ax.plot(step_counts, results['DDIM(η=0.1)'], 'yD-', lw=2, markersize=6, label='DDIM(η=0.1)')
ax.plot(step_counts, results['AB2(2阶多步)'], 'r^-', lw=2, markersize=8, label='AB2(2nd, multistep)')
ax.set_xlabel('采样步数')
ax.set_ylabel('KS统计量（越小越好）')
ax.set_title('采样器步数-质量曲线')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xscale('log')

fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '步骤3_采样器对比.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# 图4：采样轨迹可视化（1×5：4个快照[含采样器终样] + Ground Truth对照）
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
vis_steps = [0, 50, 150, 200]

for i, step in enumerate(vis_steps):
    if step in snapshots:
        data = snapshots[step]
        t_val = max(1.0 - step / N_steps_vis, 0)
        axes[i].hist(data, bins=50, density=True, alpha=0.6, color='steelblue')
        axes[i].plot(x_grid, gm1d_pdf(x_grid), 'r--', lw=2, label='Target')
        title = f'step {step} ($t={t_val:.2f}$)' + ('，采样器终样' if step == 200 else '')
        axes[i].set_title(title)
        axes[i].set_xlim(-6, 6)
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)

# 第5个位置展示目标分布（Ground Truth）作为参考
np.random.seed(42)
x0_gt = gm1d_sample(5000)
axes[4].hist(x0_gt, bins=50, density=True, alpha=0.6, color='steelblue')
axes[4].plot(x_grid, gm1d_pdf(x_grid), 'r--', lw=2, label='Target')
axes[4].set_title('Ground Truth ($x_0 \\sim p_{\\mathrm{data}}$)')
axes[4].set_xlim(-6, 6)
axes[4].legend(fontsize=8)
axes[4].grid(alpha=0.3)

fig.suptitle('DDIM采样轨迹：从噪声到数据', fontsize=14)
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
print(f"1. 噪声调度设计：几何调度dB尺度严格均匀(std=0，定义使然)；余弦调度设计目标是ᾱ空间")
print(f"   两端平滑而非dB-SNR均匀；内部区间变化率std: 线性({np.std(snr_rate_lin[_interior]):.1f})")
print(f"   小于余弦({np.std(snr_rate_cos[_interior]):.1f})；全区间std受端点裁剪主导，不作依据")
print("2. 训练目标：ε-prediction的预测目标ε~N(0,I)量级不随t变化，训练更稳定；")
print("   loss随t变化反映Var[ε|x_t]的差异（由SNR决定），余弦调度损失更均匀")
print("3. 采样器对比（DDPM/DDIM/AB2步数-质量曲线）：")
print("   - 小步数：DDPM最优（接近KS基线），AB2(二阶多步)明显优于DDIM(一阶)但不敌DDPM")
print("   - 大步数：三者均收敛，差异不显著")
print("   - 解析得分理想设定下DDPM 5步即收敛；少步数下DDPM噪声注入恰好补偿一阶离散偏差，")
print("     是'低维+极少步数'的场景特性；图像任务中ODE求解器的省步优势来自高阶收敛")
print("   - 多seed严格统计检验见DIAGNOSE=True（默认关闭）")
print("4. 采样轨迹：从纯噪声→模糊轮廓→清晰结构，得分函数逐步将噪声'引导'到数据分布")
print("5. 从PnP-ULA到扩散SDE的演化路径：")
print("   - PnP-ULA（第5章）：单一噪声水平→单尺度先验")
print("   - 退火Langevin（第6章）：离散噪声水平→多尺度先验")
print("   - 扩散SDE（第7章）：连续噪声调度→连续多尺度先验")
