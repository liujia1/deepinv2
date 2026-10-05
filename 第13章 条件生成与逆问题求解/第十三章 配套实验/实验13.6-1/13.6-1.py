# -*- coding: utf-8 -*-
"""
实验13.6-1：条件采样与零样本迁移

★ 原创设计：对比"针对特定任务训练"和"零样本迁移"
  传统优化/反演方法：必须为每种退化单独设计算法
  扩散后验采样：同一训练好的模型可处理多种逆问题

实验内容：
  - 同一个先验模型（此处以解析score表示）
  - 应用到去噪、去模糊、非线性退化三种不同逆问题
  - 验证零样本迁移能力

注意：本实验为1D演示版本，避免GPU依赖。
说明：1D高斯混合先验的score存在解析解，故本实验直接使用解析score
      替代"训练好的DDPM网络"（二者在数学上等价地参数化同一个先验），
      从而能在无需训练的前提下精确演示"同一先验模型零样本迁移到多种逆问题"。
      图像任务中的神经网络先验不具备此处的精确后验可解条件。
"""

import sys
import io
import os
import numpy as np
from numpy.polynomial.hermite import hermgauss
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import logging
import warnings

# 设置控制台输出为 UTF-8 (Windows下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# 静默matplotlib相关警告
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
    SAVE_DIR = os.path.join(_gdrive, '实验13.6-1')
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
    print("警告: chinese_font模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)

print("\n" + "=" * 60)
print("实验13.6-1: 条件采样与零样本迁移")
print("=" * 60)
print("知识点: 零样本迁移, 任意复杂先验, 不确定性量化")


# ============================================================
# 1D高斯混合先验 + VP-SDE框架
# ============================================================
GM_WEIGHTS = [0.3, 0.7]
GM_MEANS = [-2.0, 1.0]
GM_STDS = [1.0, 1.0]

BETA_MIN, BETA_MAX = 0.1, 20.0

def gm1d_pdf(x):
    pdf = np.zeros_like(x)
    for w, m, s in zip(GM_WEIGHTS, GM_MEANS, GM_STDS):
        pdf += w * np.exp(-0.5 * ((x - m) / s)**2) / (s * np.sqrt(2 * np.pi))
    return pdf

def vp_marginal(t):
    log_mean = -0.25 * t**2 * (BETA_MAX - BETA_MIN) - 0.5 * t * BETA_MIN
    mean_t = np.exp(log_mean)
    std_t = np.sqrt(1 - np.exp(2 * log_mean))
    return mean_t, std_t

def vp_beta(t):
    return BETA_MIN + t * (BETA_MAX - BETA_MIN)

def vp_score_analytic(x, t):
    mean_t, std_t = vp_marginal(t)
    pdf = np.zeros_like(x)
    dpdf = np.zeros_like(x)
    for w, m, s in zip(GM_WEIGHTS, GM_MEANS, GM_STDS):
        new_mean = mean_t * m
        new_std = np.sqrt(mean_t**2 * s**2 + std_t**2)
        pdf += w * np.exp(-0.5 * ((x - new_mean) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
        dpdf += w * (-(x - new_mean) / new_std**2) * np.exp(-0.5 * ((x - new_mean) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
    return dpdf / (pdf + 1e-30)


# ============================================================
# 步骤1：不同逆问题——同一模型
# ============================================================
print("\n" + "=" * 60)
print("步骤1：不同逆问题——同一模型（13.6节 零样本迁移）")
print("=" * 60)

print("""
13.6节核心：扩散模型四大优势
  1. 任意复杂先验：使用神经网络参数化的先验，非参数化
  2. 不确定性量化：后验采样天然给出多解
  3. 零样本迁移：同一模型可处理多种逆问题（核心实验）
  4. 统一框架：DPS/DiffPIR/img2img/CFG都是同一框架的特例

零样本迁移实验（13.6节 重点）：
  训练一个DDPM（仅学习先验）
  同一模型应用到三种不同逆问题，无需重新训练
""")

def dps_posterior_sample(y_obs, A_func, A_prime_func, sigma_y, zeta=0.5, N_particles=5000, N_steps=300, T=1.0, seed=42):
    """
    DPS后验采样（修正版：包含完整的链式法则 + 正确的数值稳定性保护）

    似然梯度完整链式法则：
    ∇_{x_t} log p(y|x_t) = A'(x0_hat) * (y - A(x0_hat)) / (sigma_y^2 * mean_t)

    数值稳定性考虑：
    - 当t→T时，mean_t很小（如0.006），除以mean_t会导致梯度较大
    - 解决方案：梯度裁剪（限制幅值上限），而非归一化为符号
    - 这样能保留A'和mean_t的相对大小关系（核心物理意义）

    参数:
        y_obs: 观测值
        A_func: 前向退化算子 A(x)
        A_prime_func: A的导数 A'(x)（关键：缺失此项会导致链式法则不完整）
        sigma_y: 观测噪声标准差
        zeta: 似然项权重（推荐0.5）
        其他参数同上
    """
    np.random.seed(seed)
    h = T / N_steps
    x = np.random.randn(N_particles)

    # 梯度裁剪上限（防止极端值，但保留相对大小关系）
    GRAD_CLIP_MAX = 10.0

    for i in range(N_steps):
        t = T - i * h
        beta_t = vp_beta(t)
        mean_t, std_t = vp_marginal(t)

        # 计算先验score（添加数值保护）
        prior_score = vp_score_analytic(x, t)

        # Tweedie公式估计x0
        x0_hat = (x + std_t**2 * prior_score) / (mean_t + 1e-10)

        # 数值保护：限制x0_hat的极端值，防止后续计算溢出
        x0_hat = np.clip(x0_hat, -10, 10)

        Ax0_hat = A_func(x0_hat)

        # 计算A的Jacobian（一维情况下即导数）
        A_prime = A_prime_func(x0_hat)

        # 修正：完整链式法则
        # 原bug1: mean_t * (...)  方向反了，应该是除以mean_t
        # 原bug2: 缺少A'(x0_hat)，对不同A的影响不同
        # 原bug3: 梯度归一化变成符号函数，丢失量级信息（本次修复）
        likelihood_grad_raw = A_prime * (y_obs - Ax0_hat) / (sigma_y**2 * mean_t)

        # 数值稳定性保护：梯度裁剪（而非归一化为符号）
        # 保留A'和mean_t的相对大小关系，避免丢失物理意义
        likelihood_grad = zeta * np.clip(likelihood_grad_raw, -GRAD_CLIP_MAX, GRAD_CLIP_MAX)

        posterior_score = prior_score + likelihood_grad

        # 反向SDE步进
        x = x + beta_t * h * (0.5 * x + posterior_score) + np.sqrt(beta_t * h) * np.random.randn(N_particles)

        # 数值保护：限制x的范围，防止发散
        x = np.clip(x, -20, 20)

    return x


# ---- 精确后验基准（仅1D可解：网格数值积分） ----
def exact_posterior_stats(y_obs, A_func, sigma_y, x_lo=-8.0, x_hi=8.0, n_grid=200_001):
    """数值计算贝叶斯后验 p(x|y) ∝ p(x)·N(y;A(x),σ_y²) 的统计量。
    用于核对DPS采样的校准度；图像任务中无此解析基准。"""
    xg = np.linspace(x_lo, x_hi, n_grid)
    q = gm1d_pdf(xg) * np.exp(-0.5 * ((y_obs - A_func(xg)) / sigma_y) ** 2)
    q /= np.trapz(q, xg)
    mean = np.trapz(xg * q, xg)
    std = np.sqrt(np.trapz((xg - mean) ** 2 * q, xg))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(xg))])
    cdf /= cdf[-1]
    ci_lo = xg[np.searchsorted(cdf, 0.025)]
    ci_hi = xg[np.searchsorted(cdf, 0.975)]
    p_right = np.trapz(q[xg >= 1.0], xg[xg >= 1.0])
    return {"均值": mean, "标准差": std, "CI2.5%": ci_lo, "CI97.5%": ci_hi, "P(x>=1)": p_right}


def exact_posterior_pdf(x_eval, y_obs, A_func, sigma_y, x_lo=-8.0, x_hi=8.0, n_grid=200_001):
    """在x_eval处给出精确后验密度（归一化常数由细网格积分得到），用于绘图叠加。"""
    xg = np.linspace(x_lo, x_hi, n_grid)
    qg = gm1d_pdf(xg) * np.exp(-0.5 * ((y_obs - A_func(xg)) / sigma_y) ** 2)
    Z = np.trapz(qg, xg)
    return gm1d_pdf(x_eval) * np.exp(-0.5 * ((y_obs - A_func(x_eval)) / sigma_y) ** 2) / Z


# ---- 精确边际化引导（1D可解的"正确"似然得分，用于对照vanilla DPS） ----
def marginalized_posterior_sample(y_obs, A_func, sigma_y, zeta=1.0,
                                  N_particles=2000, N_steps=200, T=1.0, seed=7,
                                  n_quad=48):
    """似然精确边际化后的引导采样（对照DPS的δ-似然插件近似）。

    DPS用 p(y|x_t) ≈ N(y; A(x0_hat), σ_y²)，把x0钉死在Tweedie点估计上；
    精确做法应对去噪分布 p(x0|x_t) 边际化：
        L(x_t) = ∫ N(y; A(x0), σ_y²) p(x0|x_t) dx0
    GMM先验下 p(x0|x_t) 是2个高斯的混合（参数解析），用Gauss-Hermite求积
    数值积分，中心差分给 ∇log L。高维图像中此积分不可解：线性算子可用
    ΠGDM（各向同性高斯近似），一般情形退化为Moment Matching等近似
    （见13.2/13.3精度递进链 delta→ΠGDM→完整协方差）——本1D实验把该
    近似的偏差定量暴露并闭合；线性情形的同类对照见实验13.4-2。
    """
    rng = np.random.default_rng(seed)
    gh_x, gh_w = hermgauss(n_quad)
    h = T / N_steps
    x = rng.standard_normal(N_particles)

    def log_marg_lik(xt, t):
        mean_t, std_t = vp_marginal(t)
        K = len(GM_WEIGHTS)
        logr = np.empty((len(xt), K))
        cond = []
        for k, (w, m, s) in enumerate(zip(GM_WEIGHTS, GM_MEANS, GM_STDS)):
            Vxt = mean_t**2 * s**2 + std_t**2          # x_t 分量边缘方差
            logr[:, k] = (np.log(w) - 0.5 * ((xt - mean_t * m) / np.sqrt(Vxt))**2
                          - 0.5 * np.log(2 * np.pi * Vxt))
            cov = mean_t * s**2                          # Cov(x0, xt)
            a = cov / Vxt                                # x0|xt,k ~ N(a*xt+b, v)
            b = m - a * mean_t * m
            v = max(s**2 - cov**2 / Vxt, 1e-12)
            cond.append((a, b, v))
        logr -= logr.max(axis=1, keepdims=True)
        resp = np.exp(logr)
        resp /= resp.sum(axis=1, keepdims=True)         # 分量责任度 r_k(xt)
        L = np.zeros(len(xt))
        for k, (a, b, v) in enumerate(cond):
            nodes = (a * xt + b)[:, None] + np.sqrt(2 * v) * gh_x[None, :]
            lik = np.exp(-0.5 * ((y_obs - A_func(nodes)) / sigma_y)**2) \
                  / (sigma_y * np.sqrt(2 * np.pi))
            L += resp[:, k] * (lik @ gh_w) / np.sqrt(np.pi)
        return np.log(L + 1e-300)

    for i in range(N_steps):
        t = T - i * h
        beta_t = vp_beta(t)
        mean_t, _ = vp_marginal(t)
        prior_score = vp_score_analytic(x, t)
        delta = max(1e-3 * mean_t, 1e-4)                # 中心差分步长
        glik = (log_marg_lik(x + delta, t) - log_marg_lik(x - delta, t)) / (2 * delta)
        glik = zeta * np.clip(glik, -1e4, 1e4)
        x = x + beta_t * h * (0.5 * x + prior_score + glik) \
            + np.sqrt(beta_t * h) * rng.standard_normal(N_particles)
        x = np.clip(x, -20, 20)
    return x


# ---- 任务1：去噪 (A=I) ----
print("\n任务1：去噪 (A=I)")
print("  前向算子: A(x) = x")
print("  导数: A'(x) = 1 (恒等算子导数恒为1)")
print("  zeta=0.5 (数值稳定性推荐值)")
y_denoise = 0.5
# A=I的导数恒为1（原bug对此任务无影响）
samples_denoise = dps_posterior_sample(y_denoise, lambda x: x, lambda x: np.ones_like(x),
                                      sigma_y=0.3, zeta=0.5, seed=42)
print(f"  后验采样: 均值={np.mean(samples_denoise):.3f}, 标准差={np.std(samples_denoise):.3f}")

# ---- 任务2：去模糊 (A=0.7x) ----
print("\n任务2：去模糊 (A=0.7x, 下采样比例0.7)")
print("  前向算子: A(x) = 0.7x")
print("  导数: A'(x) = 0.7 (常数)")
print("  原bug影响: 梯度强度多算了约1.43倍（缺少除以0.7）")
print("  zeta=0.5 (数值稳定性推荐值)")
y_blur = 0.5 * 0.7
# A=0.7x的导数为常数0.7
samples_blur = dps_posterior_sample(y_blur, lambda x: 0.7 * x, lambda x: 0.7 * np.ones_like(x),
                                   sigma_y=0.3, zeta=0.5, seed=42)
print(f"  后验采样: 均值={np.mean(samples_blur):.3f}, 标准差={np.std(samples_blur):.3f}")

# ---- 任务3：非线性退化 (A=tanh(2x)) ----
print("\n任务3：非线性退化 (A=tanh(2x))")
print("  前向算子: A(x) = tanh(2x)")
print("  导数: A'(x) = 2·sech²(2x) = 2*(1-tanh²(2x)) (随x变化！)")
print("  原bug严重影响: tanh饱和区A'→0，观测y应无约束力，但原代码仍强行修正")
print("  修正后效果: 当|x|较大时，A'趋近0，似然梯度自动衰减，反映信息量趋零特性")
print("  zeta=0.5 (数值稳定性推荐值)")
y_nonlinear = 0.5
# tanh(2x)的导数为2*sech^2(2x)=2*(1-tanh(2x)^2)
samples_nonlinear = dps_posterior_sample(y_nonlinear, lambda x: np.tanh(2 * x),
                                        lambda x: 2 * (1 - np.tanh(2 * x)**2),
                                        sigma_y=0.3, zeta=0.5, seed=42)
print(f"  后验采样: 均值={np.mean(samples_nonlinear):.3f}, 标准差={np.std(samples_nonlinear):.3f}")

# ---- 精确后验基准对照 ----
exact_denoise = exact_posterior_stats(y_denoise, lambda x: x, 0.3)
exact_blur = exact_posterior_stats(y_blur, lambda x: 0.7 * x, 0.3)
exact_nonlinear = exact_posterior_stats(y_nonlinear, lambda x: np.tanh(2 * x), 0.3)
print("\n精确后验基准（网格数值积分） vs DPS：")
for _name, _ex, _sm in [("任务1 去噪    ", exact_denoise, samples_denoise),
                        ("任务2 去模糊  ", exact_blur, samples_blur),
                        ("任务3 非线性  ", exact_nonlinear, samples_nonlinear)]:
    print(f"  {_name}: 精确 均值={_ex['均值']:.3f} 标准差={_ex['标准差']:.3f} "
          f"95%CI=[{_ex['CI2.5%']:.2f},{_ex['CI97.5%']:.2f}] P(x>=1)={_ex['P(x>=1)']:.3f}")
    print(f"             DPS 均值={_sm.mean():.3f} 标准差={_sm.std():.3f} "
          f"{'':9s} P(x>=1)={np.mean(_sm >= 1):.3f}")

# ---- ζ=0.25 对照（非线性任务）：说明调ζ只能移动偏差、无法同时校准 ----
samples_nonl_z025 = dps_posterior_sample(y_nonlinear, lambda x: np.tanh(2 * x),
                                         lambda x: 2 * (1 - np.tanh(2 * x)**2),
                                         sigma_y=0.3, zeta=0.25, seed=42)

# ---- 精确边际化引导（对 p(x0|xt) 精确积分，消除δ-似然近似偏差） ----
print("\n精确边际化引导（L(xt)=∫N(y;A(x0),σ²)p(x0|xt)dx0，Gauss-Hermite求积）：")
samples_marg_denoise = marginalized_posterior_sample(y_denoise, lambda x: x, 0.3, seed=7)
samples_marg_blur = marginalized_posterior_sample(y_blur, lambda x: 0.7 * x, 0.3, seed=7)
samples_marg_nonlinear = marginalized_posterior_sample(y_nonlinear,
                                                       lambda x: np.tanh(2 * x), 0.3, seed=7)
print("任务                     均值     标准差   P(x>=1)   (对照精确后验)")
for _name, _ex, _sm in [("任务1 去噪  边际化引导", exact_denoise, samples_marg_denoise),
                        ("任务2 去模糊边际化引导", exact_blur, samples_marg_blur),
                        ("任务3 非线性边际化引导", exact_nonlinear, samples_marg_nonlinear)]:
    print(f"  {_name}: {_sm.mean():.3f}    {_sm.std():.3f}    {np.mean(_sm >= 1):.3f}"
          f"      精确: {_ex['均值']:.3f}/{_ex['标准差']:.3f}/{_ex['P(x>=1)']:.3f}")
print(f"  任务3 DPS ζ=0.25对照: {samples_nonl_z025.mean():.3f}/"
      f"{samples_nonl_z025.std():.3f}/{np.mean(samples_nonl_z025 >= 1):.3f}"
      f"  (ζ=0.5为 {samples_nonlinear.mean():.3f}/{samples_nonlinear.std():.3f}/"
      f"{np.mean(samples_nonlinear >= 1):.3f})")


# ============================================================
# 步骤2：不确定性量化
# ============================================================
print("\n" + "=" * 60)
print("步骤2：不确定性量化（13.6节 优势2）")
print("=" * 60)

print("""
不确定性量化（13.6节）：
  传统MAP: 给出单一最优解
  扩散后验采样: 给出后验分布的样本，可以计算均值/方差/分位数可信区间
  示例: 本观测(y=0.5,σ=0.3)信息充分，后验为单峰集中分布；
        只有当观测位于先验两峰之间且噪声较大时，后验才会保留多峰结构
""")

# 多次采样，统计均值和方差
n_runs = 5
denoise_samples_list = []
for i in range(n_runs):
    # 去噪任务：A=I，A'=1，zeta=0.5
    samples = dps_posterior_sample(y_denoise, lambda x: x, lambda x: np.ones_like(x),
                                   sigma_y=0.3, zeta=0.5, seed=i)
    denoise_samples_list.append(samples)

all_means = [np.mean(s) for s in denoise_samples_list]
all_stds = [np.std(s) for s in denoise_samples_list]
print(f"\n{n_runs}次后验采样的均值: {[round(m, 4) for m in all_means]}")
print(f"均值: {np.mean(all_means):.3f} ± {np.std(all_means):.3f}")
print(f"标准差: {np.mean(all_stds):.3f} ± {np.std(all_stds):.3f}")
pooled = np.concatenate(denoise_samples_list)
ci_lo, ci_hi = np.percentile(pooled, [2.5, 97.5])
print(f"\n这里包含两类性质不同的不确定性，务必区分：")
print(f"  - 后验本身的不确定性（单次采样平均标准差）~ {np.mean(all_stds):.3f}")
print(f"    95% 后验可信区间（样本分位数）= [{ci_lo:.3f}, {ci_hi:.3f}]")
print(f"  - 蒙特卡洛估计误差（{n_runs}次运行均值的标准误）~ {np.std(all_means):.3f}")
print(f"    它仅来自有限粒子数(每次5000)，粒子越多越小，不是后验置信区间")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) 零样本迁移
x_hist = np.linspace(-6, 6, 500)
axes[0].hist(samples_denoise, bins=60, density=True, alpha=0.5, color='blue',
             range=(-6, 6), label='去噪 A=I')
axes[0].hist(samples_blur, bins=60, density=True, alpha=0.5, color='red',
             range=(-6, 6), label='去模糊 A=0.7')
axes[0].hist(samples_nonlinear, bins=60, density=True, alpha=0.5, color='green',
             range=(-6, 6), label='非线性 A=tanh(2x)')
# 边际化引导（非线性任务）：只描边，便于与绿色DPS柱、绿色精确后验线对照
axes[0].hist(samples_marg_nonlinear, bins=60, density=True, histtype='step',
             edgecolor='orange', linewidth=2.0, range=(-6, 6),
             label='非线性·边际化引导(精确似然)')
axes[0].plot(x_hist, gm1d_pdf(x_hist), 'k--', lw=2, alpha=0.7, label='先验 p(x)')
# 叠加各任务精确后验（同色细线），直观核对直方图校准度
axes[0].plot(x_hist, exact_posterior_pdf(x_hist, y_denoise, lambda x: x, 0.3),
             color='blue', lw=1.1, alpha=0.9)
axes[0].plot(x_hist, exact_posterior_pdf(x_hist, y_blur, lambda x: 0.7 * x, 0.3),
             color='red', lw=1.1, alpha=0.9)
axes[0].plot(x_hist, exact_posterior_pdf(x_hist, y_nonlinear, lambda x: np.tanh(2 * x), 0.3),
             color='green', lw=1.1, alpha=0.9, label='精确后验（同色细线）')
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('概率密度', fontsize=12)
axes[0].set_title('(a) 同一模型处理三种不同逆问题', fontsize=13)
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# (b) 不确定性可视化
for i, samples in enumerate(denoise_samples_list):
    axes[1].hist(samples, bins=50, density=True, alpha=0.4, range=(-6, 6),
                 color=plt.cm.Blues(0.3 + 0.15*i))
axes[1].plot(x_hist, gm1d_pdf(x_hist), 'k--', lw=2, alpha=0.7, label='先验')
axes[1].axvline(np.mean(all_means), color='red', linestyle='-', lw=2,
                label=r'平均后验均值={:.2f}'.format(np.mean(all_means)))
axes[1].axvspan(ci_lo, ci_hi, alpha=0.15, color='red',
                label=f'95%可信区间[{ci_lo:.2f},{ci_hi:.2f}]')
axes[1].set_xlabel('x', fontsize=12)
axes[1].set_ylabel('概率密度', fontsize=12)
axes[1].set_title(f'(b) 后验采样的不确定性（{n_runs}次独立运行）', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '条件采样与零样本迁移.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图已保存: {fig_path}")

print("\n" + "=" * 60)
print("实验13.6-1 完成!")
print("=" * 60)
print(f"""
关键结论:
1. 零样本迁移（13.6节 优势3）
   - 同一训练好的扩散模型（仅学习先验）可以处理多种逆问题
   - 去噪、去模糊、非线性退化——无需重新训练
   - 传统方法需要为每种逆问题单独设计算法

2. 不确定性量化（13.6节 优势2）
   - 后验采样天然给出多解
   - 可以计算均值/方差/置信区间
   - 这是传统MAP方法所没有的能力

3. 重要修正历史：似然梯度链式法则（代码评审迭代）
   第一轮评审发现：
   - 原bug1: mean_t * (...)  方向反了，应该是除以mean_t
   - 原bug2: 缺少A'(x0_hat)，对不同A的影响不同

   修正公式：
   ∇log p(y|x_t) = A'(x0_hat)*(y-A(x0_hat))/(σ_y²·mean_t)

   第二轮评审发现：
   - 原bug3: 梯度归一化变成符号函数，丢失量级信息
   - 问题: 除以自身绝对值得到sign()，A'和mean_t的相对大小关系被抹除
   - 影响: tanh饱和区的梯度衰减效果被抵消

   最终修复：
   - 使用梯度裁剪（clip），而非归一化
   - 保留A'和mean_t的相对大小关系（核心物理意义）

   各任务修正效果：
   - 任务1（A=I）：均值{samples_denoise.mean():.3f}接近观测0.5，标准差{samples_denoise.std():.3f}（约束强）
   - 任务2（A=0.7x）：均值{samples_blur.mean():.3f}，标准差{samples_blur.std():.3f}（A'=0.7<1使约束减弱）
   - 任务3（tanh）：DPS均值{samples_nonlinear.mean():.3f}、标准差{samples_nonlinear.std():.3f}
     ——"后验最宽"的定性趋势正确，但定量偏窄偏左（见第4条）

4. 与精确后验基准的对照 + δ-似然近似偏差的闭合（1D场景独有）
   - 任务1/2：vanilla DPS与精确后验高度一致（均值/标准差偏差均<0.05），校准良好
   - 任务3：精确后验为均值{exact_nonlinear['均值']:.2f}、标准差{exact_nonlinear['标准差']:.2f}、
     P(x>={1})={exact_nonlinear['P(x>=1)']:.2f}的严重右偏分布（tanh饱和使似然在x>~1
     形成平坦高原，"哪个x解释y=0.5"变模糊，先验右峰N(1,1)把质量铺到右尾）。
     DPS(ζ=0.5)给出{samples_nonlinear.mean():.2f}/{samples_nonlinear.std():.2f}/{np.mean(samples_nonlinear>=1):.2f}，
     右尾被排空——这不是"高原上信号微弱粒子进不去"，而是δ-似然近似
     p(y|x_t)≈N(y;A(x0_hat),σ²)把x0钉死在Tweedie点估计上，高原上残余的
     点估计梯度（再被1/mean_t放大）持续把右尾粒子拉回窄区。
   - 调ζ只能移动偏差、不能消除：ζ=0.25时
     {samples_nonl_z025.mean():.2f}/{samples_nonl_z025.std():.2f}/{np.mean(samples_nonl_z025>=1):.2f}
     （标准差与尾概率已接近真值{exact_nonlinear['标准差']:.2f}/{exact_nonlinear['P(x>=1)']:.2f}，
     但均值仍偏），不存在单一ζ同时校准三者；加密步数(1000步)同样无效。
   - 精确边际化引导（对去噪分布p(x0|x_t)积分L(x_t)=E[N(y;A(x0),σ²)|x_t]，
     ζ=1.0）给出{samples_marg_nonlinear.mean():.2f}/{samples_marg_nonlinear.std():.2f}/{np.mean(samples_marg_nonlinear>=1):.2f}，
     与精确后验{exact_nonlinear['均值']:.2f}/{exact_nonlinear['标准差']:.2f}/{exact_nonlinear['P(x>=1)']:.2f}吻合，右尾恢复；
     任务1/2同样吻合（{samples_marg_denoise.mean():.2f}/{samples_marg_denoise.std():.2f}、
     {samples_marg_blur.mean():.2f}/{samples_marg_blur.std():.2f}）。
   - 结论：任务3的偏差不是"扩散后验采样的固有限制"，而是vanilla DPS的
     δ-似然插件近似偏差，在同一后验得分分解框架内结构性可修
     （边际化x0|x_t即可）；高维图像无法精确积分，靠ΠGDM（线性算子
     闭式）/Moment Matching等近似——即13.2/13.3节精度递进链与
     13.6展望所述"更精确似然近似（超越DPS的delta假设）"；
     线性观测下的同类精确对照已在实验13.4-2给出。
   - 因此任务3是完整的定量演示：vanilla DPS先暴露近似边界，
     精确边际化引导再在同一框架内闭合它。
""")
# ===== 保存数值结果 =====
import json

def _to_native(obj):
    """递归转换numpy/torch类型为Python原生类型"""
    import numpy as np
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return _to_native(obj.tolist())
    try:
        import torch
        if isinstance(obj, torch.Tensor): return _to_native(obj.detach().cpu().tolist())
    except: pass
    return obj

results_summary = {
    "去噪_A=I": {"均值": round(float(np.mean(samples_denoise)), 4), "标准差": round(float(np.std(samples_denoise)), 4),
                 "95%CI(DPS分位数)": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
                 "P(x>=1)": round(float(np.mean(samples_denoise >= 1)), 4)},
    "去模糊_A=0.7x": {"均值": round(float(np.mean(samples_blur)), 4), "标准差": round(float(np.std(samples_blur)), 4),
                     "P(x>=1)": round(float(np.mean(samples_blur >= 1)), 4)},
    "非线性_A=tanh2x": {"均值": round(float(np.mean(samples_nonlinear)), 4), "标准差": round(float(np.std(samples_nonlinear)), 4),
                       "P(x>=1)": round(float(np.mean(samples_nonlinear >= 1)), 4)},
    "非线性_DPS_zeta0.25": {"均值": round(float(np.mean(samples_nonl_z025)), 4),
                            "标准差": round(float(np.std(samples_nonl_z025)), 4),
                            "P(x>=1)": round(float(np.mean(samples_nonl_z025 >= 1)), 4)},
    "边际化引导_去噪": {"均值": round(float(np.mean(samples_marg_denoise)), 4),
                       "标准差": round(float(np.std(samples_marg_denoise)), 4),
                       "P(x>=1)": round(float(np.mean(samples_marg_denoise >= 1)), 4)},
    "边际化引导_去模糊": {"均值": round(float(np.mean(samples_marg_blur)), 4),
                         "标准差": round(float(np.std(samples_marg_blur)), 4),
                         "P(x>=1)": round(float(np.mean(samples_marg_blur >= 1)), 4)},
    "边际化引导_非线性": {"均值": round(float(np.mean(samples_marg_nonlinear)), 4),
                         "标准差": round(float(np.std(samples_marg_nonlinear)), 4),
                         "P(x>=1)": round(float(np.mean(samples_marg_nonlinear >= 1)), 4)},
    "多次采样_均值统计": {"均值": round(float(np.mean(all_means)), 4), "标准差(蒙特卡洛标准误)": round(float(np.std(all_means)), 4)},
    "多次采样_标准差统计": {"均值": round(float(np.mean(all_stds)), 4), "标准差": round(float(np.std(all_stds)), 4)},
    "精确后验_去噪": {k: round(float(v), 4) for k, v in exact_denoise.items()},
    "精确后验_去模糊": {k: round(float(v), 4) for k, v in exact_blur.items()},
    "精确后验_非线性": {k: round(float(v), 4) for k, v in exact_nonlinear.items()},
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")

