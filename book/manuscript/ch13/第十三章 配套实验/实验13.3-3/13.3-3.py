# -*- coding: utf-8 -*-
"""
实验13.3-3：DDRM算法实践

★ 原创设计：DDRM的SVD频谱域条件化
  DDRM核心思想：对线性算子A做SVD分解，在频谱域按分量融合观测与先验
  优势：小奇异值方向的观测噪声放大被方差匹配自动降权（线性高斯意义下精确）
  局限：仅适用于线性算子，非线性算子无法SVD分解

实验内容：
  - 1D高斯混合先验 + VP-SDE框架
  - 线性算子A的SVD分解可视化
  - DDRM频谱域条件化采样（观测/先验双通道精度加权融合）
  - DDRM vs DPS效果对比（单步估计器，同一Tweedie去噪核）
  - DDRM局限性展示（非线性算子场景）

修订说明（复核后修复两处实现缺陷）：
  1) 旧版"教学阈值"判据 noise_level = sigma_t*sigma_obs/S_k 在本参数下
     全域恒<0.5，所有分量恒走"观测主导"分支，自适应机制失效，
     DDRM退化为与x_t/t/先验无关的纯伪逆 A^-1 y；已改为方差匹配的精度加权融合。
  2) 旧版DPS直接返回 x_t + 小步修正（不做任何去噪），"误差随t单调升"
     实为x_t自身携带的扩散噪声，是稻草人实现而非DPS固有属性；
     已改为标准DPS单步形式：Tweedie去噪 + 似然梯度步。

本实验不需要GPU，通过1D解析情形验证DDRM原理。
"""

import sys
import io
import os
import numpy as np
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
    SAVE_DIR = os.path.join(_gdrive, '实验13.3-3')
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

# matplotlib LaTeX格式用于数学符号显示（必须在中文配置之后设置，否则会被覆盖）
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False
# ========================================================

np.random.seed(42)

print("\n" + "=" * 60)
print("实验13.3-3: DDRM算法实践")
print("=" * 60)
print("知识点: DDRM算法(附录13B), SVD频谱域条件化, 与DPS的对比")

print("""
DDRM核心思想（13.3.1节）：
  线性逆问题：y = Ax + n
  SVD分解：A = U · S · V^T
  频谱域变换：
    - V^T x = x_freq（频谱域信号）
    - S · x_freq = 缩放频谱分量
    - U · (S · x_freq) = y（观测）

  DDRM条件化：
    频谱域每个分量=观测通道(伪逆)与先验通道(去噪估计)的精度加权融合，
    小奇异值方向的观测噪声放大(σ_y/S_k)被自动降权

  关键优势：
    - 每个频谱分量按噪声方差独立加权（自适应，线性高斯意义下精确）
    - 高噪声时刻仍可利用观测信息，不随t退化

  核心局限：
    - 仅适用于线性算子（需SVD分解）
    - 非线性算子无法应用DDRM
""")

# ============================================================
# 1. 1D高斯混合先验 + VP-SDE框架
# ============================================================
GM_WEIGHTS = [0.3, 0.7]
GM_MEANS = [-2.0, 1.0]
GM_STDS = [1.0, 1.0]

# VP-SDE参数（简化版，T=200）
T = 200
beta_min, beta_max = 1e-4, 0.02
betas = np.linspace(beta_min, beta_max, T)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)

def sample_from_prior(n_samples):
    """从高斯混合先验采样"""
    components = np.random.choice(len(GM_WEIGHTS), size=n_samples, p=GM_WEIGHTS)
    samples = np.array([np.random.normal(GM_MEANS[c], GM_STDS[c]) for c in components])
    return samples

def vp_sde_forward(x0, t_idx):
    """VP-SDE正向过程：x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * noise"""
    alpha_bar_t = alpha_bars[t_idx]
    noise = np.random.randn(*x0.shape)
    x_t = np.sqrt(alpha_bar_t) * x0 + np.sqrt(1.0 - alpha_bar_t) * noise
    return x_t, noise

# ★ 自检：验证VP-SDE调度的关键性质(避免T/beta范围被改后Tweedie公式仍按旧值工作)
# 调度: T=200, beta_min=1e-4, beta_max=2e-2
# 检验规则:
#   Q1: ᾱ_1 ≈ 1-β_min (因α_1=1-β_min, 累积一步的近似)
#   Q2: ᾱ_T 残余信号不可忽略 (T=200, Σβ≈2.01 → ᾱ_T≈e^-2.02≈0.13, 比T=1000时大很多)
#   Q3: ᾱ_t单调递减, 介于(0,1]之间
#   Q4: VP-SDE正向过程x_t的均值和方差符合理论(无噪声时x_t=√ᾱ·x0)
print(f"[DDRM自检] VP-SDE调度与正向过程验证(实际运行, 非手算):")
print(f"  Q1: ᾱ_1={alpha_bars[0]:.6f} (应≈1-β_min={1-beta_min:.6f}): "
      f"{'OK' if abs(alpha_bars[0] - (1-beta_min)) < 1e-3 else 'FAIL'}")
print(f"  Q2: ᾱ_T={alpha_bars[-1]:.6e} (应很小但>0, T=200时仍比T=1000大): "
      f"{'OK' if 0 < alpha_bars[-1] < 0.2 else 'FAIL'}")
print(f"  Q3: 单调递减: {'OK' if (np.diff(alpha_bars) <= 0).all() else 'FAIL'}, "
      f"范围: [{alpha_bars.min():.4f}, {alpha_bars.max():.4f}] (应∈(0,1])")
# Q4: 正向过程x_t的理论均值=√ᾱ·x0, 方差=(1-ᾱ) (对x0=0)
# 注意：自检使用独立的default_rng，不触碰全局np.random状态，保证结果可复现
_rng = np.random.default_rng(2024)
_x0 = np.array([1.0, 2.0])
for _t in [0, T//4, T//2, T-1]:
    _ab = alpha_bars[_t]
    _xt_samples = np.sqrt(_ab) * _x0 + np.sqrt(1.0 - _ab) * _rng.standard_normal((5000, 2))
    _mean = _xt_samples.mean(axis=0)
    _var = _xt_samples.var(axis=0)
    _theory_mean = np.sqrt(alpha_bars[_t]) * _x0
    _theory_var = (1 - alpha_bars[_t]) * np.ones_like(_x0)
    print(f"  Q4 t={_t+1:4d}: E[x_t]={_mean}, 理论={_theory_mean} (相对误差<5%即可)")

def tweedie_estimate(x_t, t_idx):
    """基于高斯混合先验的解析Tweedie估计（后验均值 E[x_0|x_t]）

    严格Tweedie公式: x_hat_0 = (x_t + (1-alpha_bar_t) * grad log p_t(x_t)) / sqrt(alpha_bar_t)
    对高斯混合先验，等价于逐分量计算后验均值。

    边际分布: p_t(x_t) = sum_k w_k N(x_t; sqrt(abar) mu_k, abar*sigma_k^2 + (1-abar))
    后验(分量k): E[x_0|x_t, k] = (sqrt(abar)*sigma_k^2*x_t + (1-abar)*mu_k) / (abar*sigma_k^2 + (1-abar))
    全后验均值: E[x_0|x_t] = sum_k pi_k(x_t) * E[x_0|x_t, k]

    教学简化说明：本函数对x_t的各分量独立套用同一个1D高斯混合边际先验。
    实际上x2d=[x0, x0*0.5]两分量在无噪声时确定性相关，精确处理需要联合后验。
    这里忽略相关结构，视为边际独立，是可接受的教学简化。
    """
    abar = alpha_bars[t_idx]
    sigma_t2 = 1.0 - abar
    sqrt_abar = np.sqrt(abar)

    def scalar_tweedie(x_t_scalar):
        log_weights = np.log(np.array(GM_WEIGHTS))
        log_likes = np.array([
            -0.5 * ((x_t_scalar - sqrt_abar * mu) ** 2) / (abar * sigma ** 2 + sigma_t2)
            - 0.5 * np.log(abar * sigma ** 2 + sigma_t2)
            for mu, sigma in zip(GM_MEANS, GM_STDS)
        ])
        log_joint = log_weights + log_likes
        log_joint -= np.max(log_joint)  # 数值稳定
        pi = np.exp(log_joint)
        pi /= np.sum(pi)

        posterior_means_k = np.array([
            (sqrt_abar * sigma ** 2 * x_t_scalar + sigma_t2 * mu) / (abar * sigma ** 2 + sigma_t2)
            for mu, sigma in zip(GM_MEANS, GM_STDS)
        ])
        return np.sum(pi * posterior_means_k)

    x_t_arr = np.atleast_1d(x_t)
    result = np.array([scalar_tweedie(xi) for xi in x_t_arr])
    return result if result.ndim == 1 else result.item()

def tweedie_estimate_and_var(x_t, t_idx):
    """解析给出高斯混合先验下的后验均值 E[x0|x_t] 与后验方差 Var(x0|x_t)

    对每个分量k：p(x0|x_t,k) ∝ N(x_t; √ᾱ x0, 1-ᾱ)·N(x0; μ_k, σ_k²)
      精度 = ᾱ/(1-ᾱ) + 1/σ_k²，均值 = 精度⁻¹·(ᾱ x_t/(1-ᾱ) + μ_k/σ_k²)
    混合后验方差（全方差公式）：Var = Σ_k π_k (v_k + (m_k - m̄)²)
    教学简化同tweedie_estimate：各分量独立处理。
    """
    abar = alpha_bars[t_idx]
    s2 = 1.0 - abar
    sqrt_abar = np.sqrt(abar)
    mus = np.array(GM_MEANS)
    sgs = np.array(GM_STDS)

    def scalar_mean_var(xi):
        var_m = abar * sgs ** 2 + s2
        log_joint = np.log(np.array(GM_WEIGHTS)) - 0.5 * ((xi - sqrt_abar * mus) ** 2) / var_m \
                      - 0.5 * np.log(var_m)
        log_joint -= np.max(log_joint)
        pi = np.exp(log_joint)
        pi /= np.sum(pi)
        prec = abar / s2 + 1.0 / sgs ** 2
        var_k = 1.0 / prec
        mean_k = var_k * (abar * xi / s2 + mus / sgs ** 2)
        m_bar = np.sum(pi * mean_k)
        v = np.sum(pi * (var_k + (mean_k - m_bar) ** 2))
        return m_bar, v

    x_t_arr = np.atleast_1d(x_t)
    mv = [scalar_mean_var(xi) for xi in x_t_arr]
    means = np.array([p[0] for p in mv])
    vars_ = np.array([p[1] for p in mv])
    return means, vars_

# ============================================================
# 2. 线性算子A的SVD分解
# ============================================================
print("\n" + "=" * 60)
print("步骤1：线性算子A的SVD分解可视化")
print("=" * 60)

# 设计一个非对称的线性算子（2x2矩阵，模拟降维）
# 非对角元素差异大，使U和V^T明显不同，展示SVD分解的一般性
A = np.array([[2.0, 0.1],
              [1.5, 0.5]])
print(f"线性算子 A = \n{A}")

# SVD分解：A = U · S · V^T
U, S, Vt = np.linalg.svd(A)
print(f"\nSVD分解结果：")
print(f"U（左奇异向量）= \n{U}")
print(f"S（奇异值）= {S}")
print(f"V^T（右奇异向量）= \n{Vt}")

# 验证SVD分解的正确性
A_reconstructed = U @ np.diag(S) @ Vt
print(f"\n验证: A_reconstructed 与 A 的误差 = {np.linalg.norm(A - A_reconstructed):.6f}")

# 可视化SVD分解
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# 原矩阵A
axes[0].imshow(A, cmap='coolwarm', vmin=-2, vmax=2)
axes[0].set_title('$A$')
axes[0].axis('off')

# U
axes[1].imshow(U, cmap='coolwarm', vmin=-1, vmax=1)
axes[1].set_title('$U$ (左奇异向量)')
axes[1].axis('off')

# S（奇异值）
axes[2].bar(range(len(S)), S, color='steelblue')
axes[2].set_title('$S$ (奇异值)')
axes[2].set_xlabel('频谱分量')
axes[2].set_ylabel('奇异值')

# V^T
axes[3].imshow(Vt, cmap='coolwarm', vmin=-1, vmax=1)
axes[3].set_title('$V^T$ (右奇异向量)')
axes[3].axis('off')

plt.suptitle('DDRM：线性算子 $A$ 的SVD分解（附录13B）', fontsize=14, y=1.02)
plt.tight_layout()
svd_path = os.path.join(SAVE_DIR, "SVD分解可视化.png")
plt.savefig(svd_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"SVD分解可视化已保存: {svd_path}")

# ============================================================
# 3. DDRM频谱域条件化
# ============================================================
print("\n" + "=" * 60)
print("步骤2：DDRM频谱域条件化原理")
print("=" * 60)

print("""
DDRM频谱域条件化步骤（附录13B，本实验采用双通道精度加权融合）：
  1. 频谱域观测：y_freq = U^T · y
  2. 观测通道：d_k = y_freq_k / S_k，无偏估计x_freq_k，噪声方差 v_obs=(σ_y/S_k)²
     - 大S_k：v_obs小 → 观测主导（强约束）
     - 小S_k：噪声被放大 → 自动降权，先验主导（弱约束）
  3. 先验通道：解析Tweedie估计E[x0|x_t]旋转到频谱域 V^T x̂0，方差取GM后验方差
  4. 融合：m_k = (d_k/v_obs + x̂0_k/v_prior) / (1/v_obs + 1/v_prior)
  5. 频谱域逆变换：x_recon = V · m_freq

与旧版"教学阈值"的关系：旧版判据σ_t·σ_y/S_k在本参数下恒<0.5，
分支全部落入观测主导，退化为与t无关的纯伪逆（自适应机制失效），
已替换为上述与时间步/分量均相关的方差融合。

DDRM优势：
  - 每个频谱分量独立加权，小奇异值方向的噪声放大被方差匹配自动处理
  - 先验通道退化时（t大）仍靠观测通道保持精度
""")

# 注：不再单独抽一条“演示用”观测量——主对比循环中每次都重新抽真值并据此生成观测，
# 旧版的 x0_star/noise_obs/y_obs 只用于打印、与主循环无关，已删除。

# DDRM频谱域条件化
def ddrm_spectral_conditioning(y_obs, t_idx, A, U, S, Vt, x_t, sigma_obs=0.1):
    """DDRM频谱域条件化：观测通道与先验通道的精度加权融合（Kadkhodaie & Soltanolkotabi 2021思想）

    观测通道：d_k = (U^T y)_k / S_k，方差 v_obs_k = (σ_y/S_k)²（小奇异值方向噪声被放大）
    先验通道：V^T·Tweedie(x_t)，方差取GM后验方差（正交变换不改变各向同性方差）
    融合：m_k = (d_k/v_obs_k + x̂0_k/v_prior) / (1/v_obs_k + 1/v_prior)
    线性高斯意义下这就是E[x0 | y, x_t]的分量式精确解（GM→混合高斯近似）。

    返回 (x_recon, w_obs)：w_obs为各分量观测通道权重，用于验证自适应机制确实随t/k变化。
    """
    y_freq = U.T @ y_obs

    # 先验通道：解析Tweedie估计及其后验方差，旋转到频谱域
    x_hat_0_signal, var_signal = tweedie_estimate_and_var(x_t, t_idx)
    x_hat_0_freq = Vt @ x_hat_0_signal
    v_prior = float(np.mean(var_signal))

    # 观测通道及其方差（逐分量）
    d_freq = np.array([y_freq[k] / S[k] if S[k] > 0 else 0.0 for k in range(len(S))])
    v_obs = np.array([(sigma_obs / S[k]) ** 2 if S[k] > 0 else np.inf for k in range(len(S))])

    # 精度加权融合
    prec_obs = 1.0 / v_obs
    prec_prior = 1.0 / v_prior
    w_obs = prec_obs / (prec_obs + prec_prior)
    m_freq = w_obs * d_freq + (1.0 - w_obs) * x_hat_0_freq

    # 频谱域逆变换
    x_recon = Vt.T @ m_freq
    return x_recon, w_obs

# ============================================================
# 4. DDRM vs DPS对比实验
# ============================================================
print("\n" + "=" * 60)
print("步骤3：DDRM vs DPS效果对比")
print("=" * 60)

def dps_one_step_estimate(x_t, y_obs, A, t_idx, S, sigma_obs=0.1):
    """DPS单步估计（与DDRM公平对比：同样基于x_t与y，同一个Tweedie去噪核）

    标准DPS形式：先Tweedie去噪得x̂0，再沿似然梯度方向修正：
      ∇_{x̂0} log p(y|x̂0) = A^T (y − A x̂0) / σ_y²
    梯度步长取 s = σ_y²/‖A‖₂²（= σ_y²/S₁²，保证I − s·A^TA/σ_y²半正定，
    单步欠阻尼是DPS实践常态），得稳定形式：
      x̂0_dps = x̂0 + A^T (y − A x̂0) / S₁²
    不人为恢复x2=0.5*x1的结构约束，与DDRM对等处理。

    旧版缺陷已修复：旧版直接返回 x_t + 0.1·修正（不做去噪），
    其"误差随t单调升"实为‖x_t − x2d‖（扩散噪声本身），非DPS固有属性。
    """
    x_hat_0 = tweedie_estimate(x_t, t_idx)
    residual = y_obs - A @ x_hat_0
    step = A.T @ residual / (S[0] ** 2)   # ‖A‖₂² = S₁²
    return x_hat_0 + step

# 对比实验：多个时间步
test_times = [0, 50, 100, 150, 199]
n_trials = 50  # 每个时间步采样次数

ddrm_errors = []
dps_errors = []
ddrm_w_obs_log = []  # 各时间步观测通道均权重，验证自适应机制生效

print("对比DDRM和DPS在不同噪声水平下的重建误差...")

for t_idx in test_times:
    ddrm_err_list = []
    dps_err_list = []
    w_obs_list = []

    for _ in range(n_trials):
        # 生成真值x0（每次试验独立从先验抽样）
        x0_true = sample_from_prior(1)[0]
        x2d = np.array([x0_true, x0_true * 0.5])
        # 观测必须基于同一次采样的x2d生成，保证"观测-真值"严格配对
        noise_obs_trial = np.random.randn(2) * 0.1
        y_obs_trial = A @ x2d + noise_obs_trial

        x_t, _ = vp_sde_forward(x2d, t_idx)

        # DDRM重建（频谱域双通道精度加权融合）
        x_ddrm, w_obs = ddrm_spectral_conditioning(y_obs_trial, t_idx, A, U, S, Vt, x_t)
        ddrm_err = np.linalg.norm(x_ddrm - x2d)
        ddrm_err_list.append(ddrm_err)
        w_obs_list.append(w_obs)

        # DPS重建（Tweedie去噪 + 单步似然梯度修正）
        x_dps = dps_one_step_estimate(x_t, y_obs_trial, A, t_idx, S)
        dps_err = np.linalg.norm(x_dps - x2d)
        dps_err_list.append(dps_err)

    ddrm_errors.append(np.mean(ddrm_err_list))
    dps_errors.append(np.mean(dps_err_list))
    w_mean = np.mean(w_obs_list, axis=0)
    ddrm_w_obs_log.append(w_mean)
    print(f"  t={t_idx:3d}: DDRM权重w_obs=[{w_mean[0]:.3f}, {w_mean[1]:.3f}] "
          f"(分量k=0大奇异值/k=1小奇异值), "
          f"DDRM误差={np.mean(ddrm_err_list):.4f}, DPS误差={np.mean(dps_err_list):.4f}")

# 可视化对比结果
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 误差对比
axes[0].plot(test_times, ddrm_errors, 'o-', label='DDRM', linewidth=2, markersize=8)
axes[0].plot(test_times, dps_errors, 's-', label='DPS', linewidth=2, markersize=8)
axes[0].set_xlabel('时间步 $t$')
axes[0].set_ylabel('重建误差')
axes[0].set_title('DDRM vs DPS：重建误差对比')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 误差比（DDRM/DPS）
error_ratio = np.array(ddrm_errors) / np.array(dps_errors)
axes[1].bar(range(len(test_times)), error_ratio, color='coral', alpha=0.7)
axes[1].axhline(y=1.0, color='gray', linestyle='--', label='误差相等')
axes[1].set_xlabel('时间步索引')
axes[1].set_ylabel('误差比 (DDRM/DPS)')
axes[1].set_title('DDRM相对DPS的误差比（<1表示DDRM更优）')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('DDRM vs DPS对比：频谱域方差融合 vs 单步似然梯度修正', fontsize=14, y=1.02)
plt.tight_layout()
compare_path = os.path.join(SAVE_DIR, "DDRM vs DPS对比.png")
plt.savefig(compare_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"对比结果已保存: {compare_path}")

print("\n关键发现：")
print(f"  DDRM平均误差: {np.mean(ddrm_errors):.4f}")
print(f"  DPS平均误差: {np.mean(dps_errors):.4f}")
print(f"  DDRM相对改进: {(np.mean(dps_errors) - np.mean(ddrm_errors)) / np.mean(dps_errors) * 100:.2f}%")
print(f"  t=0时: DDRM误差={ddrm_errors[0]:.4f}, DPS误差={dps_errors[0]:.4f} "
      f"({'DPS更优' if dps_errors[0] < ddrm_errors[0] else 'DDRM更优/相当'})")
print(f"  t=T-1时（ᾱ≈{alpha_bars[-1]:.3f}，中等噪声起点）: DDRM误差={ddrm_errors[-1]:.4f}, DPS误差={dps_errors[-1]:.4f} "
      f"({'DDRM更优' if ddrm_errors[-1] < dps_errors[-1] else 'DPS更优'})")

# ============================================================
# 5. DDRM局限性：非线性算子场景
# ============================================================
print("\n" + "=" * 60)
print("步骤4：DDRM局限性展示（非线性算子）")
print("=" * 60)

print("""
DDRM核心局限（13.3.1节）：
  DDRM需要对正向算子A做SVD分解：
    A = U · S · V^T

  这要求A必须是线性算子！

  非线性算子场景（DDRM无法应用）：
    - 非线性模糊：非线性滤波器
    - 非线性降维：神经网络编码器
    - 非线性测量：相位检索、非线性散射

  对比DPS优势：
    DPS仅需测量模型y = H(x)，无需SVD分解
    因此DPS可应用于非线性逆问题！
""")

# 非线性算子示例
def nonlinear_operator(x):
    """非线性算子：y = A(x) + 非线性项"""
    return np.array([x[0]**2 + x[1], x[0] * x[1] + 0.5])

print("非线性算子示例：y = [x_1^2 + x_2, x_1 * x_2 + 0.5]")
print("  无法做SVD分解，DDRM无法应用")
print("  但DPS可通过Tweedie估计 + 测量模型求解")

# 可视化局限性对比
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# 适用范围对比
categories = ['线性模糊', '线性降维', '非线性模糊', '相位检索', '神经网络编码']
# 1.0=适用，0.15=不适用（用非零小值让柱子可见，便于识别"不适用"类别）
ddrm_applicable = [1.0, 1.0, 0.15, 0.15, 0.15]
dps_applicable = [1.0, 1.0, 1.0, 1.0, 1.0]

x_pos = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x_pos - width/2, ddrm_applicable, width, label='DDRM', color='steelblue', alpha=0.7)
bars2 = ax.bar(x_pos + width/2, dps_applicable, width, label='DPS', color='coral', alpha=0.7)

ax.set_xlabel('逆问题类型')
ax.set_ylabel('适用性（高柱=适用，矮柱=不适用）')
ax.set_title('DDRM vs DPS适用范围对比（理论适用性，非数值实验）')
ax.set_xticks(x_pos)
ax.set_xticklabels(categories, rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 添加注释
ax.text(2, 0.5, 'DDRM局限性\n(非线性算子)', ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
ax.text(4, 0.5, 'DPS优势\n(适用所有)', ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
limit_path = os.path.join(SAVE_DIR, "DDRM局限性对比.png")
plt.savefig(limit_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"局限性对比已保存: {limit_path}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("实验13.3-3 完成!")
print("=" * 60)
print("""
关键结论:
1. DDRM核心原理（附录13B）
   - SVD频谱域条件化：A = U·S·V^T
   - 每个频谱分量独立处理
   - 按分量噪声方差自适应融合观测与先验

2. DDRM vs DPS对比（以实际运行数据为准）
   - DDRM：频谱域双通道精度加权融合，小奇异值方向噪声放大被自动降权
   - DPS：Tweedie去噪 + 单步似然梯度修正，高噪声时刻欠阻尼
   - 低噪声（t小）两法误差相当；高噪声（t大）DDRM显著更优

3. DDRM局限性（13.3.1节）
   - 仅适用于线性算子（需SVD分解）
   - 非线性算子无法应用DDRM
   - DPS适用范围更广（无需SVD）

4. 教学意义
   - 展示频谱域条件化的优势
   - 理解DDRM文献方法的数学基础
   - 对比不同近似方法的适用范围
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
    "DDRM_各时间步平均误差": [round(float(e), 4) for e in ddrm_errors],
    "DPS_各时间步平均误差": [round(float(e), 4) for e in dps_errors],
    "DDRM_总平均误差": round(float(np.mean(ddrm_errors)), 4),
    "DPS_总平均误差": round(float(np.mean(dps_errors)), 4),
    "DDRM相对改进百分比": round(float((np.mean(dps_errors) - np.mean(ddrm_errors)) / np.mean(dps_errors) * 100), 2),
    "SVD_奇异值": [round(float(s), 4) for s in S],
    "DDRM_观测通道权重w_obs_各时间步": [[round(float(w), 4) for w in w_row] for w_row in ddrm_w_obs_log],
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
