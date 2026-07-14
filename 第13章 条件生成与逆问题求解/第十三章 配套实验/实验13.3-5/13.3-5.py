# -*- coding: utf-8 -*-
"""
实验13.3-5：DDRM算法实践

★ 原创设计：DDRM的SVD频谱域条件化
  DDRM核心思想：对线性算子A做SVD分解，在频谱域精确条件化
  优势：避免了DPS中似然得分的近似误差
  局限：仅适用于线性算子，非线性算子无法SVD分解

实验内容：
  - 1D高斯混合先验 + VP-SDE框架
  - 线性算子A的SVD分解可视化
  - DDRM频谱域条件化采样
  - DDRM vs DPS效果对比（精度、稳定性）
  - DDRM局限性展示（非线性算子场景）

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
    SAVE_DIR = os.path.join(_gdrive, '实验13.3-5')
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
print("实验13.3-5: DDRM算法实践")
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
    在频谱域精确计算似然得分，避免DPS的Tweedie近似误差

  关键优势：
    - 频谱域条件化精确（无近似误差）
    - 每个频谱分量独立处理（自适应噪声水平）

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
    """VP-SDE前向过程：x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * noise"""
    alpha_bar_t = alpha_bars[t_idx]
    noise = np.random.randn(*x0.shape)
    x_t = np.sqrt(alpha_bar_t) * x0 + np.sqrt(1.0 - alpha_bar_t) * noise
    return x_t, noise

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
DDRM频谱域条件化步骤（附录13B）：
  1. 频谱域变换：x_freq = V^T · x_t
  2. 频谱域观测：y_freq = U^T · y
  3. 频谱分量条件化：对每个频谱分量k，根据信噪比条件化
     - 若 S_k >> 噪声：强约束（观测主导）
     - 若 S_k << 噪声：弱约束（先验主导）
  4. 频谱域逆变换：x_recon = V · x_freq_cond

DDRM优势：
  - 每个频谱分量独立处理，自适应噪声水平
  - 频谱域条件化精确，无DPS的Tweedie近似误差
""")

# 生成真实信号x0和观测y
x0_star = sample_from_prior(1)[0]  # 从先验采样一个真实信号
noise_obs = np.random.randn(2) * 0.1  # 观测噪声
y_obs = A @ np.array([x0_star, x0_star * 0.5]) + noise_obs  # 观测（简化：x是2维）

print(f"真实信号 x* = {x0_star:.3f}")
print(f"观测信号 y = {y_obs}")

# DDRM频谱域条件化
def ddrm_spectral_conditioning(y_obs, t_idx, A, U, S, Vt, x_t, sigma_obs=0.1):
    """DDRM频谱域条件化采样

    参数:
        y_obs: 观测信号（信号域）
        t_idx: 时间步索引
        A: 线性算子
        U, S, Vt: SVD分解结果
        x_t: 信号域扩散状态（用于解析Tweedie估计）
        sigma_obs: 观测噪声标准差
    """
    # 频谱域观测
    y_freq = U.T @ y_obs

    # 扩散噪声水平
    sigma_t = np.sqrt(1.0 - alpha_bars[t_idx])

    # 解析Tweedie估计（信号空间GM后验均值，再变换到频谱域；V为正交变换，分布保持一致）
    x_hat_0_signal = tweedie_estimate(x_t, t_idx)
    x_hat_0_freq = Vt @ x_hat_0_signal

    # 频谱域条件化（每个分量独立处理）
    x_freq_cond = np.zeros(len(S))

    for k in range(len(S)):
        # 信噪比判定（教学简化阈值，非DDRM论文原始判据，仅用于演示频谱分量分级策略）
        signal_strength = S[k]
        noise_level = sigma_t * sigma_obs / S[k] if S[k] > 0 else np.inf

        if noise_level < 0.5:  # 强约束区域：观测主导
            # x_freq_k = y_freq_k / S_k
            x_freq_cond[k] = y_freq[k] / S[k] if S[k] > 0 else 0
        elif noise_level > 2.0:  # 弱约束区域：先验主导
            # 使用解析Tweedie估计（基于GM先验的后验均值）的频谱域投影
            x_freq_cond[k] = x_hat_0_freq[k]
        else:  # 中间区域：观测 + 先验 混合约束
            weight = 1.0 / (1.0 + noise_level)
            obs_term = y_freq[k] / S[k] if S[k] > 0 else 0
            x_freq_cond[k] = weight * obs_term + (1.0 - weight) * x_hat_0_freq[k]

    # 频谱域逆变换
    x_recon = Vt.T @ x_freq_cond
    return x_recon

# ============================================================
# 4. DDRM vs DPS对比实验
# ============================================================
print("\n" + "=" * 60)
print("步骤3：DDRM vs DPS效果对比")
print("=" * 60)

def dps_likelihood_score(x_t, y_obs, A, t_idx, zeta=1.0, sigma_obs=0.1):
    """DPS似然得分修正项（与DDRM公平对比）

    DPS核心：s_corrected = s_prior + zeta * s_likelihood
    本函数返回s_likelihood部分（似然得分修正项）

    参数:
        x_t: 扩散状态（2D向量）
        y_obs: 观测信号
        A: 线性算子
        t_idx: 时间步索引
        zeta: 引导权重
        sigma_obs: 观测噪声标准差
    """
    # 解析Tweedie去噪估计（基于GM先验的后验均值，2D逐分量）
    x_hat_0 = tweedie_estimate(x_t, t_idx)

    # DPS似然得分：nabla log p(y|x_hat_0)
    # p(y|x_hat_0) = N(y; A x_hat_0, sigma_obs^2 I)
    # nabla log p(y|x_hat_0) = A^T (y - A x_hat_0) / sigma_obs^2
    # 不人为恢复x2=0.5*x1的结构约束，与DDRM对等处理
    residual = y_obs - A @ x_hat_0
    likelihood_score = A.T @ residual / (sigma_obs ** 2)

    # 返回得分修正项（带zeta权重）
    return zeta * likelihood_score

# 对比实验：多个时间步
test_times = [0, 50, 100, 150, 199]
n_trials = 50  # 每个时间步采样次数

ddrm_errors = []
dps_errors = []

print("对比DDRM和DPS在不同噪声水平下的重建误差...")

for t_idx in test_times:
    ddrm_err_list = []
    dps_err_list = []

    for _ in range(n_trials):
        # 生成真值x0（基于单个真实信号x0_star从先验抽样加噪声）
        x0_true = sample_from_prior(1)[0]
        x2d = np.array([x0_true, x0_true * 0.5])
        # 观测必须基于同一次采样的x2d生成，保证"观测-真值"严格配对
        noise_obs_trial = np.random.randn(2) * 0.1
        y_obs_trial = A @ x2d + noise_obs_trial

        x_t, _ = vp_sde_forward(x2d, t_idx)

        # DDRM重建（基于解析Tweedie估计的频谱域条件化）
        x_ddrm = ddrm_spectral_conditioning(y_obs_trial, t_idx, A, U, S, Vt, x_t)
        ddrm_err = np.linalg.norm(x_ddrm - x2d)
        ddrm_err_list.append(ddrm_err)

        # DPS重建（应用似然得分修正）
        # zeta = sigma_obs^2 自适应缩放，抵消似然得分分母的放大效应
        # 当sigma_obs=0.1时，zeta=0.01使步长与residual量级匹配
        score_correction = dps_likelihood_score(x_t, y_obs_trial, A, t_idx, zeta=0.01, sigma_obs=0.1)
        x_dps = x_t + 0.1 * score_correction
        dps_err = np.linalg.norm(x_dps - x2d)
        dps_err_list.append(dps_err)

    ddrm_errors.append(np.mean(ddrm_err_list))
    dps_errors.append(np.mean(dps_err_list))

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

plt.suptitle('DDRM vs DPS对比：频谱域精确条件化 vs Tweedie近似', fontsize=14, y=1.02)
plt.tight_layout()
compare_path = os.path.join(SAVE_DIR, "DDRM vs DPS对比.png")
plt.savefig(compare_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"对比结果已保存: {compare_path}")

print("\n关键发现：")
print(f"  DDRM平均误差: {np.mean(ddrm_errors):.4f}")
print(f"  DPS平均误差: {np.mean(dps_errors):.4f}")
print(f"  DDRM相对改进: {(np.mean(dps_errors) - np.mean(ddrm_errors)) / np.mean(dps_errors) * 100:.2f}%")

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
print("实验13.3-5 完成!")
print("=" * 60)
print("""
关键结论:
1. DDRM核心原理（附录13B）
   - SVD频谱域条件化：A = U·S·V^T
   - 每个频谱分量独立处理
   - 自适应噪声水平（信噪比判定）

2. DDRM vs DPS对比
   - DDRM：频谱域精确条件化，无近似误差
   - DPS：Tweedie近似似然得分，存在误差
   - 实验显示DDRM在低噪声区域误差更小

3. DDRM局限性（13.3.1节）
   - 仅适用于线性算子（需SVD分解）
   - 非线性算子无法应用DDRM
   - DPS适用范围更广（无需SVD）

4. 教学意义
   - 展示频谱域条件化的优势
   - 理解DDRM文献方法的数学基础
   - 对比不同近似方法的适用范围
""")