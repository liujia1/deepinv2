# -*- coding: utf-8 -*-
"""
实验13.4-1：引导权重ζ权衡曲线
对应章节：13.4.3节 引导权重与质量-多样性权衡

★ 原创设计：固定随机种子，用不同zeta执行后验采样，
  对比采样分布的均值（->数据一致性）和方差（->多样性）

实验内容：
  - 不同zeta值下的后验采样分布
  - zeta-数据一致性 / zeta-多样性权衡曲线
  - 与第2-3章正则化参数lambda的类比

本实验不需要GPU，通过1D解析情形研究zeta效应。
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
    SAVE_DIR = os.path.join(_gdrive, '实验13.4-1')
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
print("实验13.4-1: 引导权重zeta权衡曲线")
print("=" * 60)
print("对应章节: 13.4.3节 引导权重与质量-多样性权衡")
print("知识点: 引导权重zeta, 质量-多样性权衡, Tweedie一致性")


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


def analytic_posterior_gm_gaussian(y_obs, A, sigma_y, gm_weights, gm_means, gm_stds):
    """解析计算GM先验+高斯似然的共轭后验。
    
    对于高斯混合先验 p(x) = Σ_k w_k * N(x; μ_k, σ_k²) 和高斯似然
    p(y|x) = N(y; Ax, σ_y²)，后验仍然是高斯混合：
    
    p(x|y) = Σ_k [w_k' * N(x; μ_k|y, σ_k|y²)]
    
    其中：
      - μ_k|y = (σ_y² * μ_k + σ_k² * y/A) / (σ_y² + σ_k²)  （假设A=1）
      - σ_k|y² = σ_y² * σ_k² / (σ_y² + σ_k²)
      - w_k' = w_k * N(y; Aμ_k, σ_y² + σ_k²) / p(y)
    
    返回：后验均值、后验标准差、后验pdf函数
    """
    # 后验各分量的参数
    post_weights = []
    post_means = []
    post_stds = []
    
    # 计算各分量的权重（需要归一化）
    log_weights_unnorm = []
    for w, m, s in zip(gm_weights, gm_means, gm_stds):
        # 后验均值和方差（假设A=1的线性情形）
        post_mean = (sigma_y**2 * m + s**2 * y_obs) / (sigma_y**2 + s**2)
        post_std = sigma_y * s / np.sqrt(sigma_y**2 + s**2)
        post_means.append(post_mean)
        post_stds.append(post_std)
        
        # 边缘似然 p(y|μ_k) = N(y; μ_k, σ_y² + σ_k²)
        marginal_var = sigma_y**2 + s**2
        log_marginal = -0.5 * ((y_obs - m)**2 / marginal_var) - 0.5 * np.log(2 * np.pi * marginal_var)
        log_weights_unnorm.append(np.log(w) + log_marginal)
    
    # 归一化权重（避免数值溢出）
    max_log = max(log_weights_unnorm)
    weights_unnorm = [np.exp(lw - max_log) for lw in log_weights_unnorm]
    total = sum(weights_unnorm)
    post_weights = [wu / total for wu in weights_unnorm]
    
    # 后验均值 = Σ_k w_k' * μ_k|y
    posterior_mean = sum(w * m for w, m in zip(post_weights, post_means))
    
    # 后验方差 = Σ_k w_k' * (σ_k|y² + μ_k|y²) - posterior_mean²
    posterior_var = sum(w * (s**2 + m**2) for w, s, m in zip(post_weights, post_stds, post_means)) - posterior_mean**2
    posterior_std = np.sqrt(posterior_var)
    
    # 后验pdf函数
    def posterior_pdf(x):
        pdf = np.zeros_like(x)
        for w, m, s in zip(post_weights, post_means, post_stds):
            pdf += w * np.exp(-0.5 * ((x - m) / s)**2) / (s * np.sqrt(2 * np.pi))
        return pdf
    
    return posterior_mean, posterior_std, posterior_pdf, post_weights, post_means, post_stds


def dps_posterior_sample(y_obs, A, sigma_y, zeta, N_particles=5000, N_steps=300, T=1.0, seed=42):
    """VP-SDE后验采样（DPS近似），可调引导权重zeta

    注意：likelihood_grad 公式中的 mean_t 缩放因子不是简单的链式法则 1/mean_t，
    而是 mean_t 本身。直觉上可能认为 dx0_hat/dx_t ≈ 1/mean_t（将 prior_score 视为常数，
    只考虑 x0_hat = (x + std_t² * prior_score) / mean_t 对 x 的显式依赖），但这种
    "朴素链式法则"忽略了 score 本身对 x_t 的强依赖。

    数值验证（t=0.9时，本实验的双峰GM先验）：
      - 真实 dx0_hat/dx_t（有限差分） ≈ 0.049
      - mean_t ≈ 0.017，二者量级一致
      - 1/mean_t ≈ 58.8，差三个数量级
    严格推导：x0_hat = (x_t + std_t² · score(x_t)) / mean_t。
      - 对单峰高斯先验，score 是 x_t 的线性函数，链式法则精确给出
        dx0_hat/dx_t = mean_t，公式完全成立；
      - 对本实验的双峰GM先验，score 还包含"responsibility 权重随 x_t 变化"
        的非线性项，单分量公式不再精确，实际比值约为 0.049 / 0.017 ≈ 2.9。
    因此结论需保守表述：用 mean_t 作缩放在量级方向上正确（比 1/mean_t
    合理得多），但对 GM 先验仅有量级一致，会带来约 3 倍的近似误差——
    这可能是均值偏差随 ζ 呈非单调的部分来源（见结论段）。
    """
    np.random.seed(seed)
    h = T / N_steps
    x = np.random.randn(N_particles)
    for i in range(N_steps):
        t = T - i * h
        beta_t = vp_beta(t)
        mean_t, std_t = vp_marginal(t)
        prior_score = vp_score_analytic(x, t)
        # 数值稳定性保护：mean_t 在 t→T（1.0）时最小（本配置下约 0.0066），
        # 在 t→0 时最大（接近 1）。clip 在 1e-6 仅在 T 取得极端大值时才会触发，
        # 当前 BETA_MAX=20.0 配置下最小值 0.0066 离阈值还有三个数量级，
        # 故此 clip 在本实验中实际未激活，但保留以防参数极端化时除零
        mean_t_safe = max(mean_t, 1e-6)
        x0_hat = (x + std_t**2 * prior_score) / mean_t_safe
        # 似然梯度：∇_x_t log p(y|x_t) ≈ mean_t * (y - A * x0_hat) / σ_y²
        # （不是 1/mean_t，详见函数文档字符串）
        likelihood_grad = mean_t * (y_obs - A * x0_hat) / sigma_y**2
        posterior_score = prior_score + zeta * likelihood_grad
        x = x + beta_t * h * (0.5 * x + posterior_score) + np.sqrt(beta_t * h) * np.random.randn(N_particles)
    return x


# ============================================================
# 步骤1：引导权重zeta与质量-多样性权衡
# ============================================================
print("\n" + "=" * 60)
print("步骤1：引导权重zeta与质量-多样性权衡（13.4.3节）")
print("=" * 60)

print("""
13.4.3节：引导权重zeta控制先验与似然的相对强度
  nabla log p(x_t|y) ~ nabla log p(x_t) + zeta * nabla log p(y|x_hat_{0|t})

  zeta大 -> 强数据一致性（低多样性）-> 类似MAP
  zeta小 -> 强先验（高多样性）-> 类似无条件采样
""")

# 逆问题设置
A_val = 1.0
sigma_y = 0.5
y_obs = 0.5

# ★ 解析计算真实后验（GM先验+高斯似然的共轭情形）
post_mean_true, post_std_true, post_pdf_true, post_w_true, post_m_true, post_s_true = \
    analytic_posterior_gm_gaussian(y_obs, A_val, sigma_y, GM_WEIGHTS, GM_MEANS, GM_STDS)
print(f"\n真实后验（解析）: mean={post_mean_true:.4f}, std={post_std_true:.4f}")
print(f"  后验分量: w={post_w_true}, μ={post_m_true}, σ={post_s_true}")

zeta_values = [0.0, 0.3, 0.7, 1.0, 2.0, 5.0]
sampling_results = {}

print("\nDPS采样 vs 真实后验对比：")
print(f"{'zeta':>6s} | {'采样均值':>8s} | {'均值偏差':>8s} | {'采样std':>7s} | {'std偏差':>7s} | {'std低估率':>9s}")
print("-" * 65)
# 注：'std低估率' = (真值std − 采样std) / 真值std × 100%
#   - 正值：采样std < 真值std（低估多样性）
#   - 负值：采样std > 真值std（实际为高估，常见于小ζ如zeta=0的无条件采样）
print("  (注: 'std低估率' 正值=低估, 负值=高估)")
for zeta in zeta_values:
    samples = dps_posterior_sample(y_obs, A_val, sigma_y, zeta)
    sampling_results[zeta] = samples
    mean_s = np.mean(samples)
    std_s = np.std(samples)
    mean_bias = mean_s - post_mean_true
    std_bias = std_s - post_std_true
    std_underest = (post_std_true - std_s) / post_std_true * 100  # 方差低估百分比
    print(f"{zeta:6.1f} | {mean_s:8.3f} | {mean_bias:+8.3f} | {std_s:7.3f} | {std_bias:+7.3f} | {std_underest:8.1f}%")

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
x_hist = np.linspace(-6, 6, 500)

for idx, zeta in enumerate(zeta_values):
    ax = axes[idx // 3, idx % 3]
    samples = sampling_results[zeta]
    ax.hist(samples, bins=60, density=True, alpha=0.5, color='steelblue',
            range=(-6, 6), label=r'后验采样 ($\zeta={}$)'.format(zeta))
    ax.plot(x_hist, gm1d_pdf(x_hist), 'k--', lw=1.5, alpha=0.7, label='先验 p(x)')
    # ★ 添加真实后验曲线
    ax.plot(x_hist, post_pdf_true(x_hist), 'g-', lw=2, alpha=0.8, label='真实后验（解析）')
    ax.axvline(y_obs, color='red', linestyle=':', lw=2, label=r'观测 $y={}$'.format(y_obs))
    # ★ 真实后验均值参考线
    ax.axvline(post_mean_true, color='green', linestyle='--', lw=1.5, alpha=0.7, 
               label=r'真实均值={:.2f}'.format(post_mean_true))
    mean_s = np.mean(samples)
    ax.axvline(mean_s, color='blue', linestyle='-', lw=1.5, alpha=0.7, 
               label=r'采样均值={:.2f}'.format(mean_s))
    ax.set_title(r'$\zeta$ = {} ({})'.format(zeta, "无条件" if zeta == 0 else "弱引导" if zeta < 0.5 else "标准" if zeta < 1.5 else "强引导"), fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim(-6, 6)
    # 自适应y上限：各zeta的histogram峰值不同（zeta=5峰值约2.3，真后验约0.87），
    # 固定ylim会截断关键的多样性坍缩形状，故改用None自动缩放
    ax.set_ylim(0, None)

fig.suptitle('引导权重zeta与质量-多样性权衡（13.4.3节）', fontsize=14, y=1.01)
plt.tight_layout()
fig_path1 = os.path.join(SAVE_DIR, '引导权重权衡.png')
plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图1已保存: {fig_path1}")

# 权衡曲线
consistency_list = []
diversity_list = []
for zeta in zeta_values:
    samples = sampling_results[zeta]
    consistency_list.append(np.mean(np.abs(samples - y_obs)))
    diversity_list.append(np.std(samples))

# ★ 真实后验的多样性（std）和一致性（E|X−y|）
# 关键：DPS曲线的一致性定义为 E|X−y| = mean(|samples - y_obs|)，
# 真实后验的参考点必须用同口径统计量 E_{x∼p(x|y)}[|X-y|]，
# 不能用 Jensen 不等式两边不相等的 |E[X]−y|。下面在解析后验 pdf 上
# 做数值积分得到精确的 E|X−y|（避免蒙特卡洛噪声）
true_diversity = post_std_true
x_consistency_grid = np.linspace(-10, 10, 100000)
true_consistency = np.trapezoid(
    np.abs(x_consistency_grid - y_obs) * post_pdf_true(x_consistency_grid),
    x_consistency_grid,
)
# 同步打印，便于交叉验证
print(f"\n真实后验 E|X−y|（与DPS曲线同口径） = {true_consistency:.4f}")
print(f"  提示：若误用 |E[X]−y| = {np.abs(post_mean_true - y_obs):.4f}，"
      f"由Jensen不等式两者不等（前者>>后者），会扭曲图2参考点位置")

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(diversity_list, consistency_list, 'ro-', markersize=8, lw=2, label='DPS采样')
for i, zeta in enumerate(zeta_values):
    ax.annotate(r'$\zeta={}$'.format(zeta), (diversity_list[i], consistency_list[i]),
                textcoords="offset points", xytext=(10, 5), fontsize=10)

# ★ 添加真实后验参考点（绿色星形）
ax.scatter([true_diversity], [true_consistency], s=150, c='green', marker='*', 
           zorder=5, label=r'真实后验 ($\sigma={:.2f}$)'.format(true_diversity))
ax.axvline(true_diversity, color='green', linestyle='--', lw=1.5, alpha=0.7,
           label=r'真实多样性={:.2f}'.format(true_diversity))

ax.set_xlabel('多样性（采样标准差）', fontsize=12)
ax.set_ylabel(r'数据一致性（$|x-y|$均值）', fontsize=12)
ax.set_title('质量-多样性权衡曲线（13.4.3节）', fontsize=13)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc='upper right')
ax.annotate('右下: 弱引导(高多样性, 低一致性)\n左上: 强引导(低多样性, 高一致性)\n★绿色星形: 真实后验（解析）\n对应第2-3章的正则化参数lambda',
            xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#dfe6e9', alpha=0.8))

plt.tight_layout()
fig_path2 = os.path.join(SAVE_DIR, '权衡曲线.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig_path2}")

print("\n" + "=" * 60)
print("实验13.4-1 完成!")
print("=" * 60)
print("""
关键结论:
1. 引导权重zeta（13.4.3节）原创设计
   - zeta控制先验与似然的相对强度
   - zeta大->数据一致性强、多样性低
   - zeta小->先验贡献大、多样性高
   - 对应第2-3章正则化参数lambda的角色

2. DPS有偏性的定量证据（本实验独特优势）
   - GM先验+高斯似然是共轭情形，真实后验可解析计算
   - 真实后验: mean={:.4f}, std={:.4f}
   - 关键发现：没有任何固定zeta能同时匹配真实后验的均值和方差
     - 方差随zeta单调收缩：在zeta≈0.7附近穿过真值（0.4615），
       小zeta（如0.3）反而高估std（先验未被充分约束），大zeta（≥2）系统性低估
     - 均值偏差呈非单调：zeta≈0.7时最小（约−0.038），
       继续增大zeta（→1.0/2.0/5.0）偏差反而变差（−0.053/−0.070/−0.075）
     - 这一非单调性可能与dps_posterior_sample中mean_t缩放在GM先验下
       约3倍的近似误差有关（详见该函数docstring）
   - 这印证13.3.2节：Laplace近似（delta函数近似）系统性低估后验方差

3. 似然梯度公式说明
   - likelihood_grad使用mean_t缩放，而非朴素链式法则的1/mean_t
   - 原因：score本身对x_t有强依赖，完整梯度包含这一贡献
   - 数值验证（t=0.9，双峰GM先验）：真实dx0_hat/dx_t≈0.049，mean_t≈0.017，
     1/mean_t≈58.8——量级方向一致，但mean_t是量级近似，差约2.9倍
   - 注意：对单峰高斯先验该缩放精确成立；对GM先验仅量级一致

4. 实际启示
   - zeta=0: 无条件采样（忽略观测），std远大于真后验（高估）
   - zeta≈0.7: 本实验的"甜蜜点"——均值偏差最小，方差也最接近真值
   - zeta=1: 标准DPS（平衡先验与似然），但方差仍有低估
   - zeta过大: 类似MAP，多样性坍缩；且均值偏差反而不如0.7好
   - 建议区间：先做小批量扫描找到偏差最小点，而非简单取1.0
""".format(post_mean_true, post_std_true))
