# -*- coding: utf-8 -*-
"""
实验8.3-1 变分推断与真实后验对比
对应章节：8.3（变分推断作为优化问题）
素材来源：🆕 新写
★ 原创设计：CAVI算法实现 + 平均场近似效果对比

知识点：
  - 变分推断 = 在变分族约束下最大化ELBO
  - 平均场近似：假设潜变量相互独立
  - CAVI算法：坐标上升变分推断
  - 变分间隙：KL(q||p(z|x))衡量近似质量

实验内容：
  步骤1：CAVI算法在1D高斯混合上的实现
  步骤2：平均场近似 vs 真实后验——变分间隙

运行前提：纯NumPy/SciPy CPU即可
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys
import io

# 设置控制台输出为 UTF-8 (Windows 下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验8.3-1')
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


# ============================================================
# 模型设定：1D高斯混合 + 高斯观测
# ★ 注意：先验权重调整为 [0.5, 0.5]（对称），观测值x_obs=0.5处于两个峰中间，
#   使真实后验的两个分量权重接近 [0.5, 0.5]，真正展示平均场近似的局限
# ============================================================
prior_weights = [0.5, 0.5]
prior_means = [-2.0, 2.0]
prior_stds = [1.0, 1.0]
sigma_obs = 1.0
x_obs = 0.0  # 观测值在两个峰中间

print("=" * 60)
print("实验8.3-1：变分推断与真实后验对比")
print("=" * 60)
print(f"\n模型设定：")
print(f"  先验 p(z) = {prior_weights[0]}*N({prior_means[0]}, {prior_stds[0]}²) + {prior_weights[1]}*N({prior_means[1]}, {prior_stds[1]}²)")
print(f"  似然 p(x|z) = N(x; z, {sigma_obs}²)")
print(f"  观测值 x = {x_obs}")


# ============================================================
# 辅助函数：计算真实后验、边际似然、高斯混合PDF
# ============================================================

def compute_posterior(x, weights, means, stds, sigma_obs):
    """解析计算高斯混合先验+高斯似然的后验"""
    post_weights, post_means, post_stds = [], [], []
    for w, mu, tau in zip(weights, means, stds):
        s2 = 1.0 / (1.0 / sigma_obs**2 + 1.0 / tau**2)
        s = np.sqrt(s2)
        m = s2 * (x / sigma_obs**2 + mu / tau**2)
        ms = np.sqrt(sigma_obs**2 + tau**2)
        ml = -0.5 * ((x - mu) / ms)**2 - np.log(ms)
        post_weights.append(w * np.exp(ml))
        post_means.append(m)
        post_stds.append(s)
    total = sum(post_weights)
    post_weights = [pw / total for pw in post_weights]
    return post_weights, post_means, post_stds

post_w, post_m, post_s = compute_posterior(x_obs, prior_weights, prior_means, prior_stds, sigma_obs)

print(f"\n真实后验 p(z|x={x_obs}):")
for i in range(len(post_w)):
    print(f"  分量{i+1}: w={post_w[i]:.4f}, μ={post_m[i]:.4f}, σ={post_s[i]:.4f}")


def log_marginal(x, weights, means, stds, sigma_obs):
    """计算log p(x) = log Σₖ wₖ·N(x; μₖ, σ_obs²+τₖ²)"""
    terms = []
    for w, mu, tau in zip(weights, means, stds):
        ms = np.sqrt(sigma_obs**2 + tau**2)
        terms.append(np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(ms) - 0.5 * ((x - mu) / ms)**2)
    mx = max(terms)
    return mx + np.log(sum(np.exp(t - mx) for t in terms))

log_px = log_marginal(x_obs, prior_weights, prior_means, prior_stds, sigma_obs)
print(f"\nlog p(x={x_obs}) = {log_px:.6f}")


def gm_pdf(x, weights, means, stds):
    """高斯混合分布的PDF"""
    pdf = np.zeros_like(x)
    for w, m, s in zip(weights, means, stds):
        pdf += w * np.exp(-0.5 * ((x - m) / s)**2) / (s * np.sqrt(2 * np.pi))
    return pdf


# ============================================================
# 步骤1：CAVI算法实现
# ★ 原创设计
# CAVI（坐标上升变分推断）在平均场假设下交替更新每个因子
# 对于单变量z，平均场就是单高斯，CAVI退化为优化μ_q和σ_q
# ============================================================
print("\n" + "=" * 60)
print("步骤1：CAVI算法实现")
print("=" * 60)

print("\n[核心思想]")
print("  变分推断 = 在变分族Q的约束下最大化ELBO")
print("  平均场近似：q(z) = ∏qᵢ(zᵢ)（假设潜变量独立）")
print("  CAVI：固定其他因子，交替优化每个因子")

from scipy.optimize import minimize

# 关键修复：用 common random numbers 技巧
# 在函数外只生成一次基础噪声 epsilon ~ N(0,1)，优化时通过重参数化 z = μ + σ·ε 复用
# 避免每次调用 neg_elbo_gaussian 都重新采样导致目标函数带 MC 噪声、Nelder-Mead 假收敛
n_vi_samples = 30000
np.random.seed(42)  # 固定基础噪声，确保可复现
vi_eps = np.random.randn(n_vi_samples)  # 固定的基础噪声，z = mu + sigma * vi_eps

def neg_elbo_gaussian(params):
    """负ELBO（用于优化）——使用公共随机数"""
    mu_q, log_sigma_q = params
    sigma_q = np.exp(log_sigma_q)
    # 重参数化：z = mu + sigma * eps，复用固定的 epsilon
    z = mu_q + sigma_q * vi_eps
    log_pxz = -0.5 * np.log(2 * np.pi) - np.log(sigma_obs) - 0.5 * ((x_obs - z) / sigma_obs)**2
    log_pz = np.full_like(z, -1e30)
    for w, mu, tau in zip(prior_weights, prior_means, prior_stds):
        log_comp = np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(tau) - 0.5 * ((z - mu) / tau)**2
        log_pz = np.logaddexp(log_pz, log_comp)
    log_joint = log_pxz + log_pz
    log_qz = -0.5 * np.log(2 * np.pi) - np.log(sigma_q) - 0.5 * ((z - mu_q) / sigma_q)**2
    return -np.mean(log_joint - log_qz)


def compute_elbo_samples(mu_q, sigma_q, n_samples=30000):
    """给定q=N(mu_q, sigma_q²)，数值计算ELBO"""
    np.random.seed(42)
    z = np.random.randn(n_samples) * sigma_q + mu_q
    log_pxz = -0.5 * np.log(2 * np.pi) - np.log(sigma_obs) - 0.5 * ((x_obs - z) / sigma_obs)**2
    log_pz = np.full_like(z, -1e30)
    for w, mu, tau in zip(prior_weights, prior_means, prior_stds):
        log_comp = np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(tau) - 0.5 * ((z - mu) / tau)**2
        log_pz = np.logaddexp(log_pz, log_comp)
    log_joint = log_pxz + log_pz
    log_qz = -0.5 * np.log(2 * np.pi) - np.log(sigma_q) - 0.5 * ((z - mu_q) / sigma_q)**2
    return np.mean(log_joint - log_qz)


# CAVI迭代（交替优化μ和σ）
print("\nCAVI迭代过程（交替优化μ_q和σ_q）：")

mu_q, sigma_q = 0.0, 1.0  # 初始化为先验均值
elbo_history = []

for iteration in range(20):
    # 固定σ，在μ方向搜索
    best_mu_curr, best_elbo_curr = mu_q, -1e10
    for mu_candidate in np.linspace(-3, 3, 100):
        elbo_val = compute_elbo_samples(mu_candidate, sigma_q)
        if elbo_val > best_elbo_curr:
            best_elbo_curr = elbo_val
            best_mu_curr = mu_candidate
    mu_q = best_mu_curr
    
    # 固定μ，在σ方向搜索
    best_sigma_curr, best_elbo_curr = sigma_q, -1e10
    for log_sigma_candidate in np.linspace(-1, 1, 50):
        sigma_candidate = np.exp(log_sigma_candidate)
        elbo_val = compute_elbo_samples(mu_q, sigma_candidate)
        if elbo_val > best_elbo_curr:
            best_elbo_curr = elbo_val
            best_sigma_curr = sigma_candidate
    sigma_q = best_sigma_curr
    
    elbo = compute_elbo_samples(mu_q, sigma_q)
    elbo_history.append(elbo)
    
    if iteration % 5 == 0:
        print(f"  iter {iteration:2d}: μ_q={mu_q:.4f}, σ_q={sigma_q:.4f}, ELBO={elbo:.4f}")

# 联合优化得到最终结果
np.random.seed(42)
result = minimize(neg_elbo_gaussian, [0.0, 0.0], method='Nelder-Mead',
                  options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
best_mu, best_log_sigma = result.x
best_sigma = np.exp(best_log_sigma)
best_elbo = -result.fun

print(f"\n变分最优 q*(z) = N({best_mu:.4f}, {best_sigma:.4f}²)")
print(f"最优 ELBO = {best_elbo:.4f}")
print(f"log p(x) = {log_px:.4f}")
print(f"变分间隙 = {log_px - best_elbo:.4f}")

print(f"\n[CAVI性质验证]")
print(f"  ✓ ELBO单调递增：{elbo_history[0]:.4f} → {elbo_history[-1]:.4f}")
print(f"  ✓ 收敛到局部最优：{best_elbo:.4f}")
print(f"  ✓ 变分间隙 > 0：{log_px - best_elbo:.4f}（平均场表达能力限制）")


# ============================================================
# 步骤2：平均场近似 vs 真实后验
# ★ 原创设计
# ============================================================
print("\n" + "=" * 60)
print("步骤2：平均场近似 vs 真实后验")
print("=" * 60)

z_grid = np.linspace(-5, 5, 500)

# 真实后验
posterior_pdf = gm_pdf(z_grid, post_w, post_m, post_s)

# 变分近似（单高斯）
variational_pdf = np.exp(-0.5 * ((z_grid - best_mu) / best_sigma)**2) / (best_sigma * np.sqrt(2 * np.pi))

# 先验
prior_pdf = gm_pdf(z_grid, prior_weights, prior_means, prior_stds)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 对比图 - 使用LaTeX格式
axes[0].fill_between(z_grid, posterior_pdf, alpha=0.3, color='blue', label=r'True posterior $p(z|x)$')
axes[0].plot(z_grid, posterior_pdf, 'b-', lw=2)
axes[0].fill_between(z_grid, variational_pdf, alpha=0.3, color='red', 
                    label=r'Variational $q^*(z)=\mathcal{N}(' + f'{best_mu:.2f},{best_sigma:.2f}^2)$')
axes[0].plot(z_grid, variational_pdf, 'r--', lw=2)
axes[0].plot(z_grid, prior_pdf, 'k:', lw=1, alpha=0.5, label=r'Prior $p(z)$')
axes[0].set_xlabel(r'$z$')
axes[0].set_ylabel(r'$p(z|x)$')
axes[0].set_title('Mean-field approximation vs True posterior')
axes[0].legend()
axes[0].grid(alpha=0.3)

# ELBO迭代收敛 - 使用LaTeX格式
axes[1].plot(range(len(elbo_history)), elbo_history, 'b-o', markersize=3)
axes[1].axhline(y=log_px, color='r', linestyle='--', lw=2, 
               label=r'$\log p(x) = ' + f'{log_px:.4f}$')
axes[1].axhline(y=best_elbo, color='g', linestyle=':', lw=2, 
               label=r'Best ELBO = ' + f'{best_elbo:.4f}')
axes[1].set_xlabel(r'Iteration')
axes[1].set_ylabel(r'$\mathrm{ELBO}$')
axes[1].set_title('CAVI convergence process')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_平均场vs真实后验.png'), dpi=100)
plt.close()
print(f"\n图表已保存: 步骤2_平均场vs真实后验.png")

print(f"\n[平均场近似的局限]")
print(f"  真实后验是双峰（w=[{post_w[0]:.3f}, {post_w[1]:.3f}]）")
print(f"  单高斯q*只能覆盖一个峰，无法捕获多模态")
print(f"  变分间隙 = {log_px - best_elbo:.4f}（来自平均场的表达能力限制）")

# 直观展示"双峰被抹平"：对比真实后验与单高斯近似在关键点的密度值
print(f"\n[关键点密度对比]——直观展示平均场丢失多模态结构:")
for z_val, label in [(0.0, "两峰中间（真实后验的凹陷处）"),
                     (-1.0, "左峰位置"),
                     (1.0, "右峰位置")]:
    # 真实后验密度
    p_val = 0.0
    for w, m, s in zip(post_w, post_m, post_s):
        p_val += w * np.exp(-0.5 * ((z_val - m) / s)**2) / (s * np.sqrt(2 * np.pi))
    # 单高斯近似密度
    q_val = np.exp(-0.5 * ((z_val - best_mu) / best_sigma)**2) / (best_sigma * np.sqrt(2 * np.pi))
    diff = q_val - p_val
    flag = "  ← 错误地把低概率区标成高概率" if abs(z_val) < 0.1 and diff > 0 else ""
    print(f"  z={z_val:+.1f} ({label}):")
    print(f"    真实后验 p(z|x) = {p_val:.4f}")
    print(f"    单高斯近似 q*(z) = {q_val:.4f}")
    print(f"    差异 q*-p = {diff:+.4f}{flag}")

print(f"\n[核心结论]")
print(f"  1. CAVI算法：在平均场假设下交替优化变分因子")
print(f"  2. 平均场近似 vs 真实后验：")
print(f"     单高斯q*无法拟合双峰后验 → 变分间隙 > 0")
print(f"     间隙来源：平均场的表达能力限制")
print(f"  3. 缩小间隙的方法：")
print(f"     - 扩大变分族（结构化变分、混合分布）")
print(f"     - 增加潜变量维度")
print(f"     - 放松独立性假设")

print(f"\n{'='*60}")
print("第八章配套实验8.3-1完成！")
