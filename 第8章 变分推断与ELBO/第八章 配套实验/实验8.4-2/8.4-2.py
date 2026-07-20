# -*- coding: utf-8 -*-
"""
实验8.4-2 ELBO = 重建 + 正则：变分推断与正则化的统一
对应章节：8.4（变分推断与正则化的统一视角）
素材来源：🆕 新写
★ 原创设计：ELBO分解为重建项与KL正则项，与MAP估计的统一

知识点：
  - ELBO = 重建项 - KL正则项
  - 重建项：E_q[log p(x|z)]，衡量数据拟合质量
  - KL正则项：KL(q||p(z))，惩罚q偏离先验
  - 与MAP估计的数据项+正则项结构对应
  - 确定性正则化是变分推断的退化形式

实验内容：
  步骤1：ELBO分解为重建项与KL正则项
  步骤2：重建-正则权衡的可视化

运行前提：纯NumPy/SciPy CPU即可
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys
import io
from scipy.optimize import minimize

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
    SAVE_DIR = os.path.join(_gdrive, '实验8.4-2')
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
# ============================================================
prior_weights = [0.3, 0.7]
prior_means = [-2.0, 1.0]
prior_stds = [1.0, 1.0]
sigma_obs = 0.5
x_obs = 0.5

print("=" * 60)
print("实验8.4-2：ELBO = 重建 + 正则")
print("=" * 60)
print(f"\n模型设定：")
print(f"  先验 p(z) = {prior_weights[0]}*N({prior_means[0]}, {prior_stds[0]}²) + {prior_weights[1]}*N({prior_means[1]}, {prior_stds[1]}²)")
print(f"  似然 p(x|z) = N(x; z, {sigma_obs}²)")
print(f"  观测值 x = {x_obs}")


# ============================================================
# 辅助函数：计算log p(x)
# ============================================================

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


# ============================================================
# 步骤1：ELBO分解为重建项与KL正则项
# ★ 原创设计
# ============================================================
print("\n" + "=" * 60)
print("步骤1：ELBO分解为重建项与KL正则项")
print("=" * 60)

print("\n[核心概念]")
print("  ELBO = E_q[log p(x|z)] - KL(q||p(z))")
print("       = 重建项 - 正则项")
print("  重建项：衡量q编码的数据信息")
print("  KL正则项：惩罚q偏离先验p(z)")

# 定义计算重建项和KL项的函数
def compute_reconstruction_and_kl(mu_q, sigma_q, n_samples=20000):
    """计算ELBO的两个组成部分

    注意：每次调用都 np.random.seed(42) 重置随机种子——这是 common random numbers 技巧，
    让不同 (μ_q, σ_q) 配置下的 MC 估计噪声完全相关，从而使扫描曲线更光滑、对比更干净。
    这意味着这份代码适合用来画趋势图，但不能用于评估 MC 估计的方差或做收敛性分析。
    """
    np.random.seed(42)
    z = np.random.randn(n_samples) * sigma_q + mu_q

    # 重建项：E_q[log p(x|z)]
    reconstruction = np.mean(-0.5 * np.log(2 * np.pi) - np.log(sigma_obs) - 0.5 * ((x_obs - z) / sigma_obs)**2)

    # KL项：KL(q||p(z)) = E_q[log q(z) - log p(z)]
    log_pz = np.full_like(z, -1e30)
    for w, mu, tau in zip(prior_weights, prior_means, prior_stds):
        log_comp = np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(tau) - 0.5 * ((z - mu) / tau)**2
        log_pz = np.logaddexp(log_pz, log_comp)
    log_qz = -0.5 * np.log(2 * np.pi) - np.log(sigma_q) - 0.5 * ((z - mu_q) / sigma_q)**2
    kl = np.mean(log_qz - log_pz)

    return reconstruction, kl

# 扫描不同的q参数，观察重建项和KL项的变化
mu_range = np.linspace(-3, 3, 100)
sigma_range = [0.3, 0.5, 1.0, 2.0]

print(f"\n扫描不同的q=N(μ,σ²)，观察重建项与KL项的变化：")
print(f"{'μ_q':>8s} | {'σ_q':>6s} | {'重建项':>10s} | {'KL项':>10s} | {'ELBO':>10s}")
print("-" * 60)

for sigma_q in [0.5, 1.0, 2.0]:
    for mu_q in [-2.0, 0.0, 0.5, 1.0]:
        recon, kl = compute_reconstruction_and_kl(mu_q, sigma_q)
        elbo = recon - kl
        print(f"{mu_q:>8.2f} | {sigma_q:>6.2f} | {recon:>10.4f} | {kl:>10.4f} | {elbo:>10.4f}")

print(f"\n[关键观察]")
print(f"  1. 当μ_q接近x_obs={x_obs}时，重建项最大")
print(f"  2. 当σ_q较小时，KL项较小（q接近先验）")
print(f"  3. ELBO最大值 = 重建项与KL项的权衡")


# ============================================================
# ★ 新增：真正优化 q* 并量化变分间隙
# 用 minimize 找出 argmax ELBO 的 q*，并报告 ELBO* vs log p(x) 的差距
# 这是"ELBO是变分下界"这一核心结论的定量证据
#
# 关键设计：所有计算（优化目标、最终报告的重建项/KL项/ELBO）都使用同一组
# 公共随机数 opt_eps，避免不同随机源造成 recon - kl ≠ elbo 的不一致。
# ============================================================
print("\n" + "=" * 60)
print("★ 变分下界定量验证：优化 q* 并量化变分间隙")
print("=" * 60)

# 用 common random numbers 技巧让优化目标光滑（且与最终报告共享同一组样本）
# 大样本量（200000）以降低MC噪声，确保 gap ≥ 0 的理论性质不被噪声打破
n_opt_samples = 200000
opt_eps = np.random.randn(n_opt_samples)  # 固定基础噪声，所有评估共享

def neg_elbo_with_decomp(params):
    """负ELBO（用于 minimize 优化）——同时返回重建项、KL项、ELBO 三元组

    所有三项基于同一组公共随机数 opt_eps 计算，保证 recon - kl ≡ elbo（恒等式）。
    优化器只用到 -elbo 部分，但返回完整三元组便于最终报告。
    """
    mu_q, log_sigma_q = params
    sigma_q = np.exp(log_sigma_q)
    z = mu_q + sigma_q * opt_eps
    # 重建项
    log_pxz = -0.5 * np.log(2 * np.pi) - np.log(sigma_obs) - 0.5 * ((x_obs - z) / sigma_obs)**2
    # KL项需要的 log p(z)
    log_pz = np.full_like(z, -1e30)
    for w, mu, tau in zip(prior_weights, prior_means, prior_stds):
        log_comp = np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(tau) - 0.5 * ((z - mu) / tau)**2
        log_pz = np.logaddexp(log_pz, log_comp)
    # log q(z)
    log_qz = -0.5 * np.log(2 * np.pi) - np.log(sigma_q) - 0.5 * ((z - mu_q) / sigma_q)**2

    recon = np.mean(log_pxz)
    kl = np.mean(log_qz - log_pz)
    elbo = recon - kl  # 与 np.mean(log_pxz + log_pz - log_qz) 数值恒等
    return -elbo, recon, kl


def neg_elbo_only(params):
    """包装函数：只返回 -ELBO（用于 minimize）"""
    return neg_elbo_with_decomp(params)[0]


# 网格搜索作为全局起点（粗网格即可，因为 Nelder-Mead 会精修）
mu_candidates = np.linspace(-3, 3, 15)
sigma_candidates = np.linspace(0.2, 2.0, 10)
best_grid_val = np.inf
best_grid_params = [0.0, 0.0]
for mu_c in mu_candidates:
    for sigma_c in sigma_candidates:
        val = neg_elbo_only([mu_c, np.log(sigma_c)])
        if val < best_grid_val:
            best_grid_val = val
            best_grid_params = [mu_c, np.log(sigma_c)]

result = minimize(neg_elbo_only, best_grid_params, method='Nelder-Mead',
                  options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
if not result.success:
    print(f"  [警告] Nelder-Mead 未完全收敛: {result.message}")
mu_q_opt = result.x[0]
sigma_q_opt = np.exp(result.x[1])

# ★ 关键：用同一组 opt_eps 重新计算最优点的三项，保证 recon - kl = elbo 严格相等
_, recon_opt, kl_opt = neg_elbo_with_decomp([mu_q_opt, np.log(sigma_q_opt)])
elbo_opt = recon_opt - kl_opt  # 精确等于（同一组样本的代数恒等）
variational_gap = log_px - elbo_opt

print(f"\n[优化结果]")
print(f"  q*(z) = N({mu_q_opt:.4f}, {sigma_q_opt:.4f}²)")
print(f"  重建项 E_q*[log p(x|z)] = {recon_opt:.4f}")
print(f"  KL正则项 KL(q*||p(z))   = {kl_opt:.4f}")
print(f"  ELBO* = 重建项 - KL项 = {recon_opt:.4f} - {kl_opt:.4f} = {elbo_opt:.4f}  (代数恒等)")
print(f"  log p(x) = {log_px:.4f}")
print(f"  变分间隙 log p(x) - ELBO* = {variational_gap:.4f}")
print(f"  相对间隙 = {variational_gap / abs(log_px) * 100:.2f}%")

# 安全检查：理论上 variational_gap ≥ 0 必须成立（变分下界）
# 但MC噪声可能让估计值出现微小负值，加保护避免学生误解
if variational_gap < -1e-3:
    # 显著为负：不应仅是MC噪声，可能是代码bug
    print(f"\n  [警告] 变分间隙显著为负（{variational_gap:.4f}），")
    print(f"         这不应仅由MC噪声引起，请检查代码逻辑。")
elif variational_gap < 0:
    # 微小负值：理论上不应出现，但MC噪声在 ±1e-3 量级属正常
    print(f"\n  [注意] 变分间隙为微小负值（{variational_gap:.2e}），属于MC噪声范围，")
    print(f"         理论上变分间隙 ≥ 0 严格成立（Jensen不等式）。")
else:
    print(f"\n[变分下界验证]")
    print(f"  ✓ ELBO* ({elbo_opt:.4f}) ≤ log p(x) ({log_px:.4f}) —— 定量验证ELBO是下界")
    print(f"  ✓ 变分间隙 ≥ 0：{variational_gap:.4f}（理论上严格成立）")

print(f"\n[为什么ELBO永远不能碰到 log p(x)？]")
print(f"  本例中真实后验 p(z|x) 是双分量高斯混合（双峰）")
print(f"  而变分族 q=N(μ,σ²) 是单峰高斯，表达能力不足以拟合双峰结构")
print(f"  即使优化到最优 q*，单高斯也无法同时覆盖两个模态")
print(f"  → 变分间隙 = KL(q* || p(z|x)) > 0 永远成立")
print(f"  → 缩小间隙的方法：扩大变分族（如混合高斯 q=N(μ₁,σ₁²)+N(μ₂,σ₂²)）")


# ============================================================
# 步骤2：重建-正则权衡的可视化
# ★ 原创设计
# ============================================================
print("\n" + "=" * 60)
print("步骤2：重建-正则权衡的可视化")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：重建项随q均值的变化
for sigma_q_fixed in sigma_range:
    reconstruction_terms = []
    for mu_q in mu_range:
        recon, _ = compute_reconstruction_and_kl(mu_q, sigma_q_fixed)
        reconstruction_terms.append(recon)
    axes[0].plot(mu_range, reconstruction_terms, label=r'$\sigma_q=' + f'{sigma_q_fixed}$')

axes[0].set_xlabel(r'$\mu_q$')
axes[0].set_ylabel(r'$\mathbb{E}_q[\log p(x|z)]$')
axes[0].set_title('Reconstruction term vs $q$ mean')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 右图：ELBO分解（σ_q=1.0）
sigma_q = 1.0
elbos_scan = []
reconstruction_terms = []
kl_terms = []

for mu_q in mu_range:
    recon, kl = compute_reconstruction_and_kl(mu_q, sigma_q)
    reconstruction_terms.append(recon)
    kl_terms.append(kl)
    elbos_scan.append(recon - kl)

axes[1].plot(mu_range, elbos_scan, 'b-', lw=2, label=r'$\mathrm{ELBO} = \mathrm{Recon} - \mathrm{KL}$')
axes[1].plot(mu_range, reconstruction_terms, 'r--', label=r'Reconstruction $\mathbb{E}_q[\log p(x|z)]$')
axes[1].plot(mu_range, [-k for k in kl_terms], 'g--', label=r'$-\mathrm{KL}(q\|p)$')
axes[1].axhline(y=log_px, color='k', linestyle=':', lw=1, label=r'$\log p(x) = ' + f'{log_px:.2f}$')

axes[1].set_xlabel(r'$\mu_q$')
axes[1].set_ylabel(r'Value')
axes[1].set_title(r'ELBO decomposition ($\sigma_q=1.0$)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_重建与正则.png'), dpi=100)
plt.close()
print(f"\n图表已保存: 步骤2_重建与正则.png")

print(f"\n[可视化分析]")
print(f"  左图：重建项随q均值的变化")
print(f"    - 当μ_q接近x_obs时，重建项最大")
print(f"    - σ_q越大，重建项越平滑（对μ_q不敏感）")
print(f"  右图：ELBO分解（σ_q=1.0）")
print(f"    - ELBO = 重建项 - KL项")
print(f"    - 重建项驱动q拟合数据")
print(f"    - KL项惩罚q偏离先验")
print(f"    - ELBO最大值 = 两者权衡的最优点")

print(f"\n[与MAP估计的统一]")
print(f"  MAP估计：min_x [-log p(y|x) + λR(x)]")
print(f"           = min_x [数据项 + 正则项]")
print(f"  变分推断：max_q [E_q[log p(x|z)] - KL(q||p(z))]")
print(f"           = max_q [重建项 - 正则项]")
print(f"  两者具有完全相同的结构！")
print(f"  关键差异：")
print(f"    - MAP是点估计（优化x）")
print(f"    - 变分推断是分布估计（优化q）")
print(f"    - 当q退化为δ函数时，ELBO退化为MAP目标")

print(f"\n[核心结论]")
print(f"  1. ELBO = 重建项 - KL正则项")
print(f"  2. 重建项驱动数据拟合，KL项约束先验偏离")
print(f"  3. 与MAP估计的数据项+正则项结构完全对应")
print(f"  4. 确定性正则化是变分推断的退化形式")

print(f"\n{'='*60}")
print("第八章配套实验8.4-2完成！")
