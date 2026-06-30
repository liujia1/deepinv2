# -*- coding: utf-8 -*-
"""
实验8.2-1 ELBO计算——1D高斯混合模型
对应章节：8.2（ELBO推导：证据下界）
素材来源：🆕 新写（教学用简化案例）
★ 原创设计：1D高斯混合ELBO的完整数值验证

知识点：
  - ELBO的定义：ELBO(q) = E_q[log p(x,z)] - E_q[log q(z)]
  - Jensen不等式推导：log p(x) ≥ ELBO(q)
  - 两种ELBO分解形式：联合-熵 vs 重建-正则化
  - 变分间隙：log p(x) - ELBO(q) = KL(q || p(z|x))

实验内容：
  步骤1：Jensen不等式验证——ELBO ≤ log p(x)
  步骤2：ELBO的两种分解形式验证
  步骤3：变分间隙与q的接近程度——q→p(z|x)时ELBO→log p(x)

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
    SAVE_DIR = os.path.join(_gdrive, '实验8.2-1')
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
# p(z) = 0.3*N(-2,1) + 0.7*N(1,1)      （先验/潜变量分布）
# p(x|z) = N(x; z, σ_obs²)              （似然/观测模型）
# p(z|x) = ?                              （后验，解析可求）
# ============================================================

# 模型参数
prior_weights = [0.3, 0.7]
prior_means = [-2.0, 1.0]
prior_stds = [1.0, 1.0]
sigma_obs = 0.5  # 观测噪声

# 观测值
x_obs = 0.5

print("=" * 60)
print("实验8.2-1：1D高斯混合ELBO计算")
print("=" * 60)
print(f"\n模型设定：")
print(f"  先验 p(z) = {prior_weights[0]}*N({prior_means[0]},{prior_stds[0]}²) + {prior_weights[1]}*N({prior_means[1]},{prior_stds[1]}²)")
print(f"  似然 p(x|z) = N(x; z, {sigma_obs}²)")
print(f"  观测值 x = {x_obs}")


# ============================================================
# 解析计算真实后验 p(z|x)
# p(z|x) ∝ p(x|z)·p(z) = N(x; z, σ²) · [π₁N(z;μ₁,τ₁²) + π₂N(z;μ₂,τ₂²)]
# 共轭高斯：N(x; z, σ²)·N(z; μₖ, τₖ²) = N(z; mₖ, sₖ²) · N(x; μₖ, σ²+τₖ²)
# 其中 mₖ = (x/σ² + μₖ/τₖ²) / (1/σ² + 1/τₖ²), sₖ² = 1/(1/σ² + 1/τₖ²)
# ============================================================

def compute_posterior(x, weights, means, stds, sigma_obs):
    """解析计算高斯混合先验+高斯似然的后验"""
    post_weights = []
    post_means = []
    post_stds = []

    for w, mu, tau in zip(weights, means, stds):
        # 后验分量参数
        s2 = 1.0 / (1.0 / sigma_obs**2 + 1.0 / tau**2)
        s = np.sqrt(s2)
        m = s2 * (x / sigma_obs**2 + mu / tau**2)

        # 后验分量权重 ∝ w * N(x; mu, sigma_obs²+tau²)
        marginal_std = np.sqrt(sigma_obs**2 + tau**2)
        marginal_log = -0.5 * ((x - mu) / marginal_std)**2 - np.log(marginal_std)

        post_weights.append(w * np.exp(marginal_log))
        post_means.append(m)
        post_stds.append(s)

    # 归一化权重
    total = sum(post_weights)
    post_weights = [pw / total for pw in post_weights]

    return post_weights, post_means, post_stds

post_w, post_m, post_s = compute_posterior(x_obs, prior_weights, prior_means, prior_stds, sigma_obs)

print(f"\n真实后验 p(z|x={x_obs}):")
for i in range(len(post_w)):
    print(f"  分量{i+1}: w={post_w[i]:.4f}, μ={post_m[i]:.4f}, σ={post_s[i]:.4f}")

# 计算log p(x)（证据）
def log_marginal(x, weights, means, stds, sigma_obs):
    """计算log p(x) = log Σₖ wₖ·N(x; μₖ, σ_obs²+τₖ²)"""
    terms = []
    for w, mu, tau in zip(weights, means, stds):
        marginal_std = np.sqrt(sigma_obs**2 + tau**2)
        log_term = np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(marginal_std) \
                   - 0.5 * ((x - mu) / marginal_std)**2
        terms.append(log_term)
    # logsumexp
    max_term = max(terms)
    return max_term + np.log(sum(np.exp(t - max_term) for t in terms))

log_px = log_marginal(x_obs, prior_weights, prior_means, prior_stds, sigma_obs)
print(f"\nlog p(x={x_obs}) = {log_px:.6f}")


# ============================================================
# 步骤1：Jensen不等式验证——ELBO ≤ log p(x)
# ★ 原创设计：用不同的q(z)验证ELBO ≤ log p(x)
# ============================================================
print("\n" + "=" * 60)
print("步骤1：Jensen不等式验证——ELBO ≤ log p(x)")
print("=" * 60)

def compute_elbo(x, q_means, q_stds, q_weights, prior_weights, prior_means, prior_stds, sigma_obs, n_samples=100000):
    """数值计算ELBO = E_q[log p(x,z)] - E_q[log q(z)]"""
    # 从q采样
    z_samples = []
    for w, m, s in zip(q_weights, q_means, q_stds):
        n_k = int(w * n_samples)
        z_samples.extend(np.random.randn(n_k) * s + m)
    z_samples = np.array(z_samples)

    # log p(x,z) = log p(x|z) + log p(z)
    log_pxz = -0.5 * np.log(2 * np.pi) - np.log(sigma_obs) - 0.5 * ((x - z_samples) / sigma_obs)**2  # log p(x|z)
    log_pz = np.full_like(z_samples, -1e30)
    for w, mu, tau in zip(prior_weights, prior_means, prior_stds):
        log_comp = np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(tau) - 0.5 * ((z_samples - mu) / tau)**2
        log_pz = np.logaddexp(log_pz, log_comp)
    log_joint = log_pxz + log_pz

    # log q(z)
    log_qz = np.full_like(z_samples, -1e30)
    for w, m, s in zip(q_weights, q_means, q_stds):
        log_comp = np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(s) - 0.5 * ((z_samples - m) / s)**2
        log_qz = np.logaddexp(log_qz, log_comp)

    elbo = np.mean(log_joint - log_qz)
    return elbo

# 测试不同的q
test_qs = [
    ("q = N(0, 1)（标准正态）", [1.0], [0.0], [1.0]),
    ("q = N(0.5, 0.5)（接近后验）", [1.0], [0.5], [0.5]),
    ("q = p(z|x)（真实后验）", post_w, post_m, post_s),
    ("q = 0.5*N(-1,0.5)+0.5*N(1,0.5)（双峰）", [0.5, 0.5], [-1.0, 1.0], [0.5, 0.5]),
    ("q = N(5, 0.1)（远离后验）", [1.0], [5.0], [0.1]),
]

print(f"{'q(z)':<35s} | {'ELBO':>10s} | {'log p(x)':>10s} | {'Gap':>10s} | {'ELBO ≤ log p(x)?':>15s}")
print("-" * 90)
for name, q_w, q_m, q_s in test_qs:
    np.random.seed(42)
    elbo = compute_elbo(x_obs, q_m, q_s, q_w, prior_weights, prior_means, prior_stds, sigma_obs)
    gap = log_px - elbo
    check = "✓" if elbo <= log_px + 1e-3 else "✗"
    print(f"{name:<35s} | {elbo:>10.4f} | {log_px:>10.4f} | {gap:>10.4f} | {check:>15s}")

print(f"\n核心结论：")
print(f"  1. 所有q的ELBO ≤ log p(x) = {log_px:.4f}（Jensen不等式验证成功）")
print(f"  2. q=p(z|x)时ELBO最接近log p(x)，Gap≈0")
print(f"  3. q偏离后验越远，ELBO越小，变分间隙越大")


# ============================================================
# 步骤2：ELBO两种分解形式验证
# 分解一：ELBO = E_q[log p(x,z)] + H(q)      （联合-熵）
# 分解二：ELBO = E_q[log p(x|z)] - KL(q||p(z)) （重建+正则）
# ============================================================
print("\n" + "=" * 60)
print("步骤2：ELBO两种分解形式验证")
print("=" * 60)

for name, q_w, q_m, q_s in test_qs[:3]:  # 取前3个q
    np.random.seed(42)
    
    # 采样
    z_samples = []
    for w, m, s in zip(q_w, q_m, q_s):
        n_k = int(w * 100000)
        z_samples.extend(np.random.randn(n_k) * s + m)
    z_samples = np.array(z_samples)

    # log p(x,z)
    log_pxz = -0.5 * np.log(2 * np.pi) - np.log(sigma_obs) - 0.5 * ((x_obs - z_samples) / sigma_obs)**2
    log_pz = np.full_like(z_samples, -1e30)
    for w, mu, tau in zip(prior_weights, prior_means, prior_stds):
        log_comp = np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(tau) - 0.5 * ((z_samples - mu) / tau)**2
        log_pz = np.logaddexp(log_pz, log_comp)
    log_joint = log_pxz + log_pz

    # log q(z)
    log_qz = np.full_like(z_samples, -1e30)
    for w, m, s in zip(q_w, q_m, q_s):
        log_comp = np.log(w) - 0.5 * np.log(2 * np.pi) - np.log(s) - 0.5 * ((z_samples - m) / s)**2
        log_qz = np.logaddexp(log_qz, log_comp)

    # 分解一：联合-熵
    eq_log_joint = np.mean(log_joint)
    H_q = -np.mean(log_qz)
    elbo_decomp1 = eq_log_joint + H_q

    # 分解二：重建+正则
    eq_log_pxz_cond = np.mean(log_pxz)  # E_q[log p(x|z)]
    # KL(q||p(z)) = E_q[log q(z)] - E_q[log p(z)]
    kl_qp = np.mean(log_qz) - np.mean(log_pz)
    elbo_decomp2 = eq_log_pxz_cond - kl_qp

    # 直接计算
    elbo_direct = np.mean(log_joint - log_qz)

    print(f"\n  q = {name}:")
    print(f"    分解一（联合-熵）:  E[log p(x,z)]={eq_log_joint:.4f} + H(q)={H_q:.4f} = {elbo_decomp1:.4f}")
    print(f"    分解二（重建+正则）: E[log p(x|z)]={eq_log_pxz_cond:.4f} - KL(q||p)={kl_qp:.4f} = {elbo_decomp2:.4f}")
    print(f"    直接计算:           ELBO = {elbo_direct:.4f}")
    # 三种分解本质上是同一组蒙特卡罗样本的代数重组，理论误差应为机器精度级别（<1e-10）
    tol = 1e-9
    print(f"    三者一致（容差{tol}）: {abs(elbo_decomp1 - elbo_decomp2) < tol and abs(elbo_decomp1 - elbo_direct) < tol}")


# ============================================================
# 步骤3：q→p(z|x)时ELBO→log p(x)——变分间隙可视化
# ★ 原创设计：参数化q从先验平滑过渡到后验
# ============================================================
print("\n" + "=" * 60)
print("步骤3：变分间隙随q接近后验的变化")
print("=" * 60)

# 参数化：q_α = (1-α)·p(z) + α·p(z|x)
# α=0: q=先验, α=1: q=后验
alphas = np.linspace(0, 1, 50)
elbos = []
gaps = []

for alpha in alphas:
    # 混合q_α
    q_w_alpha = [(1 - alpha) * prior_weights[0] + alpha * post_w[0],
                 (1 - alpha) * prior_weights[1] + alpha * post_w[1]]
    q_m_alpha = [(1 - alpha) * prior_means[0] + alpha * post_m[0],
                 (1 - alpha) * prior_means[1] + alpha * post_m[1]]
    q_s_alpha = [np.sqrt((1 - alpha) * prior_stds[0]**2 + alpha * post_s[0]**2),
                 np.sqrt((1 - alpha) * prior_stds[1]**2 + alpha * post_s[1]**2)]

    np.random.seed(42)
    elbo = compute_elbo(x_obs, q_m_alpha, q_s_alpha, q_w_alpha,
                        prior_weights, prior_means, prior_stds, sigma_obs, n_samples=50000)
    elbos.append(elbo)
    gaps.append(log_px - elbo)

# 可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ELBO vs α - 使用LaTeX格式显示数学符号
ax1.plot(alphas, elbos, 'b-', lw=2, label=r'$\mathrm{ELBO}(q_\alpha)$')
ax1.axhline(y=log_px, color='r', linestyle='--', lw=2, 
           label=r'$\log p(x) = ' + f'{log_px:.4f}$')
ax1.set_xlabel(r'$\alpha$（0=先验, 1=后验）')
ax1.set_ylabel(r'$\mathrm{ELBO}$')
ax1.set_title('ELBO随q接近后验的变化')
ax1.legend()
ax1.grid(alpha=0.3)

# 变分间隙 vs α - 使用LaTeX格式显示数学符号
ax2.plot(alphas, gaps, 'g-', lw=2, 
        label=r'变分间隙 = $\log p(x) - \mathrm{ELBO}$')
ax2.set_xlabel(r'$\alpha$（0=先验, 1=后验）')
ax2.set_ylabel('变分间隙')
ax2.set_title('变分间隙随q接近后验的变化')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_变分间隙.png'), dpi=100)
plt.close()
print(f"\n图表已保存: 步骤3_变分间隙.png")

print(f"\n当α→1（q→p(z|x)）时：")
print(f"  ELBO → log p(x) = {log_px:.4f}")
print(f"  变分间隙 → 0")
print(f"这验证了8.2节的核心结论：ELBO最大化的等价于最小化KL(q||p(z|x))")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验8.2-1 总结")
print("=" * 60)
print("1. Jensen不等式验证：对任意q，ELBO(q) ≤ log p(x)")
print("   当q=p(z|x)时，ELBO=log p(x)，变分间隙=0")
print("2. ELBO两种分解：")
print("   联合-熵：ELBO = E[log p(x,z)] + H(q)")
print("   重建-正则：ELBO = E[log p(x|z)] - KL(q||p(z))")
print("3. 变分间隙 = log p(x) - ELBO = KL(q||p(z|x))")
print("   q越接近后验，间隙越小；q=p(z|x)时间隙=0")

print(f"\n{'='*60}")
print("第八章配套实验8.2-1完成！")