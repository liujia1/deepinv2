# -*- coding: utf-8 -*-
"""
实验10.4-1 层级VAE→扩散的连续极限
对应章节: 10.4 层级VAE→扩散的极限

知识点:
  - VP-SDE连续极限: 离散乘积→连续指数
  - 层级ELBO与VLB的对应关系
  - 扩散模型=固定编码器的层级VAE极限

实验内容:
  步骤1: 离散→连续极限验证 (不同L下的ab(t)对比)
  步骤2: 层级ELBO→VLB对应关系
  步骤3: VP-SDE形式推导

本实验不需要GPU，通过数值实验验证10.4节的核心数学结论。
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
    SAVE_DIR = os.path.join(_gdrive, '实验10.4-1')
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
import torch
torch.manual_seed(42)

print("\n" + "="*60)
print("实验10.4-1: 层级VAE→扩散的连续极限")
print("="*60)
print("对应章节: 10.4 层级VAE→扩散的极限")
print("使用设备: CPU (本实验不需要GPU)")


# ============================================================
# 步骤1：离散→连续极限验证 —— 10.4节核心
# ============================================================
print("\n" + "="*60)
print("步骤1：离散→连续极限验证")
print("="*60)
print("\n[核心思想]")
print("  当层级数L→∞时，离散乘积收敛到连续指数:")
print("  ab_L = prod_{i=1}^L (1 - beta/L) → exp(-beta t)")
print("  其中 t = i/L 为归一化时间")

beta_total = 1.0  # 总扩散量

print("\n不同层级数L下的离散ab vs 连续极限:")
print("     L      离散ab_L      连续极限      相对误差")
print("-" * 45)

for L in [5, 10, 50, 100, 500, 1000]:
    dt = beta_total / L
    alpha_per_step = 1 - dt
    # 离散乘积: ab_L = (1 - beta/L)^L 在 t=1 处
    alpha_bar_discrete = alpha_per_step ** L
    # 连续极限: ab(t=1) = exp(-beta)
    alpha_bar_cont = np.exp(-beta_total)

    rel_err = abs(alpha_bar_discrete - alpha_bar_cont) / alpha_bar_cont
    print(str(L).rjust(6) + "     " + str(round(alpha_bar_discrete, 6)).rjust(12) + "     " + str(round(alpha_bar_cont, 6)).rjust(12) + "     " + str(round(rel_err, 4)).rjust(10))

print("\n→ L增大时，离散乘积逐渐逼近连续指数 exp(-beta t)")
print("  这是VP-SDE的理论基础")


# ============================================================
# 步骤2：VP-SDE形式推导
# ============================================================
print("\n" + "="*60)
print("步骤2：VP-SDE形式推导")
print("="*60)
print("\n[VP-SDE的定义]")
print("  前向SDE: dx = -beta(t)/2 · x dt + sqrt(beta(t)) dW")
print("  边际分布: q(x_t|x_0) = N(x_t; sqrt(ab(t)) x_0, (1-ab(t))I)")
print("  其中 ab(t) = exp(-int_0^t beta(s) ds)")

print("\n[逆向SDE]")
print("  逆向SDE: dx = [beta(t)/2 · x - beta(t) · s_theta(x,t)] dt + sqrt(beta(t)) dW_bar")
print("  或用eps-prediction参数化:")
print("  dx = [beta(t)/2 · x + beta(t)/sqrt(1-ab(t)) · hat_eps_theta(x,t)] dt + sqrt(beta(t)) dW_bar")

print("\n[ODE形式 (Probability Flow ODE)]")
print("  去除随机项后得到确定性ODE:")
print("  dx = [beta(t)/2 · x - beta(t)/2 · s_theta(x,t)] dt")
print("  用于确定性采样和似然计算")


# ============================================================
# 步骤3：层级ELBO→VLB对应关系
# ============================================================
print("\n" + "="*60)
print("步骤3：层级ELBO→VLB对应关系")
print("="*60)
print("\n[对应关系表]")
print("""
层级VAE概念            →  扩散模型概念
────────────────────────────────────
编码器 q(z_l|z_{l-1})  →  前向加噪 q(x_t|x_{t-1})
解码器 p(z_{l-1}|z_l)  →  反向去噪 p_theta(x_{t-1}|x_t)
隐变量 z_l             →  噪声状态 x_t
先验 p(z_L)            →  先验 p(x_T) = N(0,I)
重参数化               →  直接采样 x_t = sqrt(ab_t) x_0 + sqrt(1-ab_t) eps
ELBO                   →  VLB

关键差异:
  层级VAE: 编码器参数被学习（自由推断）
  扩散模型: 编码器参数被固定（高斯转移，仅beta_t是超参数）
  → 扩散模型的"推断"无需训练，VLB只优化解码器（去噪网络）
""")

# 数值验证: 不同L下的ELBO近似
np.random.seed(42)
# 简化的2D高斯混合数据
x0_gmm = np.vstack([np.random.randn(500, 2) * 0.5 + [1, 0],
                     np.random.randn(500, 2) * 0.5 + [-1, 0]])

print("不同层级数L下的ELBO近似（2D高斯混合, beta_total=1.0）:")
print("     L      KL总和估计      每步beta")
print("-" * 35)

for L in [5, 10, 50, 100]:
    dt = 1.0 / L
    beta_step = dt
    alpha_step = 1 - beta_step

    # 简单估计: KL项的总和
    total_kl = 0
    for t in range(1, L + 1):
        ab_t = alpha_step ** t
        # KL(N(sqrt(ab_t)·mu0, ab_t·sigma0^2+(1-ab_t)) || N(0,I))
        # 对各向同性简化: KL ≈ 0.5*(ab_t*(mu0^2+sigma0^2) - 1 - log(ab_t*sigma0^2+(1-ab_t)))
        mu2 = 1.0  # E[mu0^2] 对双模态约1
        sigma2 = 0.25  # sigma0^2
        mean_sq = ab_t * (mu2 + sigma2) + (1 - ab_t)
        log_var = np.log(ab_t * sigma2 + (1 - ab_t))
        kl_t = 0.5 * (mean_sq - 1 - log_var)
        total_kl += kl_t

    print(str(L).rjust(6) + "        " + str(round(total_kl, 4)).rjust(12) + "        " + str(round(beta_step, 6)).rjust(10))

print("\n→ L增大时ELBO(=负KL)逐渐稳定，对应10.4节L→∞的连续极限")


# ============================================================
# 可视化
# ============================================================
print("\n" + "="*60)
print("生成可视化图表...")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) 不同L下的离散ab vs 连续极限
ax = axes[0, 0]
for L in [5, 10, 50, 100, 1000]:
    beta_total = 1.0
    dt = beta_total / L
    alpha_per_step = 1 - dt
    alpha_bar_L = torch.cumprod(torch.full((L,), alpha_per_step), dim=0)
    t_norm = torch.linspace(1/L, 1, L)  # 归一化时间
    ax.plot(t_norm.numpy(), alpha_bar_L.numpy(), linewidth=1.5 if L >= 50 else 1,
            label='$L=' + str(L) + '$')

# 连续极限: ab(t) = exp(-beta_total · t)
t_cont = torch.linspace(0, 1, 200)
alpha_bar_cont = torch.exp(-beta_total * t_cont)
ax.plot(t_cont.numpy(), alpha_bar_cont.numpy(), 'k--', linewidth=2, label='连续极限: $\\exp(-\\beta t)$')

ax.set_xlabel('归一化时间 $t/T$', fontsize=12)
ax.set_ylabel(r'$\bar{\alpha}(t)$', fontsize=12)
ax.set_title('(a) $L\\to\\infty$: 离散乘积→连续指数', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.annotate('$L\\to\\infty$', xy=(0.35, 0.5), xycoords='axes fraction',
            fontsize=10, color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeaa7', alpha=0.8))

# (b) VP-SDE的边际分布演化
ax = axes[0, 1]
t_cont = torch.linspace(0, 1, 100)
beta_func = torch.ones_like(t_cont) * beta_total  # beta(t) = beta (常数)
alpha_bar_t = torch.exp(-beta_func * t_cont)

ax.plot(t_cont.numpy(), torch.sqrt(alpha_bar_t).numpy(), 'b-', linewidth=2, label=r'$\sqrt{\bar{\alpha}(t)}$ ($x_0$系数)')
ax.plot(t_cont.numpy(), torch.sqrt(1 - alpha_bar_t).numpy(), 'r-', linewidth=2, label=r'$\sqrt{1-\bar{\alpha}(t)}$ (噪声系数)')
ax.set_xlabel('归一化时间 $t$', fontsize=12)
ax.set_ylabel('系数值', fontsize=12)
ax.set_title('(b) VP-SDE边际分布系数', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# (c) 不同噪声调度的对比
ax = axes[1, 0]
# DDPM线性调度
T = 1000
betas_linear = torch.linspace(1e-4, 0.02, T)
alphas_linear = 1.0 - betas_linear
alpha_bars_linear = torch.cumprod(alphas_linear, dim=0)

# 改进调度 (cosine schedule)
def cosine_beta_schedule(t, s=0.008):
    """Cosine schedule as proposed in improved DDPM"""
    return torch.cos((t + s) / (1 + s) * np.pi / 2) ** 2

t_norm = torch.linspace(0, 1, T)
alpha_bars_cosine = cosine_beta_schedule(t_norm)

ax.plot(range(1, T+1), alpha_bars_linear.numpy(), 'b-', linewidth=2, label='DDPM线性调度')
ax.plot(range(1, T+1), alpha_bars_cosine.numpy(), 'g-', linewidth=2, label='Cosine调度')
ax.set_xlabel('时间步 $t$', fontsize=12)
ax.set_ylabel(r'$\bar{\alpha}_t$', fontsize=12)
ax.set_title('(c) 不同噪声调度对比', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# (d) VP-SDE示意图
ax = axes[1, 1]
# 绘制VP-SDE的前向过程示意图
np.random.seed(42)
n_pts = 50
x0_pts = np.random.randn(n_pts) * 0.5

t_steps = [0, 0.2, 0.5, 0.8, 1.0]
colors = plt.cm.viridis(np.linspace(0, 1, len(t_steps)))

for i, t_val in enumerate(t_steps):
    ab = np.exp(-beta_total * t_val)
    noise = np.random.randn(n_pts)
    x_t = np.sqrt(ab) * x0_pts + np.sqrt(1 - ab) * noise
    offset = i * 3
    ax.scatter([offset] * n_pts, x_t, c=[colors[i]], s=30, alpha=0.7, label='$t=' + str(round(t_val, 1)) + '$')
    ax.text(offset, 3, '$t=' + str(round(t_val, 1)) + '$', ha='center', fontsize=9)

ax.set_xlabel('VP-SDE前向过程 →', fontsize=12)
ax.set_ylabel('$x_t$', fontsize=12)
ax.set_title('(d) VP-SDE前向过程演化', fontsize=13)
ax.set_ylim(-3, 3.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤1_离散到连续极限.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print("图表已保存: 步骤1_离散到连续极限.png")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "="*60)
print("实验10.4-1 总结")
print("="*60)
print("\n关键结论:")
print("\n1. 离散→连续极限 (步骤1)")
print("   - 当L→∞时: prod_{i=1}^L (1 - beta/L) → exp(-beta t)")
print("   - 这是VP-SDE的理论基础")
print("\n2. VP-SDE形式 (步骤2)")
print("   - 前向SDE: dx = -beta(t)/2 · x dt + sqrt(beta(t)) dW")
print("   - 边际分布: ab(t) = exp(-int_0^t beta(s) ds)")
print("   - 逆向SDE: 由得分函数或eps-prediction驱动")
print("\n3. 层级ELBO→VLB对应 (步骤3)")
print("   - 扩散模型 = 固定编码器的层级VAE连续极限")
print("   - 编码器固定 → 无需训练推断网络")
print("   - 只优化解码器（去噪网络）")
print("\n4. 三者的统一视角:")
print("   - 层级VAE: 通用框架，编码器可学习")
print("   - 扩散模型: 固定编码器，L=T步离散化")
print("   - VP-SDE: L→∞的连续极限，SDE形式")

print("\n" + "="*60)
print("实验10.4-1 完成!")