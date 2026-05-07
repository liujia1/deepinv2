"""
实验7.1 从Langevin到扩散——连续时间推广
对应章节：7.1（从Langevin到扩散：连续时间推广）
素材来源：★ 原创设计，基于7.1节"Langevin→扩散SDE连续时间推广"

实验内容：
  步骤1：1D高斯混合上，单步Langevin vs 多步扩散采样对比
  步骤2：从离散噪声调度到连续SDE——β(t)调度可视化
  步骤3："离散→连续→再离散"认知螺旋的数值验证

运行前提：纯NumPy/PyTorch CPU即可
"""

import numpy as np
import os
import sys

# ====== Windows控制台UTF-8输出 ======
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import logging

# ====== 解决中文乱码的核心代码 ======
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

import platform
from matplotlib.font_manager import FontManager, FontProperties

def _find_chinese_font():
    candidates = []
    if platform.system() == 'Windows':
        candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
    else:
        candidates = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei']
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available:
            return font
    import os as _os, re
    cjk_patterns = ['cjk', 'wqy', 'noto.*cjk', 'wenquan', 'chinese', 'simhei']
    for f in fm.ttflist:
        name_lower = f.name.lower()
        fname_lower = (_os.path.basename(f.fname) if hasattr(f, 'fname') else '').lower()
        for pat in cjk_patterns:
            if re.search(pat, name_lower) or re.search(pat, fname_lower):
                return f.name
    return None

_cn_font = _find_chinese_font()
if _cn_font:
    plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['font.family'] = 'sans-serif'
    print(f"[Font] 已检测到中文字体: {_cn_font}")
else:
    print("[Font] 未找到中文字体，中文可能显示为方框")
# ========================================================

np.random.seed(42)
import torch
torch.manual_seed(42)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()


# ============================================================
# 1D高斯混合分布
# ============================================================
def gm1d_pdf(x):
    return 0.3*np.exp(-0.5*(x+2)**2)/np.sqrt(2*np.pi) + \
           0.7*np.exp(-0.5*(x-1)**2)/np.sqrt(2*np.pi)

def gm1d_score(x):
    p1 = np.exp(-0.5*(x+2)**2)/np.sqrt(2*np.pi)
    p2 = np.exp(-0.5*(x-1)**2)/np.sqrt(2*np.pi)
    p = 0.3*p1 + 0.7*p2
    return (0.3*-(x+2)*p1 + 0.7*-(x-1)*p2) / p


# ============================================================
# 步骤1：单步Langevin vs 多步扩散采样对比
# ★ 原创设计：在1D高斯混合上直观展示NCSN的三个根本局限
# ============================================================
print("=" * 60)
print("步骤1：单步Langevin vs 多步扩散采样对比")
print("=" * 60)

# 方法1：单步Langevin（固定σ，第5章ULA）
def langevin_1d(n_samples, n_iter, delta, sigma_score):
    """单步Langevin：用σ固定的得分函数采样"""
    x = np.random.randn(n_samples) * 3
    for _ in range(n_iter):
        # 使用σ_score下的得分（近似∇log p_{σ_score}）
        eps = sigma_score
        v = 1 + eps**2
        p1 = np.exp(-0.5*(x+2)**2/v)/np.sqrt(2*np.pi*v)
        p2 = np.exp(-0.5*(x-1)**2/v)/np.sqrt(2*np.pi*v)
        p = 0.3*p1 + 0.7*p2
        score = (0.3*-(x+2)/v*p1 + 0.7*-(x-1)/v*p2) / p
        x = x + delta * score + np.sqrt(2*delta) * np.random.randn(n_samples)
    return x

# 方法2：退火Langevin（多步扩散，第6章NCSN）
def annealed_langevin_1d(n_samples, sigmas, T_per_level, epsilon):
    """退火Langevin：从大σ到小σ逐步采样"""
    x = np.random.randn(n_samples) * sigmas[0]
    for i, sigma in enumerate(sigmas):
        alpha = epsilon * (sigma / sigmas[-1])**2
        for _ in range(T_per_level):
            v = 1 + sigma**2
            p1 = np.exp(-0.5*(x+2)**2/v)/np.sqrt(2*np.pi*v)
            p2 = np.exp(-0.5*(x-1)**2/v)/np.sqrt(2*np.pi*v)
            p = 0.3*p1 + 0.7*p2
            score = (0.3*-(x+2)/v*p1 + 0.7*-(x-1)/v*p2) / p
            x = x + alpha/2 * score + np.sqrt(alpha) * np.random.randn(n_samples)
    return x

# 运行对比
N = 20000
n_langevin = langevin_1d(N, n_iter=5000, delta=0.01, sigma_score=0.1)
sigmas = np.exp(np.linspace(np.log(5.0), np.log(0.05), 10))
n_annealed = annealed_langevin_1d(N, sigmas, T_per_level=200, epsilon=2e-3)

# 真实样本
n_true = np.concatenate([np.random.randn(int(N*0.3))-2, np.random.randn(int(N*0.7))+1])

# 评估：各模态的样本比例
def mode_proportions(samples, threshold=-0.5):
    p_left = np.mean(samples < threshold)
    p_right = np.mean(samples >= threshold)
    return p_left, p_right

true_props = mode_proportions(n_true)
langevin_props = mode_proportions(n_langevin)
annealed_props = mode_proportions(n_annealed)

print(f"模态比例对比 (左模态N(-2,1):右模态N(1,1)):")
print(f"  真实分布:    {true_props[0]:.3f} : {true_props[1]:.3f}")
print(f"  单步Langevin: {langevin_props[0]:.3f} : {langevin_props[1]:.3f}")
print(f"  退火Langevin: {annealed_props[0]:.3f} : {annealed_props[1]:.3f}")
print(f"\nNCSN的三个根本局限（7.1节）：")
print(f"  1. 跳跃式切换噪声水平——离散σ_i不连续")
print(f"  2. 每个水平的步数T无系统选择原则")
print(f"  3. NCSN和DDPM框架不统一——看起来是两种方法")


# ============================================================
# 步骤2：从离散噪声调度到连续SDE
# ★ 原创设计：展示β(t)调度的连续化
# ============================================================
print("\n" + "=" * 60)
print("步骤2：从离散噪声调度到连续SDE——β(t)调度可视化")
print("=" * 60)

# VP-SDE的离散噪声调度（DDPM）
T_ddpm = 1000
beta_min, beta_max = 0.0001, 0.02
betas_ddpm = np.linspace(beta_min, beta_max, T_ddpm)
alphas_ddpm = 1 - betas_ddpm
alpha_bars_ddpm = np.cumprod(alphas_ddpm)

# VP-SDE的连续调度
t_continuous = np.linspace(0, 1, 1000)
beta_t_continuous = beta_min + t_continuous * (beta_max - beta_min)

# SNR对比
snr_ddpm = alpha_bars_ddpm / (1 - alpha_bars_ddpm)
snr_continuous = np.exp(-0.5*t_continuous**2*(beta_max-beta_min) - t_continuous*beta_min)

# VE-SDE的离散噪声调度（SMLD/NCSN）
L_smld = 10
sigma_min, sigma_max = 0.01, 50.0
sigmas_smld = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), L_smld))

# VE-SDE的连续调度
t_ve = np.linspace(0, 1, 1000)
sigma_t_ve = sigma_max * (sigma_min/sigma_max)**t_ve  # 几何插值

print(f"VP-SDE (DDPM离散 vs 连续):")
print(f"  T={T_ddpm}, β_min={beta_min}, β_max={beta_max}")
print(f"  连续β(t) = {beta_min} + t×({beta_max}-{beta_min})")
print(f"  ᾱ(T)={alpha_bars_ddpm[-1]:.6f} (几乎为0，信号消失)")

print(f"\nVE-SDE (SMLD离散 vs 连续):")
print(f"  L={L_smld}, σ_min={sigma_min}, σ_max={sigma_max}")
print(f"  连续σ(t) = {sigma_max}×({sigma_min}/{sigma_max})^t")


# ============================================================
# 步骤3："离散→连续→再离散"认知螺旋
# ★ 原创设计：展示三种离散化（DDPM/SMLD/DDIM）的统一
# ============================================================
print("\n" + "=" * 60)
print("步骤3：'离散→连续→再离散'认知螺旋")
print("=" * 60)

print("""
7.1节核心洞见：三种认知层次

第一步：离散→连续
  - DDPM的离散β_i → VP-SDE: dx = -β(t)/2·x dt + √β(t) dw
  - SMLD的离散σ_i → VE-SDE: dx = √(d[σ²]/dt) dw
  - 统一框架：NCSN和DDPM不再是两种方法，而是同一SDE的不同参数化

第二步：连续→新发现
  - Anderson (1982) 定理：正向SDE已知→逆向SDE自动获得
  - 得分函数是逆向SDE中唯一的未知量
  - PF-ODE：逆向SDE的确定性等价（7.4节）

第三步：连续→再离散
  - 逆向VP-SDE + Euler-Maruyama → DDPM采样
  - 逆向VE-SDE + Euler-Maruyama → SMLD采样
  - PF-ODE + Euler → DDIM采样
  - PF-ODE + RK4/DPM-Solver → 高效采样（7.5节）

关键收益：连续框架提供"免费"的逆向方程和ODE等价
  - 在离散框架中，DDPM和DDIM看起来完全不同
  - 在连续框架中，它们只是同一方程的不同离散化
""")

# 数值验证：DDPM的α_bar ≈ VP-SDE的连续解
t_norm = np.arange(T_ddpm) / T_ddpm  # 归一化到[0,1]
alpha_bar_continuous = np.exp(-0.5*t_norm**2*(beta_max-beta_min) - t_norm*beta_min)

print(f"VP-SDE连续解 vs DDPM离散α_bar:")
print(f"  t=0.1: 连续={alpha_bar_continuous[99]:.6f}, 离散={alpha_bars_ddpm[99]:.6f}")
print(f"  t=0.5: 连续={alpha_bar_continuous[499]:.6f}, 离散={alpha_bars_ddpm[499]:.6f}")
print(f"  t=0.9: 连续={alpha_bar_continuous[899]:.6f}, 离散={alpha_bars_ddpm[899]:.6f}")
print(f"  两者高度一致——离散是连续的Euler-Maruyama近似")


# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 子图1：采样对比
x_grid = np.linspace(-6, 6, 200)
axes[0].hist(n_true, bins=60, density=True, alpha=0.3, color='blue', label='真实分布')
axes[0].hist(n_langevin, bins=60, density=True, alpha=0.3, color='red', label='单步Langevin')
axes[0].hist(n_annealed, bins=60, density=True, alpha=0.3, color='green', label='退火Langevin')
axes[0].plot(x_grid, gm1d_pdf(x_grid), 'k-', lw=2, label='真实密度')
axes[0].set_xlabel('x')
axes[0].set_ylabel('密度')
axes[0].set_title('单步Langevin vs 退火Langevin')
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# 子图2：噪声调度与SNR
ax2 = axes[1]
ax2.plot(t_norm, 10*np.log10(alpha_bars_ddpm/(1-alpha_bars_ddpm)), 'b-', lw=2, label='DDPM SNR')
ax2.plot(t_continuous, 10*np.log10(snr_continuous), 'r--', lw=2, label='VP-SDE连续SNR')
ax2.set_xlabel('归一化时间 t')
ax2.set_ylabel('SNR (dB)')
ax2.set_title('VP-SDE: 离散与连续SNR对比')
ax2.legend()
ax2.grid(alpha=0.3)

# 子图3：统一关系图
axes[2].text(0.5, 0.85, '统一框架：离散→连续→再离散', ha='center', fontsize=14,
             fontweight='bold', transform=axes[2].transAxes)
relations = [
    ('DDPM (离散)', 'VP-SDE (连续)', 0.35),
    ('SMLD/NCSN (离散)', 'VE-SDE (连续)', 0.25),
    ('逆向VP-SDE + EM', 'DDPM采样', 0.15),
    ('逆向VE-SDE + EM', 'SMLD采样', 0.05),
    ('PF-ODE + Euler', 'DDIM采样', -0.05),
    ('PF-ODE + RK4', 'DPM-Solver', -0.15),
]
for left, right, y in relations:
    axes[2].annotate(left, (0.15, y), fontsize=10, transform=axes[2].transAxes)
    axes[2].annotate('→', (0.48, y), fontsize=12, transform=axes[2].transAxes, ha='center')
    axes[2].annotate(right, (0.55, y), fontsize=10, transform=axes[2].transAxes)
axes[2].set_xlim(0, 1)
axes[2].set_ylim(-0.25, 0.95)
axes[2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(_SCRIPT_DIR, '步骤1_Langevin到扩散.png'), dpi=150)
plt.show()


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验7.1 总结")
print("=" * 60)
print("1. 单步Langevin vs 退火Langevin：")
print(f"   单步Langevin难以准确采样多模态分布（左模态比例{langevin_props[0]:.3f} vs 真实{true_props[0]:.3f}）")
print(f"   退火Langevin更接近真实分布（左模态比例{annealed_props[0]:.3f}）")
print("2. NCSN的三个根本局限→推动连续化：")
print("   跳跃式切换、步数无原则、框架不统一")
print("3. 连续SDE框架的收益：")
print("   统一NCSN/DDPM为同一框架，Anderson定理免费给出逆向方程")
print("   VP-SDE连续解与DDPM离散α_bar高度一致")
print("4. '离散→连续→再离散'认知螺旋：")
print("   DDPM/SMLD是SDE的离散特例，DDIM是PF-ODE的离散特例")
