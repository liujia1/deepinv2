# -*- coding: utf-8 -*-
"""
实验9.2-1 重参数化技巧的数值验证
对应知识点：9.2节 重参数化技巧；附录9A REINFORCE梯度估计器

★ 原创设计：本实验通过数值对比系统验证重参数化技巧相对于REINFORCE的方差优势。
对测试函数 f(z)=z²，REINFORCE 的精确单样本方差为 μ⁴/σ² + 14μ² + 15σ²
（L个样本取均值再除以L），在σ→0时按 μ⁴/σ² 发散；
重参数化方差精确为 4σ²/L，在σ→0时趋于0。
注意：附录9A的 O(σ² + μ²/σ²) 只是μ固定、σ→0时的定性渐近标度，
不能当作系数为1的精确公式直接与实验数据比较。

实验内容:
  步骤1: 梯度可传播性验证
  步骤2: REINFORCE与重参数化梯度估计器对比
  步骤3: 方差随σ变化的标度行为（附录9A核心结论）
  步骤4: 样本效率对比（原创设计）

素材来源:
  - 实验9.2.py
  - 附录9A方差分析

运行前提: PyTorch, NumPy, CPU即可
"""

import sys
import io
import os
import numpy as np
import warnings
import logging

# 设置控制台输出为 UTF-8 (Windows 下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# 静默 matplotlib 相关警告
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
# 仅压制中文字体相关的告警，不做类别级通杀，以免吞掉真实问题
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*[Gg]lyph.*")
warnings.filterwarnings("ignore", message=".*findfont.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====== matplotlib 静默模式 ======
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
# =================================

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验9.2-1')
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

# 设置随机种子
np.random.seed(42)

import torch
torch.manual_seed(42)

# Checkpoint路径（保存数值计算结果）
RESULTS_CACHE_PATH = os.path.join(SAVE_DIR, 'numerical_results_cache.npz')

# ====== 全局配置参数（字面常量，不需要缓存） ======
# 步骤2使用的测试参数
MU_TEST = 2.0
SIGMA_TEST = 1.0
L_SAMPLES = 50  # 每次估计使用的样本数
TRUE_GRAD = 2 * MU_TEST  # ∇_μ E[z²] = 2μ
SIGMA_SMALL = 0.1  # 小σ测试

# 步骤3使用的参数（方差随σ变化）
MU_FIXED = 2.0
L_VAR = 50

# 步骤4使用的参数（样本效率对比）
MU_EFF = 2.0
SIGMA_EFF = 0.5
TRUE_GRAD_EFF = 2 * MU_EFF

# 试验规模与随机种子
N_TRIALS = 5000       # 步骤2重复试验次数
N_TRIALS_VAR = 3000   # 步骤3每个σ的重复次数
N_TRIALS_EFF = 2000   # 步骤4每个L的重复次数
SEED_MAIN, SEED_VAR, SEED_EFF = 42, 123, 456

# 缓存版本：任何实验参数/口径变更都应升版本号，使旧npz自动失效并重算
CACHE_VERSION = 2

# 全部实验参数（写入npz与json，保证结果脱离代码也可追溯、口径可核对）
EXPERIMENT_PARAMS = {
    'mu_test': MU_TEST, 'sigma_test': SIGMA_TEST, 'L_samples': L_SAMPLES,
    'sigma_small': SIGMA_SMALL,
    'mu_fixed': MU_FIXED, 'L_var': L_VAR,
    'mu_eff': MU_EFF, 'sigma_eff': SIGMA_EFF,
    'n_trials': N_TRIALS, 'n_trials_var': N_TRIALS_VAR, 'n_trials_eff': N_TRIALS_EFF,
    'seed_main': SEED_MAIN, 'seed_var': SEED_VAR, 'seed_eff': SEED_EFF,
}
# ===================================================

print(f"\n{'='*60}")
print(f"实验9.2-1: 重参数化技巧的数值验证")
print(f"{'='*60}")
print("\n[核心思想]")
print("  9.2节: 重参数化将随机性外移到与参数无关的噪声变量ε")
print("  附录9A: REINFORCE方差在σ→0时发散，重参数化方差趋于0")


# ============================================================
# 步骤1：梯度可传播性验证 —— 重参数化使梯度穿过随机节点
# 对应9.2节核心思想：将随机性外移到与参数无关的噪声变量ε
# ============================================================
print(f"\n{'='*60}")
print("步骤1: 梯度可传播性验证")
print(f"{'='*60}")

# --- 重参数化采样：梯度可传播 ---
mu = torch.tensor(2.0, requires_grad=True)
logvar = torch.tensor(0.0, requires_grad=True)  # σ=1

std = torch.exp(0.5 * logvar)
eps = torch.randn(1)
z = mu + std * eps  # z = μ + σε，梯度可通过μ和σ传播
loss = z ** 2
loss.backward()

print(f"重参数化: μ={mu.item():.1f}, σ={std.item():.1f}")
print(f"  ∂loss/∂μ = {mu.grad.item():.4f}  (这是单个样本的含噪值；其期望解析解 E[∂loss/∂μ] = 2μ = {2*2.0:.1f})")
print(f"  ∂loss/∂logvar = {logvar.grad.item():.4f}  (单样本含噪值；期望解析解 E[∂loss/∂logvar] = σ² = {std.item()**2:.1f})")
print("  → 梯度成功穿过随机采样节点！（无偏性的定量验证见步骤2的5000次重复）")

# --- 直接采样（非重参数化）：使用 .sample() ---
from torch.distributions import Normal
mu2 = torch.tensor(2.0, requires_grad=True)
dist = Normal(mu2, 1.0)

# .sample() 不支持梯度传播
z_no_grad = dist.sample()
print(f"\n直接采样 (.sample()): z = {z_no_grad.item():.4f}")
print(f"  requires_grad={z_no_grad.requires_grad}, grad_fn={z_no_grad.grad_fn}")
print("  → .sample() 产生的样本无法反传梯度（requires_grad=False, grad_fn=None）")

# .rsample() 使用重参数化技巧
z_with_grad = dist.rsample()
loss2 = z_with_grad ** 2
loss2.backward()
print(f"重参数化采样 (.rsample()): z = {z_with_grad.item():.4f}")
print(f"  ∂loss/∂μ = {mu2.grad.item():.4f}  → 梯度可传播！")

print("\n→ 核心区别：重参数化将随机性从 z=Sample(μ,σ) 改写为 z=μ+σε，")
print("  使梯度可通过确定性路径（μ,σ）反传。")


# ============================================================
# 步骤2：REINFORCE 与重参数化梯度估计器实现
# 对应9.2节和附录9A：两种梯度估计方法
# ============================================================
print(f"\n{'='*60}")
print("步骤2: REINFORCE 与重参数化梯度估计器对比")
print(f"{'='*60}")

def reinforce_grad_mu(mu_val, sigma_val, n_samples=100):
    """
    REINFORCE梯度估计（得分函数估计器）:
    ∇_μ E_{q(z|μ,σ)}[f(z)] ≈ (1/L) Σ f(z_l) · ∇_μ log q(z_l|μ,σ)
    
    对于高斯: ∇_μ log q(z|μ,σ) = (z-μ)/σ²
    测试函数: f(z) = z²
    真实梯度: ∇_μ E[z²] = 2μ
    """
    z = np.random.normal(mu_val, sigma_val, n_samples)
    f_z = z ** 2
    score_mu = (z - mu_val) / sigma_val ** 2
    return np.mean(f_z * score_mu)


def reparam_grad_mu(mu_val, sigma_val, n_samples=100):
    """
    重参数化梯度估计:
    ∇_μ E[f(z)] = ∇_μ E_ε[f(μ+σε)] ≈ (1/L) Σ ∇_μ f(μ+σε_l)
    
    对于 f(z)=z²: ∇_μ f(μ+σε) = 2(μ+σε) = 2z
    """
    eps = np.random.normal(0, 1, n_samples)
    z = mu_val + sigma_val * eps
    return np.mean(2 * z)  # ∂f/∂μ = 2z (chain rule: ∂z/∂μ = 1)


# 缓存中必须包含的数据数组
_CACHE_DATA_KEYS = [
    'reinforce_estimates', 'reparam_estimates',
    'reinforce_small', 'reparam_small',
    'sigma_list', 'reinforce_vars', 'reparam_vars',
    'L_list', 'reinforce_mse', 'reparam_mse',
]


def _cache_is_valid(cache):
    """校验缓存版本、全部实验参数与数据键；任一不匹配即作废重算，杜绝参数-结果张冠李戴"""
    if 'cache_version' not in cache.files:
        return False
    if int(np.asarray(cache['cache_version']).item()) != CACHE_VERSION:
        return False
    for k, v in EXPERIMENT_PARAMS.items():
        if k not in cache.files:
            return False
        if not np.isclose(float(np.asarray(cache[k]).item()), float(v), rtol=1e-9, atol=0.0):
            return False
    return all(k in cache.files for k in _CACHE_DATA_KEYS)


# 检查是否有缓存的数值结果
cached_results = None
if os.path.exists(RESULTS_CACHE_PATH):
    print(f"\n检测到已保存的数值结果: {RESULTS_CACHE_PATH}")
    _loaded = np.load(RESULTS_CACHE_PATH, allow_pickle=True)
    if _cache_is_valid(_loaded):
        print(f"✓ 缓存版本(v{CACHE_VERSION})与全部参数一致, 直接加载, 跳过数值计算")
        cached_results = _loaded
        reinforce_estimates = cached_results['reinforce_estimates']
        reparam_estimates = cached_results['reparam_estimates']
        reinforce_small = cached_results['reinforce_small']
        reparam_small = cached_results['reparam_small']
        sigma_list = cached_results['sigma_list']
        reinforce_vars = cached_results['reinforce_vars']
        reparam_vars = cached_results['reparam_vars']
        L_list = cached_results['L_list']
        reinforce_mse = cached_results['reinforce_mse']
        reparam_mse = cached_results['reparam_mse']
    else:
        print("  缓存版本或实验参数已变更/数据不完整, 作废并重新计算")

if cached_results is None:
    # 验证无偏性
    np.random.seed(SEED_MAIN)
    reinforce_estimates = np.array([reinforce_grad_mu(MU_TEST, SIGMA_TEST, L_SAMPLES) for _ in range(N_TRIALS)])
    reparam_estimates = np.array([reparam_grad_mu(MU_TEST, SIGMA_TEST, L_SAMPLES) for _ in range(N_TRIALS)])

    # σ=0.1 时的梯度估计
    reinforce_small = np.array([reinforce_grad_mu(MU_TEST, SIGMA_SMALL, L_SAMPLES) for _ in range(N_TRIALS)])
    reparam_small = np.array([reparam_grad_mu(MU_TEST, SIGMA_SMALL, L_SAMPLES) for _ in range(N_TRIALS)])


print(f"\n测试参数: μ={MU_TEST}, σ={SIGMA_TEST}, L={L_SAMPLES}")
print(f"真实梯度: ∇_μ E[z²] = 2μ = {TRUE_GRAD:.4f}")
print(f"\nREINFORCE:  均值={np.mean(reinforce_estimates):.4f}, "
      f"标准差={np.std(reinforce_estimates):.4f}, "
      f"方差={np.var(reinforce_estimates):.4f}")
print(f"重参数化:   均值={np.mean(reparam_estimates):.4f}, "
      f"标准差={np.std(reparam_estimates):.4f}, "
      f"方差={np.var(reparam_estimates):.4f}")
print(f"\n→ 两种估计器均无偏（均值≈{TRUE_GRAD}），但重参数化方差远低于REINFORCE")
print(f"  方差比: REINFORCE/重参数化 = {np.var(reinforce_estimates)/np.var(reparam_estimates):.1f}x")


# ============================================================
# 步骤3：方差随σ变化的标度行为 —— 附录9A核心结论
# ============================================================
print(f"\n{'='*60}")
print("步骤3: 方差随σ变化的标度行为（附录9A核心结论）")
print(f"{'='*60}")

if cached_results is None:
    sigma_list = np.logspace(-1.5, 0.7, 25)  # σ from ~0.03 to ~5

    np.random.seed(SEED_VAR)
    reinforce_vars = []
    reparam_vars = []

    for sigma in sigma_list:
        rg = [reinforce_grad_mu(MU_FIXED, sigma, L_VAR) for _ in range(N_TRIALS_VAR)]
        rpg = [reparam_grad_mu(MU_FIXED, sigma, L_VAR) for _ in range(N_TRIALS_VAR)]
        reinforce_vars.append(np.var(rg))
        reparam_vars.append(np.var(rpg))

    reinforce_vars = np.array(reinforce_vars)
    reparam_vars = np.array(reparam_vars)

# 理论预测 (附录9A)
# REINFORCE单样本梯度 x=z^2*(z-μ)/σ^2 的精确方差 (对高斯解析积分):
#   Var[x] = 15σ^4 + 14μ^2σ^2 + μ^4  (除以σ^2后), 即 Var[(1/L)Σx] = (15σ^2 + 14μ^2 + μ^4/σ^2)/L
#   σ→0 时 μ^4/σ^2 项主导 → 发散; σ→∞ 时 15σ^2 主导。附录9A的 O(σ^2+μ^2/σ^2) 是其标度近似。
# 重参数化单样本梯度 2z 的方差 = 4σ^2, 即 Var = 4σ^2/L (精确)
theory_reinforce = (15 * sigma_list**2 + 14 * MU_FIXED**2 + MU_FIXED**4 / sigma_list**2) / L_VAR
theory_reparam = 4 * sigma_list**2 / L_VAR

print("σ\t\tREINFORCE方差\t重参数化方差\t方差比")
print("-" * 65)
for i in [0, len(sigma_list)//4, len(sigma_list)//2, 3*len(sigma_list)//4, -1]:
    idx = i if i >= 0 else len(sigma_list) + i
    ratio = reinforce_vars[idx] / max(reparam_vars[idx], 1e-10)
    print(f"{sigma_list[idx]:.4f}\t\t{reinforce_vars[idx]:.4f}\t\t{reparam_vars[idx]:.4f}\t\t{ratio:.1f}x")

print(f"\n关键发现:")
print(f"  σ={sigma_list[0]:.3f} (小): REINFORCE方差比重参数化高 {reinforce_vars[0]/max(reparam_vars[0],1e-10):.0f}倍 → REINFORCE在σ小时几乎不可用")
print(f"  σ={sigma_list[-1]:.3f} (大): REINFORCE方差比重参数化高 {reinforce_vars[-1]/max(reparam_vars[-1],1e-10):.1f}倍 → 差距缩小但仍然显著")


# ============================================================
# 步骤4：★ 原创设计 - 样本效率对比
# 达到相同精度所需的样本数
# ============================================================
print(f"\n{'='*60}")
print("步骤4: 样本效率对比（★ 原创设计）")
print(f"{'='*60}")

if cached_results is None:
    L_list = [10, 50, 100, 500, 1000, 5000]

    np.random.seed(SEED_EFF)
    reinforce_mse = []
    reparam_mse = []

    for L_eff in L_list:
        rg = np.array([reinforce_grad_mu(MU_EFF, SIGMA_EFF, L_eff) for _ in range(N_TRIALS_EFF)])
        rpg = np.array([reparam_grad_mu(MU_EFF, SIGMA_EFF, L_eff) for _ in range(N_TRIALS_EFF)])
        reinforce_mse.append(np.mean((rg - TRUE_GRAD_EFF) ** 2))
        reparam_mse.append(np.mean((rpg - TRUE_GRAD_EFF) ** 2))

    reinforce_mse = np.array(reinforce_mse)
    reparam_mse = np.array(reparam_mse)

    # 保存全部数值结果、缓存版本号与实验参数（参数口径随结果一起持久化，可追溯）
    np.savez(RESULTS_CACHE_PATH,
             cache_version=CACHE_VERSION,
             **EXPERIMENT_PARAMS,
             reinforce_estimates=reinforce_estimates,
             reparam_estimates=reparam_estimates,
             reinforce_small=reinforce_small,
             reparam_small=reparam_small,
             sigma_list=sigma_list,
             reinforce_vars=reinforce_vars,
             reparam_vars=reparam_vars,
             L_list=L_list,
             reinforce_mse=reinforce_mse,
             reparam_mse=reparam_mse)
    print(f"\n✓ 数值计算完成, 结果已保存: {RESULTS_CACHE_PATH}")

print(f"参数: μ={MU_EFF}, σ={SIGMA_EFF}, 真实梯度={TRUE_GRAD_EFF}")
print(f"{'L':>6s}  {'REINFORCE MSE':>15s}  {'重参数化 MSE':>15s}  {'MSE比':>10s}")
print("-" * 55)
for i, L_eff in enumerate(L_list):
    ratio = reinforce_mse[i] / max(reparam_mse[i], 1e-10)
    print(f"{L_eff:6d}  {reinforce_mse[i]:15.4f}  {reparam_mse[i]:15.4f}  {ratio:10.1f}x")


# ============================================================
# 可视化
# ============================================================
print(f"\n{'='*60}")
print("生成可视化图表...")
print(f"{'='*60}")

# --- 图1: 梯度估计分布对比 + 方差随σ变化 ---
# 布局：左列(a)/(c)；右上(b)拆成上下两个共享【全范围】横轴的子图；右下(d)
# σ=0.1时两者标准差相差约200倍，若共用一个y轴或裁窄x轴都会误导：
#   - 双y轴+裁x轴会把REINFORCE约86%的样本裁到窗外，画成一块"平板"
#   - 上下分栏共享全范围横轴，各自独立纵轴，宽尾与尖峰都能完整呈现
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2)
ax_a = fig.add_subplot(gs[0, 0])
gs_b = gs[0, 1].subgridspec(2, 1, hspace=0.08)
ax_b1 = fig.add_subplot(gs_b[0])
ax_b2 = fig.add_subplot(gs_b[1], sharex=ax_b1)
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# (a) σ=1.0 时的梯度估计直方图
ax_a.hist(reinforce_estimates, bins=60, alpha=0.6, label='REINFORCE', density=True, color='#e74c3c')
ax_a.hist(reparam_estimates, bins=60, alpha=0.6, label='重参数化', density=True, color='#3498db')
ax_a.axvline(TRUE_GRAD, color='black', linestyle='--', linewidth=2, label=f'真实梯度 = {TRUE_GRAD:.1f}')
ax_a.set_xlabel('梯度估计值', fontsize=12)
ax_a.set_ylabel('概率密度', fontsize=12)
ax_a.set_title(f'(a) 梯度分布 ($\\mu={MU_TEST}$, $\\sigma={SIGMA_TEST}$, $L={L_SAMPLES}$)', fontsize=13)
ax_a.legend(fontsize=10)

# (b) σ=0.1：上下两子图共享全范围横轴（覆盖REINFORCE约99.9%的样本），纵轴各自独立
std_small_r = np.std(reinforce_small)
std_small_p = np.std(reparam_small)
x_lo, x_hi = np.quantile(reinforce_small, [0.0005, 0.9995])
x_lo = min(x_lo, TRUE_GRAD - 5 * std_small_p)
x_hi = max(x_hi, TRUE_GRAD + 5 * std_small_p)
bins_small = np.linspace(x_lo, x_hi, 100)

ax_b1.hist(reparam_small, bins=bins_small, alpha=0.6, density=True, color='#3498db')
ax_b1.axvline(TRUE_GRAD, color='black', linestyle='--', linewidth=2)
ax_b1.set_ylabel('概率密度\n(重参数化)', fontsize=11, color='#21618c')
ax_b1.tick_params(axis='y', labelcolor='#21618c')
ax_b1.text(0.97, 0.90, f'标准差 ≈ {std_small_p:.3f}\n分布窄尖峰',
           transform=ax_b1.transAxes, ha='right', va='top', fontsize=10,
           bbox=dict(boxstyle='round', fc='white', alpha=0.75, ec='#3498db'))
ax_b1.set_title(
    f'(b) $\\sigma={SIGMA_SMALL}$ 梯度分布：上下两图共享同一全范围横轴，纵轴各自独立；'
    f'两者标准差相差约 {std_small_r/std_small_p:.0f} 倍',
    fontsize=12)
plt.setp(ax_b1.get_xticklabels(), visible=False)

ax_b2.hist(reinforce_small, bins=bins_small, alpha=0.6, density=True, color='#e74c3c')
ax_b2.axvline(TRUE_GRAD, color='black', linestyle='--', linewidth=2, label=f'真实梯度 = {TRUE_GRAD:.1f}')
ax_b2.set_ylabel('概率密度\n(REINFORCE)', fontsize=11, color='#922b21')
ax_b2.tick_params(axis='y', labelcolor='#922b21')
ax_b2.set_xlabel('梯度估计值（全范围，未裁剪）', fontsize=12)
ax_b2.text(0.97, 0.90, f'标准差 ≈ {std_small_r:.2f}\n分布宽、重尾明显',
           transform=ax_b2.transAxes, ha='right', va='top', fontsize=10,
           bbox=dict(boxstyle='round', fc='white', alpha=0.75, ec='#e74c3c'))
ax_b2.legend(loc='upper left', fontsize=9)

# (c) 方差随σ变化 (log-log)
ax_c.loglog(sigma_list, reinforce_vars, 'o-', label='REINFORCE (实验)', color='#e74c3c', markersize=5)
ax_c.loglog(sigma_list, reparam_vars, 's-', label='重参数化 (实验)', color='#3498db', markersize=5)
ax_c.loglog(sigma_list, theory_reinforce, '--', label='精确理论: $(15\\sigma^2+14\\mu^2+\\mu^4/\\sigma^2)/L$', color='#e74c3c', alpha=0.6)
ax_c.loglog(sigma_list, theory_reparam, '--', label='精确理论: $4\\sigma^2/L$', color='#3498db', alpha=0.6)
ax_c.set_xlabel('$\\sigma$ (对数刻度)', fontsize=12)
ax_c.set_ylabel('梯度方差 (对数刻度)', fontsize=12)
ax_c.set_title(f'(c) 方差 vs $\\sigma$ ($\\mu={MU_FIXED}$, $L={L_VAR}$)', fontsize=13)
ax_c.legend(fontsize=9, loc='upper left')
ax_c.grid(True, alpha=0.3)

# (d) 方差比
ax_d.semilogy(sigma_list, reinforce_vars / np.maximum(reparam_vars, 1e-10), 'D-', color='#8e44ad', markersize=5)
ax_d.axhline(1, color='gray', linestyle='--', alpha=0.5, label='方差比 = 1')
ax_d.set_xlabel('$\\sigma$', fontsize=12)
ax_d.set_ylabel('方差比 (REINFORCE / 重参数化)', fontsize=12)
ax_d.set_title(f'(d) 方差比 ($\\mu={MU_FIXED}$)', fontsize=13)
ax_d.grid(True, alpha=0.3)
ax_d.legend(fontsize=10)

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤1_REINFORCE与重参数化方差.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图1已保存: {fig_path}")


# --- 图2: 样本效率对比 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) MSE vs L
ax = axes[0]
ax.loglog(L_list, reinforce_mse, 'o-', label='REINFORCE', color='#e74c3c', markersize=8)
ax.loglog(L_list, reparam_mse, 's-', label='重参数化', color='#3498db', markersize=8)
ax.set_xlabel('样本数 $L$ (对数刻度)', fontsize=12)
ax.set_ylabel('MSE (对数刻度)', fontsize=12)
ax.set_title(f'(a) 梯度MSE vs 样本数 ($\\mu={MU_EFF}$, $\\sigma={SIGMA_EFF}$)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# (b) 达到目标精度的样本数对比
target_mse = 0.1
from scipy.interpolate import interp1d
# MSE关于L严格单调递减，log-log下线性插值稳定；允许少量外推（重参数化在L=10处MSE已≈0.1）
f_reinforce = interp1d(np.log10(reinforce_mse), np.log10(L_list), fill_value='extrapolate')
f_reparam = interp1d(np.log10(reparam_mse), np.log10(L_list), fill_value='extrapolate')
L_reinforce_needed = 10 ** f_reinforce(np.log10(target_mse))
L_reparam_needed = 10 ** f_reparam(np.log10(target_mse))

print(f"\n达到MSE={target_mse}所需样本数:")
print(f"  REINFORCE:  L ≈ {L_reinforce_needed:.0f}")
print(f"  重参数化:   L ≈ {L_reparam_needed:.0f}")
print(f"  比值: {L_reinforce_needed/L_reparam_needed:.1f}x")

ax = axes[1]
methods = ['REINFORCE', '重参数化']
L_needed = [L_reinforce_needed, L_reparam_needed]
bars = ax.bar(methods, L_needed, color=['#e74c3c', '#3498db'], alpha=0.8, edgecolor='black')
ax.set_ylabel('所需样本数 $L$', fontsize=12)
ax.set_title(f'(b) 达到MSE={target_mse}所需样本数', fontsize=13)
for bar, val in zip(bars, L_needed):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
            f'{val:.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤2_样本效率对比.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig_path}")


# ============================================================
# 实验总结
# ============================================================
print(f"\n{'='*60}")
print("实验9.2-1 总结")
print(f"{'='*60}")
print("\n核心验证点（对应9.2节和附录9A）：")
print("\n1. 重参数化使梯度可穿过随机采样节点：")
print("   - z = μ + σε 将随机性外移到与参数无关的ε")
print("   - 梯度可通过μ和σ的确定性路径反传")
print("   ✓ 实验验证: 步骤1展示梯度成功穿过随机节点")
print("\n2. REINFORCE与重参数化均给出无偏梯度估计：")
print("   - 两者在大量重复实验下均值均等于真实梯度")
print("   - 但REINFORCE方差远高于重参数化")
print("   ✓ 实验验证: 步骤2验证无偏性和方差差异")
print("\n3. 方差的σ标度行为是关键差异（附录9A核心结论）：")
print("   - REINFORCE精确方差 = (μ⁴/σ² + 14μ² + 15σ²)/L → σ→0时按μ⁴/σ²发散")
print("   - 重参数化精确方差 = 4σ²/L → σ→0时趋于0")
print("   - 附录9A的 O(σ²+μ²/σ²) 是μ固定时的定性标度，省略了常数与14μ²平台项")
print("   - 这意味着在编码器输出小方差时，REINFORCE几乎不可用")
print("   ✓ 实验验证: 步骤3的实验点与精确理论曲线定量重合")
print("\n4. 重参数化所需样本数远少于REINFORCE：")
print("   - 达到相同MSE，REINFORCE需要更多样本")
print("   - 这是VAE选择重参数化而非REINFORCE的直接原因")
print("   ✓ 实验验证: 步骤4量化样本效率差异")

print(f"\n{'='*60}")
print("第九章配套实验 9.2-1 完成!")

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
    'cache_version': CACHE_VERSION,
    'parameters': dict(EXPERIMENT_PARAMS),
    'theory': {
        # f(z)=z² 时梯度估计量方差的精确闭式解（L个样本取均值）
        'reinforce_variance_formula': '(mu^4/sigma^2 + 14*mu^2 + 15*sigma^2) / L',
        'reparam_variance_formula': '4*sigma^2 / L',
        'note': '附录9A的 O(sigma^2+mu^2/sigma^2) 仅为mu固定、sigma->0的定性渐近标度',
    },
    'REINFORCE_mean': np.mean(reinforce_estimates),
    'REINFORCE_var': np.var(reinforce_estimates),
    'REPARAM_mean': np.mean(reparam_estimates),
    'REPARAM_var': np.var(reparam_estimates),
    'variance_ratio': np.var(reinforce_estimates) / np.var(reparam_estimates),
    'true_gradient': TRUE_GRAD,
    'REINFORCE_small_sigma_std': float(np.std(reinforce_small)),
    'REPARAM_small_sigma_std': float(np.std(reparam_small)),
    'REINFORCE_small_sigma_var': np.var(reinforce_small),
    'REPARAM_small_sigma_var': np.var(reparam_small),
    'small_sigma_variance_ratio': np.var(reinforce_small) / max(np.var(reparam_small), 1e-10),
    'sigma_list': sigma_list,
    'reinforce_vars': reinforce_vars,
    'reparam_vars': reparam_vars,
    'L_list': L_list,
    'reinforce_mse': reinforce_mse,
    'reparam_mse': reparam_mse,
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")