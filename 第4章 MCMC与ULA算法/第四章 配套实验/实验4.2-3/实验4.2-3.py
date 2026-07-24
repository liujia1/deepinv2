# ============================================================
# 实验4.2-3 MH接受率随维数衰减（随机游走 Metropolis）
# ============================================================
# 对应章节：4.2 Metropolis-Hastings算法
# 知识点：
#   - 随机游走 Metropolis (RWM) 在高维下接受率随维数衰减（维数诅咒的采样版）
#   - Roberts, Gelman & Gilks (1997)：RWM 最优接受率 ≈ 0.234，
#     对应最优提议步长 σ_opt = 2.38 / √d （需随维数缩小）
#   - 即便按最优步长缩放，RWM 的有效样本率 ESS/N 仍随维数下降 → 效率崩塌
#
# 实验步骤：
#   步骤1：固定步长 σ=1 下，RWM 接受率随维数 d 指数式衰减（直观演示维数诅咒）
#   步骤2：最优步长缩放 σ = 2.38/√d，接受率收敛到渐近值 ≈ 0.234，但 σ 必须随 d 缩小
#   步骤3：最优步长下有效样本率 ESS/N 仍随 d 衰减 → 高维采样效率崩塌
# ============================================================

import os
import sys
import io
import json
import logging
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置控制台输出为 UTF-8（Windows 下避免中文乱码）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace', line_buffering=True)

# 静默 matplotlib 相关警告（matlab 静默模式）
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.rcParams['axes.unicode_minus'] = False

# ── 中文显示（复制到其它机器/环境时若缺字体，自动降级）──────────────
# ====== 中文字体配置（兼容本地和 Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules
if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验4.2-3')
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

OUT_DIR = os.path.join(SAVE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)
RNG = np.random.default_rng(20240723)


# ── 目标分布：d 维标准高斯 N(0, I)（最干净的“后验”演示对象）─────────
def log_target_gaussian(x):
    """标准高斯 N(0, I) 的对数密度（省略与 x 无关的常数项）。"""
    return -0.5 * np.dot(x, x)


def rwm_sampler(log_target, d, n_samples, sigma=1.0, burn_in=0, rng=None):
    """随机游走 Metropolis (RWM) 采样器。"""
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.zeros(d)
    log_x = log_target(x)
    samples = np.empty((n_samples, d))
    n_accept = 0
    for i in tqdm(range(n_samples + burn_in), desc="RWM采样", leave=False):
        proposal = x + sigma * rng.standard_normal(d)
        log_p = log_target(proposal)
        alpha = min(1.0, np.exp(log_p - log_x))
        accepted = False
        if rng.random() < alpha:
            x, log_x = proposal, log_p
            accepted = True
        if i >= burn_in:
            samples[i - burn_in] = x
            if accepted:
                n_accept += 1
    accept_rate = n_accept / n_samples if n_samples > 0 else 0.0
    return samples, accept_rate


# ── 有效样本数 (Geyer 交织法) ─────────────────────────────────────
def autocov_batch(x):
    x = x - x.mean()
    n = len(x)
    max_lag = min(n - 1, 10000)
    acov = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        acov[lag] = np.mean(x[:n - lag] * x[lag:])
    return acov


def ess(x):
    acov = autocov_batch(x)
    if acov[0] <= 0:
        return 0.0
    s = acov[0] + 2.0 * np.sum(acov[1:])
    s = max(s, acov[0])
    return len(x) * acov[0] / s


# ════════════════════════════════════════════════════════════════
print("实验4.2-3 MH接受率随维数衰减（随机游走 Metropolis）")
print("=" * 60)

# ── 步骤1：固定步长 σ=1，接受率随维数衰减 ─────────────────────────
print("\n[步骤1] 固定步长 σ=1，RWM 接受率随维数 d 衰减")
dims_s1 = [1, 2, 5, 10, 20, 50, 100, 200]
acc_s1 = []
for d in dims_s1:
    _, ar = rwm_sampler(log_target_gaussian, d, n_samples=20000, sigma=1.0,
                        burn_in=1000, rng=RNG)
    acc_s1.append(ar)
    print(f"  d={d:4d}  接受率={ar:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].semilogy(dims_s1, acc_s1, 'o-', color='C0')
ax[0].set_xlabel(r'维数 $d$', fontsize=12)
ax[0].set_ylabel('接受率（对数轴）', fontsize=12)
ax[0].set_title(r'步骤1: 固定步长 $\sigma=1$ 下接受率随维数衰减', fontsize=12)
ax[0].grid(True, which='both', alpha=0.3)

ax[1].plot(dims_s1, acc_s1, 'o-', color='C0')
ax[1].set_xscale('log')
ax[1].axhline(0.234, ls='--', color='C3', label=r'渐近最优接受率 $\approx 0.234$')
ax[1].set_xlabel(r'维数 $d$（对数轴）', fontsize=12)
ax[1].set_ylabel('接受率', fontsize=12)
ax[1].set_title(r'步骤1: 接受率随维数（线性坐标）', fontsize=12)
ax[1].legend(fontsize=10)
ax[1].grid(True, which='both', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'exp4_2-3_step1_acceptance_decay.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  图已保存: {os.path.join(OUT_DIR, 'exp4_2-3_step1_acceptance_decay.png')}")

# ── 步骤2：最优步长缩放 σ = 2.38/√d，接受率收敛到 ≈0.234 ──────────
print("\n[步骤2] 最优步长缩放 σ = 2.38/√d，接受率收敛到渐近值 ≈ 0.234")
dims_s2 = [1, 2, 5, 10, 20, 50, 100, 200, 500]
acc_s2, sigma_s2 = [], []
for d in dims_s2:
    sigma = 2.38 / np.sqrt(d)           # RWM 高维最优步长（Roberts et al. 1997）
    sigma_s2.append(sigma)
    _, ar = rwm_sampler(log_target_gaussian, d, n_samples=30000, sigma=sigma,
                        burn_in=2000, rng=RNG)
    acc_s2.append(ar)
    print(f"  d={d:4d}  σ={sigma:.4f}  接受率={ar:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(dims_s2, acc_s2, 's-', color='C1')
ax[0].axhline(0.234, ls='--', color='C3', label=r'渐近最优接受率 $\approx 0.234$')
ax[0].set_xscale('log')
ax[0].set_xlabel(r'维数 $d$（对数轴）', fontsize=12)
ax[0].set_ylabel('接受率', fontsize=12)
ax[0].set_title(r'步骤2: 最优步长下接受率 $\to$ 0.234', fontsize=12)
ax[0].legend(fontsize=10)
ax[0].grid(True, which='both', alpha=0.3)

ax[1].plot(dims_s2, sigma_s2, '^-', color='C2')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_xlabel(r'维数 $d$（对数轴）', fontsize=12)
ax[1].set_ylabel(r'最优步长 $\sigma=2.38/\sqrt{d}$（对数轴）', fontsize=12)
ax[1].set_title(r'步骤2: 步长必须随 $\sqrt{d}$ 缩小', fontsize=12)
ax[1].grid(True, which='both', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'exp4_2-3_step2_optimal_scaling.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  图已保存: {os.path.join(OUT_DIR, 'exp4_2-3_step2_optimal_scaling.png')}")

# ── 步骤3：有效样本率 ESS/N 随维数衰减 ───────────────────────────
print("\n[步骤3] 最优步长下有效样本率 ESS/N 随维数衰减（效率崩塌）")
dims_s3 = [1, 5, 10, 20, 50, 100]
ess_ratio = []
for d in dims_s3:
    sigma = 2.38 / np.sqrt(d)
    samples, _ = rwm_sampler(log_target_gaussian, d, n_samples=60000, sigma=sigma,
                             burn_in=2000, rng=RNG)
    # 取第一个分量估计 ESS（各分量同分布）
    e = ess(samples[:, 0])
    ess_ratio.append(e / samples.shape[0])
    print(f"  d={d:4d}  ESS(分量0)={e:8.1f}  ESS/N={e/samples.shape[0]:.4f}")

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(dims_s3, ess_ratio, 'o-', color='C4')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'维数 $d$（对数轴）', fontsize=12)
ax.set_ylabel(r'有效样本率 $\mathrm{ESS}/N$（对数轴）', fontsize=12)
ax.set_title(r'步骤3: 高维下 RWM 有效样本率衰减', fontsize=12)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'exp4_2-3_step3_ess_decay.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  图已保存: {os.path.join(OUT_DIR, 'exp4_2-3_step3_ess_decay.png')}")

# ── 小结 ──────────────────────────────────────────────────────────
summary = {
    "实验": "实验4.2-3 MH接受率随维数衰减（随机游走 Metropolis）",
    "目标分布": "d 维标准高斯 N(0, I)",
    "步骤1_固定步长_接受率": {str(d): round(a, 4) for d, a in zip(dims_s1, acc_s1)},
    "步骤2_最优步长_接受率": {str(d): round(a, 4) for d, a in zip(dims_s2, acc_s2)},
    "步骤3_有效样本率ESS/N": {str(d): round(r, 4) for d, r in zip(dims_s3, ess_ratio)},
    "结论": ("固定步长下接受率随维数指数衰减；按最优步长 σ=2.38/√d 缩放后接受率"
             "收敛到 ≈0.234，但步长必须随维数缩小，且有效样本率 ESS/N 仍随维数下降，"
             "说明 RWM 在高维下效率崩塌，需要维度无关的提议（见实验4.2-4 的 pCN）。")
}
with open(os.path.join(OUT_DIR, 'exp4_2-3_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("结论:")
print("  1. 固定步长 σ=1：接受率随维数指数衰减（维数诅咒）")
print("  2. 最优步长缩放 σ=2.38/√d：接受率收敛到 ≈0.234，但 σ 必须随 √d 缩小")
print("  3. 即便最优缩放，有效样本率 ESS/N 仍随维数下降 → 高维效率崩塌")
print("  → 这正是需要维度无关提议核（pCN，见实验4.2-4）的原因")
print(f"\n实验完成。结果已保存至: {OUT_DIR}")
