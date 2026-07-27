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
        # 对数域比较：log_alpha = min(0, log_p - log_x)；接受 iff log(u) < log_alpha
        # 等价但避免 np.exp 在 |log_p-log_x| 大时溢出/下溢的 RuntimeWarning
        log_alpha = min(0.0, log_p - log_x)
        accepted = np.log(rng.random()) < log_alpha
        if accepted:
            x, log_x = proposal, log_p
        if i >= burn_in:
            samples[i - burn_in] = x
            if accepted:
                n_accept += 1
    accept_rate = n_accept / n_samples if n_samples > 0 else 0.0
    return samples, accept_rate


# ── 有效样本数 (Geyer 1992 initial positive sequence estimator, IPSE) ─
def autocov_batch(x):
    """FFT 版偏差自协方差估计（O(n log n)，比逐 lag 循环快 1-2 个数量级）。
    返回 lag=0..max_lag 的 acov 数组。"""
    x = x - x.mean()
    n = len(x)
    # 零填充到 2n 以避免循环卷积的"绕回"伪影
    f = np.fft.fft(x, n=2 * n)
    acov = np.fft.ifft(np.abs(f) ** 2).real[:n] / n
    max_lag = min(n - 1, 10000)
    return acov[:max_lag + 1]


def ess(x):
    """Geyer (1992) initial positive sequence estimator (IPSE)。

    把自协方差**归一化**为自相关系数 ρ(k)=acov(k)/acov(0) 后两两配对为
        Gamma_m = rho[2m] + rho[2m+1]   (m=0,1,2,...)
    当 m 由小到大时，Gamma_m 起初为正、最终应趋于 0；找到首次
    Gamma_m <= 0 的位置 m* 截断，剔除尾部噪声。tau = -1 + 2*sum Gamma_m，
    ESS = n / tau。

    注意：归一化非常关键——若直接对未归一化的 acov 配对，tau 的尺度会被
    acov(0)（即样本方差）污染，导致对非单位方差的目标分布系统性偏差。
    """
    acov = autocov_batch(x)
    if acov[0] <= 0:
        return 0.0
    rho = acov / acov[0]                       # 归一化自相关系数
    m_max = (len(acov) - 1) // 2
    gamma = rho[0:2 * m_max:2] + rho[1:2 * m_max:2]   # Gamma_m = rho[2m]+rho[2m+1]
    sum_pos = 0.0
    for m in range(m_max):
        if gamma[m] <= 0.0:
            break
        sum_pos += gamma[m]
    else:
        # 循环正常结束：未在 max_lag 内找到非正 Gamma，链混合很慢，
        # 后续 Gamma 仍可能为正但被截断，ESS 估计会偏高
        print(f"  [警告] ess()：未在 max_lag={len(acov)-1} 内找到 Gamma<=0，"
              f"ESS 可能被高估（建议增大 max_lag 或检查链混合）")
    tau = max(-1.0 + 2.0 * sum_pos, 1.0)
    return len(x) / tau


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
# semilogy 遇到 0 会直接丢点（log(0) 未定义），造成"断点"假象；
# 这里把 0 值替换为 0.5/n_samples 作为下界并打特殊标记，让读者看清
floor = 0.5 / 20000
acc_s1_log = [a if a > 0 else floor for a in acc_s1]
zero_dims = [d for d, a in zip(dims_s1, acc_s1) if a == 0]
ax[0].semilogy(dims_s1, acc_s1_log, 'o-', color='C0',
               label='接受率（=0 的点用 0.5/N 作下界）')
if zero_dims:
    ax[0].plot(zero_dims, [floor] * len(zero_dims), 'rx', ms=10, mew=2,
               label=f'实际为 0 的维度: {zero_dims}')
ax[0].set_xlabel(r'维数 $d$', fontsize=12)
ax[0].set_ylabel('接受率（对数轴）', fontsize=12)
ax[0].set_title(r'步骤1: 固定步长 $\sigma=1$ 下接受率随维数衰减', fontsize=12)
ax[0].legend(fontsize=9)
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
# 注：σ_opt = 2.38/√d 是 Roberts, Gelman & Gilks (1997) 渐近 (d→∞) 最优
# 公式，对中等 d（如 d=1 时真实最优接受率约 0.44）并非严格最优；
# 这里仅作为"按理论最优缩放"的代表演示，验证"高维 RWM 仍崩塌"的定性结论。
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
    "ESS算法": "Geyer (1992) IPSE：先归一化 rho=acov/acov[0]，再配对 Gamma_m=rho[2m]+rho[2m+1]，首次非正处截断",
    "步骤1_固定步长_接受率": {str(d): round(a, 4) for d, a in zip(dims_s1, acc_s1)},
    "步骤2_最优步长_接受率": {str(d): round(a, 4) for d, a in zip(dims_s2, acc_s2)},
    "步骤3_有效样本率ESS/N": {str(d): round(r, 4) for d, r in zip(dims_s3, ess_ratio)},
    "结论": ("固定步长下接受率随维数指数衰减；按渐近最优步长 σ=2.38/√d 缩放后接受率"
             "收敛到 ≈0.234（注：2.38/√d 是 d→∞ 渐近最优，小 d 时并非严格最优，如 d=1"
             "对应最优接受率约 0.44，这里仅作理论代表演示），但步长必须随维数缩小，"
             "且有效样本率 ESS/N 仍随维数下降（Geyer IPSE 截断估计），说明 RWM 在高维下"
             "效率崩塌，需要维度无关的提议（见实验4.2-4 的 pCN）。")
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
