# ============================================================
# 实验4.2-4 pCN-MCMC 实现：维度无关性（dimension-free）的直接验证
# ============================================================
# 对应章节：4.2 Metropolis-Hastings算法（三种经典提议核之 pCN）
# 知识点：
#   - pCN（preconditioned Crank-Nicolson）提议核：
#         X* = sqrt(1-beta^2) * X + beta * xi,    xi ~ N(0, C)，C 为先验协方差
#   - 由于提议在“先验度量”下可逆，先验在对数接受率中约去，
#     接受率只取决于似然比 p(y|X*)/p(y|X)
#   - 因此 pCN 接受率不随离散化维数 d 变化（维度无关），
#     而各向同性随机游走 Metropolis (RWM) 在平滑/各向异性后验上彻底失效
#   - 这是函数空间（无限维）MCMC 在有限维离散化下保持稳定的关键性质
#
# 实验设定：1D 周期信号的贝叶斯反卷积（先验为平滑高斯过程，似然为高斯模糊观测）
#   通过 FFT 高效构造 circulant 先验协方差 C 与模糊算子 B；
#   因所有算子 circulant，后验均值可解析求出（Fourier 对角），用于校验 pCN 正确性
#
# 实验步骤：
#   步骤1：设定高斯线性逆问题（先验 N(0,C)、似然 N(BX, sigma^2 I)）
#   步骤2：RWM（白噪声提议）在该平滑后验上接受率几乎为 0——基线失效
#   步骤3：pCN（先验度量提议）接受率不随维数变化——核心论证（维度无关）
#   步骤3b：RWM 调参扫描 sigma=c/sqrt(d)——即使把接受率调高，每步净移动
#           仍按 1/sqrt(d) 收缩（遍历时间 O(d)），实证"调参无法挽救"
#   步骤4：解析后验均值 vs pCN 经验后验均值（含 Fourier 模态级校验与 IACT）
#
# 口径说明：dimension-free 定理针对同一无限维问题的逐级离散化；本实验各档
#   分辨率独立生成数据实例，跨档差异已包含实例间涨落，是更严格的检验。
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

# ====== 中文字体配置（兼容本地和 Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules
if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验4.2-4')
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

OUT_DIR = SAVE_DIR
os.makedirs(OUT_DIR, exist_ok=True)
RNG = np.random.default_rng(20240724)


# ── 先验协方差（circulant，可用 FFT 采样与计算二次型）──────────────
def prior_eigs(d, tau):
    """周期离散 Laplacian 精度特征值：omega_k = 2 - 2 cos(2*pi*k/d)。
    精度矩阵 L = I + tau^2*(-Delta)，先验协方差 C = L^{-1}；lam 为其特征值。"""
    k = np.arange(d)
    omega2 = 2.0 - 2.0 * np.cos(2.0 * np.pi * k / d)
    return 1.0 + tau ** 2 * omega2          # lambda_k > 0


def sample_prior(lam, rng):
    """从 N(0, C) 采样（C 在 Fourier 域为 diag(1/lambda_k)）。"""
    w = rng.standard_normal(len(lam))
    wf = np.fft.fft(w)
    xf = wf / np.sqrt(lam)
    return np.fft.ifft(xf).real


def log_prior_quad(x, lam):
    """log 先验（省略常数）：-1/2 x^T L x = -(1/2d) sum_k lambda_k |xhat_k|^2。
    这里 d = len(x) 是 numpy FFT 的非归一化约定（sum|x_n|^2 = (1/d) sum|xhat_k|^2），
    因此二次型必须除以 d 才能与时域 x^T L x 一致。"""
    xf = np.fft.fft(x)
    d = len(x)
    return -0.5 / d * np.sum(lam * np.abs(xf) ** 2)


# ── 模糊前向算子 B（circulant，低通）──────────────────────────────
def blur_envelope(d, sigma_blur):
    """B 在 Fourier 域的乘子：高斯低通。
    频率约定：fftfreq(d)*d 是整数波数 k（每周网格数，而非角频率 2*pi*k/d），
    故乘子为 exp(-0.5*(k*sigma_blur)^2)；对应网格单位下高斯核的物理标准差为
    sigma_blur/(2*pi)。本实验只利用其“低通”作用，尺度自洽，命名不影响结论。"""
    freqs = np.fft.fftfreq(d) * d           # 整数波数 k
    return np.exp(-0.5 * (freqs * sigma_blur) ** 2)


def apply_blur(x, blur_g):
    """B x = ifft(blur_g * fft(x)).real。"""
    return np.fft.ifft(blur_g * np.fft.fft(x)).real


def log_likelihood(x, y, blur_g, sigma_noise):
    """高斯似然 log p(y|X) = N(BX, sigma^2 I)（省略常数）。"""
    Bx = apply_blur(x, blur_g)
    return -0.5 / sigma_noise ** 2 * np.sum((y - Bx) ** 2)


def analytic_posterior_mean(y, blur_g, lam, sigma_noise):
    """高斯线性逆问题的解析后验均值（Fourier 对角）：
    posterior mean = (B^T B/sigma^2 + L)^{-1} (B^T/sigma^2) y。"""
    yhat = np.fft.fft(y)
    prec_k = blur_g ** 2 / sigma_noise ** 2 + lam          # 后验精度特征值
    mu_hat = (blur_g / sigma_noise ** 2) * yhat / prec_k
    return np.fft.ifft(mu_hat).real


# ── 两类 Metropolis 采样器 ──────────────────────────────────────
def rwm_sampler(log_target, d, n_samples, sigma, burn_in, rng, x0=None):
    """随机游走 Metropolis：X* = X + sigma * xi, xi~N(0,I)。接受率用完整后验比。
    x0 为初始状态；建议传入先验抽样（平稳起点），避免链在高维全程冻结时
    接受率统计依赖非平稳起点。
    返回 (samples, 接受率, 采样窗内每步净移动 RMS = sqrt(sum||x'-x||^2/N))，
    该量除以先验典型半径 sqrt(tr C) 即单位化的混合速度（步骤3b 使用）。"""
    x = np.zeros(d) if x0 is None else np.asarray(x0, dtype=float).copy()
    log_x = log_target(x)
    samples = np.empty((n_samples, d))
    n_acc = 0
    move_sq = 0.0
    for i in tqdm(range(n_samples + burn_in), desc="RWM采样", leave=False):
        prop = x + sigma * rng.standard_normal(d)
        log_p = log_target(prop)
        # 对数域比较，避免 np.exp 在 |log_p-log_x| 大时溢出/下溢的 RuntimeWarning
        log_alpha = min(0.0, log_p - log_x)
        accepted = np.log(rng.random()) < log_alpha
        if accepted:
            move_sq += float(np.sum((prop - x) ** 2))
            x, log_x = prop, log_p
        if i >= burn_in:
            samples[i - burn_in] = x
            n_acc += int(accepted)
    rms_move = np.sqrt(move_sq / n_samples)
    return samples, n_acc / n_samples, rms_move


def pcn_sampler(log_like, lam, blur_g, sigma_noise, d, n_samples, beta,
                burn_in, rng, y, x0=None):
    """pCN-MCMC：X* = sqrt(1-beta^2) X + beta * xi, xi~N(0,C)。
    接受率只用似然比（先验约去）。x0 为初始状态（默认 0，建议传先验抽样）。
    返回 (samples, 接受率, 采样窗内每步净移动 RMS)。"""
    x = np.zeros(d) if x0 is None else np.asarray(x0, dtype=float).copy()
    ll_x = log_likelihood(x, y, blur_g, sigma_noise)
    samples = np.empty((n_samples, d))
    n_acc = 0
    move_sq = 0.0
    for i in tqdm(range(n_samples + burn_in), desc="pCN采样", leave=False):
        xi = sample_prior(lam, rng)                     # xi ~ N(0, C)
        prop = np.sqrt(1.0 - beta ** 2) * x + beta * xi
        ll_p = log_likelihood(prop, y, blur_g, sigma_noise)
        # 对数域比较，同 rwm_sampler
        log_alpha = min(0.0, ll_p - ll_x)
        accepted = np.log(rng.random()) < log_alpha
        if accepted:
            move_sq += float(np.sum((prop - x) ** 2))
            x, ll_x = prop, ll_p
        if i >= burn_in:
            samples[i - burn_in] = x
            n_acc += int(accepted)
    rms_move = np.sqrt(move_sq / n_samples)
    return samples, n_acc / n_samples, rms_move


# ════════════════════════════════════════════════════════════════
print("实验4.2-4 pCN-MCMC 实现：维度无关性的直接验证")
print("=" * 60)

# ── 步骤1：设定高斯线性逆问题 ────────────────────────────────────
TAU = 2.0          # 先验平滑强度
SIGMA_BLUR = 2.0   # 模糊算子带宽
SIGMA_NOISE = 0.05 # 观测噪声标准差
BETA = 0.15        # pCN 步长
dims = [64, 128, 256, 512, 1024]

print(f"[步骤1] 高斯线性逆问题：先验 N(0,C)（C=(I+tau^2*(-Delta))^-1, tau={TAU}）")
print(f"        似然 p(y|X)=N(BX, sigma^2 I)，模糊带宽 sigma_blur={SIGMA_BLUR}，"
      f"噪声 sigma={SIGMA_NOISE}")
print("        每档分辨率独立生成数据（避免下采样混叠），以保证前后向一致")
print("        口径说明：dimension-free 定理针对同一无限维问题的逐级离散化；")
print("        本实验各档独立生成数据实例，跨档差异已含实例间涨落，是更严格的检验")

# ── 步骤2 & 步骤3：扫描分辨率，对比 RWM 与 pCN 接受率 ─────────────
print("\n[步骤2-3] 扫描分辨率 d，对比 RWM 与 pCN 接受率")
acc_rwm, acc_pcn, mv_ratio_pcn = [], [], []
problems = {}          # 缓存各档问题实例，供步骤3b 在同一数据上做调参扫描
for d in dims:
    blur_g = blur_envelope(d, SIGMA_BLUR)
    lam = prior_eigs(d, TAU)
    xt = sample_prior(lam, RNG)                               # 该分辨率真值
    yt = apply_blur(xt, blur_g) + SIGMA_NOISE * RNG.standard_normal(d)  # 观测
    # 该分辨率下后验的完整对数密度（RWM 用）
    def log_post(xi, _yt=yt, _bg=blur_g, _lam=lam):
        return (log_likelihood(xi, _yt, _bg, SIGMA_NOISE)
                + log_prior_quad(xi, _lam))
    problems[d] = dict(blur_g=blur_g, lam=lam, yt=yt, log_post=log_post,
                       x0=sample_prior(lam, RNG))    # 共用的平稳起点（先验抽样）
    radius = np.sqrt(np.sum(1.0 / lam))             # 先验典型物理半径 sqrt(tr C)
    # RWM：最优缩放步长 sigma=2.38/sqrt(d)（各向同性白噪声提议）
    # 注：2.38/sqrt(d) 是 Roberts, Gelman & Gilks (1997) 在各向同性目标
    # 下的渐近 (d→∞) 最优公式；本实验的后验是平滑先验 + 模糊似然，**显著各向异性**，
    # 严格说并不满足该公式的推导前提。这里**刻意**复用一个不匹配的"教科书
    # 调参公式"作为反例；步骤3b 进一步对 c 做扫描，用实证排除"只是没调好参"
    # 的辩解——RWM 失效是后验几何（无函数空间极限）的固有属性。
    sig = 2.38 / np.sqrt(d)
    _, ar_rwm, _ = rwm_sampler(log_post, d, n_samples=20000, sigma=sig,
                               burn_in=1000, rng=RNG, x0=problems[d]['x0'])
    # pCN：先验度量提议
    _, ar_pcn, mv_pcn = pcn_sampler(log_likelihood, lam, blur_g, SIGMA_NOISE, d,
                                    n_samples=20000, beta=BETA, burn_in=1000,
                                    rng=RNG, y=yt, x0=problems[d]['x0'])
    acc_rwm.append(ar_rwm)
    acc_pcn.append(ar_pcn)
    mv_ratio_pcn.append(float(mv_pcn / radius))
    print(f"  d={d:5d}  RWM(sigma={sig:.4f}) 接受率={ar_rwm:.4f}   "
          f"pCN(beta={BETA}) 接受率={ar_pcn:.4f}   "
          f"pCN 步长/半径={mv_pcn/radius:.4f}")

# ── 步骤3b：RWM 调参扫描——证明"调参无法挽救" ─────────────────────
# 对 sigma = c/sqrt(d) 扫描 c：既看接受率，也看每步净移动相对先验典型半径
# sqrt(tr C) 的比例。若在任何 c 下该比例都随 1/sqrt(d) 收缩，则遍历整个
# 支撑所需步数 ~ O(d)；pCN 的同口径步长/半径则与 d 无关 → RWM 无函数空间
# 极限是固有属性，不是调参问题。
print("\n[步骤3b] RWM 调参扫描 sigma=c/sqrt(d)：接受率与每步净移动/典型半径")
rwm_cs = [0.1, 0.2, 0.33, 0.5, 1.0, 2.38]
sweep = {}        # c -> {'acc': [...按dims...], 'move_ratio': [...]}
for c in rwm_cs:
    accs_c, mvs_c = [], []
    for d in dims:
        p = problems[d]
        radius = np.sqrt(np.sum(1.0 / p['lam']))
        _, ar, mv = rwm_sampler(p['log_post'], d, n_samples=10000,
                                sigma=c / np.sqrt(d), burn_in=500,
                                rng=RNG, x0=p['x0'])
        accs_c.append(float(ar))
        mvs_c.append(float(mv / radius))
    sweep[c] = {'acc': accs_c, 'move_ratio': mvs_c}
    print("  c=%4.2f  接受率  =" % c + " ".join(f"{a:6.3f}" for a in accs_c))
    print("          步长/半径=" + " ".join(f"{m:6.4f}" for m in mvs_c))

# 绘图：接受率 vs 维数
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(dims, acc_rwm, 's-', color='C0', label='RWM (sigma=2.38/sqrt(d))')
ax[0].plot(dims, acc_pcn, 'o-', color='C2', label=f'pCN (beta={BETA})')
ax[0].set_xscale('log')
ax[0].set_xlabel(r'维数（分辨率） $d$', fontsize=12)
ax[0].set_ylabel('接受率', fontsize=12)
ax[0].set_title(r'接受率随维数变化', fontsize=12)
ax[0].legend(fontsize=10)
ax[0].grid(True, which='both', alpha=0.3)
ax[1].semilogy(dims, acc_rwm, 's-', color='C0', label='RWM (sigma=2.38/sqrt(d))')
ax[1].semilogy(dims, acc_pcn, 'o-', color='C2', label=f'pCN (beta={BETA})')
ax[1].set_xlabel(r'维数（分辨率） $d$', fontsize=12)
ax[1].set_ylabel('接受率（对数轴）', fontsize=12)
ax[1].set_title(r'对数轴：pCN 平坦，RWM 失效', fontsize=12)
ax[1].legend(fontsize=10)
ax[1].grid(True, which='both', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'acceptance_vs_dim.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  图已保存: {os.path.join(OUT_DIR, 'acceptance_vs_dim.png')}")

# 绘图：RWM 调参扫描
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for i, c in enumerate(rwm_cs):
    ax[0].plot(dims, sweep[c]['acc'], 's-', color=f'C{i}',
               label=f'RWM c={c:g}')
    ax[1].loglog(dims, sweep[c]['move_ratio'], 's-', color=f'C{i}',
                 label=f'RWM c={c:g}')
ax[0].plot(dims, acc_pcn, 'o-', color='green', lw=2,
           label=f'pCN (beta={BETA})')
# 1/sqrt(d) 参考线（以 c=0.5 档在最小 d 处锚定，仅用于指示斜率）
_anchor = sweep[0.5]['move_ratio'][0]
_ref = [_anchor * np.sqrt(dims[0] / dd) for dd in dims]
ax[1].loglog(dims, _ref, 'k:', lw=1.6, label=r'参考斜率 $d^{-1/2}$')
ax[1].loglog(dims, mv_ratio_pcn, 'o-', color='green', lw=2,
             label=f'pCN (beta={BETA})')
for _a in ax:
    _a.set_xscale('log')
    _a.set_xlabel(r'维数（分辨率） $d$', fontsize=12)
    _a.grid(True, which='both', alpha=0.3)
    _a.legend(fontsize=8, ncol=2)
ax[0].set_ylabel('接受率', fontsize=12)
ax[0].set_title(r'RWM 调参扫描：小 $c$ 可换高接受率', fontsize=12)
ax[1].set_ylabel(r'每步净移动 / 先验典型半径 $\sqrt{\mathrm{tr}\,C}$', fontsize=12)
ax[1].set_title(r'但移动速度按 $d^{-1/2}$ 收缩（遍历时间 $O(d)$），pCN 平坦',
                fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'rwm_sweep.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  图已保存: {os.path.join(OUT_DIR, 'rwm_sweep.png')}")

# ── 步骤4：pCN 经验后验均值 vs 解析后验均值（校验正确性）──────────
print("\n[步骤4] d=256 下 pCN 经验后验均值 vs 解析后验均值（校验收敛）")
d = 256
blur_g = blur_envelope(d, SIGMA_BLUR)
lam = prior_eigs(d, TAU)
xt = sample_prior(lam, RNG)
yt = apply_blur(xt, blur_g) + SIGMA_NOISE * RNG.standard_normal(d)
x0_4 = sample_prior(lam, RNG)                     # 平稳起点
mu_post = analytic_posterior_mean(yt, blur_g, lam, SIGMA_NOISE)   # 解析解
N4 = 40000
samples_pcn, ar_pcn4, _ = pcn_sampler(log_likelihood, lam, blur_g, SIGMA_NOISE, d,
                                      n_samples=N4, beta=BETA, burn_in=2000,
                                      rng=RNG, y=yt, x0=x0_4)
pcn_mean = samples_pcn.mean(axis=0)

mse_pcn_vs_truth = float(np.mean((pcn_mean - xt) ** 2))
mse_pcn_vs_analytic = float(np.mean((pcn_mean - mu_post) ** 2))
mse_analytic_vs_truth = float(np.mean((mu_post - xt) ** 2))
print(f"  pCN 接受率={ar_pcn4:.4f}")
print(f"  解析后验均值 vs 真值        MSE={mse_analytic_vs_truth:.4e}")
print(f"  pCN 经验均值 vs 真值        MSE={mse_pcn_vs_truth:.4e}")
print(f"  pCN 经验均值 vs 解析后验均值 MSE={mse_pcn_vs_analytic:.4e}")

# Fourier 模态级校验：模糊只保留 |k|=0,1（g_1=e^-2≈0.135，|k|>=2 时
# g_k<=3.4e-4，后验均值≈0、本就不可辨识）。逐模态比较 pCN 均值与解析均值；
# 能量按 Parseval 除以 d（比值时常数约去）。
mu_hat4 = np.fft.fft(mu_post)
pm_hat4 = np.fft.fft(pcn_mean)
k_modes = np.arange(0, 11)
mu_mode_e, mc_mode_e = [], []
for kk in k_modes:
    idx = [kk] if kk == 0 else [kk, d - kk]
    mu_mode_e.append(float(np.mean([abs(mu_hat4[j]) ** 2 for j in idx]) / d))
    mc_mode_e.append(float(np.mean([abs(pm_hat4[j] - mu_hat4[j]) ** 2
                                    for j in idx]) / d))
informative = [int(kk) for kk in k_modes if blur_g[kk] >= 0.01]   # 可辨识模态
mode_rel_err = {str(kk): float(mc_mode_e[kk] / mu_mode_e[kk])
                for kk in informative if mu_mode_e[kk] > 1e-12}
# 积分自相关时间（IACT）估计：逐坐标样本均值误差 / 逐坐标 iid 理论下限。
# 注：mse_pcn_vs_analytic 由 np.mean 得到、已是 d 个坐标的平均，故 iid 下限
#     必须写成 tr(C_post)/(d*N)；若只除 tr(C_post)/N，IACT 会被低估 d 倍。
tr_cpost = float(np.sum(1.0 / (blur_g ** 2 / SIGMA_NOISE ** 2 + lam)))
iact_est = float(mse_pcn_vs_analytic / (tr_cpost / (d * N4)))
print("  可辨识模态（g_k>=0.01）上 pCN 均值相对误差："
      + ", ".join(f"|k|={k}: {mode_rel_err[str(k)]:.2e}" for k in informative))
print(f"  IACT 估计（实测逐坐标MC误差/iid下限 tr(Cpost)/(d*N)={tr_cpost/(d*N4):.2e}）"
      f"≈ {iact_est:.1f}，ESS≈{N4/iact_est:.1f}（IACT≈1 表示样本接近独立）")

fig, ax = plt.subplots(1, 4, figsize=(18.5, 4.2))
t = np.arange(d)
ax[0].plot(t, xt, 'k-', lw=1.5, label='真值 x_true')
ax[0].plot(t, mu_post, 'C3', lw=1.2, label='解析后验均值')
ax[0].set_title('真值 vs 解析后验均值', fontsize=12); ax[0].legend(fontsize=9)
ax[1].plot(t, xt, 'k-', lw=1.5, label='真值 x_true')
ax[1].plot(t, pcn_mean, 'C2', lw=1.2, label='pCN 经验后验均值')
ax[1].set_title('真值 vs pCN 经验后验均值', fontsize=12); ax[1].legend(fontsize=9)
ax[2].plot(t, pcn_mean - mu_post, 'C4', lw=1.2,
           label='pCN均值 - 解析均值')
ax[2].axhline(0, color='k', lw=0.8)
ax[2].set_title(r'两者偏差（应$\approx 0$）', fontsize=12); ax[2].legend(fontsize=9)
ax[3].semilogy(k_modes, np.maximum(mu_mode_e, 1e-20), 'C3o-',
               label='解析均值模态能量')
ax[3].semilogy(k_modes, np.maximum(mc_mode_e, 1e-20), 'C4x--',
               label='pCN 均值 MC 误差能量')
ax[3].axvspan(max(informative) + 0.5, k_modes[-1] + 0.5,
              color='0.9', label='不可辨识（被模糊抹除）')
ax[3].set_xlabel(r'空间频率 $|k|$', fontsize=11)
ax[3].set_title('Fourier 模态级校验（可辨识模态误差极小）', fontsize=12)
ax[3].legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'reconstruction.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  图已保存: {os.path.join(OUT_DIR, 'reconstruction.png')}")

# ── 叠加图：真值 / 带噪观测 / 解析后验均值 / pCN 经验后验均值 ──────
# 把解析均值（红色实线）与 pCN 均值（绿色虚线）画在同一坐标上，
# 两者几乎重合时绿虚线恰好覆盖红实线，直观展示 pCN 收敛到正确后验。
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.plot(t, xt, 'k-', lw=1.0, alpha=0.85, label='真值 $x_{\\rm true}$')
ax.plot(t, yt, color='0.6', lw=0.8, alpha=0.7, label='带噪观测 $y=Bx+\\eta$')
ax.plot(t, mu_post, 'C3-', lw=2.2, label='解析后验均值')
ax.plot(t, pcn_mean, 'C2--', lw=1.2, label='pCN 经验后验均值')
ax.set_xlabel(r'位置索引 $i$', fontsize=12)
ax.set_title(r'$d=256$：pCN 经验后验均值（绿虚线）叠加于解析后验均值（红实线）', fontsize=12)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'overlay.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  图已保存: {os.path.join(OUT_DIR, 'overlay.png')}")

# ── 小结 ──────────────────────────────────────────────────────────
# 动态计算统计量，避免硬编码结论
pcn_mean_acc = float(np.mean(acc_pcn))
pcn_spread_acc = float(np.max(acc_pcn) - np.min(acc_pcn))
rwm_mean_acc = float(np.mean(acc_rwm))
# 各档在“最优 c”下的步长/半径（取所有 c 的最大值），用于量化 1/sqrt(d) 收缩
best_move = [max(sweep[c]['move_ratio'][j] for c in rwm_cs)
             for j in range(len(dims))]
shrink = best_move[-1] / best_move[0]
pcn_move_mean = float(np.mean(mv_ratio_pcn))
worst_mode_err = max(mode_rel_err.values())
summary = {
    "实验": "实验4.2-4 pCN-MCMC 实现：维度无关性验证",
    "设定": (f"1D 周期信号贝叶斯反卷积；先验 N(0,C), C=(I+{TAU}**2*(-Delta))^-1; "
             f"模糊带宽 sigma_blur={SIGMA_BLUR}; 噪声 sigma={SIGMA_NOISE}; "
             f"每档分辨率独立生成数据（实例间涨落已包含在跨档差异中）；"
             f"两条链均从先验抽样的平稳起点出发"),
    "分辨率列表": dims,
    "RWM_接受率(sigma=2.38/sqrt(d))": {str(dd): round(a, 4) for dd, a in zip(dims, acc_rwm)},
    "pCN_接受率(beta={})".format(BETA): {str(dd): round(a, 4) for dd, a in zip(dims, acc_pcn)},
    "pCN_接受率_均值": round(pcn_mean_acc, 4),
    "pCN_接受率_最大最小差": round(pcn_spread_acc, 4),
    "RWM_接受率_均值": round(rwm_mean_acc, 4),
    "pCN_步长半径比(beta={})".format(BETA): {str(dd): round(m, 4)
                                              for dd, m in zip(dims, mv_ratio_pcn)},
    "RWM调参扫描_sigma=c/sqrt(d)": {
        f"c={c:g}": {
            "接受率": {str(dd): round(a, 4) for dd, a in zip(dims, sweep[c]['acc'])},
            "步长半径比": {str(dd): round(m, 5)
                          for dd, m in zip(dims, sweep[c]['move_ratio'])},
        } for c in rwm_cs},
    "RWM_各档最优c_步长半径比": {str(dd): round(m, 5)
                                for dd, m in zip(dims, best_move)},
    "d256_解析均值_vs_真值_MSE": mse_analytic_vs_truth,
    "d256_pCN均值_vs_真值_MSE": mse_pcn_vs_truth,
    "d256_pCN均值_vs_解析均值_MSE": mse_pcn_vs_analytic,
    "d256_可辨识模态(g_k>=0.01)_pCN均值相对误差": mode_rel_err,
    "d256_MC_IACT估计": round(iact_est, 2),
    "结论": (f"扫描分辨率 d 时，各向同性 RWM 在该平滑/各向异性后验上接受率均值≈"
             f"{rwm_mean_acc:.4f}；pCN 接受率在 beta 固定时基本不随 d 变化"
             f"（均值≈{pcn_mean_acc:.3f}，最大最小差≈{pcn_spread_acc:.3f}）"
             f"→ 维度无关（dimension-free）。步骤3b 对 sigma=c/sqrt(d) 扫描 c："
             f"小 c 可把接受率维持在较高水平，但即使逐档取最优 c，每步净移动/典型半径"
             f"仍从 d={dims[0]} 的 {best_move[0]:.4f} 收缩到 d={dims[-1]} 的 "
             f"{best_move[-1]:.4f}（比值 {shrink:.2f}，理论 d^-1/2 比值 "
             f"{np.sqrt(dims[0]/dims[-1]):.2f}；遍历时间 O(d)），而 pCN 同口径比值"
             f"稳定在 {pcn_move_mean:.3f} 附近 → RWM 失效是后验几何（无函数空间极限）"
             f"的固有属性，非调参问题。步骤4 Fourier 对角解析校验：pCN 均值与解析解 "
             f"MSE={mse_pcn_vs_analytic:.3e}；数据可辨识模态（|k|={','.join(map(str, informative))}）"
             f"相对误差≤{worst_mode_err:.1e}，IACT≈{iact_est:.1f}；"
             f"|k|>=2 模态被模糊抹除（后验均值本就≈0），其重建残差不反映算法问题。")
}
with open(os.path.join(OUT_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("结论:")
print("  1. RWM 接受率均值≈{:.4f}（白噪声提议无法适配先验几何）".format(rwm_mean_acc))
print(f"  2. pCN(beta={BETA}) 接受率不随维数变化 → 维度无关（dimension-free）")
print(f"  3. RWM 调参扫描：高接受率换不来混合速度，最优步长/半径仍按 1/sqrt(d) "
      f"收缩（{best_move[0]:.3f}→{best_move[-1]:.3f}），pCN 稳定在 {pcn_move_mean:.3f}")
print(f"  4. 可辨识模态 |k|={','.join(map(str, informative))} 上 pCN 均值相对误差"
      f"≤{worst_mode_err:.1e}，IACT≈{iact_est:.1f} → pCN 收敛到正确后验")
print("  5. 高频模态不可辨识（模糊固有），其残差不构成算法缺陷")
print(f"\n实验完成。结果已保存至: {OUT_DIR}")
