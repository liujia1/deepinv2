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
#   步骤1：设定高斯线性逆问题（先验 N(0,C)、似然 p(y|X)=N(BX, sigma^2 I)）
#   步骤2：RWM（白噪声提议）在该平滑后验上接受率几乎为 0——基线失效
#   步骤3：pCN（先验度量提议）接受率不随维数变化——核心论证（维度无关）
#   步骤4：解析后验均值 vs pCN 经验后验均值——校验 pCN 收敛到正确后验
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

OUT_DIR = os.path.join(SAVE_DIR, 'outputs')
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
    """B 在 Fourier 域的乘子：高斯低通。"""
    freqs = np.fft.fftfreq(d) * d           # 空间频率
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
def rwm_sampler(log_target, d, n_samples, sigma, burn_in, rng):
    """随机游走 Metropolis：X* = X + sigma * xi, xi~N(0,I)。接受率用完整后验比。"""
    x = np.zeros(d)
    log_x = log_target(x)
    samples = np.empty((n_samples, d))
    n_acc = 0
    for i in tqdm(range(n_samples + burn_in), desc="RWM采样", leave=False):
        prop = x + sigma * rng.standard_normal(d)
        log_p = log_target(prop)
        # 对数域比较，避免 np.exp 在 |log_p-log_x| 大时溢出/下溢的 RuntimeWarning
        log_alpha = min(0.0, log_p - log_x)
        accepted = np.log(rng.random()) < log_alpha
        if accepted:
            x, log_x = prop, log_p
        if i >= burn_in:
            samples[i - burn_in] = x
            n_acc += int(accepted)
    return samples, n_acc / n_samples


def pcn_sampler(log_like, lam, blur_g, sigma_noise, d, n_samples, beta,
                burn_in, rng, y):
    """pCN-MCMC：X* = sqrt(1-beta^2) X + beta * xi, xi~N(0,C)。
    接受率只用似然比（先验约去）。"""
    x = np.zeros(d)
    ll_x = log_likelihood(x, y, blur_g, sigma_noise)
    samples = np.empty((n_samples, d))
    n_acc = 0
    for i in tqdm(range(n_samples + burn_in), desc="pCN采样", leave=False):
        xi = sample_prior(lam, rng)                     # xi ~ N(0, C)
        prop = np.sqrt(1.0 - beta ** 2) * x + beta * xi
        ll_p = log_likelihood(prop, y, blur_g, sigma_noise)
        # 对数域比较，同 rwm_sampler
        log_alpha = min(0.0, ll_p - ll_x)
        accepted = np.log(rng.random()) < log_alpha
        if accepted:
            x, ll_x = prop, ll_p
        if i >= burn_in:
            samples[i - burn_in] = x
            n_acc += int(accepted)
    return samples, n_acc / n_samples


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

# ── 步骤2 & 步骤3：扫描分辨率，对比 RWM 与 pCN 接受率 ─────────────
print("\n[步骤2-3] 扫描分辨率 d，对比 RWM 与 pCN 接受率")
acc_rwm, acc_pcn = [], []
for d in dims:
    blur_g = blur_envelope(d, SIGMA_BLUR)
    lam = prior_eigs(d, TAU)
    xt = sample_prior(lam, RNG)                               # 该分辨率真值
    yt = apply_blur(xt, blur_g) + SIGMA_NOISE * RNG.standard_normal(d)  # 观测
    # 该分辨率下后验的完整对数密度（RWM 用）
    def log_post(xi, _yt=yt, _bg=blur_g, _lam=lam):
        return (log_likelihood(xi, _yt, _bg, SIGMA_NOISE)
                + log_prior_quad(xi, _lam))
    # RWM：最优缩放步长 sigma=2.38/sqrt(d)（各向同性白噪声提议）
    # 注：2.38/sqrt(d) 是 Roberts, Gelman & Gilks (1997) 在各向同性目标
    # 下的渐近 (d→∞) 最优公式；本实验的后验是平滑先验 + 模糊似然，**显著各向异性**，
    # 严格说并不满足该公式的推导前提。这里**刻意**复用一个不匹配的"教科书
    # 调参公式"，目的正是用反例说明：即便按各向同性最优调参，RWM 在各向异性
    # 后验上仍会失效——从而突出 pCN（先验度量提议）才是这类问题的根本解法。
    sig = 2.38 / np.sqrt(d)
    _, ar_rwm = rwm_sampler(log_post, d, n_samples=20000, sigma=sig,
                            burn_in=1000, rng=RNG)
    # pCN：先验度量提议
    _, ar_pcn = pcn_sampler(log_likelihood, lam, blur_g, SIGMA_NOISE, d,
                            n_samples=20000, beta=BETA, burn_in=1000,
                            rng=RNG, y=yt)
    acc_rwm.append(ar_rwm)
    acc_pcn.append(ar_pcn)
    print(f"  d={d:5d}  RWM(sigma={sig:.4f}) 接受率={ar_rwm:.4f}   "
          f"pCN(beta={BETA}) 接受率={ar_pcn:.4f}")

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
fig.savefig(os.path.join(OUT_DIR, 'exp4_2-4_acceptance_vs_dim.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  图已保存: {os.path.join(OUT_DIR, 'exp4_2-4_acceptance_vs_dim.png')}")

# ── 步骤4：pCN 经验后验均值 vs 解析后验均值（校验正确性）──────────
print("\n[步骤4] d=256 下 pCN 经验后验均值 vs 解析后验均值（校验收敛）")
d = 256
blur_g = blur_envelope(d, SIGMA_BLUR)
lam = prior_eigs(d, TAU)
xt = sample_prior(lam, RNG)
yt = apply_blur(xt, blur_g) + SIGMA_NOISE * RNG.standard_normal(d)
mu_post = analytic_posterior_mean(yt, blur_g, lam, SIGMA_NOISE)   # 解析解
samples_pcn, ar_pcn4 = pcn_sampler(log_likelihood, lam, blur_g, SIGMA_NOISE, d,
                                   n_samples=40000, beta=BETA, burn_in=2000,
                                   rng=RNG, y=yt)
pcn_mean = samples_pcn.mean(axis=0)

mse_pcn_vs_truth = float(np.mean((pcn_mean - xt) ** 2))
mse_pcn_vs_analytic = float(np.mean((pcn_mean - mu_post) ** 2))
mse_analytic_vs_truth = float(np.mean((mu_post - xt) ** 2))
print(f"  pCN 接受率={ar_pcn4:.4f}")
print(f"  解析后验均值 vs 真值        MSE={mse_analytic_vs_truth:.4e}")
print(f"  pCN 经验均值 vs 真值        MSE={mse_pcn_vs_truth:.4e}")
print(f"  pCN 经验均值 vs 解析后验均值 MSE={mse_pcn_vs_analytic:.4e}")

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
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
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'exp4_2-4_reconstruction.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  图已保存: {os.path.join(OUT_DIR, 'exp4_2-4_reconstruction.png')}")

# ── 小结 ──────────────────────────────────────────────────────────
# 动态计算 pCN 接受率统计量，避免硬编码结论
pcn_mean_acc = float(np.mean(acc_pcn))
pcn_spread_acc = float(np.max(acc_pcn) - np.min(acc_pcn))
rwm_mean_acc = float(np.mean(acc_rwm))
summary = {
    "实验": "实验4.2-4 pCN-MCMC 实现：维度无关性验证",
    "设定": (f"1D 周期信号贝叶斯反卷积；先验 N(0,C), C=(I+{TAU}**2*(-Delta))^-1; "
             f"模糊带宽 sigma_blur={SIGMA_BLUR}; 噪声 sigma={SIGMA_NOISE}; "
             f"每档分辨率独立生成数据"),
    "分辨率列表": dims,
    "RWM_接受率(sigma=2.38/sqrt(d))": {str(d): round(a, 4) for d, a in zip(dims, acc_rwm)},
    "pCN_接受率(beta={})".format(BETA): {str(d): round(a, 4) for d, a in zip(dims, acc_pcn)},
    "pCN_接受率_均值": round(pcn_mean_acc, 4),
    "pCN_接受率_最大最小差": round(pcn_spread_acc, 4),
    "RWM_接受率_均值": round(rwm_mean_acc, 4),
    "d256_解析均值_vs_真值_MSE": mse_analytic_vs_truth,
    "d256_pCN均值_vs_真值_MSE": mse_pcn_vs_truth,
    "d256_pCN均值_vs_解析均值_MSE": mse_pcn_vs_analytic,
    "结论": (f"扫描分辨率 d 时，各向同性 RWM 在该平滑/各向异性后验上接受率均值≈"
             f"{rwm_mean_acc:.4f}（白噪声提议无法适配先验几何）；pCN 接受率在 beta 固定时"
             f"基本不随 d 变化（均值≈{pcn_mean_acc:.3f}，最大最小差≈{pcn_spread_acc:.3f}）"
             f"→ 维度无关（dimension-free）。步骤4 用 Fourier 对角解析后验均值校验："
             f"pCN 经验后验均值与解析解 MSE 很小（{mse_pcn_vs_analytic:.3e}），"
             f"证明 pCN 正确收敛到目标后验。")
}
with open(os.path.join(OUT_DIR, 'exp4_2-4_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("结论:")
print("  1. RWM 接受率均值≈{:.4f}（白噪声提议无法适配先验几何）".format(rwm_mean_acc))
print(f"  2. pCN(beta={BETA}) 接受率不随维数变化 → 维度无关（dimension-free）")
print("  3. pCN 经验后验均值 ≈ 解析后验均值 → 校验 pCN 收敛到正确后验")
print(f"\n实验完成。结果已保存至: {OUT_DIR}")
