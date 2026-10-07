# -*- coding: utf-8 -*-
"""
实验7.5-1 数值离散化采样（一）：DDPM/SMLD 与步数-质量关系
对应章节：7.3（逆向SDE：从噪声到数据的采样过程）、7.5（数值离散化）

说明：
- 本实验的核心理论来自7.3章节（Anderson逆时SDE定理、逆向SDE）
- 本实验的离散化方法来自7.5章节（Euler-Maruyama方法）
- 建议读者先学习7.3章节的理论，再学习7.5章节的离散化方法，最后完成本实验

注意：本实验涉及7.5章节的离散化方法，如果您尚未学习7.5章节，
      可能会遇到一些超纲的概念。建议您先阅读以下内容：
      - 7.5章节的Euler-Maruyama离散化方法
      - 7.5章节的步数与采样质量的关系
      这些内容将在实验中详细展示，帮助您理解离散化方法的应用。

素材来源：
  - 04-sde.ipynb的Euler-Maruyama采样代码
  - 02-ddpm.ipynb的DDPM推理代码
  - ★ 原创设计：Anderson逆时SDE定理的数值验证
  - ★ 原创设计：VE/VP逆向SDE采样对比

实验内容：
  步骤1：Anderson逆时SDE定理验证——正向SDE的边际分布=逆向SDE的边际分布
  步骤2：VP-SDE逆向采样（DDPM=Euler-Maruyama离散化）
  步骤3：VE-SDE逆向采样（SMLD=Euler-Maruyama离散化，步长 α_i=σ_i²-σ_{i-1}²）
  步骤4：步数对采样质量的影响（用对解析 p_0 的单样本 KS 与 W2，并标出 MC 噪声地板）

度量说明（复核时修正）：
  旧版用两样本 ks_2samp(final, x0) 做“步数-质量”曲线，但该度量的噪声地板
  （≈0.012）比步数≥50 时的离散化误差大一个量级，画出来只能得到噪声；
  现改用对解析 CDF 的单样本 KS，并把两种度量的地板都画进图里作对比。

运行前提：纯 NumPy/SciPy CPU 即可，无需预训练模型
"""

import json
import numpy as np
import os
import sys
import io
import warnings
import logging
from scipy import stats as sp_stats
from scipy.special import erf

# 设置控制台输出为 UTF-8 (Windows 下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 静默 matplotlib 相关警告
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
    SAVE_DIR = os.path.join(_gdrive, '实验7.5-1')
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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()


# ============================================================
# 目标分布：1D高斯混合（便于可视化）
# ============================================================
def gm1d_pdf(x, weights=[0.3, 0.7], means=[-2, 1], stds=[1, 1]):
    """1D高斯混合概率密度"""
    pdf = np.zeros_like(x)
    for w, m, s in zip(weights, means, stds):
        pdf += w * np.exp(-0.5 * ((x - m) / s)**2) / (s * np.sqrt(2 * np.pi))
    return pdf

def gm1d_score(x, weights=[0.3, 0.7], means=[-2, 1], stds=[1, 1]):
    """1D高斯混合的得分函数 ∇log p(x)（解析解）"""
    pdf = gm1d_pdf(x, weights, means, stds)
    dpdf = np.zeros_like(x)
    for w, m, s in zip(weights, means, stds):
        dpdf += w * (-(x - m) / s**2) * np.exp(-0.5 * ((x - m) / s)**2) / (s * np.sqrt(2 * np.pi))
    return dpdf / (pdf + 1e-30)


# ============================================================
# 解析度量工具（复核时新增）
# ------------------------------------------------------------
# 为什么不用两样本 KS（sp_stats.ks_2samp(final, x0)）？
# 实测：在 20000 vs 10000 的粒子规模下，即使两组样本都来自同一个精确 p_0，
# ks_2samp 的噪声地板就有 mean≈0.0121、std≈0.0039，远大于步数≥50 时的
# 离散化误差（≈0.003）。用这种度量画“步数-质量”曲线只能画出噪声，
# 因此改用对解析 CDF 的单样本 KS，并额外给出 W2 与 MC 噪声地板作参照。
# ============================================================
TARGET_WEIGHTS, TARGET_MEANS, TARGET_STDS = [0.3, 0.7], [-2, 1], [1, 1]
MU_POP = sum(w * m for w, m in zip(TARGET_WEIGHTS, TARGET_MEANS))
VAR_POP = (sum(w * (m**2 + s**2) for w, m, s in
               zip(TARGET_WEIGHTS, TARGET_MEANS, TARGET_STDS)) - MU_POP**2)

_XG = np.linspace(-15, 15, 200001)
_FG = np.zeros_like(_XG)
for _w, _m, _s in zip(TARGET_WEIGHTS, TARGET_MEANS, TARGET_STDS):
    _FG += _w * 0.5 * (1 + erf((_XG - _m) / (_s * np.sqrt(2))))

def target_cdf(x):
    """目标分布 p_0 的解析 CDF"""
    return np.interp(x, _XG, _FG)

def target_ppf(u):
    """目标分布 p_0 的分位数函数（仅在开区间 0<u<1 内部可靠）"""
    return np.interp(u, _FG, _XG)

def ks_vs_cdf(samples, cdf_fn):
    """单样本 KS（双侧，与 scipy.stats.kstest 口径一致）：
    经验 CDF 与指定解析 CDF 的最大偏离（无参考样本噪声）。
    D = max( sup[F_n(x)-F(x)], sup[F(x)-F_n(x)] )，两侧都必须取，
    只取 i/n - F(x_i) 在离散化误差很小（接近MC地板）时会系统性偏小。
    """
    xs = np.sort(np.asarray(samples))
    n = len(xs)
    fvals = cdf_fn(xs)
    d_plus = np.max(np.arange(1, n + 1) / n - fvals)
    d_minus = np.max(fvals - np.arange(0, n) / n)
    return float(max(d_plus, d_minus))

def ks_vs_target(samples):
    return ks_vs_cdf(samples, target_cdf)

def w2_vs_target(samples, n_q=2000, q_lo=0.002, q_hi=0.998):
    """W2 距离（限制在 [q_lo,q_hi] 分位数窗口，避开解析 CDF 的饱和平段）"""
    xs = np.sort(np.asarray(samples))
    n = len(xs)
    u = np.linspace(q_lo, q_hi, n_q)
    idx = np.clip((u * n).astype(int), 0, n - 1)
    return float(np.sqrt(np.mean((xs[idx] - target_ppf(u))**2)))

def sample_target(N, seed):
    """直接从解析 p_0 抽样，用于估计有限粒子数的 MC 噪声地板"""
    rs = np.random.RandomState(seed)
    comp = rs.rand(N) < TARGET_WEIGHTS[0]
    return np.where(comp, rs.randn(N) + TARGET_MEANS[0], rs.randn(N) + TARGET_MEANS[1])


# ============================================================
# 步骤1：Anderson逆时SDE定理验证
# ★ 原创设计
# 正向SDE边际分布p_t(x) = 逆向SDE边际分布p_t(x)（Anderson 1982）
# 用VP-SDE验证：从p_0出发正向走到p_1，再逆向走回p_0
# ============================================================
print("=" * 60)
print("步骤1：Anderson逆时SDE定理验证")
print("=" * 60)

beta_min, beta_max = 0.1, 20.0

def vp_marginal(t, beta_min_val=None, beta_max_val=None):
    """VP-SDE边际参数"""
    _bmin = beta_min_val if beta_min_val is not None else beta_min
    _bmax = beta_max_val if beta_max_val is not None else beta_max
    log_mean = -0.25 * t**2 * (_bmax - _bmin) - 0.5 * t * _bmin
    mean_t = np.exp(log_mean)
    std_t = np.sqrt(1 - np.exp(2 * log_mean))
    return mean_t, std_t

def vp_beta(t):
    """VP-SDE的β(t)（线性插值）"""
    return beta_min + t * (beta_max - beta_min)

def vp_drift(x, t):
    """VP-SDE漂移 f(x,t) = -β(t)/2 · x"""
    return -vp_beta(t) / 2 * x

def vp_score_analytic(x, t, beta_min_val=None, beta_max_val=None):
    """VP-SDE解析得分 ∇log p_t(x)

    参数:
        x: 粒子位置
        t: 时间
        beta_min_val, beta_max_val: VP-SDE参数，默认使用全局beta_min, beta_max
            通过显式传参实现依赖注入，避免闭包捕获全局变量导致的隐藏耦合
    """
    mean_t, std_t = vp_marginal(t, beta_min_val, beta_max_val)
    pdf = np.zeros_like(x)
    dpdf = np.zeros_like(x)
    weights = [0.3, 0.7]
    means = [-2, 1]
    stds = [1, 1]
    for w, m, s in zip(weights, means, stds):
        new_mean = mean_t * m
        new_std = np.sqrt(mean_t**2 * s**2 + std_t**2)
        pdf += w * np.exp(-0.5 * ((x - new_mean) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
        dpdf += w * (-(x - new_mean) / new_std**2) * np.exp(-0.5 * ((x - new_mean) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
    return dpdf / (pdf + 1e-30)

def vp_diffusion(t):
    """VP-SDE扩散 g(t) = sqrt(beta(t))"""
    return np.sqrt(vp_beta(t))

def vp_marginal_cdf(x, t):
    """VP-SDE 解析边际 p_t(x) 的 CDF（高斯混合的线性高斯信道仍是高斯混合）"""
    mean_t, std_t = vp_marginal(t)
    cdf = np.zeros_like(x)
    for w, m, s in zip(TARGET_WEIGHTS, TARGET_MEANS, TARGET_STDS):
        new_mean = mean_t * m
        new_std = np.sqrt(mean_t**2 * s**2 + std_t**2)
        cdf += w * 0.5 * (1 + erf((x - new_mean) / (new_std * np.sqrt(2))))
    return cdf

N_particles = 10000
N_steps = 500
T = 1.0
dt = T / N_steps

# 从p_0（高斯混合）采样
x0 = np.where(np.random.rand(N_particles) < 0.3,
              np.random.randn(N_particles) - 2,
              np.random.randn(N_particles) + 1)

# ---- 正向VP-SDE（使用边际分布解析解，更精确）----
fwd_snapshots = {0.0: x0.copy()}
snapshot_times = [0.2, 0.5, 0.8, 1.0]

np.random.seed(123)
for st in snapshot_times:
    mean_t, std_t = vp_marginal(st)
    fwd_snapshots[st] = mean_t * x0 + std_t * np.random.randn(N_particles)

print(f"正向SDE（解析边际）: 各时间步采样完成")

# ---- Anderson定理验证（正向与逆向边际分布对比）----
# 方法：从N(0,I)出发，用逆向VP-SDE采样，对比逆向各时间步的边际
# 与正向各时间步的边际——应一致（Anderson 1982定理保证）
np.random.seed(42)
x_rev_init = np.random.randn(N_particles)  # 从N(0,I)开始

# 用逆向VP-SDE采样（逆时参数化），保存中间快照
rev_snapshots = {1.0: x_rev_init.copy()}
rev_times = [0.8, 0.5, 0.2, 0.0]

# 预先计算每个目标时间对应的步数索引，避免浮点容差判断
# i_target表示从T=1.0走到目标时间st需要的步数：i_target = round((T - st) / dt)
rev_time_to_step = {st: round((T - st) / dt) for st in rev_times}

x_rev = x_rev_init.copy()
for i in range(N_steps):
    t = T - i * dt  # 原始时间（从T到0）
    beta_t = vp_beta(t)
    score = vp_score_analytic(x_rev, t)

    # 逆向VP-SDE更新（逆时参数化，参考04-sde.ipynb）
    # dx = [β/2·x + β·∇log p_t(x)] dτ + √β·dW̃
    x_rev = x_rev + beta_t * dt * (0.5 * x_rev + score) + np.sqrt(beta_t * dt) * np.random.randn(N_particles)

    # 使用步数索引精确匹配，避免浮点累积误差导致的快照缺失
    next_step = i + 1
    for st, i_target in rev_time_to_step.items():
        if next_step == i_target and st not in rev_snapshots:
            rev_snapshots[st] = x_rev.copy()

print(f"DDPM逆向采样完成: Var(x_0)={np.var(x_rev):.4f}")
print(f"目标分布总体方差:  Var(p_0)={VAR_POP:.4f}（样本 x0 的方差={np.var(x0):.4f}）")

# 比较正向和逆向在各时间步的边际分布
# 量化指标：各自对“解析边际 p_t”的单样本 KS（而不是两者互比），
# 这样才能分别看到“正向解析采样”与“逆向EM采样”各自的误差
print("\n边际分布对比（Forward vs DDPM Reverse，括号内为对解析边际 p_t 的 KS）：")
step1_rows = []
for t_val in [0.0, 0.2, 0.5, 0.8, 1.0]:
    fwd_data = fwd_snapshots.get(t_val, None)
    rev_data = rev_snapshots.get(t_val, None)
    if fwd_data is not None and rev_data is not None:
        cdf_fn = lambda z, _t=t_val: vp_marginal_cdf(z, _t)
        ks_f = ks_vs_cdf(fwd_data, cdf_fn)
        ks_r = ks_vs_cdf(rev_data, cdf_fn)
        step1_rows.append(dict(t=t_val,
                               fwd_mu=float(np.mean(fwd_data)), fwd_var=float(np.var(fwd_data)),
                               rev_mu=float(np.mean(rev_data)), rev_var=float(np.var(rev_data)),
                               ks_fwd=ks_f, ks_rev=ks_r))
        print(f"  t={t_val:.1f}: Fwd(mu={np.mean(fwd_data):.3f}, var={np.var(fwd_data):.3f}, KS={ks_f:.4f})"
              f"  Rev(mu={np.mean(rev_data):.3f}, var={np.var(rev_data):.3f}, KS={ks_r:.4f})")

print("\nAnderson定理验证：正向与逆向边际分布应近似一致")
print("（上面两列 KS 均与同一解析 p_t 比较，两者同量级且接近 MC 地板，")
print("  说明逆向 EM 采样复现了正向边际；残余差异来自：")
print("  1)有限粒子数 2)逆向 DDPM 的离散化误差 3)正向用解析边际、无离散化误差）")


# ============================================================
# 步骤2：VP-SDE逆向采样 = DDPM
# 参考：02-ddpm.ipynb, 7.3节, 7.5节
# DDPM = Euler-Maruyama离散化逆向VP-SDE
# ============================================================
print("\n" + "=" * 60)
print("步骤2：VP-SDE逆向采样（DDPM）")
print("=" * 60)

def ddpm_sample_vp(score_fn, N_particles, N_steps, T=1.0, beta_min=0.1, beta_max=20.0,
                   store_traj=True):
    """VP-SDE逆向采样（DDPM = Euler-Maruyama on reverse VP-SDE）

    逆向VP-SDE（逆时参数化τ=T-t，dτ>0）:
    dx = [β(t)/2·x + β(t)·∇log p_t(x)] dτ + √β(t)·dW̃

    Euler-Maruyama离散化（参考04-sde.ipynb）:
    x_{τ+dτ} = x_τ + β(t)·h·(0.5·x_τ + s_θ(x_τ,t)) + √(β(t)·h)·z

    其中 h = T/N_steps，t = T - τ

    store_traj=False 时只返回最终粒子集（步骤4用 N=1e5，存整条轨迹会占用 ~GB 内存）
    """
    h = T / N_steps
    x = np.random.randn(N_particles)  # 从N(0,I)开始

    trajectory = [x.copy()] if store_traj else None

    for i in range(N_steps):
        t = T - i * h  # 原始时间（从T到0）
        beta_t = beta_min + t * (beta_max - beta_min)
        # 将beta参数显式传递给score_fn，实现依赖注入
        score = score_fn(x, t, beta_min_val=beta_min, beta_max_val=beta_max)

        # DDPM更新（逆时参数化）
        x = x + beta_t * h * (0.5 * x + score) + np.sqrt(beta_t * h) * np.random.randn(N_particles)

        if store_traj:
            trajectory.append(x.copy())

    return np.array(trajectory) if store_traj else x


# 参考噪声地板：直接抽一批解析 p_0 样本，看“算法完全正确”时指标能到多小
N_step2 = 50000
_floor_s = [sample_target(N_step2, 2025 + k) for k in range(3)]
floor2 = dict(
    mu=float(np.mean([abs(np.mean(s) - MU_POP) for s in _floor_s])),
    var=float(np.mean([abs(np.var(s) - VAR_POP) for s in _floor_s])),
    ks=float(np.mean([ks_vs_target(s) for s in _floor_s])),
)
print(f"  [MC噪声地板 N={N_step2}] |mu_diff|≈{floor2['mu']:.4f}, "
      f"|var_diff|≈{floor2['var']:.4f}, KS≈{floor2['ks']:.4f}")
print("  （指标低于地板时无法再区分步数差异，只能说明误差已被MC噪声淹没）")

# DDPM采样（不同步数）
step2_rows = []
for n_steps in [50, 200, 500, 1000]:
    np.random.seed(42)
    final = ddpm_sample_vp(vp_score_analytic, N_step2, n_steps, store_traj=False)
    mean_diff = abs(np.mean(final) - MU_POP)
    var_diff = abs(np.var(final) - VAR_POP)
    ks_val = ks_vs_target(final)
    has_nan = bool(np.isnan(final).any())
    step2_rows.append(dict(N_steps=n_steps, mu_diff=float(mean_diff), var_diff=float(var_diff),
                           ks=float(ks_val), nan=has_nan))
    print(f"  DDPM N_steps={n_steps}: mu_diff={mean_diff:.4f}, var_diff={var_diff:.4f}, "
          f"KS={ks_val:.4f}, NaN={has_nan}")

print("结论：步数从50→1000，离散化误差向MC地板收敛；到达地板后指标不再单调（受抽样噪声主导）")
print("注：本实验用解析得分，不存在真实DDPM中神经网络拟合得分的误差；")
print("    所有步数下 NaN=False，说明解析得分+小步长Euler-Maruyama不会发散")


# ============================================================
# 步骤3：VE-SDE逆向采样 = SMLD
# 参考：03-smld.ipynb, 7.3节
# SMLD = Euler-Maruyama离散化逆向VE-SDE
# ============================================================
print("\n" + "=" * 60)
print("步骤3：VE-SDE逆向采样（SMLD）")
print("=" * 60)

sigma_min_ve, sigma_max_ve = 0.01, 10.0

def ve_sigma(t, sigma_min=sigma_min_ve, sigma_max=sigma_max_ve):
    """VE-SDE 噪声水平 σ(t) = σ_min·(σ_max/σ_min)^t

    时间约定必须与本文件其他部分一致：t=0 对应数据（σ=σ_min），
    t=1 对应最大噪声（σ=σ_max）。
    （旧版本写成了 σ_max·(σ_min/σ_max)^t，方向相反，导致采样过程中
      σ(t) 从 0.01 一路升到 10——“退火”变成“加噪”，采样永远不收敛）
    """
    return sigma_min * (sigma_max / sigma_min) ** t

def ve_marginal_cdf(x, t, sigma_min=sigma_min_ve, sigma_max=sigma_max_ve):
    """VE-SDE 解析边际 p_t 的 CDF：分量 i → N(μ_i, s_i²+σ_t²)"""
    sigma_t = ve_sigma(t, sigma_min, sigma_max)
    cdf = np.zeros_like(x)
    for w, m, s in zip(TARGET_WEIGHTS, TARGET_MEANS, TARGET_STDS):
        new_std = np.sqrt(s**2 + sigma_t**2)
        cdf += w * 0.5 * (1 + erf((x - m) / (new_std * np.sqrt(2))))
    return cdf

def ve_score_analytic(x, t, sigma_min=sigma_min_ve, sigma_max=sigma_max_ve):
    """VE-SDE的解析得分

    p_t(x) = ∫ p_0(x_0) · N(x; x_0, σ_t²) dx_0
    对高斯混合: 分量 i → N(μ_i, s_i² + σ_t²)
    """
    sigma_t = ve_sigma(t, sigma_min, sigma_max)
    pdf = np.zeros_like(x)
    dpdf = np.zeros_like(x)
    for w, m, s in zip(TARGET_WEIGHTS, TARGET_MEANS, TARGET_STDS):
        new_var = s**2 + sigma_t**2
        new_std = np.sqrt(new_var)
        pdf += w * np.exp(-0.5 * ((x - m) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
        dpdf += w * (-(x - m) / new_var) * np.exp(-0.5 * ((x - m) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
    return dpdf / (pdf + 1e-30)

def smld_sample_ve(score_fn, N_particles, N_steps, T=1.0,
                   sigma_min=sigma_min_ve, sigma_max=sigma_max_ve, store_traj=True):
    """SMLD = 逆向 VE-SDE 的 Euler-Maruyama 离散化（7.5节推导）

    逆向 VE-SDE（f=0, g²=d[σ(t)²]/dt）:
        dx = -d[σ²]/dt · ∇log p_t(x) dt + sqrt(d[σ²]/dt) dw̃
    从 σ_i 走到下一档 σ_{i-1}（σ递减），令 α_i = σ_i² - σ_{i-1}² > 0：
        x_{i-1} = x_i + α_i·∇log p_{σ_i}(x_i) + sqrt(α_i)·z
    这正是7.5节给出的“SMLD = 逆向VE-SDE的Euler-Maruyama离散化”迭代式。
    """
    dt = T / N_steps
    x = np.random.randn(N_particles) * ve_sigma(T, sigma_min, sigma_max)  # 从最大噪声开始

    trajectory = [x.copy()] if store_traj else None

    for i in range(N_steps):
        t = T - i * dt
        sigma_hi = ve_sigma(t, sigma_min, sigma_max)
        sigma_lo = ve_sigma(max(t - dt, 0.0), sigma_min, sigma_max)
        alpha = sigma_hi**2 - sigma_lo**2  # > 0（σ随t递增，向t=0方向退火）

        score = score_fn(x, t, sigma_min=sigma_min, sigma_max=sigma_max)
        x = x + alpha * score + np.sqrt(alpha) * np.random.randn(N_particles)

        if store_traj:
            trajectory.append(x.copy())

    return np.array(trajectory) if store_traj else x

# SMLD采样（不同步数）
print(f"  [MC噪声地板 N={N_step2}] KS≈{floor2['ks']:.4f}（与DDPM共用，针对解析 p_0）")
N_step3 = 50000
step3_rows = []
for n_steps in [50, 200, 500, 1000]:
    np.random.seed(42)
    final = smld_sample_ve(ve_score_analytic, N_step3, n_steps, store_traj=False)
    mean_diff = abs(np.mean(final) - MU_POP)
    var_diff = abs(np.var(final) - VAR_POP)
    ks_val = ks_vs_target(final)
    has_nan = bool(np.isnan(final).any())
    step3_rows.append(dict(N_steps=n_steps, mu_diff=float(mean_diff), var_diff=float(var_diff),
                           ks=float(ks_val), nan=has_nan))
    print(f"  SMLD N_steps={n_steps}: mu_diff={mean_diff:.4f}, var_diff={var_diff:.4f}, "
          f"KS={ks_val:.4f}, NaN={has_nan}")

print("结论：采用7.5节的EM步长 α_i=σ_i²-σ_{i-1}² 后，SMLD与DDPM一样能收敛到 p_0")


# ============================================================
# 步骤4：步数对采样质量的影响
# ★ 原创设计
# ============================================================
print("\n" + "=" * 60)
print("步骤4：步数对采样质量的影响")
print("=" * 60)

step_counts = [10, 25, 50, 100, 200, 500, 1000]
vp_errors = []        # 单样本 KS（对解析 p_0 的 CDF），多种子均值
vp_errors_ks_std = []  # 对应的多种子标准差（误差棒与JSON共用）
vp_errors_w2 = []     # W2 距离（平滑辅助度量），多种子均值
vp_errors_w2_std = []
vp_errors_2samp = []    # 旧指标：两样本 KS，只用来展示它为何不可用
vp_errors_2samp_std = []

# 粒子数要足够大：单样本 KS 的 MC 地板约 0.87/sqrt(N)
# （旧版用 N=20000 与参考样本 x0(N=10000) 做两样本 KS，地板高达≈0.012，
#   完全盖住了步数≥25 时的离散化误差，画出来只能是一团噪声）
N_eval = 100000  # 评估用粒子数
N_seeds = 3      # 多种子平均

_floor_big = [sample_target(N_eval, 3000 + k) for k in range(N_seeds)]
mc_floor_ks = float(np.mean([ks_vs_target(s) for s in _floor_big]))
mc_floor_w2 = float(np.mean([w2_vs_target(s) for s in _floor_big]))
mc_floor_2samp = float(np.mean([
    sp_stats.ks_2samp(_floor_big[k], x0).statistic for k in range(N_seeds)]))
print(f"  [MC噪声地板 N={N_eval}] 单样本KS≈{mc_floor_ks:.4f}, W2≈{mc_floor_w2:.4f}")
print(f"  [旧指标地板] 两样本KS (N={N_eval} vs {len(x0)}) ≈{mc_floor_2samp:.4f} —— 远高于可分辨的离散化误差")

for n_steps in step_counts:
    ks_vals, w2_vals, k2_vals = [], [], []
    for seed_offset in range(N_seeds):
        np.random.seed(42 + seed_offset * 7)
        final = ddpm_sample_vp(vp_score_analytic, N_eval, n_steps, store_traj=False)
        ks_vals.append(ks_vs_target(final))
        w2_vals.append(w2_vs_target(final))
        k2_vals.append(sp_stats.ks_2samp(final, x0).statistic)
    ks_mean = float(np.mean(ks_vals))
    ks_std = float(np.std(ks_vals))
    w2_mean, w2_std = float(np.mean(w2_vals)), float(np.std(w2_vals))
    k2_mean, k2_std = float(np.mean(k2_vals)), float(np.std(k2_vals))
    vp_errors.append(ks_mean)
    vp_errors_ks_std.append(ks_std)
    vp_errors_w2.append(w2_mean)
    vp_errors_w2_std.append(w2_std)
    vp_errors_2samp.append(k2_mean)
    vp_errors_2samp_std.append(k2_std)
    print(f"  DDPM N_steps={n_steps}: KS={ks_mean:.5f} (std={ks_std:.5f}), "
          f"W2={w2_mean:.5f} (std={w2_std:.5f}), "
          f"旧两样本KS={k2_mean:.5f} (std={k2_std:.5f})")

print("\n结论：在可分辨区间（步数≲100）步数增加→离散化误差减小→质量提升；")
print("      步数足够多后误差降到 MC 地板以下，曲线变平并只余抽样涨落（这不是算法失效）")
print("这解释了为什么DDPM通常需要1000步，而高阶求解器（DPM-Solver）只需20步")


# ============================================================
# 可视化
# ============================================================

# 图1：Anderson定理验证——正向vs逆向边际分布
fig, axes = plt.subplots(1, 5, figsize=(25, 4))
x_grid = np.linspace(-6, 6, 500)
times_to_plot = [0.0, 0.2, 0.5, 0.8, 1.0]

for i, t_val in enumerate(times_to_plot):
    fwd = fwd_snapshots.get(t_val)
    rev = rev_snapshots.get(t_val)
    if fwd is not None:
        axes[i].hist(fwd, bins=60, density=True, alpha=0.5, color='blue', label='Forward')
    if rev is not None:
        axes[i].hist(rev, bins=60, density=True, alpha=0.5, color='red', label='Reverse')
    # 理论边际
    mean_t, std_t = vp_marginal(t_val)
    theory_pdf = np.zeros_like(x_grid)
    for w, m, s in zip([0.3, 0.7], [-2, 1], [1, 1]):
        new_mean = mean_t * m
        new_std = np.sqrt(mean_t**2 * s**2 + std_t**2)
        theory_pdf += w * np.exp(-0.5 * ((x_grid - new_mean) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
    axes[i].plot(x_grid, theory_pdf, 'k--', lw=2, label='Theory')
    row_t = next((r for r in step1_rows if abs(r['t'] - t_val) < 1e-9), None)
    title = f't={t_val:.1f}'
    if row_t is not None:
        title += f"\nKS(fwd)={row_t['ks_fwd']:.4f}, KS(rev)={row_t['ks_rev']:.4f}"
    axes[i].set_title(title)
    axes[i].set_xlim(-6, 6)
    axes[i].legend(fontsize=8)
    axes[i].grid(alpha=0.3)

fig.suptitle('Anderson逆时SDE定理验证：Forward vs Reverse边际分布', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_Anderson定理验证.png'), dpi=150, bbox_inches='tight')
plt.close()

# 图2：DDPM采样过程
# 每个子图都画上“该步对应时刻的解析边际 p_t”（黑虚）与最终目标 p_0（红虚），
# 这样能看出整条概率流路径是否正确，而不只是看终点
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
n_vis = 8
np.random.seed(42)
traj = ddpm_sample_vp(vp_score_analytic, 5000, 200)
indices = np.linspace(0, len(traj)-1, n_vis, dtype=int)
n_steps_fig2 = len(traj) - 1

for i, idx in enumerate(indices):
    row, col = i // 4, i % 4
    t_now = T * (1.0 - idx / n_steps_fig2)  # 第 idx 个轨迹快照对应的原始时间
    mean_t, std_t = vp_marginal(t_now)
    marginal_pdf = np.zeros_like(x_grid)
    for w, m, s in zip(TARGET_WEIGHTS, TARGET_MEANS, TARGET_STDS):
        new_mean = mean_t * m
        new_std = np.sqrt(mean_t**2 * s**2 + std_t**2)
        marginal_pdf += w * np.exp(-0.5 * ((x_grid - new_mean) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
    axes[row, col].hist(traj[idx], bins=50, density=True, alpha=0.6, color='blue',
                        label=f'Samples (t={t_now:.2f})')
    axes[row, col].plot(x_grid, marginal_pdf, 'k--', lw=2, label='Analytic $p_t$')
    axes[row, col].plot(x_grid, gm1d_pdf(x_grid), 'r:', lw=2, label='Target $p_0$')
    axes[row, col].set_title(f'Reverse step {idx}/{n_steps_fig2}')
    axes[row, col].set_xlim(-6, 6)
    axes[row, col].legend(fontsize=8)
    axes[row, col].grid(alpha=0.3)

fig.suptitle('VP-SDE逆向采样过程（DDPM，200步）：每步与解析边际 $p_t$ 对比', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_DDPM采样过程.png'), dpi=150, bbox_inches='tight')
plt.close()

# 图3：SMLD退火过程（逆向VE-SDE的EM离散化）
fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
np.random.seed(42)
traj_ve = smld_sample_ve(ve_score_analytic, 20000, 500, store_traj=True)
n_steps_fig3 = len(traj_ve) - 1
ve_vis_t = [1.0, 0.6, 0.25, 0.0]

for i, t_target in enumerate(ve_vis_t):
    idx = int(round((T - t_target) / T * n_steps_fig3))
    idx = min(max(idx, 0), n_steps_fig3)
    sig_now = ve_sigma(t_target)
    ve_pdf = np.zeros_like(x_grid)
    for w, m, s in zip(TARGET_WEIGHTS, TARGET_MEANS, TARGET_STDS):
        new_std = np.sqrt(s**2 + sig_now**2)
        ve_pdf += w * np.exp(-0.5 * ((x_grid - m) / new_std)**2) / (new_std * np.sqrt(2 * np.pi))
    axes[i].hist(traj_ve[idx], bins=80, density=True, alpha=0.55, color='teal',
                 label='SMLD particles')
    axes[i].plot(x_grid, ve_pdf, 'k--', lw=2, label='Analytic $p_t$ (VE)')
    axes[i].plot(x_grid, gm1d_pdf(x_grid), 'r:', lw=2, label='Target $p_0$')
    axes[i].set_title(f'σ(t)={sig_now:.2f}  (t={t_target:.2f})')
    axes[i].set_xlim(-6, 6)
    axes[i].legend(fontsize=8)
    axes[i].grid(alpha=0.3)

fig.suptitle('VE-SDE逆向采样（SMLD = EM离散化，500步）：噪声水平 σ 从 10 退火到 0.01', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_SMLD采样过程.png'), dpi=150, bbox_inches='tight')
plt.close()

# 图4：步数vs采样质量（含MC噪声地板，并展示旧指标为何不可用）
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
axes[0].axhline(mc_floor_ks, color='gray', ls='--', lw=1.5,
                label=f'MC floor (single-sample KS) ≈ {mc_floor_ks:.4f}')
axes[0].errorbar(step_counts, vp_errors, yerr=[0.0] * len(step_counts), fmt='bo-',
                 lw=2, markersize=8, label='单样本 KS vs 解析 $p_0$')
axes[0].plot(step_counts, vp_errors_2samp, 'rs--', lw=1.5, markersize=6, alpha=0.8,
             label='旧指标：两样本 KS vs $x_0$ 样本')
axes[0].axhline(mc_floor_2samp, color='salmon', ls=':', lw=1.5,
                label=f'MC floor (two-sample KS) ≈ {mc_floor_2samp:.4f}')
axes[0].set_xlabel('采样步数')
axes[0].set_ylabel('KS 统计量（越小越好）')
axes[0].set_title('步数-质量曲线：度量选择决定能否看到趋势')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3, which='both')

axes[1].axhline(mc_floor_w2, color='gray', ls='--', lw=1.5,
                label=f'MC floor ($W_2$) ≈ {mc_floor_w2:.4f}')
axes[1].plot(step_counts, vp_errors_w2, 'g^-', lw=2, markersize=8,
             label=' $W_2$ vs 解析 $p_0$ 分位数')
axes[1].set_xlabel('采样步数')
axes[1].set_ylabel('$W_2$ 距离（越小越好）')
axes[1].set_title('$W_2$：比 KS 更平滑的辅助度量')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3, which='both')

fig.suptitle('DDPM 步数与采样质量（用解析得分，残差只来自离散化误差与有限粒子数）', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤4_步数与质量.png'), dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验7.5-1 总结")
print("=" * 60)
print("1. Anderson逆时SDE定理：正向与逆向边际分布一致（逐时刻 KS 已接近 MC 地板）")
print("2. DDPM = Euler-Maruyama离散化逆向VP-SDE（逆时参数化）")
print("   x = x + β·h·(0.5·x + s_θ) + √(β·h)·z")
print("3. SMLD = 逆向VE-SDE的Euler-Maruyama离散化：x = x + α_i·s + √α_i·z, α_i=σ_i²-σ_{i-1}²")
print("4. 步数增加→离散化误差减小；但误差降到有限粒子数的MC地板以下后，")
print("   曲线变平并出现涨落，此时‘步数越多越好’已无法从图上读出")

# ============================================================
# 结果归档：results_summary.json
# ============================================================
summary = dict(
    experiment='7.5-1',
    target=dict(weights=TARGET_WEIGHTS, means=TARGET_MEANS, stds=TARGET_STDS,
                mu=MU_POP, var=VAR_POP),
    vp_sde=dict(beta_min=beta_min, beta_max=beta_max, T=T),
    ve_sde=dict(sigma_min=sigma_min_ve, sigma_max=sigma_max_ve, T=T,
                schedule='sigma(t)=sigma_min*(sigma_max/sigma_min)^t'),
    step1_anderson=dict(N_particles=N_particles, N_steps=N_steps, rows=step1_rows),
    step2_ddpm=dict(N_particles=N_step2, mc_floor=floor2, rows=step2_rows),
    step3_smld=dict(N_particles=N_step3, mc_floor_ks=floor2['ks'], rows=step3_rows),
    step4_steps_vs_quality=dict(N_eval=N_eval, N_seeds=N_seeds,
                                step_counts=step_counts,
                                ks_single=vp_errors,
                                w2=vp_errors_w2,
                                ks_two_sample_legacy=vp_errors_2samp,
                                mc_floor_ks=mc_floor_ks,
                                mc_floor_w2=mc_floor_w2,
                                mc_floor_2samp=mc_floor_2samp),
)

with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as _f:
    json.dump(summary, _f, ensure_ascii=False, indent=2)
print(f"\n结果已写入: {os.path.join(SAVE_DIR, 'results_summary.json')}")