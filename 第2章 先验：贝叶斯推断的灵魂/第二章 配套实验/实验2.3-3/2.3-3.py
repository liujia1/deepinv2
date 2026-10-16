"""
实验2.3-3 不同先验下的去噪器函数曲线 D(y) vs y
对应章节：2.3 先验的质量：MMSE vs MAP 估计器 - “正向映射：给定先验 p(x) → 去噪器 D_σ(y)”
知识点：
  - 去噪器 D_σ(y) = E[x|y] (MMSE) 由先验 p(x) 与噪声水平 σ 共同决定；
  - 不同先验给出形状迥异的 D_σ(y)：
      无先验   → 恒等 D(y)=y (对角直线)
      高斯先验 → 线性缩放 D(y)=γ²/(γ²+σ²)·y (相对收缩恒定，无阈值)
      Laplace  → MMSE(后验均值)无初等闭式，但存在只用标准正态 Φ/φ 的截断正态
                 混合解析式（本实验用它交叉校验数值积分）；其 MAP 才有初等闭式软阈值
                 D_MAP(y)=sign(y)·max(|y|-σ²/b, 0)   ← 阈值是 σ²/b，不是 b·σ²
      Student-t→ 重尾先验的 MMSE：近零处温和收缩，大 |y| 处最接近恒等
  - 同一 Laplace 先验下同时画出 MMSE 与 MAP 两条曲线，直接呈现 2.3 节
    “MMSE vs MAP 分歧”；这正说明“先验的质量”体现在去噪器曲线形状上 (对应正文图2-5)。

说明：本实验为 1D 贝叶斯去噪器的数值积分与可视化，无 GPU 训练；
      “是否训练”相关规则在此不适用。所有先验的 D_σ(y) 通过对 y 网格的逐点
      1D 后验积分求得，并分别与共轭闭式解、Laplace 截断正态解析式、
      以及“同间距、更宽支撑域”的积分网格做三重交叉校验（见“自检”输出）。
"""

import numpy as np
import os
import sys
# ====== 静默模式配置 (matlab 静默模式) ======
SILENT_MODE = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
# 只静默“字形缺失”这一类可预期告警（无中文字体时）；
# 不再全局 ignore UserWarning，避免把其他真实问题也一并吞掉
warnings.filterwarnings('ignore', message='Glyph .* missing from font.*')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.3-3', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.3-3')
    os.makedirs(SAVE_DIR, exist_ok=True)
else:
    # 用脚本自身所在目录拼绝对路径：从任意工作目录启动都能找到字体模块
    _chinese_path = os.path.join(SCRIPT_DIR, '.chinese')
    SAVE_DIR = SCRIPT_DIR
if os.path.isdir(_chinese_path):
    sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    cn_font = setup_chinese_font(save_dir=_chinese_path)
    if cn_font:
        plt.rcParams['font.sans-serif'] = [cn_font] + plt.rcParams.get('font.sans-serif', [])
        plt.rcParams['font.family'] = 'sans-serif'
except ImportError:
    print(f"警告: chinese_font 模块未找到 (查找路径: {_chinese_path})，中文字体可能无法正常显示")

# ══════════════════════════════════════════════════════════
# 设置：噪声水平与先验参数
# ══════════════════════════════════════════════════════════
sigma = 0.5        # 观测噪声标准差
gamma = 1.0        # 高斯先验 N(0, gamma^2) 的尺度
laplace_b = 0.5    # Laplace 先验尺度 b:  p(x) ∝ exp(-|x|/b)
nu = 2.0           # Student-t 自由度 (小 → 重尾)

# Laplace(0,b) + 高斯似然的 MAP：min (y-x)^2/(2σ²) + |x|/b
#   对 x>0 求导置零 → x = y - σ²/b，即软阈值水平 λ_MAP = σ²/b (与正文一致)
laplace_threshold = sigma ** 2 / laplace_b    # = 0.5  (注意: 不是 b·σ² = 0.125)

# ══════════════════════════════════════════════════════════
# 1) 闭式去噪器 (用于与数值积分结果对照)
# ══════════════════════════════════════════════════════════
def denoiser_identity(y):
    return y


def denoiser_gaussian(y, gamma, sigma):
    # 高斯先验共轭 → MMSE = MAP = 线性缩放
    return (gamma ** 2 / (gamma ** 2 + sigma ** 2)) * y


def denoiser_laplace_map(y, threshold):
    # 这是 Laplace 先验的 MAP(后验众数)去噪器 = 软阈值；
    # Laplace 的 MMSE(后验均值)无初等闭式，实验统一用数值积分 (D_num['laplace'])，
    # 另有截断正态解析式 denoiser_laplace_mmse_closed 仅作交叉校验
    return np.sign(y) * np.maximum(np.abs(y) - threshold, 0.0)


def denoiser_laplace_mmse_closed(y, b, s):
    r"""Laplace(0,b) + N(0,s²) 似然的 MMSE 去噪器（标量），截断正态混合解析式。

    后验去掉与 x 无关的因子后是两段截断正态：
      x ≥ 0 支: 密度核 N(y-λ, s²) 截断到 [0,∞)，非归一化质量 A₊ ∝ e^{-y/b+λ/(2b)} Φ(z₊)
      x < 0 支: 密度核 N(y+λ, s²) 截断到 (-∞,0]，非归一化质量 A₋ ∝ e^{ y/b+λ/(2b)} Φ(-z₋)
    其中 λ=s²/b、z₊=(y-λ)/s、z₋=(y+λ)/s；两段的条件均值为
      m₊ = y-λ + s·φ(z₊)/Φ(z₊)，  m₋ = y+λ - s·φ(z₋)/Φ(-z₋)
    D_MMSE(y) = (A₊m₊ + A₋m₋)/(A₊+A₋)。
    只用标准正态 Φ(math.erf) 与 φ，不含数值积分；用于校验 posterior_mean。
    """
    import math
    lam = s ** 2 / b
    sqrt2, sqrt2pi = math.sqrt(2.0), math.sqrt(2.0 * math.pi)
    z_p, z_m = (y - lam) / s, (y + lam) / s
    Phi_p = 0.5 * (1.0 + math.erf(z_p / sqrt2))
    Phi_m = 0.5 * (1.0 + math.erf(-z_m / sqrt2))
    phi_p = math.exp(-0.5 * z_p ** 2) / sqrt2pi
    phi_m = math.exp(-0.5 * z_m ** 2) / sqrt2pi
    A_p = math.exp(-y / b + lam / (2 * b)) * Phi_p
    A_m = math.exp(y / b + lam / (2 * b)) * Phi_m
    m_p = (y - lam) + s * (phi_p / Phi_p) if Phi_p > 1e-300 else 0.0
    m_m = (y + lam) - s * (phi_m / Phi_m) if Phi_m > 1e-300 else 0.0
    return (A_p * m_p + A_m * m_m) / (A_p + A_m)


# ══════════════════════════════════════════════════════════
# 2) 通用数值积分求 D_σ(y) = E[x|y] = ∫ x p(x|y) dx / ∫ p(x|y) dx
#    对每个先验、每个 y 在 x 网格上算后验并求均值 (多轮计算 → tqdm)
# ══════════════════════════════════════════════════════════
y_grid = np.linspace(-3, 3, 400)
x_grid = np.linspace(-6, 6, 2000)
# 截断自检网格：必须与 x_grid 逐点重合、只在两侧增加外侧点 —— 否则网格相位
# 错位半个步长也会改变 Laplace 在 x=0 不可导尖点处的离散化误差，与“截断误差”
# 混淆。做法：在 x_grid 两端各接出整数个 dx（不能直接用对称 linspace，因为
# N=2000 为偶数时原网格不含 x=0，另起的对称网格会有 dx/2 相位差）。
_dx_grid = float(x_grid[1] - x_grid[0])
_x_wide_half_target = 12.0
_n_extend = int(round((_x_wide_half_target - abs(float(x_grid[0]))) / _dx_grid))
x_grid_wide = np.concatenate([
    float(x_grid[0]) - np.arange(_n_extend, 0, -1) * _dx_grid,   # 左侧外接点(升序)
    x_grid,                                                      # 原网格逐点保留
    float(x_grid[-1]) + np.arange(1, _n_extend + 1) * _dx_grid,  # 右侧外接点
])
_x_wide_lo, _x_wide_hi = float(x_grid_wide[0]), float(x_grid_wide[-1])


def log_prior_identity(x):
    return np.zeros_like(x)


def log_prior_gaussian(x, gamma):
    return -x ** 2 / (2 * gamma ** 2)


def log_prior_laplace(x, b):
    return -np.abs(x) / b


def log_prior_student_t(x, nu):
    return -((nu + 1) / 2) * np.log1p(x ** 2 / nu)


priors = {
    'identity':   log_prior_identity,
    'gaussian':   lambda x: log_prior_gaussian(x, gamma),
    'laplace':    lambda x: log_prior_laplace(x, laplace_b),
    'student_t':  lambda x: log_prior_student_t(x, nu),
}


def posterior_mean(lp, y, xg):
    """给定 log 先验、观测 y、x 网格 → 后验均值 E[x|y] (未归一化权重直接求比)"""
    log_post = lp(xg) - (y - xg) ** 2 / (2 * sigma ** 2)
    log_post = log_post - log_post.max()
    # 未归一化权重 w_i ∝ p(x_i|y); 后验均值用 PMF 形式 E[x] = Σ x_i w_i / Σ w_i
    # (等价于先归一化为密度再积分，此写法与 dx 无关、对截断网格更稳健)
    w = np.exp(log_post)
    return np.sum(xg * w) / np.sum(w)


D_num = {k: np.zeros_like(y_grid) for k in priors}
for j, y in enumerate(tqdm(y_grid, desc='计算去噪器曲线 D(y)', ncols=80)):
    for k, lp in priors.items():
        D_num[k][j] = posterior_mean(lp, y, x_grid)

# 闭式曲线 (与数值结果对照)
D_closed = {
    'identity': denoiser_identity(y_grid),
    'gaussian': denoiser_gaussian(y_grid, gamma, sigma),
    'laplace_map': denoiser_laplace_map(y_grid, laplace_threshold),
}

# ══════════════════════════════════════════════════════════
# 3) 自检：三重交叉校验
#    (a) 共轭先验的闭式解 (恒等/高斯)
#    (b) Laplace MMSE 的截断正态解析式（只用 Φ/φ，不做积分）——
#        数值与它之差主要是 x=0 尖点处的网格离散化误差
#    (c) 同间距 dx、支撑域 [-6,6]→[-12,12] —— 之差才是纯粹的截断误差
# ══════════════════════════════════════════════════════════
chk_closed = float(max(np.max(np.abs(D_num['identity'] - D_closed['identity'])),
                       np.max(np.abs(D_num['gaussian'] - D_closed['gaussian']))))
# (b) Laplace MMSE 数值积分 vs 截断正态解析式
D_laplace_analytic = np.array(
    [denoiser_laplace_mmse_closed(float(y), laplace_b, sigma) for y in y_grid])
chk_laplace_analytic = float(np.max(np.abs(D_num['laplace'] - D_laplace_analytic)))
# (c) 纯截断误差（宽网格与原网格间距相同、网格点重合，仅多出外侧支撑）
chk_trunc_laplace = float(np.max(np.abs(
    [posterior_mean(priors['laplace'], y, x_grid_wide)
     - posterior_mean(priors['laplace'], y, x_grid) for y in y_grid])))
chk_trunc_student_t = float(np.max(np.abs(
    [posterior_mean(priors['student_t'], y, x_grid_wide)
     - posterior_mean(priors['student_t'], y, x_grid) for y in y_grid])))
chk_trunc = max(chk_trunc_laplace, chk_trunc_student_t)
# Laplace 的 MMSE 与 MAP(软阈值) 分歧：最大偏差、出现位置、以及大 |y| 后“重合”的起点
_diff_lap = D_num['laplace'] - D_closed['laplace_map']
chk_laplace_mmse_vs_map = float(np.max(np.abs(_diff_lap)))
_ys = y_grid[y_grid > 0]
_tail = np.abs(_diff_lap[y_grid > 0])
chk_laplace_mmse_vs_map_at = float(_ys[int(np.argmax(_tail))])
ALIGN_TOL = 0.01          # 认为“基线重合”的判据: 偏差 ≤ 0.01 ≈ σ/50
_k = len(_tail)
while _k > 0 and _tail[_k - 1] <= ALIGN_TOL:
    _k -= 1
chk_laplace_align_from_y = float(_ys[_k]) if _k < len(_tail) else float('nan')

# ══════════════════════════════════════════════════════════
# 绘图 (文字说明一律用 print，不写入图片)
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 7))

# 参考：无先验(对角直线)
ax.plot(y_grid, D_closed['identity'], 'k-.', linewidth=1.5,
        label='无先验 (恒等 $D(y)=y$)')

ax.plot(y_grid, D_closed['gaussian'], 'b-', linewidth=2,
        label=f'高斯先验 MMSE $\\mathcal{{N}}(0,\\gamma^2)$, $\\gamma={gamma}$')
ax.plot(y_grid, D_num['laplace'], 'g-', linewidth=2,
        label=f'Laplace 先验 MMSE (数值积分), $b={laplace_b}$')
ax.plot(y_grid, D_closed['laplace_map'], 'g--', linewidth=1.6,
        label=f'Laplace 先验 MAP = 软阈值, 阈值 $\\sigma^2/b={laplace_threshold}$')
ax.plot(y_grid, D_num['student_t'], 'r-', linewidth=2,
        label=f'Student-t 先验 MMSE (重尾, 数值积分), $\\nu={nu}$')

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel('观测值 $y$', fontsize=13)
ax.set_ylabel('去噪输出 $D_\\sigma(y)$', fontsize=13)
ax.set_title('不同先验下的贝叶斯去噪器 $D_\\sigma(y)$ 随 $y$ 变化曲线',
             fontsize=14, fontweight='bold', pad=14)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(SAVE_DIR, '图2-5_不同先验下的去噪器曲线.png'), dpi=150, bbox_inches='tight')
plt.close()

if not SILENT_MODE:
    plt.show()

# ══════════════════════════════════════════════════════════
# 文字说明 (用 print，不放入图片)
# ══════════════════════════════════════════════════════════
print("=" * 70)
print("实验2.3-3 不同先验下的去噪器函数曲线 D(y) vs y")
print("=" * 70)
print(f"\n实验设定: 噪声水平 σ={sigma}")
print(f"  高斯先验尺度 γ={gamma}  → 线性缩放系数 γ²/(γ²+σ²)={gamma**2/(gamma**2+sigma**2):.3f}")
print(f"  Laplace 先验尺度 b={laplace_b} (p∝e^(-|x|/b))  → MAP 软阈值 σ²/b={laplace_threshold:.3f}")
print(f"  Student-t 自由度 ν={nu}  → 重尾先验 (方差无穷)")

print("\n[自检] 三重交叉校验：闭式 / 截断正态解析式 / 同间距扩支撑域")
print(f"  (a) 恒等与高斯: |数值 - 闭式|max = {chk_closed:.1e}   (共轭情形达机器精度 → 积分实现正确)")
print(f"  (b) Laplace MMSE 数值 vs 截断正态解析式(Φ/φ，无积分): |ΔD|max = {chk_laplace_analytic:.1e}")
print(f"      残差来自先验在 x=0 不可导尖点的网格离散化（dx={_dx_grid:.4f}），非截断误差")
print(f"  (c) 同间距 dx、逐点外接把支撑域 [-6,6] 扩大到 [{_x_wide_lo:.1f},{_x_wide_hi:.1f}] 后的纯截断误差:")
print(f"      Laplace |ΔD|max = {chk_trunc_laplace:.1e},  Student-t |ΔD|max = {chk_trunc_student_t:.1e}  → [-6,6] 截断可忽略")
print(f"  Laplace MMSE vs MAP: 最大分歧 {chk_laplace_mmse_vs_map:.3f} 出现在 y≈{chk_laplace_mmse_vs_map_at:.2f} (死区边缘),")
print(f"    而 |y| ≥ {chk_laplace_align_from_y:.2f} 之后两者差一直 ≤ {ALIGN_TOL} (≈σ/{sigma/ALIGN_TOL:.0f}) → 软阈值近似只在大幅度处可靠，在死区边缘最差")

print("\n[曲线形状对照] (取若干 y 的 D_σ(y) 数值结果)")
sample_y = [-3, -1, -0.2, 0.2, 1, 3]
print(f"  {'y':>6} | {'高斯MMSE':>9} | {'Lap MMSE':>9} | {'Lap MAP':>8} | {'Student-t':>10}")
for yy in sample_y:
    iy = np.argmin(np.abs(y_grid - yy))
    print(f"  {yy:>6.1f} | {D_closed['gaussian'][iy]:>9.3f} | {D_num['laplace'][iy]:>9.3f} | "
          f"{D_closed['laplace_map'][iy]:>8.3f} | {D_num['student_t'][iy]:>10.3f}")

# 量化近零处收缩：用网格点(最接近 0.2)，保证文字与数值输出完全一致
_iy02 = np.argmin(np.abs(y_grid - 0.2))
_y02 = y_grid[_iy02]
_Dg02 = D_closed['gaussian'][_iy02]
_Dlap02 = D_num['laplace'][_iy02]
_Dlapmap02 = D_closed['laplace_map'][_iy02]
_Dst02 = D_num['student_t'][_iy02]
_g_supp = (1 - _Dg02 / _y02) * 100
_lap_supp = (1 - _Dlap02 / _y02) * 100
_lapmap_supp = (1 - _Dlapmap02 / _y02) * 100
_st_supp = (1 - _Dst02 / _y02) * 100

# 近零处的收缩强弱：取 y→0 极限斜率 lim D(y)/y = Var(x|y=0)/σ² (数值积分直接算)
_y0 = 1e-3
slope_gauss = float(gamma ** 2 / (gamma ** 2 + sigma ** 2))
slope_laplace = float(posterior_mean(priors['laplace'], _y0, x_grid) / _y0)
slope_t = float(posterior_mean(priors['student_t'], _y0, x_grid) / _y0)

# 收缩强弱的交叉点：|y-D_t| 首次小于 |y-D_gauss| 的 y
_gap_t = y_grid - D_num['student_t']
_gap_g = y_grid - D_closed['gaussian']
_pos = np.where((y_grid > 0) & (_gap_t < _gap_g))[0]
_y_cross = float(y_grid[_pos[0]]) if len(_pos) else float('nan')

print("\n[核心观察] (以下数值直接引用上方表格，与曲线一致)")
print("  - 无先验: D(y)=y，对角直线，完全不做收缩 (没有利用任何先验)。")
print(f"  - 高斯先验: 处处线性、斜率 {gamma**2/(gamma**2+sigma**2):.2f}<1，无阈值；相对收缩恒为 {_g_supp:.0f}%"
      f"（绝对收缩量 σ²/(γ²+σ²)={sigma**2/(gamma**2+sigma**2):.2f}·|y| 随 |y| 线性增长），故大 |y| 处收缩最凶。")
print(f"  - Laplace 先验 MMSE: 近零处收缩最强 (y≈{_y02:.2f} 压至 {_Dlap02:.3f}，约 {_lap_supp:.0f}%)，")
print("    但因后验尾部拉回，均值不为 0 —— 没有真正的“死区”，只是斜率明显变小。")
print(f"  - Laplace 先验 MAP: 才有死区 |y|≤σ²/b={laplace_threshold}，同一 y 输出 {_Dlapmap02:.3f} (压制 {_lapmap_supp:.0f}%)；")
print(f"    两条绿色曲线在 y=0 附近最大相差 {chk_laplace_mmse_vs_map:.3f} —— 这就是“同一先验、两种估计器形状迥异”。")
print(f"  - Student-t 先验(重尾) MMSE: 近零处压制约 {_st_supp:.0f}%，比 Laplace MMSE ({_lap_supp:.0f}%) 温和，")
print(f"    但收缩仍略强于高斯的 {_g_supp:.0f}%（近零斜率 y→0 时的 D/y: 高斯 {slope_gauss:.2f} > t {slope_t:.2f} > Laplace MMSE {slope_laplace:.2f} > Laplace MAP 0）；")
print(f"    其绝对收缩量有界且随 |y| 衰减 (~σ²(ν+1)/y)，|y| ≳ {_y_cross:.2f} 后在所有“会收缩的先验”中最弱（恒等先验不收缩，自然是严格的恒等）、最接近恒等。")
print("    所以“重尾=收缩最温和”只在大 |y| 成立，近零处最温和的是高斯。")
print("  → 注意: 人们常把重尾先验与'硬阈值/死区'联系在一起，但那是 MAP(后验众数)")
print("    估计器的行为；MMSE 去噪器(本实验)对重尾先验是平滑弱收缩。")
print("    这恰是 2.3 节'MMSE vs MAP 分歧'的又一例证 —— 同一先验，两种估计器形状迥异。")
print("  → 先验 p(x) 的形状，直接决定了去噪器 D_σ(y) 的形状，")
print("    这正是'先验的质量'在去噪器层面的具体体现。")

# ===== 保存数值结果 (JSON) =====
import json


def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    return obj


_idx = [int(np.argmin(np.abs(y_grid - v))) for v in sample_y]
results_summary = {
    'experiment': '2.3-3',
    'title': '不同先验下的去噪器函数曲线 D(y) vs y',
    'noise_sigma': float(sigma),
    'priors': {
        'gaussian': {'gamma': float(gamma),
                     'scale': float(gamma ** 2 / (gamma ** 2 + sigma ** 2))},
        'laplace': {'b': float(laplace_b),
                    'map_threshold_sigma2_over_b': float(laplace_threshold)},
        'student_t': {'nu': float(nu)},
    },
    'sample_curves': {
        'y': [float(v) for v in sample_y],
        'grid_y': [float(y_grid[i]) for i in _idx],
        'gaussian_mmse': [float(D_closed['gaussian'][i]) for i in _idx],
        'laplace_mmse': [float(D_num['laplace'][i]) for i in _idx],
        'laplace_map_soft': [float(D_closed['laplace_map'][i]) for i in _idx],
        'student_t_mmse': [float(D_num['student_t'][i]) for i in _idx],
    },
    'self_checks': {
        # (a) 共轭闭式
        'numeric_vs_closed_max_abs': chk_closed,
        # (b) 数值积分 vs Laplace 截断正态解析式（残差≈x=0 尖点离散化）
        'laplace_numeric_vs_truncnorm_analytic_max_abs': chk_laplace_analytic,
        # (c) 同间距 dx、扩大支撑域得到的纯截断误差
        'grid_truncation_laplace_max_abs': chk_trunc_laplace,
        'grid_truncation_student_t_max_abs': chk_trunc_student_t,
        'grid_truncation_narrow_support': [-6.0, 6.0],
        'grid_truncation_wide_support': [_x_wide_lo, _x_wide_hi],
        'grid_dx': _dx_grid,
        # MMSE vs MAP 分歧
        'laplace_mmse_vs_map_max_abs': chk_laplace_mmse_vs_map,
        'laplace_mmse_vs_map_max_at_y': chk_laplace_mmse_vs_map_at,
        'laplace_mmse_vs_map_align_from_y': chk_laplace_align_from_y,
        'laplace_align_tol': float(ALIGN_TOL),
    },
    'shrinkage': {
        'suppress_pct_at_y0p2': {
            'gaussian': float(_g_supp),
            'laplace_mmse': float(_lap_supp),
            'laplace_map': float(_lapmap_supp),
            'student_t': float(_st_supp),
        },
        'student_t_shrink_crossover_y': _y_cross,
        'near_zero_slope_D_over_y': {'gaussian': slope_gauss,
                                     'student_t': slope_t,
                                     'laplace_mmse': slope_laplace,
                                     'laplace_map': 0.0},
    },
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"\n数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
print(f"图片已保存至: {SAVE_DIR}")
print("=" * 70)
