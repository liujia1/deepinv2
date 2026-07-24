"""
实验2.3-3 不同先验下的去噪器函数曲线 D(y) vs y
对应章节：2.3 先验的质量：MMSE vs MAP 估计器 - “正向映射：给定先验 p(x) → 去噪器 D_σ(y)”
知识点：
  - 去噪器 D_σ(y) = E[x|y] 由先验 p(x) 与噪声水平 σ 共同决定；
  - 不同先验给出形状迥异的 D_σ(y)：
      无先验   → 恒等 D(y)=y (对角直线)
      高斯先验 → 线性缩放 D(y)=γ²/(γ²+σ²)·y (远离原点时仍接近 y)
      Laplace  → 软阈值 D(y)=sign(y)·max(|y|-bσ²,0) (近零处收缩为 0)
      Student-t→ 重尾先验，收缩更强、近硬阈值的“死区”去噪器
  - 这正说明“先验的质量”体现在去噪器曲线形状上 (对应正文图2-5)。

说明：本实验为 1D 贝叶斯去噪器的数值积分与可视化，无 GPU 训练；
      “是否训练”相关规则在此不适用。Student-t 先验的 D_σ(y) 通过对 y 网格
      的逐点 1D 后验积分求得，用 tqdm 进度条显示多轮计算过程。
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
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
from tqdm import tqdm

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.3-3', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.3-3')
    os.makedirs(SAVE_DIR, exist_ok=True)
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    cn_font = setup_chinese_font(save_dir=_chinese_path)
    if cn_font:
        plt.rcParams['font.sans-serif'] = [cn_font] + plt.rcParams.get('font.sans-serif', [])
        plt.rcParams['font.family'] = 'sans-serif'
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")

# 兼容不同 numpy 版本的梯形积分
_trapz = getattr(np, 'trapezoid', np.trapz)

# ══════════════════════════════════════════════════════════
# 设置：噪声水平与先验参数
# ══════════════════════════════════════════════════════════
sigma = 0.5        # 观测噪声标准差
gamma = 1.0        # 高斯先验 N(0, gamma^2) 的尺度
laplace_b = 0.5    # Laplace 先验尺度 b
nu = 2.0           # Student-t 自由度 (小 → 重尾)

# ══════════════════════════════════════════════════════════
# 1) 闭式去噪器 (用于与数值积分结果对照)
# ══════════════════════════════════════════════════════════
def denoiser_identity(y):
    return y


def denoiser_gaussian(y, gamma, sigma):
    return (gamma ** 2 / (gamma ** 2 + sigma ** 2)) * y


def denoiser_laplace(y, b, sigma):
    # MMSE 软阈值: sign(y)·max(|y| - b·σ², 0)
    return np.sign(y) * np.maximum(np.abs(y) - b * sigma ** 2, 0.0)


# ══════════════════════════════════════════════════════════
# 2) 通用数值积分求 D_σ(y) = E[x|y] = ∫ x p(x|y) dx / ∫ p(x|y) dx
#    对每个先验、每个 y 在 x 网格上算后验并求均值 (多轮计算 → tqdm)
# ══════════════════════════════════════════════════════════
y_grid = np.linspace(-3, 3, 400)
x_grid = np.linspace(-6, 6, 2000)
dx = x_grid[1] - x_grid[0]


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

D_num = {k: np.zeros_like(y_grid) for k in priors}
for j, y in enumerate(tqdm(y_grid, desc='计算去噪器曲线 D(y)', ncols=80)):
    for k, lp in priors.items():
        log_post = lp(x_grid) - (y - x_grid) ** 2 / (2 * sigma ** 2)
        log_post -= log_post.max()
        w = np.exp(log_post)
        w /= w.sum()
        D_num[k][j] = _trapz(x_grid * w, x_grid)

# 闭式曲线 (与数值结果一致，作为清晰对照)
D_closed = {
    'identity': denoiser_identity(y_grid),
    'gaussian': denoiser_gaussian(y_grid, gamma, sigma),
    'laplace':  denoiser_laplace(y_grid, laplace_b, sigma),
}

# ══════════════════════════════════════════════════════════
# 绘图 (文字说明一律用 print，不写入图片)
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 7))

# 参考：无先验(对角直线)
ax.plot(y_grid, D_closed['identity'], 'k-.', linewidth=1.5,
        label='无先验 (恒等 $D(y)=y$)')

ax.plot(y_grid, D_closed['gaussian'], 'b-', linewidth=2,
        label=f'高斯先验 $\\mathcal{{N}}(0,\\gamma^2)$, $\\gamma={gamma}$')
ax.plot(y_grid, D_closed['laplace'], 'g-', linewidth=2,
        label=f'Laplace 先验, $b={laplace_b}$')
ax.plot(y_grid, D_num['student_t'], 'r-', linewidth=2,
        label=f'Student-t 先验 (重尾), $\\nu={nu}$')

ax.plot([-3, 3], [-3, 3], 'gray', linewidth=0.8, alpha=0.5)
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
print(f"  Laplace 先验尺度 b={laplace_b}  → 软阈值阈值 b·σ²={laplace_b*sigma**2:.3f}")
print(f"  Student-t 自由度 ν={nu}  → 重尾先验，收缩更强")

print("\n[曲线形状对照] (取若干 y 的 D_σ(y) 数值结果)")
sample_y = [-3, -1, -0.2, 0.2, 1, 3]
print(f"  {'y':>6} | {'高斯':>8} | {'Laplace':>8} | {'Student-t':>10}")
for yy in sample_y:
    iy = np.argmin(np.abs(y_grid - yy))
    print(f"  {yy:>6.1f} | {D_closed['gaussian'][iy]:>8.3f} | "
          f"{D_closed['laplace'][iy]:>8.3f} | {D_num['student_t'][iy]:>10.3f}")

print("\n[核心观察]")
print("  - 无先验: D(y)=y，对角直线，完全不做收缩 (没有利用任何先验)。")
print("  - 高斯先验: 处处线性、斜率<1，远离原点仍接近 y —— 只做轻微均匀收缩。")
print("  - Laplace 先验: 软阈值，近零处被压成 0，产生稀疏(边缘保持)效果。")
print("  - Student-t 先验: 重尾，收缩更强、近硬阈值的'死区'去噪器，")
print("    对小信号抑制更狠 —— 体现'重尾先验更偏好稀疏/近零解'。")
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


results_summary = {
    'experiment': '2.3-3',
    'title': '不同先验下的去噪器函数曲线 D(y) vs y',
    'noise_sigma': float(sigma),
    'priors': {
        'gaussian': {'gamma': float(gamma),
                     'scale': float(gamma ** 2 / (gamma ** 2 + sigma ** 2))},
        'laplace': {'b': float(laplace_b),
                    'threshold': float(laplace_b * sigma ** 2)},
        'student_t': {'nu': float(nu)},
    },
    'sample_curves': {
        'y': [float(v) for v in sample_y],
        'gaussian': [float(D_closed['gaussian'][np.argmin(np.abs(y_grid - v))]) for v in sample_y],
        'laplace': [float(D_closed['laplace'][np.argmin(np.abs(y_grid - v))]) for v in sample_y],
        'student_t': [float(D_num['student_t'][np.argmin(np.abs(y_grid - v))]) for v in sample_y],
    },
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"\n数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
print(f"图片已保存至: {SAVE_DIR}")
print("=" * 70)
