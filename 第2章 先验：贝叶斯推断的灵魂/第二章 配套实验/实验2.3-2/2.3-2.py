"""
实验2.3-2 偏斜/多峰后验下 MMSE vs MAP 分歧可视化
对应章节：2.3 先验的质量：MMSE vs MAP 估计器
知识点：
  - 非对称(偏斜)后验下，MAP(众数)偏向峰值/零点，MMSE(均值)被尾部质量拉远；
  - 多峰(双峰)后验下，MAP 选最高峰，MMSE 取两峰加权平均，落在两峰之间。

素材来源：
  - temp/图2-3_偏斜分布MMSE_vs_MAP.py (原型)
  - temp/图2-4_多峰后验MMSE_vs_MAP.py (原型)

说明：本实验仅为 1D 后验的数值积分与可视化，无 GPU 训练；
      “是否训练”相关规则在此不适用。
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

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.3-2', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.3-2')
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
# 子图(a)：偏斜后验 (Laplace 先验 + 高斯似然)
# ══════════════════════════════════════════════════════════
# 先验 p(x) ∝ exp(-|x|/b)  (Laplace, b=0.8)
# 似然 p(y|x) ∝ exp(-(y-x)^2/(2σ^2))  (σ=1.0)
# 后验 log p(x|y) = -|x|/b - (y-x)^2/(2σ^2)，峰值偏向零点、尾部向远离零点方向延伸
b = 0.8
sigma = 1.0
y_obs = 2.0


def log_posterior_skew(x, y, b, sigma):
    """对数后验：log p(x|y) = -|x|/b - (y-x)^2/(2σ^2) + const"""
    return -np.abs(x) / b - (y - x) ** 2 / (2 * sigma ** 2)


x_range_a = np.linspace(-2, 3, 1000)
log_p_a = log_posterior_skew(x_range_a, y_obs, b, sigma)
p_a = np.exp(log_p_a - log_p_a.max())
p_a = p_a / (p_a.sum() * (x_range_a[1] - x_range_a[0]))

map_idx_a = np.argmax(p_a)
x_map_a = x_range_a[map_idx_a]
x_mmse_a = _trapz(x_range_a * p_a, x_range_a)


# ══════════════════════════════════════════════════════════
# 子图(b)：双峰后验 (二值图像场景)
# ══════════════════════════════════════════════════════════
# 像素值应为 0 或 1，噪声使其后验呈双峰混合
# p(x|y) = 0.6 * N(x; μ1=0, σ=0.15) + 0.4 * N(x; μ2=1, σ=0.15)
mu1, sigma1, w1 = 0.0, 0.15, 0.6
mu2, sigma2, w2 = 1.0, 0.15, 0.4


def gaussian_pdf(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


x_range_b = np.linspace(-0.5, 1.5, 1000)
p_b = w1 * gaussian_pdf(x_range_b, mu1, sigma1) + w2 * gaussian_pdf(x_range_b, mu2, sigma2)

map_idx_b = np.argmax(p_b)
x_map_b = x_range_b[map_idx_b]
x_mmse_b = _trapz(x_range_b * p_b, x_range_b)
x_mmse_b_theory = w1 * mu1 + w2 * mu2


# ══════════════════════════════════════════════════════════
# 绘图 (文字说明一律用 print，不写入图片)
# ══════════════════════════════════════════════════════════
# ---- 图2-3：偏斜后验 ----
fig, ax = plt.subplots(figsize=(11, 6))
ax.fill_between(x_range_a, p_a, alpha=0.3, color='steelblue')
ax.plot(x_range_a, p_a, 'b-', linewidth=2, label='后验分布 $p(x|y)$')

ax.axvline(x=x_map_a, color='red', linestyle='--', linewidth=2, alpha=0.8)
ax.plot(x_map_a, p_a[map_idx_a], 'ro', markersize=12, zorder=5)
ax.annotate(f'MAP = {x_map_a:.2f}\n(峰值，偏向零点)',
            xy=(x_map_a, p_a[map_idx_a]), xytext=(-1.5, p_a[map_idx_a] * 0.7),
            fontsize=11, ha='center', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

ax.axvline(x=x_mmse_a, color='green', linestyle='--', linewidth=2, alpha=0.8)
mmse_p_idx_a = np.argmin(np.abs(x_range_a - x_mmse_a))
ax.plot(x_mmse_a, p_a[mmse_p_idx_a], 'go', markersize=12, zorder=5)
ax.annotate(f'MMSE = {x_mmse_a:.2f}\n(均值，被尾部拉远)',
            xy=(x_mmse_a, p_a[mmse_p_idx_a]), xytext=(2.5, p_a[mmse_p_idx_a] * 0.5),
            fontsize=11, ha='center', color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

tail_mask = x_range_a > 1.5
ax.fill_between(x_range_a[tail_mask], p_a[tail_mask], alpha=0.2, color='orange')
ax.annotate('尾部质量\n拉动 MMSE 向右', xy=(2.0, 0.02), xytext=(2.3, 0.08),
            fontsize=10, ha='center', color='darkorange',
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))

ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.text(0.05, ax.get_ylim()[1] * 0.9, '零点', fontsize=9, color='gray')
ax.axvline(x=y_obs, color='purple', linestyle=':', linewidth=1.5, alpha=0.6)
ax.text(y_obs + 0.05, ax.get_ylim()[1] * 0.8, f'观测 $y={y_obs}$', fontsize=9, color='purple')

ax.set_xlabel('$x$', fontsize=13)
ax.set_ylabel('后验概率密度 $p(x|y)$', fontsize=13)
ax.set_title('偏斜后验下 MAP vs MMSE 分歧\nLaplace 先验 + 高斯似然 $\\rightarrow$ 偏斜后验',
             fontsize=14, fontweight='bold', linespacing=1.8, pad=20)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(SAVE_DIR, '图2-3_偏斜分布MMSE_vs_MAP.png'), dpi=150, bbox_inches='tight')
plt.close()

# ---- 图2-4：双峰后验 ----
fig, ax = plt.subplots(figsize=(11, 6))
ax.fill_between(x_range_b, p_b, alpha=0.3, color='steelblue')
ax.plot(x_range_b, p_b, 'b-', linewidth=2, label='后验分布 $p(x|y)$')

p1 = w1 * gaussian_pdf(x_range_b, mu1, sigma1)
p2 = w2 * gaussian_pdf(x_range_b, mu2, sigma2)
ax.plot(x_range_b, p1, 'b--', linewidth=1, alpha=0.4, label=f'分量1: $w_1={w1}$, $\\mu_1={mu1}$')
ax.plot(x_range_b, p2, 'b--', linewidth=1, alpha=0.4, label=f'分量2: $w_2={w2}$, $\\mu_2={mu2}$')

ax.axvline(x=x_map_b, color='red', linestyle='--', linewidth=2, alpha=0.8)
ax.plot(x_map_b, p_b[map_idx_b], 'ro', markersize=12, zorder=5)
ax.annotate(f'MAP = {x_map_b:.2f}\n(选最高的峰)',
            xy=(x_map_b, p_b[map_idx_b]), xytext=(x_map_b - 0.35, p_b[map_idx_b] + 0.15),
            fontsize=11, ha='center', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

ax.axvline(x=x_mmse_b, color='green', linestyle='--', linewidth=2, alpha=0.8)
mmse_p_idx_b = np.argmin(np.abs(x_range_b - x_mmse_b))
ax.plot(x_mmse_b, p_b[mmse_p_idx_b], 'go', markersize=12, zorder=5)
ax.annotate(f'MMSE = {x_mmse_b:.2f}\n(加权平均)',
            xy=(x_mmse_b, p_b[mmse_p_idx_b]), xytext=(x_mmse_b + 0.35, p_b[mmse_p_idx_b] + 0.3),
            fontsize=11, ha='center', color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

ax.annotate('', xy=(x_mmse_b, 0.05), xytext=(mu1, 0.05),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
ax.annotate('', xy=(mu2, 0.05), xytext=(x_mmse_b, 0.05),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
ax.text(x_mmse_b, 0.08, 'MMSE在两峰之间', fontsize=9, ha='center', color='gray')

ax.text(mu1, -0.15, '像素值=0\n(背景)', fontsize=9, ha='center', color='blue')
ax.text(mu2, -0.15, '像素值=1\n(前景)', fontsize=9, ha='center', color='blue')

ax.set_xlabel('$x$ (像素值)', fontsize=13)
ax.set_ylabel('后验概率密度 $p(x|y)$', fontsize=13)
ax.set_title('多峰后验下 MAP vs MMSE 分歧\n二值图像场景：像素应为 0 或 1，但噪声使后验呈双峰',
             fontsize=15, fontweight='bold', pad=12)
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(bottom=-0.2)
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(SAVE_DIR, '图2-4_多峰后验MMSE_vs_MAP.png'), dpi=150, bbox_inches='tight')
plt.close()

if not SILENT_MODE:
    plt.show()

# ══════════════════════════════════════════════════════════
# 文字说明 (用 print，不放入图片)
# ══════════════════════════════════════════════════════════
print("=" * 70)
print("实验2.3-2 偏斜/多峰后验下 MMSE vs MAP 分歧可视化")
print("=" * 70)
print(f"\n[子图(a) 偏斜后验] Laplace 先验(b={b}) + 高斯似然(σ={sigma})，观测 y={y_obs}")
print(f"  MAP  = {x_map_a:.4f}  (后验峰值，偏向零点)")
print(f"  MMSE = {x_mmse_a:.4f}  (后验均值，被右侧尾部质量拉远)")
print(f"  分歧: MMSE - MAP = {x_mmse_a - x_map_a:.4f}")
print("  核心观察: MAP 选后验峰值(最可能值)，偏零点；MMSE 取后验均值，")
print("            被尾部质量拉离峰值。非对称分布下两者产生明显分歧。")

print(f"\n[子图(b) 双峰后验] 二值图像双峰混合: {w1}·N({mu1},{sigma1}^2) + {w2}·N({mu2},{sigma2}^2)")
print(f"  MAP  = {x_map_b:.4f}  (最高的峰，此处为 0)")
print(f"  MMSE = {x_mmse_b:.4f}  (数值积分)")
print(f"  MMSE(理论) = {x_mmse_b_theory:.4f}  (两峰加权平均)")
print(f"  分歧: MMSE - MAP = {x_mmse_b - x_map_b:.4f}")
print("  核心观察: MAP 选概率最高的峰(像素判为背景 0)；MMSE 取两峰加权")
print("            平均(0.4)，落在两峰之间、给出模糊值。两者代表不同决策哲学。")

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
    'experiment': '2.3-2',
    'title': '偏斜/多峰后验下 MMSE vs MAP 分歧可视化',
    'skewed_posterior': {
        'prior': 'Laplace', 'b': b, 'likelihood_sigma': sigma, 'y_obs': y_obs,
        'MAP': float(x_map_a), 'MMSE': float(x_mmse_a),
        'divergence_MMSE_minus_MAP': float(x_mmse_a - x_map_a),
    },
    'bimodal_posterior': {
        'prior': 'binary_mixture',
        'w1': w1, 'mu1': mu1, 'sigma1': sigma1,
        'w2': w2, 'mu2': mu2, 'sigma2': sigma2,
        'MAP': float(x_map_b), 'MMSE': float(x_mmse_b),
        'MMSE_theory': float(x_mmse_b_theory),
        'divergence_MMSE_minus_MAP': float(x_mmse_b - x_map_b),
    },
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"\n数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
print(f"图片已保存至: {SAVE_DIR}")
print("=" * 70)
