import numpy as np, matplotlib, os, sys
matplotlib.use('Agg')
sys.path.insert(0, '.chinese')
from chinese_font import setup_chinese_font
cn = setup_chinese_font(save_dir='.chinese')
print('cn font =', cn)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = os.path.abspath('.')

def pow10_fmt(v, _pos):
    e = int(round(np.log10(v)))
    return f'$10^{{{e}}}$' if e != 1 else '$10$'

# 变体 A：仅 mathtext.fontset='cm'
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plt.rcParams['mathtext.fontset'] = 'cm'
for ax, mode in zip(axes, ['A: fontset=cm 默认刻度', 'B: fontset=cm + FuncFormatter', 'C: 注释文本变体']):
    x = np.logspace(0, 12, 10); y = np.logspace(-8, 3, 10)
    ax.loglog(x, y, 'o-')
    ax.set_title(mode, fontsize=10)
    ax.grid(True, alpha=.3)
    if mode.startswith('B'):
        ax.yaxis.set_major_formatter(FuncFormatter(pow10_fmt))
        ax.xaxis.set_major_formatter(FuncFormatter(pow10_fmt))
    if mode.startswith('C'):
        ax.text(1e8, 1e-4, '开头悬念：$\\kappa=10^{10}$\n($10^{-8}$ 误差 → 完全失真)', fontsize=9, color='purple',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lavender', alpha=0.6))
        ax.set_ylabel(r'重建相对误差 $\|x-\hat{x}\|/\|x\|$')
        ax.set_title(r'不同条件数的奇异值分布\n（小 $\sigma_i$ → $1/\sigma_i$ 爆炸）')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fix_test_A.png'), dpi=130, bbox_inches='tight')
print('saved', os.path.join(OUT, 'fix_test_A.png'))
