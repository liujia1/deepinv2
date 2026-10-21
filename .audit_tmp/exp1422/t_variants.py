"""复现步骤5 图的 fig 参数，二分定位 ylabel 异常触发条件。"""
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=False)
sys.path.insert(0, '.chinese')
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir='.chinese')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = r'h:\deepinv\deepinv2\.audit_tmp\exp1422'


def variant(name, ylabel, rect, use_tight, figsize=(12, 4.2)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax = axes[0]
    xs = np.random.randn(4000) * 3
    ax.hist(xs, bins=60, density=True, alpha=0.4)
    ax.set_xlabel(r'$x_2$')
    ax.set_ylabel(ylabel)
    ax.set_title('(a) y=0 时 $x_2$ 的边缘：解析为双峰')
    ax.grid(alpha=0.3)
    ax2 = axes[1]
    ax2.bar(['解析', 'MAF', 'x1 透传'], [0.66, 0.54, 0.33])
    ax2.set_ylabel('双峰系数 BC')
    if use_tight:
        plt.tight_layout(rect=rect)
    fig.suptitle('实验14.2-2 步骤5：后验形状诊断（y=0）', fontsize=12)
    p = OUT + '\\' + name + '.png'
    plt.savefig(p, dpi=150)
    plt.close()
    print('saved', p)


variant('v1_rect_density', '密度', [0, 0, 1, 0.94], True)
variant('v2_rect_densityP', '密度 p', [0, 0, 1, 0.94], True)
variant('v3_notight', '密度', None, False)
variant('v4_rect_en', 'density', [0, 0, 1, 0.94], True)
