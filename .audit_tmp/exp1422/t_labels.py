"""对照：多个 ylabel 文本，定位哪些会触发字形重叠。"""
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=False)
sys.path.insert(0, '.chinese')
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir='.chinese')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

print('font.family   =', mpl.rcParams['font.family'])
print('font.sans-serif =', mpl.rcParams['font.sans-serif'])
print('text.usetex   =', mpl.rcParams.get('text.usetex'))
print('mathtext.fontset =', mpl.rcParams.get('mathtext.fontset'))

texts = ['密度', '概率密度', '密度 p', 'density', '双峰系数', '样本数']
fig, axes = plt.subplots(1, len(texts), figsize=(3.2 * len(texts), 3.0))
for ax, t in zip(axes, texts):
    ax.set_ylabel(t)
    ax.set_title(t, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(r'h:\deepinv\deepinv2\.audit_tmp\exp1422\t_labels.png', dpi=150)
print('saved')

# 直接测宽度：单位 em
from matplotlib.font_manager import FontProperties
fp = FontProperties(size=10)
for t in texts:
    w, h, d = 0, 0, 0
    try:
        from matplotlib.textpath import TextPath
        tp = TextPath((0, 0), t, size=10, prop=fp)
        bb = tp.get_extents()
        print(f'{t!r:12s} TextPath width = {bb.width:.2f} pt (期望≈{10*len(t):.0f})')
    except Exception as e:
        print(t, 'err', e)
