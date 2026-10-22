"""验证 ylabel='密度' 是否真的缺字形（与 (b) 的中文 ylabel 对比）。"""
import sys, io, os
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=False)

sys.path.insert(0, '.chinese')
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir='.chinese')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont

fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
ax[0].set_ylabel('密度')
ax[0].set_title('(a) ylabel=密度')
ax[1].set_ylabel('双峰系数 BC')
ax[1].set_title('(b) ylabel=双峰系数 BC')
plt.tight_layout()
plt.savefig(r'h:\deepinv\deepinv2\.audit_tmp\exp1422\t_ylabel.png', dpi=150)
print('saved')

cm_font = FontProperties()
f = findfont(cm_font)
print('font.family =', matplotlib.rcParams['font.family'])
print('resolved ->', f)

from matplotlib.ft2font import FT2Font
ft = FT2Font(f)
print('FT2Font file =', f)
for ch in '密度双峰系数':
    print(f'  {ch} U+{ord(ch):04X} glyph_index={ft.get_char_index(ord(ch))}')
