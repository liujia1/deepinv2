# -*- coding: utf-8 -*-
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_DIR = r"d:/works/deepinv/第13章 条件生成与逆问题求解/第十三章 配套实验/实验13.4-4"
sys.path.insert(0, os.path.join(SAVE_DIR, '.chinese'))
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=os.path.join(SAVE_DIR, '.chinese'))
plt.rcParams['mathtext.fontset'] = 'cm'

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].text(0.5, 0.5, '无需训练 观察要点', ha='center', fontsize=14, style='italic')
axes[0].set_title('带 style=italic（假设=豆腐块）')
axes[1].text(0.5, 0.5, '无需训练 观察要点', ha='center', fontsize=14)
axes[1].set_title('不带 italic（假设=正常）')
for a in axes:
    a.axis('off')
out = r"d:/works/deepinv/.audit_tmp/_italic_test_1344.png"
plt.savefig(out, dpi=120, bbox_inches='tight')
print("saved", out)
