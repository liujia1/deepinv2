# -*- coding: utf-8 -*-
"""验证修复: axis('off')后ylabel不可见, 需 ax.yaxis.label.set_visible(True) 或用 ax.text"""
import sys
sys.path.insert(0, r'd:\works\deepinv\第13章 条件生成与逆问题求解\第十三章 配套实验\实验13.6-2\.chinese')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=r'd:\works\deepinv\第13章 条件生成与逆问题求解\第十三章 配套实验\实验13.6-2\.chinese')

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
rng = np.random.default_rng(0)
# 方法1: set_ylabel + yaxis.label.set_visible(True)
axes[0,0].imshow(rng.random((28,28)), cmap='gray'); axes[0,0].axis('off')
axes[0,0].set_ylabel('FIX1 ζ=1.0\nPSNR=20.1dB', rotation=0, labelpad=40, fontsize=9)
axes[0,0].yaxis.label.set_visible(True)
# 方法2: ax.text transAxes
axes[0,1].imshow(rng.random((28,28)), cmap='gray'); axes[0,1].axis('off')
axes[0,1].text(-0.08, 0.5, 'FIX2 ζ=3.0\nPSNR=21.2dB', transform=axes[0,1].transAxes,
               rotation=0, va='center', ha='right', fontsize=9)
# 对照: 仅set_ylabel(应不可见)
axes[1,0].imshow(rng.random((28,28)), cmap='gray'); axes[1,0].axis('off')
axes[1,0].set_ylabel('BROKEN ζ=5.0', rotation=0, labelpad=40, fontsize=9)
axes[1,1].imshow(rng.random((28,28)), cmap='gray'); axes[1,1].axis('off')
plt.tight_layout()
fig.savefig(r'd:\works\deepinv\.audit_tmp\exp1362_fixtest.png', dpi=120, bbox_inches='tight')
print('saved')
