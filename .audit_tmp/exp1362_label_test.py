# -*- coding: utf-8 -*-
"""最小复现：检验 ylabel(rotation=0,labelpad=50) 是否能在保存图中显示"""
import sys, os
sys.path.insert(0, r'd:\works\deepinv\第13章 条件生成与逆问题求解\第十三章 配套实验\实验13.6-2\.chinese')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=r'd:\works\deepinv\第13章 条件生成与逆问题求解\第十三章 配套实验\实验13.6-2\.chinese')

zeta_values = [1.0, 3.0, 5.0]
psnr_list = [20.1, 21.2, 22.3]
n_rows = len(zeta_values) + 1
fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4*n_rows))
rng = np.random.default_rng(0)
for i in range(4):
    axes[0, i].imshow(rng.random((28, 28)), cmap='gray', vmin=0, vmax=1)
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_ylabel('观测y\n(遮挡)', fontsize=11, rotation=0, labelpad=50)
    for row_idx, z in enumerate(zeta_values):
        axes[row_idx+1, i].imshow(rng.random((28, 28)), cmap='gray', vmin=0, vmax=1)
        axes[row_idx+1, i].axis('off')
        if i == 0:
            axes[row_idx+1, i].set_ylabel(f'ζ={z}\nPSNR={psnr_list[row_idx]:.1f}dB',
                                          fontsize=10, rotation=0, labelpad=50)
plt.suptitle('测试标题', fontsize=14, y=1.02)
plt.tight_layout()
out = r'd:\works\deepinv\.audit_tmp\exp1362_labeltest.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print('saved', out)
