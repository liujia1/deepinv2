# -*- coding: utf-8 -*-
"""定位13.6-2配图ylabel缺失的根因: 测试4种变体"""
import sys
sys.path.insert(0, r'd:\works\deepinv\第13章 条件生成与逆问题求解\第十三章 配套实验\实验13.6-2\.chinese')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
print('matplotlib', matplotlib.__version__)
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=r'd:\works\deepinv\第13章 条件生成与逆问题求解\第十三章 配套实验\实验13.6-2\.chinese')

def make(variant, out):
    zetas = [1.0, 3.0, 5.0]
    n_rows = 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4*n_rows))
    rng = np.random.default_rng(0)
    for i in range(4):
        axes[0, i].imshow(rng.random((28, 28)), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('观测y\n(遮挡)', fontsize=11, rotation=0, labelpad=50)
        for r, z in enumerate(zetas):
            axes[r+1, i].imshow(rng.random((28, 28)), cmap='gray', vmin=0, vmax=1)
            axes[r+1, i].axis('off')
            if i == 0:
                axes[r+1, i].set_ylabel(f'ζ={z}\nPSNR={20+r:.1f}dB', fontsize=10, rotation=0, labelpad=50)
    plt.suptitle('变体 ' + variant, fontsize=14, y=1.02)
    if 'notight' not in variant:
        plt.tight_layout()
    bbox = None if 'nobbox' in variant else 'tight'
    fig.savefig(out, dpi=100, bbox_inches=bbox)
    plt.close(fig)
    # 检查左侧是否有文本像素
    from PIL import Image
    im = np.array(Image.open(out).convert('L'))
    h, w = im.shape
    left = im[:, :int(w*0.12)]
    # 文本行: 白色背景上的深色小像素团
    print(variant, 'size', im.shape, 'left12% darkfrac', round((left < 100).mean(), 4))

make('A_current', r'd:\works\deepinv\.audit_tmp\exp1362_v_a.png')
make('B_notight', r'd:\works\deepinv\.audit_tmp\exp1362_v_b.png')
make('C_notight_nobbox', r'd:\works\deepinv\.audit_tmp\exp1362_v_c.png')

# D: labelpad=5
import matplotlib.pyplot as plt
def make_d():
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    rng = np.random.default_rng(0)
    for i in range(4):
        for r in range(4):
            axes[r, i].imshow(rng.random((28, 28)), cmap='gray', vmin=0, vmax=1)
            axes[r, i].axis('off')
            if i == 0:
                axes[r, i].set_ylabel(f'ζ测试 row{r}', fontsize=10, rotation=0, labelpad=5)
    plt.tight_layout()
    fig.savefig(r'd:\works\deepinv\.audit_tmp\exp1362_v_d.png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    from PIL import Image
    im = np.array(Image.open(r'd:\works\deepinv\.audit_tmp\exp1362_v_d.png').convert('L'))
    h, w = im.shape
    print('D_labelpad5 left12% darkfrac', round((im[:, :int(w*0.12)] < 100).mean(), 4))
make_d()
