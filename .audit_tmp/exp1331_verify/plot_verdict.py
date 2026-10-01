# -*- coding: utf-8 -*-
"""免责裁决图: 归一化 vs 不归一化 的 PSNR-ζ 权衡曲线"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SAVE_DIR, '.chinese'))
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=os.path.join(SAVE_DIR, '.chinese'))

zeta = np.array([1, 3, 5])
unnorm = np.array([24.04, 25.83, 26.18])       # B组: 不归一化
norm_small = np.array([8.51, 11.10, 13.59])     # 原实验: 归一化(小ζ)
norm_large_z = np.array([15, 30, 60])
norm_large = np.array([26.27, 25.99, 24.69])    # C组: 归一化(大ζ)
OBS = 16.47
UNCOND = 7.58

fig, ax = plt.subplots(figsize=(8, 5.2))
ax.axhline(OBS, color='tab:orange', ls='--', lw=1.5, label=f'含噪观测 = {OBS} dB')
ax.axhline(UNCOND, color='gray', ls=':', lw=1.2, label=f'无条件采样 = {UNCOND} dB')
ax.plot(zeta, norm_small, 'o-', color='tab:red', label='归一化梯度 (代码实际做法)', ms=8)
ax.plot(zeta, unnorm, 's-', color='tab:green', label='不归一化梯度 (讲义输入表声称)', ms=8)
ax.plot(norm_large_z, norm_large, '^--', color='tab:blue', label='归一化 + 加大 ζ', ms=8)
ax.annotate('原实验全部低于观测\n→ 讲义归因于"容量/轮数"',
            xy=(3, 11.1), xytext=(3.3, 15.6), fontsize=9, color='tab:red',
            arrowprops=dict(arrowstyle='->', color='tab:red'))
ax.annotate('同一模型仅改梯度处理\n即远超观测 (≈26 dB)',
            xy=(3, 25.83), xytext=(4.2, 21.0), fontsize=9, color='tab:green',
            arrowprops=dict(arrowstyle='->', color='tab:green'))
ax.set_xlabel('引导权重 $\\zeta$')
ax.set_ylabel('重建 PSNR (dB)')
ax.set_title('实验13.3-1 免责裁决：低 PSNR 源于梯度归一化（配置错误），非模型容量')
ax.set_xscale('log')
ax.set_xticks(list(zeta) + list(norm_large_z))
ax.set_xticklabels([str(v) for v in list(zeta) + list(norm_large_z)])
ax.grid(alpha=0.3)
ax.legend(loc='lower right', fontsize=8.5)
fig.tight_layout()
out = os.path.join(SAVE_DIR, 'ablation_verdict.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print('saved', out)
