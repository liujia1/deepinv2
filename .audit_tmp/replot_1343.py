# -*- coding: utf-8 -*-
"""临时脚本：用 13.4-3 的 JSON 数据快速试排标注位置（不改动正式产物）"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

EXP_DIR = r'h:\deepinv\deepinv2\第13章 条件生成与逆问题求解\第十三章 配套实验\实验13.4-3'
sys.path.insert(0, os.path.join(EXP_DIR, '.chinese'))
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=os.path.join(EXP_DIR, '.chinese'))
except ImportError:
    print('警告: chinese_font 未找到')

with open(os.path.join(EXP_DIR, 'results_summary.json'), encoding='utf-8') as f:
    d = json.load(f)

zeta_values = d['zeta_values']
psnr_results = [d['PSNR_各zeta'][f'zeta={z}'] for z in zeta_values]
zeta_values_chung = d['对照_Chung原版']['zeta_prime_values']
psnr_results_chung = [d['对照_Chung原版']["PSNR_各zeta_prime"][f"zeta'={z}"] for z in zeta_values_chung]
psnr_obs = d['含噪观测PSNR参照']

fig2, ax2 = plt.subplots(1, 1, figsize=(9, 6))
ax2.plot(zeta_values, psnr_results, 'ro-', markersize=8, lw=2,
         label='单位归一化实现(本实验主曲线)')
ax2.plot(zeta_values_chung, psnr_results_chung, 'bs--', markersize=8, lw=2,
         label=r"Chung et al. (2022) 原版步长 $\zeta'\!\cdot\!\nabla\Vert r\Vert$(对照)")
ax2.axhline(psnr_obs, color='gray', linestyle=':', lw=1.5)
_trans_obs = mtransforms.blended_transform_factory(ax2.transAxes, ax2.transData)
ax2.text(0.985, psnr_obs + 0.15, f'含噪观测 PSNR={psnr_obs:.1f}dB',
         transform=_trans_obs, ha='right', va='bottom', fontsize=9, color='dimgray')
ax2.set_xscale('symlog', linthresh=1.0)
ax2.set_xticks([0, 0.3, 0.7, 1.5, 3.0, 6.0, 10.0, 15.0, 30.0, 50.0, 100.0])
ax2.set_xticklabels(['0', '0.3', '0.7', '1.5', '3', '6', '10', '15', '30', '50', '100'])
ax2.set_xlabel(r'引导权重 $\zeta$', fontsize=12)
ax2.set_ylabel('PSNR (dB)', fontsize=12)
ax2.set_title(r'$\zeta$-重建质量权衡曲线（13.4.3节）', fontsize=13)
ax2.grid(alpha=0.3, which='both')
ax2.legend(loc='lower right', fontsize=9)

best_idx = int(np.argmax(psnr_results))
best_zeta = zeta_values[best_idx]
best_psnr = psnr_results[best_idx]
ax2.plot(best_zeta, best_psnr, 'b*', markersize=16, zorder=5,
         markeredgecolor='navy', markeredgewidth=0.5)
ax2.annotate(rf'归一化实现最优 $\zeta$={best_zeta:g}' + '\n' + rf'PSNR={best_psnr:.2f}dB',
             xy=(best_zeta, best_psnr),
             xytext=(5.0, 23.6),
             fontsize=9, ha='left', va='center',
             arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeaa7', alpha=0.85))
bc_idx = int(np.argmax(psnr_results_chung))
bc_zeta = zeta_values_chung[bc_idx]
bc_psnr = psnr_results_chung[bc_idx]
ax2.plot(bc_zeta, bc_psnr, 'g*', markersize=16, zorder=5,
         markeredgecolor='darkgreen', markeredgewidth=0.5)
ax2.annotate(rf"Chung对照最优 $\zeta'$={bc_zeta:g}" + '\n' + rf'PSNR={bc_psnr:.2f}dB',
             xy=(bc_zeta, bc_psnr),
             xytext=(0.5, 21.0),
             fontsize=9, ha='left', va='center',
             arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#dff9e3', alpha=0.85))

fig2.tight_layout()
out = r'h:\deepinv\deepinv2\.audit_tmp\replot_A.png'
fig2.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig2)
print('saved', out)
