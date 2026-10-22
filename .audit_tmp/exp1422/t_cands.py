import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.2-2\.chinese')
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.2-2\.chinese')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cands = [r'概率密度', r'概率密度 p', r'概率密度 $p(x_2)$', r'样本密度', r'密度', r'密度 p']
fig, axes = plt.subplots(1, len(cands), figsize=(3.0 * len(cands), 4.2))
for ax, t in zip(axes, cands):
    ax.set_ylabel(t)
    ax.set_title(t, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.suptitle('ylabel candidates', fontsize=12)
plt.savefig(r'h:\deepinv\deepinv2\.audit_tmp\exp1422\v5_cands.png', dpi=150)
print('ok')
