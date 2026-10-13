import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, '.chinese')
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir='.chinese')

fig, ax = plt.subplots(1, 3, figsize=(12, 3))
ax[0].plot([0, 1], [0, 1], label='1 \u2212 归一化 MSE')   # U+2212
ax[1].plot([0, 1], [0, 1], label='1 - 归一化 MSE')        # ASCII hyphen
ax[2].plot([0, 1], [0, 1], label='$1-$ 归一化 MSE')       # mathtext
for a in ax:
    a.legend(loc='lower right')
    a.axis('off')
plt.savefig('_minus_test.png', dpi=150)
print('font:', matplotlib.rcParams['font.family'])
print('done')
