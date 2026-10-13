# 频谱面板配色方案对比：log10|H| (banding?) vs |H| 线性
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.data import astronaut
from skimage.color import rgb2gray

x = rgb2gray(astronaut()); n = x.shape[0]
radius = 15; sigma = 5.0
s = 2*radius+1
ax_ = np.arange(-radius, radius+1); xx, yy = np.meshgrid(ax_, ax_)
k = np.exp(-(xx**2+yy**2)/(2*sigma**2)); k /= k.sum()
h = np.zeros((n, n)); h[:s, :s] = k
H = np.abs(np.fft.fftshift(np.fft.fft2(h)))
print("min|H| =", H.min(), " max =", H.max())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(np.log10(H + 1e-15), cmap='hot')
axes[0].set_title('log10|H| + 1e-15 floor (current)')
im1 = axes[1].imshow(H, cmap='hot')
axes[1].set_title('|H| linear')
plt.colorbar(im1, ax=axes[1], fraction=0.046)
im2 = axes[2].imshow(np.log10(H), cmap='hot', vmin=-12, vmax=0)
axes[2].set_title('log10|H|, vmin=-12')
plt.colorbar(im2, ax=axes[2], fraction=0.046)
for a in axes: a.axis('off')
plt.tight_layout()
plt.savefig(r'D:\works\deepinv\.audit_tmp\exp121_spec_cmp.png', dpi=110)
print('saved')
