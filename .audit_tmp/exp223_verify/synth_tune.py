# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
import sys, os
sys.path.insert(0, r"d:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\第二章 配套实验\实验2.2-3\.chinese")
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=r"d:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\第二章 配套实验\实验2.2-3\.chinese")

sigma = 0.1
m = 128
tt = np.arange(m)
x_true = np.zeros(m)
x_true[:32] = 0.20
x_true[32:80] = 0.20 + 0.55*(tt[32:80]-32)/(80-32)   # 线性渐变区(ramp)
x_true[80:96] = 0.75
x_true[96:] = 0.30                                     # 阶跃边缘(step)
rng = np.random.default_rng(0)
x_noisy = x_true + rng.normal(0, sigma, m)

def sobolev_1d(y, lam):
    N = len(y)
    f = np.fft.fftfreq(N)*2*np.pi
    lap = 2*(np.cos(f)-1)
    return np.real(np.fft.ifft(np.fft.fft(y)/(1-lam*lap)))

# check 1D TV works
tv = denoise_tv_chambolle(x_noisy, weight=0.15)
print("1D TV ok, shape", tv.shape, "range", round(tv.min(),3), round(tv.max(),3))

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
for j, lam in enumerate([1.0, 3.0, 6.0]):
    ax = axes[0, j]
    sob = sobolev_1d(x_noisy, lam)
    ax.plot(tt, x_true, 'k', lw=2, label='真实')
    ax.plot(tt, x_noisy, color='gray', lw=0.5, alpha=0.5, label='含噪')
    ax.plot(tt, sob, 'steelblue', lw=1.5, label=f'Sobolev λ={lam}')
    ax.plot(tt, tv, 'crimson', lw=1.5, label='TV α=0.15')
    ax.set_title(f'lam={lam} 全信号')
    ax.legend(fontsize=8)
for j, w in enumerate([0.1, 0.2, 0.35]):
    ax = axes[1, j]
    tvc = denoise_tv_chambolle(x_noisy, weight=w)
    ax.plot(tt[28:100], x_true[28:100], 'k', lw=2, label='真实')
    ax.plot(tt[28:100], tvc[28:100], 'crimson', lw=1.5, label=f'TV w={w}')
    ax.plot(tt[28:100], sobolev_1d(x_noisy, 3.0)[28:100], 'steelblue', lw=1.5, label='Sob λ=3')
    ax.set_title(f'TV w={w} 斜坡放大')
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r"d:\works\deepinv\.audit_tmp\exp223_verify\synth_tune.png", dpi=110)
print("saved synth_tune.png")
