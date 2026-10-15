# -*- coding: utf-8 -*-
import numpy as np
from skimage.restoration import denoise_tv_chambolle

sigma = 0.1
m = 128
tt = np.arange(m)
x_true = np.zeros(m)
x_true[:32] = 0.20
x_true[32:80] = 0.20 + 0.55*(tt[32:80]-32)/(80-32)
x_true[80:96] = 0.75
x_true[96:] = 0.30
rng = np.random.default_rng(0)
x_noisy = x_true + rng.normal(0, sigma, m)

def sobolev_1d(y, lam):
    N = len(y); f = np.fft.fftfreq(N)*2*np.pi
    return np.real(np.fft.ifft(np.fft.fft(y)/(1-lam*2*(np.cos(f)-1))))

tv = denoise_tv_chambolle(x_noisy, weight=0.15)
sob = sobolev_1d(x_noisy, 3.0)

ramp = slice(32, 80)
step = slice(88, 104)   # around the jump at 96
def mse(a,b): return float(np.mean((a-b)**2))
print("ramp  MSE  TV=%.5f  Sob=%.5f  (staircase: TV worse on smooth ramp)"%(mse(tv[ramp],x_true[ramp]), mse(sob[ramp],x_true[ramp])))
print("step  MSE  TV=%.5f  Sob=%.5f  (edge: TV better)"%(mse(tv[step],x_true[step]), mse(sob[step],x_true[step])))
print("overall MSE TV=%.5f Sob=%.5f"%(mse(tv,x_true), mse(sob,x_true)))
# staircase step count on ramp: number of local jumps in TV
d = np.diff(tv[ramp])
print("TV ramp: #|grad|>0.02 jumps =", int(np.sum(np.abs(d)>0.02)), " #near-zero-grad =", int(np.sum(np.abs(d)<0.005)), "/", len(d))
ds = np.diff(sob[ramp])
print("Sob ramp: #|grad|>0.02 jumps =", int(np.sum(np.abs(ds)>0.02)), " #near-zero-grad =", int(np.sum(np.abs(ds)<0.005)), "/", len(ds))
