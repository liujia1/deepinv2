# 校验讲义中引用的两个量：未clip标准差、放大倍数
import numpy as np
from skimage.data import astronaut
from skimage.color import rgb2gray

np.random.seed(42)
x = rgb2gray(astronaut()); n = x.shape[0]

def gaussian_psf(sigma, radius):
    s = 2 * radius + 1
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)); k /= k.sum()
    h = np.zeros((n, n)); h[:s, :s] = k
    return h

h = gaussian_psf(5.0, 15)
H = np.fft.fft2(h)
y = np.real(np.fft.ifft2(H * np.fft.fft2(x)))
yn = y + 0.01 * np.random.randn(n, n)
xdn = np.real(np.fft.ifft2(np.fft.fft2(yn) / H))
print("x.std =", float(x.std()), " 未clip重建.std =", float(xdn.std()),
      " 未clip重建.max/min =", float(xdn.max()), float(xdn.min()))
print("min|H| =", float(np.abs(H).min()), " 1/min|H| =", float(1 / np.abs(H).min()))
print("clip后 rel_err =", float(np.linalg.norm(np.clip(xdn,0,1)-x)/np.linalg.norm(x)))
print("未clip rel_err =", float(np.linalg.norm(xdn-x)/np.linalg.norm(x)))
