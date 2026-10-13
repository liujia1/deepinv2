import numpy as np, os, sys, json
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio

np.random.seed(42)
x = rgb2gray(astronaut())
n = x.shape[0]

def gaussian_psf(size, sigma):
    ax = np.fft.fftfreq(size) * size
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return h / h.sum()

def blur(x, h):
    return np.real(np.fft.ifft2(np.fft.fft2(h) * np.fft.fft2(x)))

sigma_A, sigma_B = 5.0, 5.3
h_A = gaussian_psf(n, sigma_A); h_B = gaussian_psf(n, sigma_B)
noise_sigma = 0.01
noise = noise_sigma * np.random.randn(n, n)
y_IC = blur(x, h_A) + noise
y_noIC = blur(x, h_B) + noise
H_A = np.fft.fft2(h_A)

def naive_deconv(y, H, eps=1e-15):
    abs_H = np.abs(H); abs_H_clamped = np.maximum(abs_H, eps)
    scale = np.ones_like(abs_H); mask = abs_H > 0
    scale[mask] = abs_H_clamped[mask] / abs_H[mask]
    H_reg = H * scale; H_reg[~mask] = eps
    return np.real(np.fft.ifft2(np.fft.fft2(y) / H_reg))

def tikhonov_deconv(y, H, lam):
    H_star = np.conj(H); H_sq = np.abs(H) ** 2
    return np.real(np.fft.ifft2(H_star / (H_sq + lam) * np.fft.fft2(y)))

y_clean = blur(x, h_A)
x_naive_clean = naive_deconv(y_clean, H_A)
x_naive_noisy = naive_deconv(y_IC, H_A)
x_tikh_IC = tikhonov_deconv(y_IC, H_A, 1e-2)
x_tikh_noIC = tikhonov_deconv(y_noIC, H_A, 1e-2)

sigma_B_list = [3.0,4.0,4.5,5.0,5.3,5.5,6.0,7.0,8.0,10.0]
psnr_curve = []
for sb in sigma_B_list:
    h_tmp = gaussian_psf(n, sb); y_tmp = blur(x, h_tmp) + noise
    x_tmp = tikhonov_deconv(y_tmp, H_A, 1e-2)
    psnr_curve.append(peak_signal_noise_ratio(x, np.clip(x_tmp,0,1), data_range=1.0))

p_nn = peak_signal_noise_ratio(x, np.clip(x_naive_noisy,0,1), data_range=1.0)
p_nc = peak_signal_noise_ratio(x, np.clip(x_naive_clean,0,1), data_range=1.0)
p_ic = peak_signal_noise_ratio(x, np.clip(x_tikh_IC,0,1), data_range=1.0)
p_noic = peak_signal_noise_ratio(x, np.clip(x_tikh_noIC,0,1), data_range=1.0)

print("naive_noisy  =", round(p_nn,2), " (json 4.67, 讲义文字 -5)")
print("naive_clean  =", round(p_nc,2), " (json 20.51)")
print("tikh_IC      =", round(p_ic,2), " (json 22.34)")
print("tikh_noIC    =", round(p_noic,2), " (json 21.93)")
print("curve        =", [round(v,2) for v in psnr_curve])

# unclipped naive noisy MSE to show why -5dB impossible after clip
print("naive_noisy raw range:", x_naive_noisy.min(), x_naive_noisy.max())
mse_clip = np.mean((x - np.clip(x_naive_noisy,0,1))**2)
print("clipped MSE =", round(mse_clip,4), "-> PSNR floor is 0dB when MSE=1")
