import numpy as np
from skimage import data
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

np.random.seed(42)
DATA_RANGE = 1.0
n = 128
camera = resize(data.camera(), (n, n))
sigma = 0.1
camera_noisy = random_noise(camera, mode='gaussian', var=sigma**2, rng=42)

def sobolev_denoise(y, lam):
    nn = y.shape[0]
    freq = np.fft.fftfreq(nn) * 2 * np.pi
    fx, fy = np.meshgrid(freq, freq)
    lap_eig = 2 * (np.cos(fx) + np.cos(fy) - 2)
    denom = 1 - lam * lap_eig
    return np.real(np.fft.ifft2(np.fft.fft2(y) / denom))

def gmag(img):
    gx = np.roll(img, -1, axis=1) - img
    gy = np.roll(img, -1, axis=0) - img
    return np.sqrt(gx**2 + gy**2)

print("=== Sobolev lambda sweep (PSNR/SSIM, residual-noise std, grad-mean) ===")
print("true grad-mean:", gmag(camera).mean().round(4))
for lam in [0.15, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
    x = sobolev_denoise(camera_noisy, lam)
    p = peak_signal_noise_ratio(camera, x, data_range=DATA_RANGE)
    s = structural_similarity(camera, x, data_range=DATA_RANGE)
    # residual high-freq noise: std of (x - x_smoothed) as proxy of leftover noise
    resid = (x - camera).std()
    print(f"lam={lam:5.2f}  PSNR={p:5.2f} SSIM={s:.3f}  err_std={resid:.4f}  gradmean={gmag(x).mean():.4f}")

print("\n=== TV alpha sweep ===")
for a in [0.05, 0.1, 0.15, 0.2, 0.3]:
    x = denoise_tv_chambolle(camera_noisy, weight=a)
    p = peak_signal_noise_ratio(camera, x, data_range=DATA_RANGE)
    s = structural_similarity(camera, x, data_range=DATA_RANGE)
    print(f"alpha={a:4.2f}  PSNR={p:5.2f} SSIM={s:.3f}  err_std={(x-camera).std():.4f}  gradmean={gmag(x).mean():.4f}")

print("\n=== edge sharpness comparison at best-denoise params ===")
# find lam that maximizes SSIM for sobolev
best = max([(structural_similarity(camera, sobolev_denoise(camera_noisy,l), data_range=DATA_RANGE), l) for l in np.arange(0.1,20,0.1)])
print("Sobolev best SSIM=%.3f at lam=%.2f" % (best[0], best[1]))
lamb = best[1]
xs = sobolev_denoise(camera_noisy, lamb)
xt = denoise_tv_chambolle(camera_noisy, weight=0.15)
print("  Sobolev@best  PSNR=%.2f SSIM=%.3f" % (peak_signal_noise_ratio(camera,xs,DATA_RANGE), structural_similarity(camera,xs,DATA_RANGE)))
print("  TV@0.15       PSNR=%.2f SSIM=%.3f" % (peak_signal_noise_ratio(camera,xt,DATA_RANGE), structural_similarity(camera,xt,DATA_RANGE)))
