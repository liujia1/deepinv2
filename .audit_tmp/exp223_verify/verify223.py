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
    n = y.shape[0]
    freq = np.fft.fftfreq(n) * 2 * np.pi
    fx, fy = np.meshgrid(freq, freq)
    lap_eig = 2 * (np.cos(fx) + np.cos(fy) - 2)
    denom = 1 - lam * lap_eig
    return np.real(np.fft.ifft2(np.fft.fft2(y) / denom))

def find_staircase_position(tv_img, sob_img, margin=20, window_size=7):
    from scipy.ndimage import uniform_filter
    grad_tv_x = np.roll(tv_img, -1, axis=1) - tv_img
    grad_tv_y = np.roll(tv_img, -1, axis=0) - tv_img
    grad_tv_mag = np.sqrt(grad_tv_x**2 + grad_tv_y**2)
    grad_sob_x = np.roll(sob_img, -1, axis=1) - sob_img
    grad_sob_y = np.roll(sob_img, -1, axis=0) - sob_img
    grad_sob_mag = np.sqrt(grad_sob_x**2 + grad_sob_y**2)
    tv_excess = np.maximum(grad_tv_mag - grad_sob_mag, 0)
    diff = np.abs(tv_img - sob_img)
    staircase_score = tv_excess * uniform_filter(diff, size=window_size)
    staircase_score[:margin, :] = 0
    staircase_score[-margin:, :] = 0
    staircase_score[:, :margin] = 0
    staircase_score[:, -margin:] = 0
    max_idx = np.unravel_index(np.argmax(staircase_score), staircase_score.shape)
    return (max_idx[1], max_idx[0])

lam_sob = 0.15
camera_sob = sobolev_denoise(camera_noisy, lam_sob)
alpha_tv = 0.15
camera_tv = denoise_tv_chambolle(camera_noisy, weight=alpha_tv)
staircase_pos = find_staircase_position(camera_tv, camera_sob)

psnr_noisy = peak_signal_noise_ratio(camera, camera_noisy, data_range=DATA_RANGE)
psnr_sob = peak_signal_noise_ratio(camera, camera_sob, data_range=DATA_RANGE)
psnr_tv = peak_signal_noise_ratio(camera, camera_tv, data_range=DATA_RANGE)
ssim_noisy = structural_similarity(camera, camera_noisy, data_range=DATA_RANGE)
ssim_sob = structural_similarity(camera, camera_sob, data_range=DATA_RANGE)
ssim_tv = structural_similarity(camera, camera_tv, data_range=DATA_RANGE)

grad_sob_x = np.roll(camera_sob, -1, axis=1) - camera_sob
grad_sob_y = np.roll(camera_sob, -1, axis=0) - camera_sob
grad_sob_mag = np.sqrt(grad_sob_x**2 + grad_sob_y**2).flatten()
grad_tv_x = np.roll(camera_tv, -1, axis=1) - camera_tv
grad_tv_y = np.roll(camera_tv, -1, axis=0) - camera_tv
grad_tv_mag = np.sqrt(grad_tv_x**2 + grad_tv_y**2).flatten()

tv_zero_grad = np.sum(grad_tv_mag < 0.01) / len(grad_tv_mag) * 100
sob_zero_grad = np.sum(grad_sob_mag < 0.01) / len(grad_sob_mag) * 100

print("psnr_noisy", round(psnr_noisy,2))
print("psnr_sob  ", round(psnr_sob,2))
print("psnr_tv   ", round(psnr_tv,2))
print("ssim_noisy", round(ssim_noisy,4))
print("ssim_sob  ", round(ssim_sob,4))
print("ssim_tv   ", round(ssim_tv,4))
print("tv_zero_grad%  ", round(tv_zero_grad,2))
print("sob_zero_grad% ", round(sob_zero_grad,2))
print("staircase_pos (x,y)", staircase_pos)

for thr in [0.001,0.005,0.01,0.02,0.05]:
    print(f"  sobolev zero-grad% at thr={thr}: {np.sum(grad_sob_mag<thr)/len(grad_sob_mag)*100:.2f}")
for thr in [0.001,0.005,0.01,0.02,0.05]:
    print(f"  TV     zero-grad% at thr={thr}: {np.sum(grad_tv_mag<thr)/len(grad_tv_mag)*100:.2f}")
