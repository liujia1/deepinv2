import numpy as np
from skimage import data
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio

np.random.seed(42)
n = 128
x = resize(data.camera(), (n, n))
x_std = np.std(x)
DATA_RANGE = x.max() - x.min()
sigma_noise = 0.1
y = x + sigma_noise * np.random.randn(n, n)
y_clipped = np.clip(y, 0, 1)

mu_x = np.mean(x)          # oracle prior mean
mu_y = np.mean(y)          # practical estimate (noise mean ~0)
print("mu_x=%.4f  mu_y=%.4f"%(mu_x, mu_y))

lam_bayes = sigma_noise**2 / x_std**2
psnr_noisy = peak_signal_noise_ratio(x, y_clipped, data_range=DATA_RANGE)
print("lam_bayes=%.4f  psnr_noisy=%.2f"%(lam_bayes, psnr_noisy))

# mean-centered Tikhonov: x_hat = mu + (y-mu)/(1+lam)
def recon(mu, lam): return mu + (y - mu)/(1+lam)

x_bayes_mc = recon(mu_x, lam_bayes)
print("[mean-centered oracle] PSNR=%.2f  (clip PSNR=%.2f)"%(
    peak_signal_noise_ratio(x, x_bayes_mc, data_range=DATA_RANGE),
    peak_signal_noise_ratio(x, np.clip(x_bayes_mc,0,1), data_range=DATA_RANGE)))

# grid scan
lams = np.logspace(-4,1,2000)
p = [peak_signal_noise_ratio(x, recon(mu_x,L), data_range=DATA_RANGE) for L in lams]
i = int(np.argmax(p))
print("grid lam_best=%.4f  psnr_best=%.2f   (ratio to bayes=%.2f)"%(lams[i], p[i], lams[i]/lam_bayes))

# using mu_y instead of oracle mu_x
x_bayes_mc2 = recon(mu_y, lam_bayes)
print("[mean-centered, mu from y] PSNR=%.2f"%peak_signal_noise_ratio(x, x_bayes_mc2, data_range=DATA_RANGE))

# sigma_x estimated from y (subtract noise var) as in original noisy-estimate branch
y_std = np.std(y_clipped)
sigma_x_est = np.sqrt(max(y_std**2 - sigma_noise**2, 1e-8))
lam_noisy = sigma_noise**2/sigma_x_est**2
print("lam_bayes_noisy(mean-centered)=%.4f  PSNR=%.2f"%(lam_noisy,
    peak_signal_noise_ratio(x, recon(mu_y, lam_noisy), data_range=DATA_RANGE)))

# does denoise beat noisy?
print("denoise(%.2f) > noisy(%.2f) ? %s"%(p[i], psnr_noisy, p[i] > psnr_noisy))
