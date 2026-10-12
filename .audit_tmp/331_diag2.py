import numpy as np
from skimage import data
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio

np.random.seed(42)
n = 128
x = resize(data.camera(), (n, n))
x_std = np.std(x); mu = np.mean(x)
DATA_RANGE = x.max() - x.min()
sigma_noise = 0.1
y = x + sigma_noise * np.random.randn(n, n)
y_clipped = np.clip(y, 0, 1)

lam_bayes = sigma_noise**2 / x_std**2
x_bayes = y / (1 + lam_bayes)

print("=== PSNR comparison (data_range=%.4f) ===" % DATA_RANGE)
print("noisy y (UNclipped)   :", round(peak_signal_noise_ratio(x, y, data_range=DATA_RANGE),2))
print("noisy y (CLIPPED)     :", round(peak_signal_noise_ratio(x, y_clipped, data_range=DATA_RANGE),2))
print("Tikhonov bayes UNclip :", round(peak_signal_noise_ratio(x, x_bayes, data_range=DATA_RANGE),2))
print("Tikhonov bayes CLIP   :", round(peak_signal_noise_ratio(x, np.clip(x_bayes,0,1), data_range=DATA_RANGE),2))

# grid best unclipped and clipped
lams = np.logspace(-4,1,50)
p_un=[peak_signal_noise_ratio(x, y/(1+L), data_range=DATA_RANGE) for L in lams]
p_cl=[peak_signal_noise_ratio(x, np.clip(y/(1+L),0,1), data_range=DATA_RANGE) for L in lams]
print("best grid UNclip lam=%.4f PSNR=%.2f"%(lams[int(np.argmax(p_un))], max(p_un)))
print("best grid CLIP lam=%.4f PSNR=%.2f"%(lams[int(np.argmax(p_cl))], max(p_cl)))

# proper Wiener shrinking toward the MEAN (correct prior center)
# x_hat = mu + (sigma_x^2/(sigma_x^2+sigma^2)) (y - mu)
w = x_std**2/(x_std**2+sigma_noise**2)
x_wiener_mean = mu + w*(y-mu)
print("Wiener shrink toward mean=%.3f PSNR=%.2f"%(mu, peak_signal_noise_ratio(x, x_wiener_mean, data_range=DATA_RANGE)))
