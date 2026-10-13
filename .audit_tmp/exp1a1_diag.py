import numpy as np
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

np.random.seed(42)
x = rgb2gray(astronaut())
noise_levels = [0.01, 0.03, 0.05, 0.10, 0.20, 0.40]

results = {}
for sigma in noise_levels:
    y = x + sigma * np.random.randn(*x.shape)
    y_clip = np.clip(y, 0, 1)   # 复现原脚本：整图裁剪
    y_noclip = y.copy()         # 同一噪声实现的未裁剪版本
    # 另：仅裁剪噪声再相加（部分文献做法）
    y_clipnoise = x + np.clip(sigma * np.random.RandomState(0).randn(*x.shape), -3.5*sigma, 3.5*sigma)
    clipfrac = float(np.mean(y < 0) + np.mean(y > 1))
    row = dict(
        clipfrac=clipfrac,
        mse_c=mean_squared_error(x, y_clip),
        mse_n=mean_squared_error(x, y_noclip),
        psnr_c=peak_signal_noise_ratio(x, y_clip),
        psnr_n=peak_signal_noise_ratio(x, y_noclip),
        ssim_c7=structural_similarity(x, y_clip, data_range=1.0, win_size=7),
        ssim_c11=structural_similarity(x, y_clip, data_range=1.0, win_size=11),
        ssim_n11=structural_similarity(x, y_noclip, data_range=1.0, win_size=11),
    )
    results[sigma] = row
    print(f"sigma={sigma:.2f} | clipfrac={row['clipfrac']:.4f} | MSE {row['mse_c']:.6f}(clip) vs {row['mse_n']:.6f}(raw=sig^2±MC) | PSNR {row['psnr_c']:.2f} vs {row['psnr_n']:.2f} | SSIM w7={row['ssim_c7']:.4f} w11={row['ssim_c11']:.4f} raw_w11={row['ssim_n11']:.4f}")

# 关键数值：图像统计量
print("\nmean(x)=", x.mean(), " std(x)=", x.std(), " frac(x==0)=", np.mean(x == 0), " frac(x==1)=", np.mean(x == 1))

# 无 clip 时 MSE 与 sigma^2 的偏差（有限样本）
print("\n理论 MSE=sigma^2 与实际无clip MSE 之比：")
rng = np.random.default_rng(0)
for sigma in noise_levels:
    ratios = []
    for k in range(20):
        z = rng.standard_normal(x.shape)
        ratios.append(mean_squared_error(x, x + sigma * z) / sigma**2)
    print(f"  sigma={sigma:.2f}: mean ratio={np.mean(ratios):.4f}, std={np.std(ratios):.4f}")
