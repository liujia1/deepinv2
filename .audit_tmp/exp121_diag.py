# 实验1.2-1 复核诊断脚本
# 目的：1) 复现 results_summary.json 两个数值；2) 定位无噪声反卷积误差 0.215 偏大的根源
import numpy as np
from skimage.data import astronaut
from skimage.color import rgb2gray

np.random.seed(42)

x = rgb2gray(astronaut())
n = x.shape[0]

def gaussian_psf(size, sigma):
    ax = np.concatenate((np.arange(0, size // 2), np.arange(-size // 2, 0)))
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    h = h / h.sum()
    return h

def blur_operator(x, h):
    return np.real(np.fft.ifft2(np.fft.fft2(h) * np.fft.fft2(x)))

sigma_psf = 5.0
h = gaussian_psf(n, sigma_psf)
y_blur = blur_operator(x, h)

H = np.fft.fft2(h)

# ---- 复现原脚本 ----
x_deconv_clean = np.real(np.fft.ifft2(np.fft.fft2(y_blur) / (H + 1e-15)))
err_clean = np.linalg.norm(x_deconv_clean - x) / np.linalg.norm(x)
print("reconstruction_error_clean =", round(float(err_clean), 6))

y_blur_noisy = y_blur + 0.01 * np.random.randn(*y_blur.shape)
x_deconv_noisy = np.real(np.fft.ifft2(np.fft.fft2(y_blur_noisy) / (H + 1e-15)))
err_noisy = np.linalg.norm(np.clip(x_deconv_noisy, 0, 1) - x) / np.linalg.norm(x)
print("reconstruction_error_noisy =", round(float(err_noisy), 6))

# ---- 诊断1：PSF 峰值位置 ----
print("PSF argmax (row,col) =", np.unravel_index(np.argmax(h), h.shape), " 期望 (0,0)")
print("h[0,0] =", h[0, 0], ", h[256,256] =", h[256, 256])

# ---- 诊断2：H 的数值范围（下溢问题）----
Hmag = np.abs(H)
print("|H| max =", Hmag.max(), ", min(非零) =", Hmag[Hmag > 0].min(), ", 精确0个数 =", (Hmag == 0).sum())
print("|H| < 1e-8 的频率个数 =", (Hmag < 1e-8).sum(), " / ", n * n)
print("float64 denormal floor ~ 5e-324; |H| < 1e-300 个数 =", (Hmag < 1e-300).sum())

# ---- 诊断3：反卷积残余误差的频域来源 ----
D = np.fft.fft2(x_deconv_clean - x)
contrib = np.abs(D)
idx = np.argsort(contrib.ravel())[::-1][:10]
print("误差频谱能量最大的10个频率的 |H| 值：", Hmag.flatten()[idx])
# 误差占总能量的比例：极端衰减频段
band_tiny = Hmag < 1e-100
rel_energy = (contrib[band_tiny] ** 2).sum() / (contrib ** 2).sum()
print("误差能量中 |H|<1e-100 频段占比 =", float(rel_energy))

# ---- 诊断4：若把 PSF 正确居中于索引(0,0)（标准循环卷积约定），误差应回到机器精度 ----
h2 = np.fft.ifftshift(h)   # 把中心峰移到 (0,0)
y_blur2 = np.real(np.fft.ifft2(np.fft.fft2(h2) * np.fft.fft2(x)))
H2 = np.fft.fft2(h2)
xd2 = np.real(np.fft.ifft2(np.fft.fft2(y_blur2) / (H2 + 1e-15)))
print("[修复后-无噪] rel_err =", float(np.linalg.norm(xd2 - x) / np.linalg.norm(x)))
# 用 np.divide where 只除 |H|>eps 的更稳健做法
xd2b = np.real(np.fft.ifft2(np.fft.fft2(y_blur2) / H2))  # 不加速生除，看原生结果
print("[修复后-无噪, 不加eps] rel_err =", float(np.linalg.norm(xd2b - x) / np.linalg.norm(x)))

# ---- 诊断5：修复约定后含噪反卷积是否仍放大噪声 ----
rng = np.random.RandomState(0)
yn2 = y_blur2 + 0.01 * rng.randn(n, n)
xdn2 = np.real(np.fft.ifft2(np.fft.fft2(yn2) / (H2 + 1e-15)))
print("[修复后-含噪] rel_err =", float(np.linalg.norm(xdn2 - x) / np.linalg.norm(x)))
print("[修复后-含噪] 未clip std =", float(xdn2.std()), " x.std =", float(x.std()))
