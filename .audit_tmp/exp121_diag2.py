# 实验1.2-1 诊断2：分解 clean 反卷积误差 0.215 的频段来源，并测试正确实现
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
    return h / h.sum()

def blur_operator(x, h):
    return np.real(np.fft.ifft2(np.fft.fft2(h) * np.fft.fft2(x)))

h = gaussian_psf(n, 5.0)
y = blur_operator(x, h)
H = np.fft.fft2(h)
Xf = np.fft.fft2(x)
Yf = np.fft.fft2(y)

xd = np.real(np.fft.ifft2(Yf / (H + 1e-15)))
E = np.fft.fft2(xd - x)
Hm = np.abs(H)
tot = (np.abs(E) ** 2).sum()

for lo, hi in [(0, 1e-300), (1e-300, 1e-100), (1e-100, 1e-15), (1e-15, 1e-12),
               (1e-12, 1e-6), (1e-6, 1e-3), (1e-3, 1), (1, 10)]:
    band = (Hm >= lo) & (Hm < hi)
    if band.sum() == 0: continue
    e2 = (np.abs(E[band]) ** 2).sum()
    # 该频段 X 的真值能量
    x2 = (np.abs(Xf[band]) ** 2).sum()
    print(f"|H|∈[{lo:.0e},{hi:.0e}): 频点数={band.sum():6d}  误差能量占比={e2/tot:.4f}  该频段|X|rms={np.sqrt(x2/band.sum()):.3e}")

# 深度阻带里 Y 是否已经被浮点舍入污染？
deep = Hm < 1e-15
print("\n|H|<1e-15 频点数 =", deep.sum())
print("这些频点上 |Y| 的 rms =", float(np.sqrt((np.abs(Yf[deep])**2).mean())),
      " 而 |H*X| 理论 rms =", float(np.sqrt(((Hm[deep]*np.abs(Xf[deep]))**2).mean())))

# ---- 正确实现A：小尺寸高斯核（截断）放入 (0,0) 处，标准循环卷积 ----
def gaussian_psf_small(sigma, radius):
    s = 2 * radius + 1
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)); k /= k.sum()
    h = np.zeros((n, n)); h[:s, :s] = k        # 峰在(0,0)，右下半为负偏移（循环约定）
    return h

for radius in (15, 30):
    hs = gaussian_psf_small(5.0, radius)
    Hs = np.fft.fft2(hs)
    ys = blur_operator(x, hs)
    xds = np.real(np.fft.ifft2(np.fft.fft2(ys) / Hs))   # 直接除（无 eps）
    rel = np.linalg.norm(xds - x) / np.linalg.norm(x)
    print(f"[截断核 radius={radius}] |H|min={np.abs(Hs).min():.3e}, clean直接除 rel_err={rel:.3e}")
    # 含噪声
    ysn = ys + 0.01 * np.random.randn(n, n)
    xdsn = np.real(np.fft.ifft2(np.fft.fft2(ysn) / Hs))
    print(f"    noisy直接除 rel_err={np.linalg.norm(np.clip(xdsn,0,1)-x)/np.linalg.norm(x):.3f}, 未clip std={xdsn.std():.1f}")

# ---- 正确实现B：原全尺寸核但用掩码除法（只 inversion |H|>tol）----
for tol in (1e-6, 1e-3):
    num = np.zeros_like(Yf)
    mask = np.abs(H) > tol
    num[mask] = Yf[mask] / H[mask]
    xdm = np.real(np.fft.ifft2(num))
    print(f"[原核+掩码除法 tol={tol:.0e}] clean rel_err={np.linalg.norm(xdm-x)/np.linalg.norm(x):.3e}")

# ---- 原脚本口径下，若把 eps 改大（Wienz 式稳定分母）会怎样 ----
for eps in (1e-15, 1e-8, 1e-4):
    xc = np.real(np.fft.ifft2(Yf / (H + eps)))
    print(f"[原实现 eps={eps:.0e}] clean rel_err={np.linalg.norm(xc-x)/np.linalg.norm(x):.3e}")
