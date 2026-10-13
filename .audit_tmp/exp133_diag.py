import numpy as np, sys
sys.path.insert(0, r'd:\works\deepinv\.audit_tmp\exp133_backup')
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

n = 64
phantom = resize(shepp_logan_phantom(), (n, n), order=3, preserve_range=True, anti_aliasing=True)
x = phantom / phantom.max()

def gaussian_psf(size, sigma):
    ax = np.concatenate((np.arange(0, size // 2), np.arange(-size // 2, 0)))
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return h / h.sum()

def blur(u, h):
    return np.real(np.fft.ifft2(np.fft.fft2(h) * np.fft.fft2(u)))

sigma = 3.0
h = gaussian_psf(n, sigma)
H = np.fft.fft2(h)  # 精确离散传递函数

xx, yy = np.meshgrid(np.arange(n), np.arange(n))
frequencies = [5, 10, 15, 20, 25]

print("=== unrounded measured attenuation vs theory ===")
for f in frequencies:
    d = np.sin(2*np.pi*f*xx/n) * np.sin(2*np.pi*f*yy/n)
    Ad = blur(d, h)
    meas = np.linalg.norm(Ad)/np.linalg.norm(d)
    exact = np.abs(H[f, f])   # sin*sin 峰在 (±f,±f)，圆周卷积下严格特征值
    cont = np.exp(-2*np.pi**2*sigma**2*(f**2+f**2)/n**2)
    ratio = meas/exact if exact > 1e-300 else float('nan')
    print(f"f={f:3d}  meas={meas:.3e}  exact|H(f,f)|={exact:.3e}  contin_approx={cont:.3e}  meas/exact={ratio:.4f}")

# noise floor: compare two mathematically identical implementations
r0 = np.random.default_rng(0).standard_normal((n, n))
p1 = np.real(np.fft.ifft2(np.fft.fft2(h) * np.fft.fft2(r0)))
Fh = np.fft.fft2(h); F0 = np.fft.fft2(r0)
p2 = np.real(np.fft.ifft2(Fh * F0) )  # same path
# independent path: spatial direct circular convolution via matrix on one row-block (float64)
# use double-double style check: convolve via np.fft with norm='forward' and manual scaling
p3 = np.fft.ifft2(np.fft.fft2(h, norm='forward') * np.fft.fft2(r0, norm='forward')).real
print(f"\nimpl-path diff p1 vs p3: {np.linalg.norm(p1-p3)/np.linalg.norm(p1):.3e}  (eps={np.finfo(float).eps:.2e})")

# main ghost case f=15 unrounded
alpha = 0.3
f = 15
d = alpha*np.sin(2*np.pi*f*xx/n)*np.sin(2*np.pi*f*yy/n)
Ax = blur(x, h); Ad = blur(d, h)
rel_meas = np.linalg.norm(Ad)/np.linalg.norm(Ax)
rel_img = np.linalg.norm(d)/np.linalg.norm(x)
print(f"\nf=15 main case (unrounded): rel_img={rel_img:.8f}  rel_meas={rel_meas:.6e}  amp={rel_img/rel_meas:.6e}")
amp_att = 1/(np.linalg.norm(Ad)/np.linalg.norm(d))
gain = np.linalg.norm(Ax)/np.linalg.norm(x)
print(f"||d||/||Ad||={amp_att:.6e};  ||Ax||/||x||={gain:.4f};  product={amp_att*gain:.6e}")

# determinism: recompute twice
Ad2 = blur(alpha*np.sin(2*np.pi*15*xx/n)*np.sin(2*np.pi*15*yy/n), h)
print("bitwise identical rerun:", np.array_equal(Ad, Ad2))
