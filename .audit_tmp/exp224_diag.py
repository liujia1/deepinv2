import numpy as np
from skimage import data
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr

np.random.seed(42)
n = 128; SIG = 0.1
x = resize(data.camera(), (n, n))
y = x + SIG * np.random.randn(n, n)

def soft_thresh(v, l):
    return np.sign(v) * np.maximum(np.abs(v) - l, 0)

# --- LASSO claims ---
lam_l = 0.02
xl = soft_thresh(y, lam_l)
print("x range", x.min(), x.max())
print("y range", y.min(), y.max())
print("LASSO zeroed pixels (==0):", np.sum(xl == 0), "/", xl.size, "=", 100*np.mean(xl==0), "%")
print("pixels with |y|<0.02:", 100*np.mean(np.abs(y) < 0.02), "%")
print("LASSO==y? (unchanged where>|lam|) frac shifted down:", 100*np.mean(xl != y), "%")
# is LASSO basically y minus a constant shift?
diff = y - xl
print("mean shift (y - lasso):", diff.mean(), "max shift:", diff.max())
print("PSNR lasso", psnr(x, xl, data_range=1.0), "PSNR y", psnr(x, y, data_range=1.0))

# --- Tikhonov ---
lam_t = 0.01
xt = y/(1+lam_t)
print("Tikhonov PSNR", psnr(x, xt, data_range=1.0))

# --- TV curve peak scan (coarse) ---
import importlib.util
spec = importlib.util.spec_from_file_location("m", "2.2-4.py")
# instead re-implement Chambolle from the file by exec-ing only the function
src = open("2.2-4.py", encoding="utf-8").read()
ns = {}
exec(src[src.index("def tv_denoise_chambolle"):src.index("# ═")], {"np": np}, ns)
tvd = ns["tv_denoise_chambolle"]
best = (-1, None)
for lam in [0.02,0.05,0.08,0.1,0.12,0.15,0.2,0.3,0.5]:
    p = psnr(x, tvd(y, lam, n_iter=300), data_range=1.0)
    print(f"TV lam={lam:.3f} PSNR={p:.3f}")
    if p > best[0]: best = (p, lam)
print("best TV (300 iters):", best)
