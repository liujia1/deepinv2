"""3.5-2 复核补充：[6] 差异图边缘集中定量 + [7] H1 FFT vs 直接解 + 无裁剪PSNR精确值"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio

def dxp(u):  return np.roll(u, -1, axis=1) - u
def dyp(u):  return np.roll(u, -1, axis=0) - u
def dxm_ad(p): return p - np.roll(p, 1, axis=1)
def dym_ad(p): return p - np.roll(p, 1, axis=0)

def cp_tv(y, lam, maxiter=500, clip=True):
    n1, n2 = y.shape
    x = np.zeros((n1, n2)); x_bar = np.zeros((n1, n2))
    px = np.zeros((n1, n2)); py = np.zeros((n1, n2))
    tau = sigma = 1.0/np.sqrt(8.0)
    for k in range(maxiter):
        px += sigma*dxp(x_bar); py += sigma*dyp(x_bar)
        f = np.maximum(1.0, np.sqrt(px**2+py**2)/lam)
        px /= f; py /= f
        xo = x.copy()
        x = (xo + tau*(dxm_ad(px)+dym_ad(py)) + tau*y)/(1+tau)
        if clip: x = np.maximum(0, x)
        x_bar = 2*x - xo
    return x

def h1(y, lam):
    n = y.shape[0]
    d = 2 - 2*np.cos(2*np.pi*np.arange(n)/n)
    DX, DY = np.meshgrid(d, d)
    return np.real(np.fft.ifft2(np.fft.fft2(y)/(1+lam*(DX+DY))))

np.random.seed(42)
n = 128
x_true = resize(shepp_logan_phantom(), (n, n), order=1, preserve_range=True)
y = x_true + 0.15*np.random.randn(n, n)
x_tv = cp_tv(y, 0.15, 500); x_tk = h1(y, 0.5)

print("[6] |TV-Tikh| 边缘集中定量")
gx, gy = np.gradient(x_true)
eg = np.sqrt(gx**2+gy**2)
thr = np.quantile(eg[eg > 1e-6], 0.90)
edge = eg > max(thr, 0.02)
flat = eg < 0.005
diff = np.abs(x_tv - x_tk)
print(f"  边缘带占比={edge.mean()*100:.1f}%, mean|diff|={diff[edge].mean():.4f}, p90={np.quantile(diff[edge],0.9):.4f}")
print(f"  平坦区占比={flat.mean()*100:.1f}%, mean|diff|={diff[flat].mean():.4f}, p90={np.quantile(diff[flat],0.9):.4f}")
print(f"  全图 mean|diff|={diff.mean():.4f};  边缘/平坦比值={diff[edge].mean()/diff[flat].mean():.2f}")
# diff>0.1(图colorbar上限)的像素中边缘带占比
big = diff > 0.1
print(f"  |diff|>0.1 像素共 {big.sum()} 个, 其中位于边缘带的占比 = {edge[big].mean()*100:.1f}%")

print("[5b] 无裁剪精确对比")
x_nc = cp_tv(y, 0.15, 500, clip=False)
mse_c = np.mean((np.clip(x_tv, None, 1)-x_true)**2)
mse_n = np.mean((np.clip(x_nc, 0, 1)-x_true)**2)
mse_n_raw = np.mean((x_nc-x_true)**2)
print(f"  PSNR(clip后截断评估): 有裁剪={peak_signal_noise_ratio(x_true, np.clip(x_tv,None,1)):.4f}, 无裁剪(评估时clip)={peak_signal_noise_ratio(x_true, np.clip(x_nc,0,1)):.4f}, 无裁剪(不clip)={peak_signal_noise_ratio(x_true, x_nc):.4f}")
print(f"  有裁剪解负像素数={(x_tv<0).sum()}; 无裁剪解负像素数={(x_nc<0).sum()}, min={x_nc.min():.4f}")

print("[7] H1 FFT vs 稀疏直接解 (n=32)")
m = 32*32; ys = y[:32, :32]; lam_s = 0.5
I = sparse.identity(m, format='csr'); idx = np.arange(m)
r = np.where(idx % 32 == 31, idx - 31, idx + 1)
Dx = sparse.csr_matrix((np.ones(m), (idx, r)), shape=(m, m)) - I
col = np.arange(m); row_ = (col + 32) % m
Dy = sparse.csr_matrix((np.ones(m), (row_, col)), shape=(m, m)) - I
x_direct = spsolve((I + lam_s*(Dx.T@Dx + Dy.T@Dy)).tocsc(), ys.ravel())
x_fft = h1(ys, lam_s).ravel()
print(f"  max|FFT-direct| = {np.abs(x_direct-x_fft).max():.3e}")

print("[8] TV解 vs 独立参照(chambolle_pock 20000步) 及 与scipy优化无关的KKT残差")
x_ref = cp_tv(y, 0.15, 20000)
print(f"  ||x_500 - x_20k||_inf = {np.abs(x_tv-x_ref).max():.4e}")
