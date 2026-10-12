"""实验3.5-2 复核脚本：
1) 差分算子伴随性数值验证（dxp/dxm_ad 是否为转置对）
2) seed=42 复现 PSNR（对照讲义表 16.45 / 22.91 / 26.57）
3) 收敛性：500 vs 5000 步，目标值差异 + 与参考解距离
4) 公平性：TV / H1 各自 λ 扫描取最优 PSNR 对比
5) 非负裁剪 max(0,x) 的影响（有/无裁剪对比）
6) 差异图"边缘集中"定量验证（边缘带 vs 平坦区平均差异）
7) H1 DFT闭式解 vs 直接稀疏求解（周期性验证）
"""
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

def chambolle_pock_tv_denoise(y, lam, maxiter=500, clip=True):
    n1, n2 = y.shape
    x = np.zeros((n1, n2)); x_bar = np.zeros((n1, n2))
    px = np.zeros((n1, n2)); py = np.zeros((n1, n2))
    L2 = 8.0; sigma = 1.0/np.sqrt(L2); tau = 1.0/np.sqrt(L2)
    for k in range(maxiter):
        px = px + sigma*dxp(x_bar); py = py + sigma*dyp(x_bar)
        norm_p = np.sqrt(px**2 + py**2); factor = np.maximum(1.0, norm_p/lam)
        px = px/factor; py = py/factor
        x_old = x.copy()
        div_p = dxm_ad(px) + dym_ad(py)
        x = (x_old + tau*div_p + tau*y)/(1+tau)
        if clip: x = np.maximum(0, x)
        x_bar = 2*x - x_old
    return x

def h1_tikhonov_denoise(y, lam):
    n = y.shape[0]
    d_hat = 2 - 2*np.cos(2*np.pi*np.arange(n)/n)
    DX, DY = np.meshgrid(d_hat, d_hat)
    X_hat = np.fft.fft2(y)/(1 + lam*(DX+DY))
    return np.real(np.fft.ifft2(X_hat))

def tv_obj(x, y, lam):
    gx, gy = dxp(x), dyp(x)
    return 0.5*np.sum((x-y)**2) + lam*np.sum(np.sqrt(gx**2+gy**2+1e-12))

# ---------- 1) 伴随性检查 ----------
print("="*70); print("[1] 差分算子伴随性检查 (小尺寸随机)")
rng = np.random.default_rng(0)
u = rng.normal(size=(9, 7)); p1 = rng.normal(size=(9, 7)); p2 = rng.normal(size=(9, 7))
lhs = np.sum(dxp(u)*p1) + np.sum(dyp(u)*p2)      # <Du, p>
rhs = -np.sum(u*dxm_ad(p1)) - np.sum(u*dym_ad(p2))  # -<u, div p>
print(f"  <Du,p> = {lhs:.10f},  -<u,div p> = {rhs:.10f},  相对误差 = {abs(lhs-rhs)/max(1,abs(lhs)):.2e}")
# 周期差分的 D^T D 由 FFT 对角化, 特征值 = (2-2cos θ_x)+(2-2cos θ_y), 上界直接算
n_ = 32
d_hat = 2 - 2*np.cos(2*np.pi*np.arange(n_)/n_)
DXe, DYe = np.meshgrid(d_hat, d_hat)
print(f"  周期差分单方向最大特征值={d_hat.max():.3f} (理论4), ||D||^2 理论={(DXe+DYe).max():.1f} (代码取 L2=8)")

# ---------- 数据 ----------
np.random.seed(42)
n = 128
x_true = resize(shepp_logan_phantom(), (n, n), order=1, preserve_range=True)
sigma_n = 0.15
y = x_true + sigma_n*np.random.randn(n, n)

# ---------- 2) 复现 PSNR ----------
print("="*70); print("[2] seed=42 复现 (500步, lam_tv=0.15, lam_tikh=0.5)")
x_tv = chambolle_pock_tv_denoise(y, 0.15, 500)
x_tk = h1_tikhonov_denoise(y, 0.5)
p_noisy = peak_signal_noise_ratio(x_true, y)
p_tk = peak_signal_noise_ratio(x_true, np.clip(x_tk, 0, 1))
p_tv = peak_signal_noise_ratio(x_true, np.clip(x_tv, None, 1))
print(f"  noisy={p_noisy:.2f} (讲义16.45)  tikh={p_tk:.2f} (讲义22.91)  tv={p_tv:.2f} (讲义26.57)")

# ---------- 3) 收敛性 ----------
print("="*70); print("[3] 收敛性: 500 vs 5000 vs 20000 步")
x_tv_5k = chambolle_pock_tv_denoise(y, 0.15, 5000)
x_tv_20k = chambolle_pock_tv_denoise(y, 0.15, 20000)
print(f"  F(x_500)={tv_obj(x_tv,y,0.15):.6f}  F(x_5k)={tv_obj(x_tv_5k,y,0.15):.6f}  F(x_20k)={tv_obj(x_tv_20k,y,0.15):.6f}")
print(f"  ||x_500-x_20k||/||x_20k|| = {np.linalg.norm(x_tv-x_tv_20k)/np.linalg.norm(x_tv_20k):.3e}")
print(f"  ||x_5k -x_20k||/||x_20k|| = {np.linalg.norm(x_tv_5k-x_tv_20k)/np.linalg.norm(x_tv_20k):.3e}")
print(f"  PSNR: 500步={p_tv:.2f}  5k步={peak_signal_noise_ratio(x_true, np.clip(x_tv_5k,None,1)):.2f}  20k步={peak_signal_noise_ratio(x_true, np.clip(x_tv_20k,None,1)):.2f}")

# ---------- 4) λ 扫描公平性 ----------
print("="*70); print("[4] λ 扫描 (TV用2000步)")
print("  TV:")
best_tv = (None, -1)
for lam in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40]:
    xt = chambolle_pock_tv_denoise(y, lam, 2000)
    p = peak_signal_noise_ratio(x_true, np.clip(xt, None, 1))
    print(f"    lam={lam:.2f}: PSNR={p:.2f}")
    if p > best_tv[1]: best_tv = (lam, p)
print("  H1-Tikhonov:")
best_tk = (None, -1)
for lam in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
    xt = h1_tikhonov_denoise(y, lam)
    p = peak_signal_noise_ratio(x_true, np.clip(xt, 0, 1))
    print(f"    lam={lam:.2f}: PSNR={p:.2f}")
    if p > best_tk[1]: best_tk = (lam, p)
print(f"  最优: TV lam={best_tv[0]} PSNR={best_tv[1]:.2f} | H1 lam={best_tk[0]} PSNR={best_tk[1]:.2f}")

# ---------- 5) 非负裁剪影响 ----------
print("="*70); print("[5] 无裁剪 vs 有裁剪 (500步)")
x_nc = chambolle_pock_tv_denoise(y, 0.15, 500, clip=False)
print(f"  无裁剪 PSNR={peak_signal_noise_ratio(x_true, np.clip(x_nc,0,1)):.2f}, min={x_nc.min():.4f}, 负像素占比={(x_nc<0).mean()*100:.2f}%")
print(f"  有/无裁剪最大差值 = {np.abs(x_nc - x_tv).max():.4e}")

# ---------- 6) 差异图边缘集中 ----------
print("="*70); print("[6] |TV-Tikh| 边缘集中定量")
gx, gy = np.gradient(x_true)
egmag = np.sqrt(gx**2+gy**2)
thr = np.quantile(egmag[egmag > 1e-6], 0.90)
edge = emag > max(thr, 0.02)
flat = emag < 0.005
diff = np.abs(x_tv - x_tk)
print(f"  边缘带像素占比={edge.mean()*100:.1f}%, 平均|diff|={diff[edge].mean():.4f}")
print(f"  平坦区占比={flat.mean()*100:.1f}%, 平均|diff|={diff[flat].mean():.4f}")
print(f"  全图平均|diff|={diff.mean():.4f}")
print(f"  边缘/平坦 比值 = {diff[edge].mean()/diff[flat].mean():.2f}")

# ---------- 7) H1 FFT vs 直接解 (周期性交叉验证) ----------
print("="*70); print("[7] H1闭式解 vs 稀疏直接解 (n=32子图)")
ys = y[:32, :32]; lam_s = 0.5
m = 32*32
I = sparse.identity(m, format='csr')
Dx = sparse.csr_matrix((m, m)); Dy = sparse.csr_matrix((m, m))
idx = np.arange(m)
# dxp: 行内 x_{j+1}-x_j, 行末回绕到行首 (周期), 不能跨行
r = np.where(idx % 32 == 31, idx - 31, idx + 1)
Dx = sparse.csr_matrix((np.ones(m), (idx, r)), shape=(m, m)) - I
col = np.arange(m); row_ = (col + 32) % m
Dy = sparse.csr_matrix((np.ones(m), (row_, col)), shape=(m, m)) - I
DtD = (Dx.T@Dx + Dy.T@Dy).tocsc()
x_direct = spsolve((I + lam_s*DtD).tocsc(), ys.ravel())
x_fft = h1_tikhonov_denoise(ys, lam_s).ravel()
print(f"  max|FFT - direct| = {np.abs(x_direct - x_fft).max():.3e}")
print("="*70); print("完成")
