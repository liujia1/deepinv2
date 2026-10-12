"""
实验4.4-1  【修复版 schemeA】Colab 验证脚本（自包含，粘贴进 GPU 单元格运行）
验证内容：
  1) 修复后的 MYULA：grad_prior = (X - prox_{λβ·TV}(X))/λ （去掉外层 β），L_g=1/λ，γ=0.98/(L_f+1/λ)
  2) 步长自检：γ<2/L_total 与 γ<λ 是否都 OK
  3) 5 个 seed 的 MMSE/single 稳定性，MMSE>single 是否仍成立
  4) 复现实验的 2×3 结果图（ground truth / 观测 / 后验std / MMSE / 单样本 / 收敛曲线）
预期：psnr_obs≈21.0，MMSE≈23.5，single≈23.3（均高于观测；较修复前的虚高 24.2 回落，符合正确 β=0.3）
注：GPU 与本地 CPU 的随机数流不同，MMSE/single 会有 ±0.1dB 抖动，属正常，验证的是"行为"而非逐位复现。
"""
import numpy as np, torch, math
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float32)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", dev)
from skimage.data import camera
from skimage.transform import resize

def make_blur(klen, size, dev):
    nx, ny = size; h = torch.zeros(nx, ny, device=dev); lx, ly = klen
    h[0:lx, 0:ly] = 1.0/(lx*ly)
    c = np.ceil((np.array([ly, lx])-1)/2).astype("int64")
    H = torch.fft.fft2(torch.roll(h, shifts=(-c[0],-c[1]), dims=(0,1))); HC = torch.conj(H)
    A = lambda x: torch.fft.ifft2(H*torch.fft.fft2(x)).real
    AT = lambda x: torch.fft.ifft2(HC*torch.fft.fft2(x)).real
    x = torch.randn(nx, ny, device=dev); x = x/torch.linalg.norm(x.flatten()); prev = 1.0
    for _ in range(10000):
        x = AT(A(x)); val = torch.linalg.norm(x.flatten()).item()
        if abs(val-prev)/prev < 1e-4: break
        prev = val; x = x/val
    return A, AT, val

def _grad(u):
    z = u[1:, :]-u[:-1, :]; dux = torch.vstack([z, torch.zeros(1, u.shape[1], device=u.device)])
    z = u[:, 1:]-u[:, :-1]; duy = torch.hstack([z, torch.zeros(u.shape[0], 1, device=u.device)])
    return dux, duy
def _div(p1, p2):
    z = p2[:, 1:-1]-p2[:, :-2]; v = torch.hstack([p2[:, :1], z, -p2[:, -2:-1]])
    z = p1[1:-1, :]-p1[:-2, :]; u = torch.vstack([p1[:1, :], z, -p1[-2:-1, :]])
    return v+u
def prox_tv(g, tau, maxiter=10):
    px = torch.zeros_like(g); py = torch.zeros_like(g); t = 0.249
    for _ in range(maxiter):
        u = _div(px, py)-g/tau; upx, upy = _grad(u); tmp = torch.sqrt(upx*upx+upy*upy)
        px = (px+t*upx)/(1+t*tmp); py = (py+t*upy)/(1+t*tmp)
    return g - tau*_div(px, py)

class Welford:
    def __init__(s, x): s.k = 1; s.M = x.clone(); s.S = torch.zeros_like(x)
    def update(s, x):
        s.k += 1; Mn = s.M + (x-s.M)/s.k; s.S += (x-s.M)*(x-Mn); s.M = Mn
    def mean(s): return s.M
    def var(s): return s.S/(s.k-1) if s.k >= 2 else s.S/s.k

def psnr(ref, est, mv=255.0):
    mse = torch.mean((ref-est)**2).item(); return 10*math.log10(mv**2/mse) if mse > 0 else float('inf')

u = resize(camera().astype(np.float64), (128, 128), anti_aliasing=True)
x_true = torch.tensor(u, device=dev); nx, ny = x_true.shape; dimx = nx*ny
A, AT, AAT = make_blur([9, 9], (nx, ny), dev)
Ax = A(x_true); BSNR = 20
sigma = torch.linalg.norm(Ax-Ax.mean())/math.sqrt(dimx*10**(BSNR/10)); sigma2 = (sigma**2).item()
beta, lam = 0.3, 1.0; lam_beta = lam*beta
L_f = AAT/sigma2; L_g = 1.0/lam; L_total = L_f+L_g; gamma = 0.98/L_total   # ← 修复：L_g=1/λ
print(f"sigma={sigma.item():.4f} L_f={L_f:.4f} L_g={L_g:.4f} L_total={L_total:.4f} gamma={gamma:.4f}")
print(f"[自检] gamma<2/L_total({2/L_total:.4f}): {'OK' if gamma<2/L_total else 'FAIL'} | "
      f"gamma<lambda({lam:.2f}): {'OK' if gamma<lam else 'FAIL'}")

def myula(seed, n=10000, burn=4000):
    torch.manual_seed(seed)
    y = Ax + sigma*torch.randn_like(x_true); X = y.clone(); W = None; curve = []
    for i in range(n):
        P = prox_tv(X, lam_beta, 10)
        gp = (X - P)/lam                       # ← 修复：不乘外层 beta
        X = X - gamma*(AT(A(X)-y)/sigma2 + gp) + math.sqrt(2*gamma)*torch.randn_like(X)
        if i >= burn:
            if W is None: W = Welford(X)
            else: W.update(X)
            if (i+1) % 1000 == 0:
                curve.append((i+1, torch.sqrt(torch.mean((W.mean()-x_true)**2)).item()))
    Xm = W.mean(); Xstd = torch.sqrt(torch.clamp(W.var(), min=0))
    return psnr(x_true, y), psnr(x_true, Xm), psnr(x_true, X), Xm, X, Xstd, curve, y

print("\n[seed 扫描] seed | psnr_obs  MMSE  single | MMSE>single")
pm_l, pl_l = [], []
for s in [42, 1, 7, 13, 2024]:
    po, pm, pl, *_ = myula(s)
    pm_l.append(pm); pl_l.append(pl)
    print(f"          {s:5d} |   {po:.2f}   {pm:.2f}  {pl:.2f}  | {pm>pl}")
print(f">> MMSE {np.mean(pm_l):.2f}±{np.std(pm_l):.2f} | single {np.mean(pl_l):.2f}±{np.std(pl_l):.2f} "
      f"| MMSE>single {sum(p>q for p,q in zip(pm_l,pl_l))}/5")

po, pm, pl, Xm, Xl, Xstd, curve, y = myula(42)
print(f"\n[seed=42 出图数据] psnr_obs={po:.2f} MMSE={pm:.2f} single={pl:.2f}")
print(f"  X_mean range [{Xm.min():.1f},{Xm.max():.1f}] | X_last range [{Xl.min():.1f},{Xl.max():.1f}]")

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax[0,0].imshow(x_true.cpu().numpy(), cmap='gray', vmin=0, vmax=255); ax[0,0].set_title('Ground truth')
ax[0,1].imshow(y.cpu().numpy(), cmap='gray', vmin=0, vmax=255); ax[0,1].set_title(f'Observation (PSNR={po:.1f})')
ax[0,2].imshow(Xstd.cpu().numpy(), cmap='gray'); ax[0,2].set_title('Posterior std')
ax[1,0].imshow(Xm.cpu().numpy(), cmap='gray', vmin=0, vmax=255); ax[1,0].set_title(f'MMSE (PSNR={pm:.1f})')
ax[1,1].imshow(Xl.cpu().numpy(), cmap='gray', vmin=0, vmax=255); ax[1,1].set_title(f'Single (PSNR={pl:.1f})')
ax[1,2].plot([c[0] for c in curve], [c[1] for c in curve]); ax[1,2].set_xlabel('iter after burn-in'); ax[1,2].set_ylabel('RMSE'); ax[1,2].grid(alpha=.3)
for a in ax.ravel(): a.set_xticks([]) if a.get_images() else None
ax[0,0].axis('off'); ax[0,1].axis('off'); ax[0,2].axis('off'); ax[1,0].axis('off'); ax[1,1].axis('off')
plt.suptitle('Experiment 4.4-1 (FIXED schemeA): MYULA posterior sampling', fontsize=13)
plt.tight_layout(); plt.show()
print("\n若 MMSE≈23.5、single≈23.3、两条自检 OK、曲线 burn-in 后走平 → 修复验证通过。")
