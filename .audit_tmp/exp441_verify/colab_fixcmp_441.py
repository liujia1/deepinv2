"""
实验4.4-1  Colab 修复对比脚本（自包含，粘贴进 Colab GPU 单元格运行）
公平比较三种 MYULA 先验实现，各自使用【自己正确的稳定步长】：
  code   : grad=β·(X-P_{λβ})/λ,  L_g=β/λ,  γ=0.98/(L_f+β/λ)   ← 现有实现（β 重复计入，有效权重≈β²）
  schemeB: grad=β·(X-P_{λ })/λ,  L_g=β/λ,  γ=0.98/(L_f+β/λ)   ← 最小修复：近端 τ 由 λβ 改成 λ（目标=exp(-f-βTV)）
  schemeA: grad=  (X-P_{λβ})/λ,  L_g=1/λ,  γ=0.98/(L_f+1/λ)   ← 正统方案：去掉外层 β（目标=exp(-f-βTV)）
关注：改成"正确 β=0.3"后 PSNR/平滑度怎么变，讲义结论是否仍成立。
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
    def __init__(s, x): s.k = 1; s.M = x.clone()
    def update(s, x): s.k += 1; s.M = s.M + (x-s.M)/s.k
    def mean(s): return s.M

def psnr(ref, est, mv=255.0):
    mse = torch.mean((ref-est)**2).item(); return 10*math.log10(mv**2/mse) if mse > 0 else float('inf')

u = resize(camera().astype(np.float64), (128, 128), anti_aliasing=True)
x_true = torch.tensor(u, device=dev); nx, ny = x_true.shape; dimx = nx*ny
A, AT, AAT = make_blur([9, 9], (nx, ny), dev)
Ax = A(x_true); BSNR = 20
sigma = torch.linalg.norm(Ax-Ax.mean())/math.sqrt(dimx*10**(BSNR/10)); sigma2 = (sigma**2).item()
beta, lam = 0.3, 1.0; L_f = AAT/sigma2
gB = 0.98/(L_f+beta/lam); gA = 0.98/(L_f+1.0/lam)
print(f"sigma={sigma.item():.4f} L_f={L_f:.4f}  γ(code/B)={gB:.4f}  γ(schemeA)={gA:.4f}")

def myula(seed, gamma, tau, outer, n=10000, burn=4000):
    torch.manual_seed(seed)
    y = Ax + sigma*torch.randn_like(x_true); X = y.clone(); W = None
    for i in range(n):
        P = prox_tv(X, tau, 10); gp = outer*(X-P)/lam
        X = X - gamma*(AT(A(X)-y)/sigma2 + gp) + math.sqrt(2*gamma)*torch.randn_like(X)
        if i >= burn:
            if W is None: W = Welford(X)
            else: W.update(X)
    Xm = W.mean(); return psnr(x_true, y), psnr(x_true, Xm), psnr(x_true, X), Xm, y

variants = {
    'code':    dict(tau=lam*beta, outer=beta, gamma=gB),
    'schemeB': dict(tau=lam,      outer=beta, gamma=gB),
    'schemeA': dict(tau=lam*beta, outer=1.0,  gamma=gA),
}
means = {}
print("\n变体            seed | psnr_obs  MMSE   single | TV(均值)")
for name, v in variants.items():
    pm_l, pl_l, tv_l = [], [], []
    for s in [42, 1, 7]:
        po, pm, pl, Xm, y = myula(s, v['gamma'], v['tau'], v['outer'])
        pm_l.append(pm); pl_l.append(pl)
        dux, duy = _grad(Xm); tv_l.append(torch.sqrt(dux*dux+duy*duy).sum().item())
        if s == 42: means[name] = (Xm, y)
    print(f"{name:9s}  mean |  {po:.2f}   {np.mean(pm_l):.2f}   {np.mean(pl_l):.2f} | {np.mean(tv_l):.0f}"
          f"   (MMSE±std {np.mean(pm_l):.2f}±{np.std(pm_l):.2f})")

fig, ax = plt.subplots(2, 3, figsize=(15, 9))
ax[0, 0].imshow(x_true.cpu().numpy(), cmap='gray', vmin=0, vmax=255); ax[0, 0].set_title('ground truth')
ax[0, 1].imshow(means['code'][1].cpu().numpy(), cmap='gray', vmin=0, vmax=255); ax[0, 1].set_title('observation y')
for j, name in enumerate(['code', 'schemeB', 'schemeA']):
    r, c = 1, j
    if name == 'code': r, c = 1, 0
    elif name == 'schemeB': r, c = 1, 1
    else: r, c = 1, 2
    po, pm, pl, Xm, _ = None, None, None, means[name][0], None
    ax[r, c].imshow(Xm.cpu().numpy(), cmap='gray', vmin=0, vmax=255); ax[r, c].set_title(f'{name} posterior mean')
for a in ax.ravel(): a.axis('off')
plt.tight_layout(); plt.show()
print("\n提示：code 有效先验≈β²=0.09（TV 更小、更欠平滑）；schemeA/B 用正确 β=0.3（TV 更大、更平滑）。")
