"""
实验4.4-1 Colab 复核脚本（自包含，可直接粘贴到 Colab 单元格运行）
目的：在 GPU 上快速完成 CPU 上因资源争抢而跑不动的三项关键验证
  1) β 因子核查：代码 grad_prior = beta*(x-P_{λβ}(x))/λ 是否多乘了一个 β
     - variant 'code' : 复现现有 4.4-1.py（含可疑的额外 β）
     - variant 'fixA' : 正确的 ∇(βTV)_λ = (x-P_{λβ}(x))/λ（去掉外层 β）
     对比二者 PSNR，判断该因子是否影响结果与"β=0.3"标注的自洽性
  2) 多 seed 方差：MMSE>单样本 是否为稳定结论，还是单次幸运抽样
  3) burn-in 充分性：30k 长链，比较 4-10k / 10-20k / 20-30k 三个窗口的均值与 PSNR
"""
import numpy as np, torch, math, time
torch.set_default_dtype(torch.float32)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", dev)

from skimage.data import camera
from skimage.transform import resize

# ---------- 正算子（均匀模糊，FFT） ----------
def make_blur(kernel_len, size, dev):
    nx, ny = size
    h = torch.zeros(nx, ny, device=dev)
    lx, ly = kernel_len
    h[0:lx, 0:ly] = 1.0/(lx*ly)
    c = np.ceil((np.array([ly, lx]) - 1)/2).astype("int64")
    H = torch.fft.fft2(torch.roll(h, shifts=(-c[0], -c[1]), dims=(0, 1)))
    HC = torch.conj(H)
    A = lambda x: torch.fft.ifft2(H*torch.fft.fft2(x)).real
    AT = lambda x: torch.fft.ifft2(HC*torch.fft.fft2(x)).real
    # 幂迭代求 ||A^T A||
    x = torch.randn(nx, ny, device=dev); x = x/torch.linalg.norm(x.flatten())
    prev = 1.0
    for _ in range(10000):
        x = AT(A(x)); val = torch.linalg.norm(x.flatten()).item()
        if abs(val-prev)/prev < 1e-4: break
        prev = val; x = x/val
    return A, AT, val

# ---------- Chambolle prox_{τ TV}（含负伴随修正） ----------
def _grad(u):
    z = u[1:, :] - u[:-1, :]
    dux = torch.vstack([z, torch.zeros(1, u.shape[1], device=u.device)])
    z = u[:, 1:] - u[:, :-1]
    duy = torch.hstack([z, torch.zeros(u.shape[0], 1, device=u.device)])
    return dux, duy
def _div(p1, p2):
    z = p2[:, 1:-1] - p2[:, :-2]
    v = torch.hstack([p2[:, :1], z, -p2[:, -2:-1]])
    z = p1[1:-1, :] - p1[:-2, :]
    u = torch.vstack([p1[:1, :], z, -p1[-2:-1, :]])
    return v + u
def prox_tv(g, tau, maxiter=10):
    px = torch.zeros_like(g); py = torch.zeros_like(g); tau_c = 0.249
    for _ in range(maxiter):
        u = _div(px, py) - g/tau
        upx, upy = _grad(u)
        tmp = torch.sqrt(upx*upx + upy*upy)
        px = (px + tau_c*upx)/(1 + tau_c*tmp)
        py = (py + tau_c*upy)/(1 + tau_c*tmp)
    return g - tau*_div(px, py)

class Welford:
    def __init__(s, x): s.k = 1; s.M = x.clone(); s.S = torch.zeros_like(x)
    def update(s, x):
        s.k += 1; Mn = s.M + (x - s.M)/s.k
        s.S += (x - s.M)*(x - Mn); s.M = Mn
    def mean(s): return s.M

def psnr(ref, est, mv=255.0):
    mse = torch.mean((ref-est)**2).item()
    return 10*math.log10(mv**2/mse) if mse > 0 else float('inf')

# ---------- 数据与超参（对齐 4.4-1.py） ----------
u = resize(camera().astype(np.float64), (128, 128), anti_aliasing=True)
x_true = torch.tensor(u, device=dev)
nx, ny = x_true.shape; dimx = nx*ny
A, AT, AAT = make_blur([9, 9], (nx, ny), dev)
Ax = A(x_true); BSNR = 20
sigma = torch.linalg.norm(Ax - Ax.mean())/math.sqrt(dimx*10**(BSNR/10))
sigma2 = (sigma**2).item()
beta, lam = 0.3, 1.0
L_f = AAT/sigma2; L_g = beta/lam
gamma = 0.98/(L_f + L_g)
print(f"sigma={sigma.item():.4f} AAT={AAT:.4f} L_f={L_f:.4f} gamma={gamma:.4f}")

def myula(seed, n_iter, burn, variant, prox_iter=10, collect_windows=None):
    torch.manual_seed(seed)
    y = Ax + sigma*torch.randn_like(x_true)
    X = y.clone(); W = None; curve = []
    for i in range(n_iter):
        P = prox_tv(X, lam*beta, prox_iter)
        if variant == 'code':      # 现有代码：外层再乘 beta
            gp = beta*(X - P)/lam
        elif variant == 'fixA':    # 正确：∇(beta TV)_lambda
            gp = (X - P)/lam
        X = X - gamma*(AT(A(X) - y)/sigma2 + gp) + math.sqrt(2*gamma)*torch.randn_like(X)
        if i >= burn:
            if W is None: W = Welford(X)
            else: W.update(X)
        if collect_windows and (i+1) in collect_windows:
            Xm = W.mean(); curve.append((i+1, psnr(x_true, Xm), torch.sqrt(torch.mean((Xm-x_true)**2)).item()))
    Xm = W.mean()
    return psnr(x_true, y), psnr(x_true, Xm), psnr(x_true, X), curve

print("\n[1] β 因子核查（seed=42, 10k）")
for v in ['code', 'fixA']:
    po, pm, pl, _ = myula(42, 10000, 4000, v)
    print(f"  {v:5s}: psnr_obs={po:.2f} MMSE={pm:.2f} single={pl:.2f}")

print("\n[2] 多 seed 方差（variant=code, 10k）")
pm_list, pl_list = [], []
for s in [42, 1, 7, 13, 2024]:
    po, pm, pl, _ = myula(s, 10000, 4000, 'code')
    pm_list.append(pm); pl_list.append(pl)
    print(f"  seed={s:5d}: MMSE={pm:.2f} single={pl:.2f} MMSE>single={pm>pl}")
pm_a, pl_a = np.mean(pm_list), np.mean(pl_list)
print(f"  >> MMSE {pm_a:.2f}±{np.std(pm_list):.2f} | single {pl_a:.2f}±{np.std(pl_list):.2f} | "
      f"MMSE 更高占比 {sum(p>q for p,q in zip(pm_list,pl_list))}/5")

print("\n[3] burn-in 充分性（seed=42, 30k, 窗口 PSNR）")
_, _, _, cw = myula(42, 30000, 4000, 'code', collect_windows={10000, 20000, 30000})
for k, p, r in cw:
    print(f"  截至 {k:5d}: 运行均值 PSNR={p:.2f} RMSE={r:.3f}")
