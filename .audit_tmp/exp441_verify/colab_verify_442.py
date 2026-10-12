"""
实验4.4-2  【Colab 复核脚本】SAPG 经验贝叶斯（自包含，粘贴进 GPU 单元格运行）
512×512 camera，均匀模糊9×9，BSNR=30。忠实复刻 4.4-2.py 的 prox/TVnorm/SAPG/MAP。
验证三件事：
  [A] 多 seed 稳健性：θ_est / σ²_est / rel_err / snr_map 的分布 → "σ²误差3%" 是否 seed=42 幸运值
  [B] σ² 初值敏感性：σ²_init ∈ {1, 真值, 20} → 是否真能"从数据推出"σ²，还是只停在真值初值附近
  [C] TVnorm 转置怪癖影响：θ-Fisher 项用 lib(带.T) vs 正确各向同性 TV → θ_est 差多少
注：本脚本重实现的全局 RNG 流与原库(max_eigenval 的 torch.normal)不同，θ_est 不会逐位=0.0231；
    验证的是"行为/分布"，不是逐位复现。原实现代码本身已确认无 θ² 双重计入(与修好的4.4-1一致)。
"""
import numpy as np, torch, math
torch.set_default_dtype(torch.float32)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", dev)
from skimage.data import camera

x_true = torch.tensor(camera().astype(np.float64), device=dev)
nx, ny = x_true.shape; dimx = nx*ny
print("image:", x_true.shape, "range", float(x_true.min()), float(x_true.max()))

def make_blur(klen, size):
    nxx, nyy = size; h = torch.zeros(nxx, nyy, device=dev); lx, ly = klen
    h[0:lx,0:ly] = 1.0/(lx*ly)
    c = np.ceil((np.array([ly,lx])-1)/2).astype("int64")
    H = torch.fft.fft2(torch.roll(h, shifts=(-c[0],-c[1]), dims=(0,1))); HC = torch.conj(H)
    A = lambda x: torch.fft.ifft2(H*torch.fft.fft2(x)).real
    AT = lambda x: torch.fft.ifft2(HC*torch.fft.fft2(x)).real
    v = torch.randn(nxx,nyy,device=dev); v=v/torch.linalg.norm(v.flatten()); prev=1.0
    for _ in range(10000):
        v = AT(A(v)); val = torch.linalg.norm(v.flatten()).item()
        if abs(val-prev)/prev < 1e-4: break
        prev=val; v=v/val
    return A, AT, val
A, AT, AAT = make_blur([9,9], (nx,ny))

def _grad(u):
    z=u[1:,:]-u[:-1,:]; dux=torch.vstack([z,torch.zeros(1,u.shape[1],device=dev)])
    z=u[:,1:]-u[:,:-1]; duy=torch.hstack([z,torch.zeros(u.shape[0],1,device=dev)])
    return dux,duy
def _div(p1,p2):
    z=p2[:,1:-1]-p2[:,:-2]; v=torch.hstack([p2[:,:1],z,-p2[:,-2:-1]])
    z=p1[1:-1,:]-p1[:-2,:]; u=torch.vstack([p1[:1,:],z,-p1[-2:-1,:]])
    return v+u
def prox_tv(g, lambd, maxiter=10):
    px=torch.zeros_like(g); py=torch.zeros_like(g); t=0.249
    for _ in range(maxiter):
        u=_div(px,py)-g/lambd; upx,upy=_grad(u); tmp=torch.sqrt(upx*upx+upy*upy)
        px=(px+t*upx)/(1+t*tmp); py=(py+t*upy)/(1+t*tmp)
    return g - lambd*_div(px,py)
gradg = lambda x,lam,lp: (x - prox_tv(x, lam))/lp

def tvnorm_lib(x):   # 复刻 Grad_Image.TVnorm（diffv 带 .T 转置怪癖）
    dh=torch.zeros_like(x); dh[:,:-1]=x[:,1:]-x[:,:-1]
    dv=torch.zeros_like(x); dv[:-1,:]=x[1:,:]-x[:-1,:]; dv=dv.T
    return torch.sum(torch.sqrt(dh*dh+dv*dv))
def tvnorm_correct(x):
    dh=torch.zeros_like(x); dh[:,:-1]=x[:,1:]-x[:,:-1]
    dv=torch.zeros_like(x); dv[:-1,:]=x[1:,:]-x[:-1,:]
    return torch.sum(torch.sqrt(dh*dh+dv*dv))

f_norm = lambda x: torch.linalg.matrix_norm(y - A(x), ord='fro')**2
def run_sapg(seed, s2_init_mode, tvfn, warm=1000, total=1000, burn=700):
    torch.manual_seed(seed)
    y = Ax + sigma*torch.randn_like(x_true)   # y 用真 sigma（与 s2_init 无关）
    globals()['y'] = y
    lam_p = 1.0; th_init=0.01; alpha=1.0
    s2_true = sigma2_true
    s2_init = {'true':s2_true, 'low':1.0, 'high':20.0}[s2_init_mode]
    c_eta, c_s2, d_exp = 10, 10000, 0.8
    dstep = lambda i: ((i+1)**(-d_exp))/dimx
    # warmup（固定 θ=th_init, σ²=s2_init）
    gamma = 0.98/(AAT/s2_init + 1.0/lam_p)
    X = y.clone(); lwu = lam_p*th_init
    for k in range(warm):
        gf = AT(A(X)-y)/s2_init; gg = gradg(X, lwu, lam_p)
        X = X - gamma*gf - gamma*gg + math.sqrt(2*gamma)*torch.randn_like(X)
    eta = math.log(th_init); theta = th_init; s2 = s2_init
    tl=[theta]; sl=[s2]
    for k in range(total):
        g = 0.98/(AAT/s2 + 1.0/lam_p)
        gg = gradg(X, lam_p*theta, lam_p); gf = AT(A(X)-y)/s2
        Z = torch.randn_like(X); X = X - g*gf - g*gg + math.sqrt(2*g)*Z
        gx = tvfn(X).item(); r2 = (f_norm(X)/(2*s2*s2)).item()
        gth = -gx + dimx/(alpha*theta); geta = gth*theta
        eta = max(math.log(0.001), min(math.log(1.0), eta + c_eta*dstep(k)*geta)); theta=math.exp(eta)
        gs2 = r2 - dimx/(2*s2)
        s2 = max(0.1, min(100.0, s2 + c_s2*dstep(k)*gs2))
        tl.append(theta); sl.append(s2)
    th = float(np.mean(tl[burn:])); s2e = float(np.mean(sl[burn:]))
    # MAP（800步：200步实测未收敛，梯度范数约10，800步降至约2且SNR进入平台）
    xm = y.clone(); lm=lam_p*th; gm=0.98/(AAT/s2e + 1.0/lam_p)
    for _ in range(800):
        xm = xm - gm*(AT(A(xm)-y)/s2e + gradg(xm, lm, lam_p))
    snr = lambda e: 10*math.log10((x_true**2).sum().item()/((x_true-e)**2).sum().item())
    return th, s2e, abs(s2e-s2_true)/s2_true*100, snr(y), snr(xm)

Ax = A(x_true)
sigma2_true = (torch.linalg.matrix_norm(Ax-Ax.mean(),ord='fro')**2/(dimx*10**(30/10))).item()
sigma = math.sqrt(sigma2_true)
print(f"true sigma2={sigma2_true:.4f}")

print("\n[A] 多 seed（σ²_init=真值, tv=correct【2026-10修复后口径】）  seed | theta   sigma2  rel_err%  snr_obs snr_map")
for s in [42,1,7,13,2024]:
    th,s2e,re,so,sm = run_sapg(s,'true',tvnorm_correct)
    print(f"                     {s:5d} | {th:.4f} {s2e:7.3f}  {re:5.1f}%   {so:5.2f} {sm:6.2f}")

print("\n[B] σ² 初值敏感性（seed=42, tv=correct）  init | theta  sigma2_est  rel_err%")
for m in ['low','true','high']:
    th,s2e,re,so,sm = run_sapg(42,m,tvnorm_correct)
    print(f"                     {m:5s} | {th:.4f}  {s2e:7.3f}   {re:5.1f}%")

print("\n[C] TVnorm 转置怪癖（seed=42, init=true）  tv | theta_est  snr_map")
for nm,fn in [('lib(.T)',tvnorm_lib), ('correct',tvnorm_correct)]:
    th,s2e,re,so,sm = run_sapg(42,'true',fn)
    print(f"                     {nm:8s} | {th:.4f}   {sm:6.2f}")

print(f"\n判读：")
print(f" - [A] 若 rel_err 明显>3%（如 8~15%）→ '3%' 是 seed=42 幸运值，讲义应给区间/多seed均值")
print(f" - [B] 若 init=1/20 也收敛到≈{sigma2_true:.2f} → 经验贝叶斯确实'从数据推出'σ²（非停在真值init）")
print(f" - [C] 若 lib 与 correct 的 θ_est 差异大 → TVnorm 转置怪癖实质影响 θ 估计，应统一或注明")
