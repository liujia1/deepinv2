# -*- coding: utf-8 -*-
"""Non-destructive reproduction & claim-check for 实验3.2-1."""
import numpy as np
from skimage.transform import resize
from skimage.data import shepp_logan_phantom
from skimage.metrics import peak_signal_noise_ratio

np.random.seed(42)
n=32; N=n*n
phantom=resize(shepp_logan_phantom(),(n,n),order=0,preserve_range=True,anti_aliasing=False)
x=phantom/phantom.max()
def gpsf(size,sigma):
    ax=np.concatenate((np.arange(0,size//2),np.arange(-size//2,0)))
    xx,yy=np.meshgrid(ax,ax); h=np.exp(-(xx**2+yy**2)/(2*sigma**2)); return h/h.sum()
h=gpsf(n,3.0); H_fft=np.fft.fft2(h)
def A_op(v): return np.real(np.fft.ifft2(H_fft*np.fft.fft2(v.reshape(n,n)))).ravel()
def AT_op(v): return np.real(np.fft.ifft2(np.conj(H_fft)*np.fft.fft2(v.reshape(n,n)))).ravel()
sn=0.05; y=A_op(x.ravel())+sn*np.random.randn(N)
lam=0.01
def objective(v,l): return 0.5*np.linalg.norm(A_op(v)-y)**2+0.5*l*np.linalg.norm(v)**2
def gradient(v,l): return AT_op(A_op(v)-y)+l*v

# power iteration L
rng=np.random.default_rng(seed=0); v=rng.standard_normal(N); v/=np.linalg.norm(v)
for _ in range(50):
    v=AT_op(A_op(v)); v/=np.linalg.norm(v)
A_norm_sq=float(np.dot(A_op(v),A_op(v))); L=A_norm_sq+lam
L_exact=float(np.max(np.abs(H_fft)**2)+lam)
# smallest eigenvalue of A^T A (spectrum exact for circulant) -> mu
spec=np.abs(H_fft)**2
lam_min=float(spec.min()); mu=lam_min+lam
kappa=L/mu
print(f"[L] power-iter L={L:.4f} exact={L_exact:.4f} (json 1.01)")
print(f"[kappa] mu=lambda_min(A^TA)+lam={mu:.6f}  kappa=L/mu={kappa:.2f}  (claim ~101)")

# closed form J*
Yf=np.fft.fft2(y.reshape(n,n)); Xf=np.conj(H_fft)*Yf/(np.abs(H_fft)**2+lam)
x_star=np.real(np.fft.ifft2(Xf)).ravel(); J_star=objective(x_star,lam)
print(f"[J*] {J_star:.4f} (json 1.3163)")

def gd(x0,l,tau,ni):
    xx=x0.copy(); hist=np.zeros(ni)
    for k in range(ni):
        xx=xx-tau*gradient(xx,l); val=objective(xx,l); hist[k]=val
        if not np.isfinite(val): hist[k:]=np.nan; break
    return xx,hist

x_init=np.zeros(N); J_init=objective(x_init,lam)
print(f"[J_init] {J_init:.4f} (claim ~11.9)")
taus=[('1/L',1/L),('0.5/L',0.5/L),('1.8/L',1.8/L),('2.1/L',2.1/L)]
for lab,tau in taus:
    ni=50 if lab=='2.1/L' else 1000
    xo,h=gd(x_init,lam,tau,ni)
    fin=h[np.isfinite(h)]
    if lab!='2.1/L':
        rises=int(np.sum(np.diff(fin)>0))
        gap_i=J_init-J_star; thr=gap_i*1e-3
        idx=np.where((h-J_star)<thr)[0]
        it=int(idx[0]+1) if len(idx) else None
        ps=peak_signal_noise_ratio(x,np.clip(xo.reshape(n,n),0,1))
        print(f"[{lab}] tau={tau:.6f} final={fin[-1]:.4f} psnr={ps:.2f} rises_in_J={rises}/{len(fin)-1} iters_to_thresh={it}")
    else:
        print(f"[{lab}] tau={tau:.6f} final={fin[-1]:.4f} (diverge) converged={np.all(np.isfinite(xo))}")
