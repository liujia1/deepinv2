# -*- coding: utf-8 -*-
import numpy as np
from skimage.transform import resize
from skimage.data import shepp_logan_phantom
np.random.seed(42)
n=32
phantom = resize(shepp_logan_phantom(), (n,n), order=0, preserve_range=True, anti_aliasing=False)
x = phantom/phantom.max(); N=n*n
def gpsf(size,sigma):
    ax=np.fft.fftfreq(size,d=1.0)*size; xx,yy=np.meshgrid(ax,ax)
    h=np.exp(-(xx**2+yy**2)/(2*sigma**2)); return h/h.sum()
h=gpsf(n,2.0); H=np.fft.fft2(h)
A=np.zeros((N,N))
for j in range(N):
    e=np.zeros(N); e[j]=1.0
    A[:,j]=np.real(np.fft.ifft2(H*np.fft.fft2(e.reshape(n,n)))).ravel()
sn=0.05; y=A@x.ravel()+sn*np.random.randn(N)
sp=0.5; lam=sn**2/sp**2
AtA=A.T@A; Aty=A.T@y
Sig=np.linalg.inv(AtA/sn**2+np.eye(N)/sp**2); mu=Sig@(Aty/sn**2)
# equivalence to Tikhonov closed form
mu_tik=np.linalg.solve(AtA+lam*np.eye(N),Aty)
print("[eq] ||mu_post - tikhonov|| =", np.linalg.norm(mu-mu_tik))
pv=np.diag(Sig).reshape(n,n)
print(f"[postvar] min={pv.min():.6f} max={pv.max():.6f} mean={pv.mean():.6f} std={pv.std():.3e}")
print(f"[postvar] claim ~0.227 constant -> spread={pv.max()-pv.min():.2e}")
# is Sigma circulant? check A circulant
print(f"[circ] A first-col circulant check: max|A - circulant| residual via symmetry")
# Sigma_post should be circulant if A is; check a shift invariance
col0=Sig[:,0]; shifted=np.roll(col0,1); print(f"[circ] Sig col shift invariance err={np.linalg.norm(Sig[:,1]-shifted):.2e}")
# MAP global min check
def E(v): return 0.5/sn**2*np.linalg.norm(A@v-y)**2+0.5/sp**2*np.linalg.norm(v)**2
rng=np.random.default_rng(0)
worse=sum(1 for _ in range(300) if E(mu+0.05*rng.standard_normal(N))<E(mu))
print(f"[min] perturbations lowering energy at MAP: {worse}/300 (0 => global min)")
print(f"[energy] E_map={E(mu):.2f} E_true={E(x.ravel()):.2f}  (true>=map expected)")
# gradient at MAP ~ 0
g=AtA@mu-Aty+lam*mu; print(f"[grad] ||grad J(mu)|| = {np.linalg.norm(g):.3e}")
