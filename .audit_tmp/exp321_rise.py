# -*- coding: utf-8 -*-
import numpy as np
from skimage.transform import resize
from skimage.data import shepp_logan_phantom
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
sn=0.05; y=A_op(x.ravel())+sn*np.random.randn(N); lam=0.01
def objective(v,l): return 0.5*np.linalg.norm(A_op(v)-y)**2+0.5*l*np.linalg.norm(v)**2
Yf=np.fft.fft2(y.reshape(n,n)); Xf=np.conj(H_fft)*Yf/(np.abs(H_fft)**2+lam)
x_star=np.real(np.fft.ifft2(Xf)).ravel(); J_star=objective(x_star,lam)
L=1.01
for tau_mult in [1.0,1.8]:
    tau=tau_mult/L; xx=np.zeros(N); hist=[]
    for k in range(1000):
        xx=xx-tau*(AT_op(A_op(xx)-y)+lam*xx); hist.append(objective(xx,lam))
    hist=np.array(hist); gap=hist-J_star; rises=np.where(np.diff(gap)>0)[0]
    if len(rises):
        print(f"tau={tau_mult}/L: rises={len(rises)}  max_gap_at_rise={gap[rises].max():.2e}  "
              f"rises_with_gap>=1e-10: {int(np.sum(gap[rises]>=1e-10))}  first_rise_iter={int(rises[0])}")
    else:
        print(f"tau={tau_mult}/L: rises=0  final_gap={gap[-1]:.2e}")
