# -*- coding: utf-8 -*-
import numpy as np
beta_min, beta_max = 0.1, 20.0
def vp_beta(t): return beta_min + t*(beta_max-beta_min)
def vp_marginal_params(t):
    log_mean = -0.25*t**2*(beta_max-beta_min) - 0.5*t*beta_min
    mc = np.exp(log_mean); return mc, np.sqrt(max(1-mc**2,0))
gmm_weights=[0.4,0.3,0.3]
gmm_means=[np.array([-2.,0.]),np.array([2.,1.]),np.array([1.,-2.])]
gmm_covs=[np.array([[0.5,0.1],[0.1,0.5]]),np.array([[0.8,-0.2],[-0.2,0.6]]),np.array([[0.6,0.],[0.,0.4]])]
def gmm_2d_score_fast(X,Y,weights,means,covs):
    pts=np.stack([X.ravel(),Y.ravel()],axis=1); pdf=np.zeros(len(pts)); grad=np.zeros((len(pts),2))
    for w,m,c in zip(weights,means,covs):
        diff=pts-m; ci=np.linalg.inv(c); maha=np.einsum('ni,ij,nj->n',diff,ci,diff)
        g=w*np.exp(-0.5*maha)/(2*np.pi*np.sqrt(np.linalg.det(c))); pdf+=g; grad+=g[:,None]*(-diff@ci.T)
    s=grad/(pdf[:,None]+1e-30); return s[:,0].reshape(X.shape),s[:,1].reshape(Y.shape)

probe = np.array([[4.,4.],[-4.,4.],[4.,-4.],[-4.,-4.],[0.,4.],[4.,0.],[2.,0.],[0.,0.]])
for t in [0.1,0.4,0.7,1.0]:
    mc,st = vp_marginal_params(t); b=vp_beta(t)
    X=probe[:,0]; Y=probe[:,1]
    sx,sy = gmm_2d_score_fast(X,Y,gmm_weights,[mc*m for m in gmm_means],[mc**2*c+st**2*np.eye(2) for c in gmm_covs])
    dx = b/2*X + b*sx; dy = b/2*Y + b*sy
    print(f"t={t}: beta={b:.2f} mc={mc:.3f}")
    for (px,py),ex,ey in zip(probe,dx,dy):
        # does drift point toward origin/data? dot with -position
        inward = -(px*ex+py*ey)  # >0 means drift points toward origin
        print(f"   pt=({px:5.1f},{py:5.1f}) drift=({ex:8.2f},{ey:8.2f})  toward-origin={inward:9.2f}")
    print()
