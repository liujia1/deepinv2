# -*- coding: utf-8 -*-
"""Reproduce & claim-check for 实验3.3-1 (A=I Tikhonov denoise)."""
import numpy as np
from skimage import data
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio

np.random.seed(42)
n=128
x=resize(data.camera(),(n,n))
x_std=np.std(x); DATA_RANGE=x.max()-x.min()
sn=0.1
y=x+sn*np.random.randn(n,n)
mu=float(np.mean(y))
y_clipped=np.clip(y,0,1)
print(f"[std] x_std={x_std:.4f} (doc 0.2815)  mu={mu:.4f} (json 0.5063)")

lam_bayes=sn**2/x_std**2
x_bayes=mu+(y-mu)/(1+lam_bayes)
psnr_noisy=peak_signal_noise_ratio(x,y_clipped,data_range=DATA_RANGE)
psnr_bayes=peak_signal_noise_ratio(x,x_bayes,data_range=DATA_RANGE)
y_std=np.std(y_clipped); sigma_x_est=np.sqrt(max(y_std**2-sn**2,1e-8))
lam_bayes_noisy=sn**2/sigma_x_est**2
x_bayes_noisy=mu+(y-mu)/(1+lam_bayes_noisy)
psnr_bayes_noisy=peak_signal_noise_ratio(x,x_bayes_noisy,data_range=DATA_RANGE)
print(f"[lam] bayes={lam_bayes:.4f}(0.1262) noisy={lam_bayes_noisy:.4f}(0.1327)")
print(f"[psnr] noisy={psnr_noisy:.2f}(20.12) bayes={psnr_bayes:.2f}(20.22) bayes_noisy={psnr_bayes_noisy:.2f}(20.22)")

# grid scan (same as script)
lambdas=np.logspace(-4,1,50)
psnr_list=np.array([peak_signal_noise_ratio(x,mu+(y-mu)/(1+l),data_range=DATA_RANGE) for l in lambdas])
bi=int(np.argmax(psnr_list))
print(f"[grid] lam_best={lambdas[bi]:.4f}(0.1151) psnr_best={psnr_list[bi]:.2f}(20.22)")
# fine scan to see true optimum vs bayes
fine=np.logspace(-2,0,2000)
pf=np.array([peak_signal_noise_ratio(x,mu+(y-mu)/(1+l),data_range=DATA_RANGE) for l in fine])
print(f"[fine] true argmax lam={fine[int(np.argmax(pf))]:.4f} psnr={pf.max():.4f}; psnr@bayes={peak_signal_noise_ratio(x,mu+(y-mu)/(1+lam_bayes),data_range=DATA_RANGE):.4f}")

# counterfactual mu=0 (doc claim: optimal lam ~0.03, ~4x off, psnr below noisy)
def best_mu0():
    pl=np.array([peak_signal_noise_ratio(x,np.clip(y/(1+l),0,1),data_range=DATA_RANGE) for l in lambdas])
    i=int(np.argmax(pl)); return lambdas[i],pl[i]
l0,p0=best_mu0()
E_x2=x_std**2+mu**2
print(f"[mu=0] optimal lam={l0:.4f} psnr={p0:.2f}; sigma^2/E[x^2]={sn**2/E_x2:.4f}; psnr_noisy={psnr_noisy:.2f}")
print(f"[mu=0] ratio lam_bayes/lam_mu0 = {lam_bayes/l0:.2f}  (doc ~4x)")
print(f"[mu=0] psnr below noisy? {p0 < psnr_noisy}")
