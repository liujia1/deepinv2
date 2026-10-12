# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(42)
n = 64
sigma = np.exp(-5*np.arange(n)/(n-1))
x_true = np.zeros(n); x_true[0:10]=1.0; x_true[15:30]=0.5; x_true[40:55]=0.8
sigma_noise = 0.05
y = sigma*x_true + sigma_noise*np.random.randn(n)
tau=1.0
maxiter=2000
xk=np.zeros(n)
errs=np.zeros(maxiter+1); res=np.zeros(maxiter+1)
errs[0]=np.linalg.norm(xk-x_true); res[0]=np.linalg.norm(sigma*xk-y)
for k in range(1,maxiter+1):
    xk=xk-tau*sigma*(sigma*xk-y)
    errs[k]=np.linalg.norm(xk-x_true); res[k]=np.linalg.norm(sigma*xk-y)
k_opt=int(np.argmin(errs))
print("k_opt",k_opt,"err_min",errs[k_opt])
print("residual at k_opt =",res[k_opt])
print("noise level sigma*sqrt(n) =",sigma_noise*np.sqrt(n))
print("residual when first <= noise:", np.argmax(res<=sigma_noise*np.sqrt(n)))
# LS asymptotic error
x_ls=y/sigma
print("||x_LS - x_true|| (asymptotic)=",np.linalg.norm(x_ls-x_true))
print("||noise/sigma||=",np.linalg.norm(sigma_noise*np.random.randn(n)/sigma))
# Picard: fraction of i where |a_i|*sigma_i > noise
a=x_true
mask=np.abs(a)*sigma>sigma_noise
print("Picard recoverable idx (|a_i|sigma_i>noise):",np.where(mask)[0])
print("count",mask.sum(),"frac",mask.sum()/n)
# threshold formula
print("i_threshold formula",int(-(n-1)/5*np.log(sigma_noise)))
# k_opt vs noise
for nl in [1e-2,2e-2,5e-2,1e-1,2e-1]:
    np.random.seed(42)
    yn=sigma*x_true+nl*np.random.randn(n)
    xkk=np.zeros(n); e=np.zeros(maxiter+1); e[0]=np.linalg.norm(x_true)
    for k in range(1,maxiter+1):
        xkk=xkk-tau*sigma*(sigma*xkk-yn); e[k]=np.linalg.norm(xkk-x_true)
    print("noise",nl,"k_opt",int(np.argmin(e)),"err",e.min(),"still decreasing?",e[-1]<e[-2])
