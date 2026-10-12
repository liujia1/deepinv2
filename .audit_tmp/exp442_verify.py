import sys, os, math, time
import numpy as np, torch
sys.path.insert(0, r"d:/works/deepinv/第4章 MCMC与ULA算法/第四章 配套实验/实验4.4-2")
from skimage.data import camera
from sampling_tools import blur_operators, chambolle_prox_TV, TVnorm

device = torch.device('cpu')

def run(seed=None, tag=''):
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.Tensor(camera().astype(np.float64)); nx, ny = x.shape; dimx = nx*ny
    A, AT, nrm = blur_operators([9,9],(nx,ny),'uniform',device)
    Ax = A(x); sigma = torch.linalg.matrix_norm(Ax-torch.mean(Ax),ord='fro')/math.sqrt(dimx*10**3)
    y = Ax + sigma*torch.randn_like(x); s2_true = float(sigma**2)
    lp = 1.0
    proxg = lambda z,lam: chambolle_prox_TV(z,device,{'lambda':lam,'MaxIter':10})
    gradg = lambda z,lam,lprox:(z-proxg(z,lam))/lprox
    grad_f_x = lambda z,s2: AT(A(z)-y)/s2
    df_dsigma2 = lambda z,s2: (torch.linalg.matrix_norm(y-A(z),ord='fro')**2)/(2*s2**2)
    theta=0.01; s2=s2_true
    Lf=nrm/s2; Lg=theta/lp; gamma=0.98/(Lf+Lg)
    X=y.clone()
    for k in range(1000):
        gf=grad_f_x(X,s2); gg=theta*gradg(X,lp*theta,lp)
        X=X-gamma*gf-gamma*gg+math.sqrt(2*gamma)*torch.randn_like(X)
    eta=math.log(theta); d_exp=0.8; c_eta=10; c_sig=10000
    mn_th,mx_th=0.001,1; mn_s,ms_s=0.1,100
    mn_eta,mx_eta=math.log(mn_th),math.log(mx_th)
    s2=s2_true; theta=math.exp(eta)
    th_list=[theta]; s2_list=[s2]
    delta=lambda i:((i+1)**(-d_exp))/dimx
    for k in range(1000):
        Lf=nrm/s2; Lg=theta/lp; gamma=0.98/(Lf+Lg)
        gg=theta*gradg(X,lp*theta,lp); gf=grad_f_x(X,s2)
        X=X-gamma*gf-gamma*gg+math.sqrt(2*gamma)*torch.randn_like(X)
        gx=TVnorm(X).item(); gs=df_dsigma2(X,s2).item()
        gt=-gx+dimx/theta; ge=gt*theta
        eta=max(mn_eta,min(mx_eta,eta+c_eta*delta(k)*ge)); theta=math.exp(eta)
        g_s2=gs-dimx/(2*s2)
        s2=max(mn_s,min(ms_s,s2+c_sig*delta(k)*g_s2))
        th_list.append(theta); s2_list.append(s2)
    tb=th_list[700:]; sb=s2_list[700:]
    te=float(np.mean(tb)); se=float(np.mean(sb))
    xm=y.clone(); lam=lp*te; Lfm=nrm/se; Lgm=te/lp; gm=0.98/(Lfm+Lgm)
    for _ in range(200):
        xm=xm-gm*(grad_f_x(xm,se)+te*gradg(xm,lam,lp))
    snr=lambda r,e:10*torch.log10(torch.sum(r**2)/torch.sum((r-e)**2)).item()
    print(f"[{tag}] seed={seed} true_s2={s2_true:.4f} theta_est={te:.5f} sigma2_est={se:.5f} "
          f"rel_err_s2={abs(se-s2_true)/s2_true*100:.1f}% snr_obs={snr(x,y):.2f} snr_map={snr(x,xm):.2f}", flush=True)
    return te,se

t0=time.time()
run(None,'noseed-1')
run(None,'noseed-2')
run(42,'seed42-a')
run(42,'seed42-b')
print('total sec', round(time.time()-t0,1))
