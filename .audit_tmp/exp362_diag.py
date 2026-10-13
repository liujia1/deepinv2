# -*- coding: utf-8 -*-
"""独立复核 3.6-2：复现 JSON 全部字段（含连续细分与30种子稳健性）"""
import numpy as np
np.random.seed(42)
n, m = 50, 40
A = np.random.randn(m, n)/np.sqrt(m)
x_true = np.zeros(n); x_true[5]=1.0; x_true[15]=0.8; x_true[30]=-0.5
sigma2 = 0.01
y = A@x_true + np.sqrt(sigma2)*np.random.randn(m)
peak = np.max(np.abs(x_true))

def ml(y,A,a,s2):
    C=s2*np.eye(m)+a*A@A.T; _,ld=np.linalg.slogdet(C); Ci=np.linalg.solve(C,y)
    return -0.5*m*np.log(2*np.pi)-0.5*ld-0.5*y@Ci
def tik(y,A,a,s2): return np.linalg.solve(A.T@A/s2+np.eye(n)/a, A.T@y/s2)
def psnr(y,A,a,xt,s2,pk): return 10*np.log10(pk**2/np.mean((tik(y,A,a,s2)-xt)**2))
def refine(fun,c,lo=1/4,hi=4,N=4000):
    loc=np.logspace(np.log10(c*lo),np.log10(c*hi),N); v=np.array([fun(a) for a in loc]); return loc[v.argmax()],v.max()

alphas=np.logspace(-4,2,400)
lm=np.array([ml(y,A,a,sigma2) for a in alphas]); ae=alphas[lm.argmax()]
aec,mlmax=refine(lambda a:ml(y,A,a,sigma2),ae)
ps=np.array([psnr(y,A,a,x_true,sigma2,peak) for a in alphas]); ao=alphas[ps.argmax()]
aoc,popt=refine(lambda a:psnr(y,A,a,x_true,sigma2,peak),ao)
peb=psnr(y,A,ae,x_true,sigma2,peak)
print("=== single seed42 ===")
print(f"alpha_eb_grid={ae:.4f}(J0.036) alpha_opt_grid={ao:.4f}(J0.0373)")
print(f"alpha_eb_cont={aec:.5f}(J0.03557) alpha_opt_cont={aoc:.5f}(J0.03695) relgap={100*(aoc/aec-1):.2f}%(J3.88)")
print(f"log_ml_max={mlmax:.2f}(J7.46) psnr_eb={peb:.3f}(J17.529) psnr_opt={popt:.3f}(J17.529) gap={popt-peb:.4f}(J0.0002)")

seeds=list(range(42,72)); eb=[];opt=[];gap=[]
for s in seeds:
    np.random.seed(s); As=np.random.randn(m,n)/np.sqrt(m)
    ys=As@x_true+np.sqrt(sigma2)*np.random.randn(m)
    lms=np.array([ml(ys,As,a,sigma2) for a in alphas]); psn=np.array([psnr(ys,As,a,x_true,sigma2,peak) for a in alphas])
    a_e=alphas[lms.argmax()]; eb.append(a_e); opt.append(alphas[psn.argmax()])
    gap.append(psn.max()-psnr(ys,As,a_e,x_true,sigma2,peak))
eb=np.array(eb);opt=np.array(opt);gap=np.array(gap);ratio=eb/opt
w2=int(np.sum((ratio<2)&(ratio>0.5))); c=np.corrcoef(np.log10(eb),np.log10(opt))[0,1]
print("=== multiseed 30 ===")
print(f"median_eb={np.median(eb):.4f}(J0.0393) median_opt={np.median(opt):.4f}(J0.0393)")
print(f"within2x={w2}(J24) corr={c:.3f}(J-0.508) gap_mean={gap.mean():.3f}(J0.133) gap_max={gap.max():.3f}(J1.235)")

aC=np.logspace(-4,2,100); lmC=np.array([ml(y,A,a,sigma2) for a in aC]); psC=np.array([psnr(y,A,a,x_true,sigma2,peak) for a in aC])
print("=== coarse 100-grid reproduce original coincidence ===")
print(f"100pt: alpha_eb={aC[lmC.argmax()]:.4f} alpha_opt={aC[psC.argmax()]:.4f} same={aC[lmC.argmax()]==aC[psC.argmax()]} step_ratio={aC[1]/aC[0]:.3f}x")
