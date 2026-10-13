# -*- coding: utf-8 -*-
import numpy as np

def legacy(seed):
    np.random.seed(seed)
    n, m = 50, 40
    A = np.random.randn(m, n)/np.sqrt(m)
    x_true = np.zeros(n); x_true[5]=1.0; x_true[15]=0.8; x_true[30]=-0.5
    sigma2 = 0.01
    y = A@x_true + np.sqrt(sigma2)*np.random.randn(m)
    return A, x_true, y, sigma2

def log_ml(y, A, alpha, sigma2):
    m = A.shape[0]
    C = sigma2*np.eye(m) + alpha*(A@A.T)
    sign, logdet = np.linalg.slogdet(C)
    return -0.5*m*np.log(2*np.pi) - 0.5*logdet - 0.5*y@np.linalg.solve(C, y)

def tik(y, A, alpha, sigma2):
    return np.linalg.solve(A.T@A/sigma2 + np.eye(A.shape[1])/alpha, A.T@y/sigma2)

peak = 1.0
def psnr(y,A,a,x_true,sigma2):
    xh = tik(y,A,a,sigma2); return 10*np.log10(peak**2/np.mean((xh-x_true)**2))

# ---- seed 42 on 400-grid + continuous refinement ----
A,x_true,y,sigma2 = legacy(42)
g400 = np.logspace(-4,2,400)
lml = np.array([log_ml(y,A,a,sigma2) for a in g400])
psn = np.array([psnr(y,A,a,x_true,sigma2) for a in g400])
ae400=g400[np.argmax(lml)]; ao400=g400[np.argmax(psn)]
print("400-grid: alpha_eb=%.5f alpha_opt=%.5f  same=%s" % (ae400, ao400, ae400==ao400))
print("400-grid: log_ml_max=%.4f  psnr_eb=%.4f psnr_opt=%.4f" % (lml.max(), psn[np.argmax(lml)], psn.max()))

fine = np.logspace(-4,2,20000)
lf = np.array([log_ml(y,A,a,sigma2) for a in fine])
pf = np.array([psnr(y,A,a,x_true,sigma2) for a in fine])
print("continuous: alpha_eb*=%.5f (logml=%.4f) alpha_opt*=%.5f (psnr=%.4f) rel_gap=%.3f%%" % (
    fine[np.argmax(lf)], lf.max(), fine[np.argmax(pf)], pf.max(),
    100*(fine[np.argmax(pf)]/fine[np.argmax(lf)]-1)))

# ---- multi-seed robustness (400-grid) ----
seeds = list(range(42,72))
eb_list, op_list = [], []
for s in seeds:
    A,x_true,y,sigma2 = legacy(s)
    lml = np.array([log_ml(y,A,a,sigma2) for a in g400])
    psn = np.array([psnr(y,A,a,x_true,sigma2) for a in g400])
    eb_list.append(g400[np.argmax(lml)]); op_list.append(g400[np.argmax(psn)])
eb=np.array(eb_list); op=np.array(op_list)
ratio = eb/op
print("\nmulti-seed (n=%d):" % len(seeds))
print("  median alpha_eb=%.4f median alpha_opt=%.4f" % (np.median(eb), np.median(op)))
print("  within 2x: %d/%d ; within 1.5x: %d/%d ; within 1.2x: %d/%d" % (
    np.sum((ratio<2)&(ratio>0.5)), len(seeds),
    np.sum((ratio<1.5)&(ratio>1/1.5)), len(seeds),
    np.sum((ratio<1.2)&(ratio>1/1.2)), len(seeds)))
print("  pearson corr(log10 eb, log10 opt)=%.3f" % np.corrcoef(np.log10(eb),np.log10(op))[0,1])
print("  mean |log10 ratio|=%.3f (decades)" % np.mean(np.abs(np.log10(ratio))))
# PSNR penalty: how much worse is EB's psnr vs oracle psnr
pen=[]
for s in seeds:
    A,x_true,y,sigma2 = legacy(s)
    lml = np.array([log_ml(y,A,a,sigma2) for a in g400])
    psn = np.array([psnr(y,A,a,x_true,sigma2) for a in g400])
    ae=g400[np.argmax(lml)]; 
    pe=10*np.log10(1/np.mean((tik(y,A,ae,sigma2)-x_true)**2))
    pen.append(psn.max()-pe)
pen=np.array(pen)
print("  PSNR(oracle)-PSNR(EB) dB gap: mean=%.3f max=%.3f (>=0 means EB slightly worse)" % (pen.mean(), pen.max()))
