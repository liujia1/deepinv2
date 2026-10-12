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

alphas = np.logspace(-4, 2, 100)
fine = np.logspace(-4, 2, 1500)
peak = 1.0

def curves(A, x_true, y, sigma2, grid):
    lml = np.array([log_ml(y,A,a,sigma2) for a in grid])
    psn = np.array([10*np.log10(peak**2/np.mean((tik(y,A,a,sigma2)-x_true)**2)) for a in grid])
    return lml, psn

print("=== legacy seed=42 (exact script path) ===")
A, x_true, y, sigma2 = legacy(42)
lml, psn = curves(A,x_true,y,sigma2,alphas)
i_eb=int(np.argmax(lml)); i_op=int(np.argmax(psn))
print(f"grid idx EB={i_eb} a={alphas[i_eb]:.5f}; idx opt={i_op} a={alphas[i_op]:.5f}; same={i_eb==i_op}")
lf, pf = curves(A,x_true,y,sigma2,fine)
print(f"continuous EB a={fine[np.argmax(lf)]:.5f}; continuous PSNR-opt a={fine[np.argmax(pf)]:.5f}")
d=np.sign(np.diff(lml)); print("log_ml unimodal (0 sign changes):", int(np.sum(d[1:]*d[:-1]<0)))
print(f"PSNR at EB grid={psn[i_eb]:.3f}; PSNR max={psn.max():.3f}")

print("\n=== multi-seed (legacy) ===")
print("seed EB_a opt_a same_grid cont_EB cont_opt")
for s in range(42,52):
    A,x_true,y,sigma2 = legacy(s)
    lml,psn = curves(A,x_true,y,sigma2,alphas)
    lf,pf = curves(A,x_true,y,sigma2,fine)
    ae=alphas[np.argmax(lml)]; ao=alphas[np.argmax(psn)]
    ce=fine[np.argmax(lf)]; co=fine[np.argmax(pf)]
    print(f"{s:3d} {ae:.5f} {ao:.5f} {str(ae==ao):5s} {ce:.5f} {co:.5f}")
