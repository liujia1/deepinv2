import numpy as np, time
N = 4; S = N*N

def grad_fwd(x):
    dh = np.zeros_like(x); dv = np.zeros_like(x)
    dh[:, :-1] = x[:, 1:] - x[:, :-1]
    dv[:-1, :] = x[1:, :] - x[:-1, :]
    return dh, dv
def sym_grad_fwd(w1, w2):
    E1 = np.zeros_like(w1); E2 = np.zeros_like(w2)
    E1[:, :-1] = w1[:, 1:] - w1[:, :-1]
    E2[:-1, :] = w2[1:, :] - w2[:-1, :]
    a = np.zeros_like(w2); b = np.zeros_like(w1)
    a[:, :-1] = w2[:, 1:] - w2[:, :-1]
    b[:-1, :] = w1[1:, :] - w1[:-1, :]
    return E1, E2, 0.5*(a + b)
def basis(k):
    e = np.zeros(S); e[k] = 1.0; return e.reshape(N, N)
Dm = np.zeros((2*S, S)); Em = np.zeros((3*S, 2*S))
for k in range(S):
    dh, dv = grad_fwd(basis(k)); Dm[:S,k]=dh.reshape(-1); Dm[S:,k]=dv.reshape(-1)
    r = sym_grad_fwd(basis(k), np.zeros((N,N))); Em[:S,k]=r[0].reshape(-1); Em[S:2*S,k]=r[1].reshape(-1); Em[2*S:,k]=r[2].reshape(-1)
    r = sym_grad_fwd(np.zeros((N,N)), basis(k)); Em[:S,S+k]=r[0].reshape(-1); Em[S:2*S,S+k]=r[1].reshape(-1); Em[2*S:,S+k]=r[2].reshape(-1)

# 正确的 -D^T / -E^T
def _mDT_h(p):
    out = np.zeros(p.shape); out[:, :-1] += p[:, :-1]; out[:, 1:] -= p[:, :-1]; return out
def _mDT_v(p):
    out = np.zeros(p.shape); out[:-1, :] += p[:-1, :]; out[1:, :] -= p[:-1, :]; return out
def div_true(p, q):  return _mDT_h(p) + _mDT_v(q)
def sdiv_true(q1, q2, q3):
    return _mDT_h(q1) + 0.5*_mDT_v(q3), _mDT_v(q2) + 0.5*_mDT_h(q3)

def mat_of_div(f):
    M = np.zeros((S, 2*S))
    for k in range(S):
        M[:S,k]=f(basis(k),np.zeros((N,N))).reshape(-1); M[:S,S+k]=f(np.zeros((N,N)),basis(k)).reshape(-1)
    return M
def mat_of_sdiv(f):
    M = np.zeros((2*S, 3*S))
    for c in range(3):
        for k in range(S):
            qs=[np.zeros((N,N)) for _ in range(3)]; qs[c]=basis(k); r=f(*qs)
            M[:S,c*S+k]=r[0].reshape(-1); M[S:,c*S+k]=r[1].reshape(-1)
    return M
print("正确div:  ||DV + D^T|| =", np.abs(mat_of_div(div_true)+Dm.T).max())
print("正确sdiv: ||SD + E^T|| =", np.abs(mat_of_sdiv(sdiv_true)+Em.T).max())

# ============ 用正确算子跑 CP, 检查收敛 ============
def tgv_pd(y, a0, a1, n_iter=1000, tau=0.25, sigma=0.25, trace_every=0):
    h,w = y.shape
    x = y.copy(); w1=np.zeros((h,w)); w2=np.zeros((h,w))
    p1=np.zeros((h,w)); p2=np.zeros((h,w)); q1=np.zeros((h,w)); q2=np.zeros((h,w)); q3=np.zeros((h,w))
    xb=x.copy(); w1b=w1.copy(); w2b=w2.copy(); tr=[]
    for k in range(n_iter):
        Dx=grad_fwd(xb)
        p1+=sigma*(Dx[0]-w1b); p2+=sigma*(Dx[1]-w2b)
        nrm=np.sqrt(p1**2+p2**2); p1/=np.maximum(1,nrm/a1); p2/=np.maximum(1,nrm/a1)
        E=sym_grad_fwd(w1b,w2b)
        q1+=sigma*E[0]; q2+=sigma*E[1]; q3+=sigma*E[2]
        nq=np.sqrt(q1**2+q2**2+2*q3**2); q1/=np.maximum(1,nq/a0); q2/=np.maximum(1,nq/a0); q3/=np.maximum(1,nq/a0)
        dp=div_true(p1,p2); sq=sdiv_true(q1,q2,q3)
        xn=(x + tau*dp + tau*y)/(1+tau)   # x-part: x - tau*D^T p = x + tau*div_true
        w1n=w1 + tau*(sq[0]+p1); w2n=w2 + tau*(sq[1]+p2)   # w-part: w - tau*(-p+E^T q)=w+tau p+tau sdiv
        xb=xn+(xn-x); w1b=w1n+(w1n-w1); w2b=w2n+(w2n-w2)
        x,w1,w2=xn,w1n,w2n
        if trace_every and (k+1)%trace_every==0:
            Dx=grad_fwd(x); E=sym_grad_fwd(w1,w2)
            obj=0.5*np.sum((x-y)**2)+a1*np.sum(np.sqrt(Dx[0]**2+Dx[1]**2+1e-24))+a0*np.sum(np.sqrt(E[0]**2+E[1]**2+2*E[2]**2+1e-24))
            tr.append((k+1,obj))
    return x,w1,w2,tr

from skimage import data
from skimage.util import random_noise
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
n=128; cam=resize(data.camera(),(n,n)); np.random.seed(42)
y=random_noise(cam,mode='gaussian',var=0.01)
x,w1,w2,tr=tgv_pd(y,0.3,0.15,n_iter=1000,trace_every=100)
print("目标函数轨迹:", ", ".join(f"{k}:{o:.1f}" for k,o in tr))
print(f"正确TGV: PSNR={psnr(cam,x):.2f} SSIM={ssim(cam,x,data_range=1):.4f}")
