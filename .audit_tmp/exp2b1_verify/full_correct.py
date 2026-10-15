import numpy as np, time
from skimage import data
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def grad_fwd(x):
    dh=np.zeros_like(x); dv=np.zeros_like(x)
    dh[:,:-1]=x[:,1:]-x[:,:-1]; dv[:-1,:]=x[1:,:]-x[:-1,:]; return dh,dv
def sym_grad_fwd(w1,w2):
    E1=np.zeros_like(w1); E2=np.zeros_like(w2)
    E1[:,:-1]=w1[:,1:]-w1[:,:-1]; E2[:-1,:]=w2[1:,:]-w2[:-1,:]
    a=np.zeros_like(w2); b=np.zeros_like(w1)
    a[:,:-1]=w2[:,1:]-w2[:,:-1]; b[:-1,:]=w1[1:,:]-w1[:-1,:]
    return E1,E2,0.5*(a+b)
def _h(p):
    o=np.zeros(p.shape); o[:,:-1]+=p[:,:-1]; o[:,1:]-=p[:,:-1]; return o
def _v(p):
    o=np.zeros(p.shape); o[:-1,:]+=p[:-1,:]; o[1:,:]-=p[:-1,:]; return o
def div_true(p,q): return _h(p)+_v(q)
def sdiv_true(q1,q2,q3): return _h(q1)+0.5*_v(q3), _v(q2)+0.5*_h(q3)

def tgv_pd(y,a0,a1,n_iter=1000,tau=0.25,sigma=0.25):
    h,w=y.shape
    x=y.copy(); w1=np.zeros((h,w)); w2=np.zeros((h,w))
    p1=np.zeros((h,w)); p2=np.zeros((h,w)); q1=np.zeros((h,w)); q2=np.zeros((h,w)); q3=np.zeros((h,w))
    xb=x.copy(); w1b=w1.copy(); w2b=w2.copy()
    for k in range(n_iter):
        Dx=grad_fwd(xb)
        p1+=sigma*(Dx[0]-w1b); p2+=sigma*(Dx[1]-w2b)
        nr=np.sqrt(p1**2+p2**2); p1/=np.maximum(1,nr/a1); p2/=np.maximum(1,nr/a1)
        E=sym_grad_fwd(w1b,w2b)
        q1+=sigma*E[0]; q2+=sigma*E[1]; q3+=sigma*E[2]
        nq=np.sqrt(q1**2+q2**2+2*q3**2); q1/=np.maximum(1,nq/a0); q2/=np.maximum(1,nq/a0); q3/=np.maximum(1,nq/a0)
        dp=div_true(p1,p2); sq=sdiv_true(q1,q2,q3)
        xn=(x+tau*dp+tau*y)/(1+tau); w1n=w1+tau*(sq[0]+p1); w2n=w2+tau*(sq[1]+p2)
        xb=xn+(xn-x); w1b=w1n+(w1n-w1); w2b=w2n+(w2n-w2); x,w1,w2=xn,w1n,w2n
    return x,w1,w2

# ===== 独立交叉验证: 32x32 用 scipy 最小化真实 TGV 目标, 对比 CP 解 =====
from scipy.optimize import minimize
m=32; cam32=resize(data.camera(),(m,m)); np.random.seed(42)
y32=random_noise(cam32,mode='gaussian',var=0.01)
a0,a1=0.3,0.15
def obj(z):
    x=z[:m*m].reshape(m,m); u=z[m*m:2*m*m].reshape(m,m); v=z[2*m*m:].reshape(m,m)
    Dh,Dv=grad_fwd(x); E=sym_grad_fwd(u,v)
    return 0.5*np.sum((x-y32)**2)+a1*np.sum(np.sqrt(Dh**2+Dv**2+1e-12))+a0*np.sum(np.sqrt(E[0]**2+E[1]**2+2*E[2]**2+1e-12))
z0=np.concatenate([y32.reshape(-1),np.zeros(2*m*m)])
r=minimize(obj,z0,method='L-BFGS-B',options=dict(maxiter=50000,ftol=1e-15,gtol=1e-12))
x_ref=r.x[:m*m].reshape(m,m); obj_ref=obj(r.x)
x_cp,w1c,w2c=tgv_pd(y32,a0,a1,n_iter=20000)
obj_cp=obj(np.concatenate([x_cp.reshape(-1),w1c.reshape(-1),w2c.reshape(-1)]))
print(f"[交叉验证 32x32] scipy obj={obj_ref:.6f}  CP obj={obj_cp:.6f}  差={abs(obj_cp-obj_ref):.2e}")
print(f"  rel||x_cp-x_ref||={np.linalg.norm(x_cp-x_ref)/np.linalg.norm(x_ref):.4f}  PSNR cp={psnr(cam32,x_cp):.2f} ref={psnr(cam32,x_ref):.2f}")

# ===== 主实验 128x128 =====
n=128; cam=resize(data.camera(),(n,n)); np.random.seed(42)
y=random_noise(cam,mode='gaussian',var=0.01)
t0=time.time(); x_tv=denoise_tv_chambolle(y,weight=0.2); t_tv=time.time()-t0
t0=time.time(); x_tgv,w1,w2=tgv_pd(y,0.3,0.15,n_iter=500); t_tgv=time.time()-t0
print(f"\n[主实验 128] 含噪 PSNR={psnr(cam,y):.2f} SSIM={ssim(cam,y,data_range=1):.4f}")
print(f"  TV  PSNR={psnr(cam,x_tv):.2f} SSIM={ssim(cam,x_tv,data_range=1):.4f}  用时{t_tv:.3f}s")
print(f"  TGV PSNR={psnr(cam,x_tgv):.2f} SSIM={ssim(cam,x_tgv,data_range=1):.4f}  用时{t_tgv:.3f}s  (TGV/TV耗时比={t_tgv/t_tv:.1f}x)")

# ===== 参数敏感性 =====
print("\n[敏感性 α1=0.15]  ratio: PSNR/SSIM")
for ratio in [0.5,1.0,2.0,4.0,8.0]:
    xx,_,_=tgv_pd(y,ratio*0.15,0.15,n_iter=500)
    print(f"  {ratio:4.1f}: {psnr(cam,xx):.2f}/{ssim(cam,xx,data_range=1):.4f}")

# ===== α0→∞ 是否退化为 TV(α1) =====
print("\n[退化检验] α0→大, 对比 TV(weight=α1=0.15):")
x_tv15=denoise_tv_chambolle(y,weight=0.15)
for big in [3.0,30.0,300.0]:
    xa,_,_=tgv_pd(y,big,0.15,n_iter=2000)
    print(f"  α0={big:6.1f}: rel||x-TVa1||={np.linalg.norm(xa-x_tv15)/np.linalg.norm(x_tv15):.4f} PSNR={psnr(cam,xa):.2f}")

# ===== 阶梯效应核心指标: 渐变区域 =====
row=n//2
def plateau_and_mse(img,sl):
    prof=img[row,sl]; d1=np.diff(prof)
    return np.mean(np.abs(d1)<1e-3), np.mean((prof-camera_ref[row,sl])**2)
camera_ref=cam
sl=slice(32,64)
print("\n[阶梯效应 渐变区32-64] 平台占比 / 该区MSE:")
for nm,img in [("TV",x_tv),("TGV",x_tgv)]:
    pf,me=plateau_and_mse(img,sl); print(f"  {nm}: 平台={pf:.3f} MSE={me:.5f}")
