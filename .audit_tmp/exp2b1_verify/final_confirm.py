import numpy as np
from skimage import data
from skimage.util import random_noise
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr

# ---- A. 噪声可复现性: 同一脚本内两次 np.random.seed(42)+random_noise ----
cam=resize(data.camera(),(128,128))
def make_noise():
    np.random.seed(42)
    return random_noise(cam, mode='gaussian', var=0.01)
y1=make_noise(); y2=make_noise()
print(f"[A 噪声可复现] 两次相同种子 random_noise 是否逐位相同: {np.array_equal(y1,y2)}, 最大差={np.abs(y1-y2).max():.2e}, PSNR {psnr(cam,y1):.3f}/{psnr(cam,y2):.3f}")
# 显式 rng=42
np.random.seed(42); ya=random_noise(cam,mode='gaussian',var=0.01)
try:
    yb=random_noise(cam,mode='gaussian',var=0.01,rng=42)
    print("   显式rng=42可用(skimage>=0.22):", np.array_equal(ya,yb))
except TypeError as e:
    print("   rng参数不支持:", e)

# ---- B. 用"真实TGV目标函数"评估: 原代码算子 vs 正确算子 ----
def grad_fwd(x):
    dh=np.zeros_like(x); dv=np.zeros_like(x)
    dh[:,:-1]=x[:,1:]-x[:,:-1]; dv[:-1,:]=x[1:,:]-x[:-1,:]; return dh,dv
def sym_grad_fwd(w1,w2):
    E1=np.zeros_like(w1); E2=np.zeros_like(w2)
    E1[:,:-1]=w1[:,1:]-w1[:,:-1]; E2[:-1,:]=w2[1:,:]-w2[:-1,:]
    a=np.zeros_like(w2); b=np.zeros_like(w1)
    a[:,:-1]=w2[:,1:]-w2[:,:-1]; b[:-1,:]=w1[1:,:]-w1[:-1,:]
    return E1,E2,0.5*(a+b)
def true_obj(x,w1,w2,y,a0,a1):
    Dh,Dv=grad_fwd(x); E=sym_grad_fwd(w1,w2)
    return 0.5*np.sum((x-y)**2)+a1*np.sum(np.sqrt(Dh**2+Dv**2))+a0*np.sum(np.sqrt(E[0]**2+E[1]**2+2*E[2]**2))

# 原代码的 div/sdiv (含 bug)
def div_orig(p,q):
    d=np.zeros(p.shape); d[:,1:]+=p[:,:-1]; d-=p; d[1:]+=q[:-1,:]; d-=q; return d
def sdiv_orig(q1,q2,q3):
    dh=np.zeros_like(q1);dv=np.zeros_like(q2)
    dh[:,1:]+=q1[:,:-1];dh-=q1; dv[1:]+=q2[:-1,:];dv-=q2
    a=np.zeros_like(q3);b=np.zeros_like(q3)
    a[:,1:]+=0.5*q3[:,:-1];a-=0.5*q3; b[1:]+=0.5*q3[:-1,:];b-=0.5*q3
    dh+=b; dv+=a; return dh,dv
def _h(p):
    o=np.zeros(p.shape);o[:,:-1]+=p[:,:-1];o[:,1:]-=p[:,:-1];return o
def _v(p):
    o=np.zeros(p.shape);o[:-1,:]+=p[:-1,:];o[1:,:]-=p[:-1,:];return o
def div_ok(p,q): return _h(p)+_v(q)
def sdiv_ok(q1,q2,q3): return _h(q1)+0.5*_v(q3), _v(q2)+0.5*_h(q3)

def run_cp(y,a0,a1,divf,sdivf,w_sign,n_iter=500):
    h,w=y.shape
    x=y.copy();w1=np.zeros((h,w));w2=np.zeros((h,w))
    p1=np.zeros((h,w));p2=np.zeros((h,w));q1=np.zeros((h,w));q2=np.zeros((h,w));q3=np.zeros((h,w))
    xb=x.copy();w1b=w1.copy();w2b=w2.copy()
    for k in range(n_iter):
        Dx=grad_fwd(xb); p1+=0.25*(Dx[0]-w1b); p2+=0.25*(Dx[1]-w2b)
        nr=np.sqrt(p1**2+p2**2); p1/=np.maximum(1,nr/a1); p2/=np.maximum(1,nr/a1)
        E=sym_grad_fwd(w1b,w2b); q1+=0.25*E[0];q2+=0.25*E[1];q3+=0.25*E[2]
        nq=np.sqrt(q1**2+q2**2+2*q3**2); q1/=np.maximum(1,nq/a0);q2/=np.maximum(1,nq/a0);q3/=np.maximum(1,nq/a0)
        dp=divf(p1,p2); sq=sdivf(q1,q2,q3)
        xn=(x+0.25*dp+0.25*y)/1.25; w1n=w1+0.25*(sq[0]+w_sign*p1); w2n=w2+0.25*(sq[1]+w_sign*p2)
        xb=xn+(xn-x);w1b=w1n+(w1n-w1);w2b=w2n+(w2n-w2);x,w1,w2=xn,w1n,w2n
    return x,w1,w2

np.random.seed(42); y=random_noise(cam,mode='gaussian',var=0.01)
# 原代码: div_orig + w_sign=-1
xo,w1o,w2o=run_cp(y,0.3,0.15,div_orig,sdiv_orig,-1,500)
print(f"\n[B 真实目标值] 原代码(div_orig,-p): obj={true_obj(xo,w1o,w2o,y,0.3,0.15):.1f}  PSNR={psnr(cam,xo):.2f}")
xk,w1k,w2k=run_cp(y,0.3,0.15,div_ok,sdiv_ok,1,500)
print(f"  正确版(div_ok,+p): obj={true_obj(xk,w1k,w2k,y,0.3,0.15):.1f}  PSNR={psnr(cam,xk):.2f}")
print(f"  (正确版目标远低于原代码 => 原代码发散, 未最小化TGV)")

# ---- C. 敏感性 收敛充分性: 500 vs 2000 迭代 ----
print("\n[C 敏感性 迭代数对照]")
for it in (500,2000):
    vals=[]
    for ratio in [0.5,1.0,2.0,4.0,8.0]:
        xx,_,_=run_cp(y,ratio*0.15,0.15,div_ok,sdiv_ok,1,it)
        vals.append(f"{ratio}:{psnr(cam,xx):.2f}")
    print(f"  n_iter={it}: "+"  ".join(vals))
