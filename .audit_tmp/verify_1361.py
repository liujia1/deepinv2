# -*- coding: utf-8 -*-
import numpy as np

GM_WEIGHTS=[0.3,0.7]; GM_MEANS=[-2.0,1.0]; GM_STDS=[1.0,1.0]
SIG=0.3
def prior(x):
    p=np.zeros_like(x)
    for w,m,s in zip(GM_WEIGHTS,GM_MEANS,GM_STDS):
        p+=w*np.exp(-0.5*((x-m)/s)**2)/(s*np.sqrt(2*np.pi))
    return p

def exact_post(A,y,sig=SIG):
    x=np.linspace(-12,12,400001)
    lik=np.exp(-0.5*((y-A(x))/sig)**2)
    post=lik*prior(x)
    Z=np.trapz(post,x)
    post_n=post/Z
    mean=np.trapz(x*post_n,x)
    var=np.trapz((x-mean)**2*post_n,x)
    return mean,np.sqrt(var),x,post_n

tasks={
 "denoise A=I":(lambda x:x,0.5),
 "blur A=0.7x":(lambda x:0.7*x,0.5*0.7),
 "nonlin tanh2x":(lambda x:np.tanh(2*x),0.5),
}
# DPS observed results
dps={"denoise A=I":(0.508,0.295),"blur A=0.7x":(0.522,0.412),"nonlin tanh2x":(0.662,0.587)}
for name,(A,y) in tasks.items():
    m,s,_,_=exact_post(A,y)
    print(f"{name}: 精确后验 mean={m:.3f} std={s:.3f} | DPS mean={dps[name][0]:.3f} std={dps[name][1]:.3f}")

# 检查 t_start: x0_hat 的极大似然点
print("\n非线性: tanh(2x)=0.5 -> x*=",np.arctanh(0.5)/2)
# 检查似然在饱和区是否近似平坦
xs=np.array([-3,-2,-1,-0.5,0.275,1,2,3])
print("A(x)=tanh(2x) at x:",np.round(np.tanh(2*xs),3))
print("A'(x)=2sech^2:",np.round(2*(1-np.tanh(2*xs)**2),4))
