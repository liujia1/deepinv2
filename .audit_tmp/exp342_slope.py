# -*- coding: utf-8 -*-
"""验证 3.4-2 讲义/注释的局部斜率论断是否属实"""
import math
import numpy as np
np.random.seed(42)
n, sigma_blur, sigma_noise, lam = 256, 2.0, 0.05, 0.1
def mk(n, seed=17):
    rng=np.random.RandomState(seed); x=np.zeros(n); i=0
    while i<n:
        s=rng.randint(15,40); x[i:i+s]=rng.uniform(0.3,1.0)*rng.choice([-1.,1.]); i+=s
    return x
x_true=mk(n)
t=np.fft.fftfreq(n)*n; h=np.exp(-t**2/(2*sigma_blur**2)); h/=h.sum(); H=np.fft.fft(h)
fwd=lambda x:np.real(np.fft.ifft(H*np.fft.fft(x)))
adj=lambda z:np.real(np.fft.ifft(np.conj(H)*np.fft.fft(z)))
y=fwd(x_true)+sigma_noise*np.random.randn(n)
L=float(np.max(np.abs(H))**2); gamma=0.99/L
grad=lambda x:adj(fwd(x)-y)
soft=lambda v,tt:np.sign(v)*np.maximum(np.abs(v)-tt,0.)
cost=lambda x:0.5*float(np.sum((fwd(x)-y)**2))+lam*float(np.sum(np.abs(x)))
def ista(mi):
    x=np.zeros(n);c=np.zeros(mi)
    for k in range(mi):
        x=soft(x-gamma*grad(x),gamma*lam);c[k]=cost(x)
    return c
def fista(mi):
    x=np.zeros(n);xo=np.zeros(n);to=1.;c=np.zeros(mi)
    for k in range(mi):
        tk=.5*(1+math.sqrt(1+4*to**2));yk=x+(to-1)/tk*(x-xo)
        xn=soft(yk-gamma*grad(yk),gamma*lam);c[k]=cost(xn);to=tk;xo=x;x=xn
    return c
ci=ista(2000);cf=fista(2000);cr=fista(20000);Fs=float(cr[-1])
gi=np.maximum(ci-Fs,1e-15);gf=np.maximum(cf-Fs,1e-15)
def loc(g,a,b):
    ka=np.arange(a,b+1);return float(np.polyfit(np.log(ka),np.log(g[a-1:b]),1)[0])
print("ISTA 局部斜率(讲义: k<1000 过渡 -0.6~-0.9, k>=1500 逼近 -1):")
for a,b in [(100,200),(200,400),(400,700),(700,1000),(1000,1500),(1500,2000)]:
    print(f"  k={a}~{b}: {loc(gi,a,b):+.3f}")
print("FISTA 局部斜率(讲义: 中前段 -2.3~-3.1, 尾部达 -6):")
for a,b in [(50,100),(100,200),(200,300),(300,500),(500,1000),(1000,2000)]:
    print(f"  k={a}~{b}: {loc(gf,a,b):+.3f}")
