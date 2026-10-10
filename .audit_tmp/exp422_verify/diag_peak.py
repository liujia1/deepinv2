import numpy as np

def tld(x): return -0.5*x**2

def mh(seed, n, sigma, burn=1000):
    rng = np.random.default_rng(seed); x=0.0; lx=tld(x); s=np.empty(n); na=0
    for i in range(n+burn):
        p=x+sigma*rng.standard_normal(); lp=tld(p); a=min(1.0,np.exp(lp-lx))
        acc=rng.random()<a
        if acc: x=p; lx=lp
        if i>=burn:
            s[i-burn]=x
            if acc: na+=1
    return s, na/n

def acf(x, ml):
    x=x-x.mean(); v=x.var(); N=len(x)
    f=np.fft.rfft(x,n=2*N); ac=np.fft.irfft(f*np.conj(f))[:ml+1]
    nl=N-np.arange(ml+1); return (ac[1:]/nl[1:])/v

def ess(s, ml=100):
    N=len(s); a=acf(s,ml); pairs=a[0::2]+a[1::2]; neg=pairs<0
    ss=np.sum(a[:2*np.argmax(neg)]) if neg.any() else np.sum(a)
    return N/(1+2*ss)

sig=np.logspace(-1,1,20)
E=np.zeros(len(sig)); A=np.zeros(len(sig))
for i,s0 in enumerate(sig):
    for seed in range(30):
        sm,a=mh(7000+seed,20000,s0); E[i]+=ess(sm,100); A[i]+=a
    E[i]/=30; A[i]/=30
k=int(np.argmax(E))
print("argmax sigma=%.3f  ESS=%.0f  accept=%.4f  (30-seed avg)"%(sig[k],E[k],A[k]))
print("sigma   ESSavg   accept")
for i in range(len(sig)):
    if sig[i]>=1.0: print("%5.2f  %7.0f  %.4f"%(sig[i],E[i],A[i]))
peak=E[k]; band=[sig[i] for i in range(len(sig)) if E[i]>0.97*peak]
print("ESS within 3pct of peak spans sigma ~ [%.2f, %.2f]"%(min(band),max(band)))
