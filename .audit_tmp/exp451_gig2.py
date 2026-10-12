import numpy as np, time
from scipy.stats import geninvgauss, invgauss

# Exact fast sampler for standard GIG(nu=1/2, chi, psi): f(z)~z^{-1/2}exp(-(chi/z+psi z)/2)
# Derivation: 1/Z ~ GIG(-1/2, psi, chi).  scipy invgauss(mu_v)=GIG(-1/2, mu_v, 1/mu_v^2).
# Scaling c*V: V~GIG(-1/2,x0,p0) -> cV~GIG(-1/2, c^2 x0, p0/c^2). Solve c^2 mu_v=psi, (1/mu_v^2)/c^2=chi
#   => mu_v = 1/(chi*psi),  c = psi*sqrt(chi*psi).  Then Z = 1/(c*invgauss(mu_v)).
def gig_half_fast(chi, psi, size, rng=None):
    chi=np.maximum(chi,1e-30)
    mu_v = 1.0/(chi*psi)
    c    = psi*np.sqrt(chi*psi)
    W = c*invgauss.rvs(mu_v, size=size, random_state=rng)
    return 1.0/W

rng=np.random.default_rng(1)
print("verify gig_half_fast vs scipy geninvgauss (code formula):")
for (t,lam) in [(0.3,10.0),(1.0,10.0),(0.05,10.0),(2.0,1.0)]:
    chi=t**2; psi=lam**2
    z=gig_half_fast(np.full(50000,chi),psi,50000,rng)
    zc=geninvgauss.rvs(0.5,lam*t,size=5000)*t/lam
    print(f"t={t},lam={lam}: fast mean={z.mean():.4f} E1z={np.mean(1/z):.3f} | code mean={zc.mean():.4f} E1z={np.mean(1/zc):.3f}")

print("\nspeed: geninvgauss vs invgauss-based, 20000 draws x 20:")
x=np.abs(np.random.randn(20000))+0.1
s=time.time(); [gig_half_fast(x**2,100,20000,rng) for _ in range(20)]; print("fast :",round(time.time()-s,2),"s")
s=time.time(); [geninvgauss.rvs(0.5,10*x) for _ in range(3)]; print("scipy(3x):",round(time.time()-s,2),"s -> ~x",round((time.time()-s)/3*20/ (time.time()-time.time()+1e-9),1) if False else "(slow, extrapolate)")
