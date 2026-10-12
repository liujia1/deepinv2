import numpy as np, torch, time
from scipy.stats import geninvgauss

# target standard GIG(nu=1/2, chi, psi): logpdf(z) = -0.5*log z -0.5*(chi/z + psi*z)
# vectorized random-walk MH in log-space across all elements (independent chains)
def gig_mh(chi, psi, rng, steps=150, init=None):
    chi = np.maximum(chi, 1e-30)
    if init is None:
        logz = np.log(np.maximum(chi/psi, 1e-12))  # near mode-ish start
    else:
        logz = np.log(init)
    step = 0.5
    def lp(lz):
        z = np.exp(lz)
        return -0.5*lz - 0.5*(chi/z + psi*z)   # -0.5 lz from jacobian (z^{-1/2} * dz=dz... ) use +lz? see note
    # density in z: z^{-1/2}exp(-(chi/z+psi z)/2). In log-space z=e^lz, target for lz:
    # p(lz) = p(z)*|dz/dlz| = e^{-lz/2}exp(...)*e^{lz} = e^{lz/2}exp(-(chi e^{-lz}+psi e^{lz})/2)
    def lpz(lz):
        return 0.5*lz - 0.5*(chi*np.exp(-lz) + psi*np.exp(lz))
    cur = lpz(logz)
    for _ in range(steps):
        prop = logz + rng.normal(0, step, size=logz.shape)
        pc = lpz(prop)
        accept = np.log(rng.random(size=logz.shape)) < (pc - cur)
        logz = np.where(accept, prop, logz)
        cur = np.where(accept, pc, cur)
    return np.exp(logz)

rng=np.random.default_rng(1)
print("validate gig_mh vs scipy code formula (chi=t^2, psi=lam^2):")
for (t,lam) in [(0.3,10.0),(1.0,10.0),(0.05,10.0),(2.0,1.0)]:
    chi=np.full(20000,t**2); psi=lam**2
    z=gig_mh(chi,psi,rng)
    zc=geninvgauss.rvs(0.5,lam*t,size=3000)*t/lam
    print(f"t={t},lam={lam}: mh mean={z.mean():.4f} E1z={np.mean(1/z):.3f} | code mean={zc.mean():.4f} E1z={np.mean(1/zc):.3f}")
