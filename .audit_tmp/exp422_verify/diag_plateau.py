import numpy as np

def tld(x):
    return -0.5 * x ** 2

def mh(seed, n, sigma, burn=1000):
    rng = np.random.default_rng(seed)
    x = 0.0
    s = []
    lx = tld(x)
    na = 0
    for i in range(n + burn):
        p = x + sigma * rng.standard_normal()
        lp = tld(p)
        a = min(1.0, np.exp(lp - lx))
        acc = rng.random() < a
        if acc:
            x = p; lx = lp
        if i >= burn:
            s.append(x)
            if acc: na += 1
    return np.array(s), na / n

def acf(x, ml):
    x = x - x.mean(); v = x.var()
    if v < 1e-12: return np.zeros(ml)
    N = len(x)
    # FFT-based full autocovariance, unbiased normalization (N-lag)
    f = np.fft.rfft(x, n=2*N)
    acov_full = np.fft.irfft(f*np.conj(f))[:ml+1]
    n_lag = N - np.arange(ml+1)
    out = (acov_full[1:]/n_lag[1:]) / v
    return out

def ess_geyer(samples, ml=100):
    N = len(samples); a = acf(samples, ml)
    pairs = a[0::2] + a[1::2]
    neg = pairs < 0
    if neg.any():
        c = np.argmax(neg); ssum = np.sum(a[:2*c])
    else:
        ssum = np.sum(a)
    return N/(1+2*ssum)

# analytic-ish acceptance rate check + fine ESS plateau
sigmas = np.logspace(-0.5, 0.7, 25)  # ~0.316 .. 5
print("sigma   accept(mean8)  ess_ml100(mean8)  ess_ml300(mean8)")
for s in sigmas:
    ar=[]; e100=[]; e300=[]
    for seed in range(6):
        sm, a = mh(1000+seed, 20000, s)
        ar.append(a); e100.append(ess_geyer(sm,100)); e300.append(ess_geyer(sm,300))
    print(f"{s:5.2f}  {np.mean(ar):.4f}         {np.mean(e100):7.0f}            {np.mean(e300):7.0f}")
