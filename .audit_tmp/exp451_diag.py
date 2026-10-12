import numpy as np
from scipy.stats import geninvgauss, kstest
import scipy

print("scipy", scipy.__version__)

# --- Understand scipy geninvgauss parameterization ---
# scipy docstring: pdf(x,p,b) = c * x**(p-1) * exp(-(b**2/x + x**2)/2)
# We want the Gibbs conditional for the Laplace/GSM scale variable:
#   p(z | t) propto z**(-1/2) * exp(-(t**2/z + lambda**2 * z)/2)   (standard GIG(1/2, chi=t^2, psi=lambda^2))

rng = np.random.default_rng(0)

def target_logpdf(z, t, lam):
    return -0.5*np.log(z) - 0.5*(t**2/z + lam**2*z)

def sample_true_gig(t, lam, size):
    # sample standard GIG(nu=1/2, chi=t^2, psi=lam^2) via rejection on a broad proposal
    # use log-target with simple adaptive rejection-free: sample from inverse-gamma-ish proposal then MH
    # Simpler: use the relation Z = (t/lam) * Y where Y ~ GIG(1/2, 1, 1)?? Instead do MH random-walk on log z.
    n = size
    z = np.abs(rng.normal(0, t/lam, n)) + 1e-6
    # run MH on log-space to mix toward target
    chain = np.log(z)
    samples = []
    step = 0.3
    cur = chain[0]
    acc = 0
    tot = 0
    burn = 2000
    thin = 20
    for i in range(burn + n*thin):
        prop = cur + rng.normal(0, step)
        lp_c = target_logpdf(np.exp(cur), t, lam) + cur   # +cur jacobian for log-space uniform ref
        lp_p = target_logpdf(np.exp(prop), t, lam) + prop
        if np.log(rng.random()) < (lp_p - lp_c):
            cur = prop
            acc += 1
        tot += 1
        if i >= burn and (i-burn) % thin == 0:
            samples.append(np.exp(cur))
    return np.array(samples), acc/tot

def code_sample(t, lam, size):
    tmp = np.maximum(t, 1e-30)
    return geninvgauss.rvs(0.5, lam*tmp, size=size) * tmp / lam

for (t, lam) in [(0.3, 10.0), (1.0, 10.0), (0.05, 10.0), (2.0, 1.0)]:
    true_s, acc = sample_true_gig(t, lam, 20000)
    code_s = code_sample(np.full(20000, t), lam, 20000)
    # KS between code samples and true samples
    ks = kstest(code_s, lambda x: np.searchsorted(np.sort(true_s), x)/len(true_s))
    print(f"t={t}, lam={lam}: true_mean={true_s.mean():.4f} code_mean={code_s.mean():.4f} "
          f"true_var={true_s.var():.4f} code_var={code_s.var():.4f} MHacc={acc:.2f} KS={ks.statistic:.3f} p={ks.pvalue:.2e}")

# Also test the analytic mean of GIG(1/2, t^2, lam^2): E[z] = sqrt(chi/psi)*G_{nu+1}/G_nu ... use numeric
# Check E[1/z] which enters the Gaussian precision: for target, and for code.
print("\n--- E[1/z] comparison (drives prior precision 1/z) ---")
for (t, lam) in [(0.3, 10.0), (1.0, 10.0), (0.05, 10.0)]:
    true_s, _ = sample_true_gig(t, lam, 20000)
    code_s = code_sample(np.full(20000, t), lam, 20000)
    print(f"t={t}, lam={lam}: true_E[1/z]={np.mean(1/true_s):.4f} code_E[1/z]={np.mean(1/code_s):.4f}")
