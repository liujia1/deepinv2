"""Independent verification of experiment 2.3-2."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# ---------- Subplot (a): skewed posterior ----------
b, sigma, y = 0.8, 1.0, 2.0

def post(x):
    return np.exp(-np.abs(x)/b - (y-x)**2/(2*sigma**2))

Z, _ = quad(post, -50, 50, limit=400, epsabs=1e-14)
m1, _ = quad(lambda x: x*post(x), -50, 50, limit=400, epsabs=1e-14)
mmse = m1/Z
# exact MAP
map_exact = y - sigma**2/b
print("[a] exact MAP (analytic y-sigma^2/b) =", map_exact)
print("[a] quad MMSE (posterior mean)      =", mmse)
print("[a] divergence MMSE-MAP             =", mmse-map_exact)

# reproduce script grid argmax & trapz
xg = np.linspace(-5, 8, 3000)
lp = -np.abs(xg)/b - (y-xg)**2/(2*sigma**2)
p = np.exp(lp-lp.max()); p = p/(p.sum()*(xg[1]-xg[0]))
print("[a] script-grid MAP =", xg[np.argmax(p)])
print("[a] script-grid MMSE(trapz) =", np.trapezoid(xg*p, xg))

# ---------- Subplot (b): bimodal ----------
mu1,s1,w1 = 0.0,0.15,0.6
mu2,s2,w2 = 1.0,0.15,0.4
def g(x,mu,s): return np.exp(-(x-mu)**2/(2*s**2))/(s*np.sqrt(2*np.pi))
Zb,_ = quad(lambda x: w1*g(x,mu1,s1)+w2*g(x,mu2,s2), -10, 10, epsabs=1e-14)
m1b,_ = quad(lambda x: x*(w1*g(x,mu1,s1)+w2*g(x,mu2,s2)), -10, 10, epsabs=1e-14)
print("[b] quad MMSE =", m1b/Zb, " theory w1mu1+w2mu2 =", w1*mu1+w2*mu2)
# MAP: which peak higher
print("[b] peak1 height=", w1*g(mu1,mu1,s1)+w2*g(mu1,mu2,s2),
      " peak2 height=", w1*g(mu2,mu1,s1)+w2*g(mu2,mu2,s2))
