import numpy as np, arviz
M_obs, ess_obs = 49, 4.35
tau = M_obs / ess_obs
print(f"med: M={M_obs}, ESS={ess_obs} -> tau=M/ESS={tau:.1f}")
print(f"  need ESS>=100 -> M>=tau*100={tau*100:.0f} samples; thinning9 -> ~{tau*100*9:.0f} iters")
print(f"  ref notebook maxit=10000 n_samples=2000 -> ESS~{2000/tau:.0f} (>100)")
print("ESS<=M cap:")
for M in (49,200,2000):
    print(f"  M={M:5d}: ideal indep ESS={M}; if tau={tau:.0f} -> ESS~{M/tau:.0f}")
print("Synthetic AR(1) (M=49, 4000 reps):")
rng=np.random.default_rng(0)
for rho in (0.35,0.6,0.8):
    tt=(1+rho)/(1-rho); el=[]
    for _ in range(4000):
        x=np.zeros(M_obs); x[0]=rng.standard_normal(); e=rng.standard_normal(M_obs)*np.sqrt(1-rho**2)
        for t in range(1,M_obs): x[t]=rho*x[t-1]+e[t]
        el.append(float(arviz.ess(x)))
    print(f"  rho1={rho}: tau_theory={tt:.1f} expESS~{M_obs/tt:.1f} measured={np.mean(el):.1f}")
