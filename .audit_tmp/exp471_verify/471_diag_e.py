"""复核脚本 E（瞬时）：ESS 与样本预算/自相关时间的关系"""
import numpy as np
import arviz

M_obs, ess_obs = 49, 4.35
tau = M_obs / ess_obs
print(f"med: M={M_obs}, ESS={ess_obs} -> tau = M/ESS = {tau:.1f}")
print(f"  need ESS>=100 -> M >= tau*100 = {tau*100:.0f} samples")
print(f"  with thinning_step=9 -> ~{tau*100*9:.0f} iters")
print(f"  ref notebook maxit=10000 n_samples=2000 -> ESS ~ {2000/tau:.0f} (>100)")

print("\nESS <= M cap:")
for M in (49, 200, 2000):
    print(f"  M={M:5d}: ideal indep ESS={M}; if tau={tau:.0f} -> ESS~{M/tau:.0f}")

print("\nSynthetic AR(1) check (M=49, 5000 reps):")
rng = np.random.default_rng(0)
for rho in (0.35, 0.6, 0.8):
    tau_theory = (1 + rho) / (1 - rho)
    ess_list = []
    for _ in range(5000):
        x = np.zeros(M_obs); x[0] = rng.standard_normal()
        e = rng.standard_normal(M_obs) * np.sqrt(1 - rho**2)
        for t in range(1, M_obs):
            x[t] = rho * x[t-1] + e[t]
        ess_list.append(float(arviz.ess(x)))
    print(f"  rho1={rho}: tau_theory={tau_theory:.1f} exp ESS~{M_obs/tau_theory:.1f}; measured ESS={np.mean(ess_list):.1f}")
