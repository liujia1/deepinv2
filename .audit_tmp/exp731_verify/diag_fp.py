# -*- coding: utf-8 -*-
import numpy as np
beta_min, beta_max = 0.1, 20.0
def vp_beta(t): return beta_min + t*(beta_max-beta_min)
def vp_marginal_params(t):
    log_mean = -0.25*t**2*(beta_max-beta_min) - 0.5*t*beta_min
    mc = np.exp(log_mean); return mc, np.sqrt(max(1-mc**2,0))
def vp_gmm_1d_marginal(x,t,weights=[0.4,0.6],means=[-2,2],stds=[1,1]):
    mc,std_t=vp_marginal_params(t); pdf=np.zeros_like(x,dtype=float)
    for w,m,s in zip(weights,means,stds):
        nm=mc*m; ns=np.sqrt(mc**2*s**2+std_t**2)
        pdf+=w*np.exp(-0.5*((x-nm)/ns)**2)/(ns*np.sqrt(2*np.pi))
    return pdf
x_grid=np.linspace(-8,8,2000); dx=x_grid[1]-x_grid[0]
for t in [0.1,0.3,0.5,0.7,0.9]:
    p_t=vp_gmm_1d_marginal(x_grid,t); dt_fp=0.01
    dp_dt=(vp_gmm_1d_marginal(x_grid,t+dt_fp)-vp_gmm_1d_marginal(x_grid,t-dt_fp))/(2*dt_fp)
    dp_dx=np.gradient(p_t,dx); d2p=np.gradient(dp_dx,dx); div_xp=p_t+x_grid*dp_dx
    b=vp_beta(t); rhs_fwd=b/2*div_xp+b/2*d2p; rhs_rev=-b/2*div_xp-b/2*d2p
    interior=slice(100,-100)
    err_f=(dp_dt-rhs_fwd)[interior]; err_r=(-dp_dt-rhs_rev)[interior]
    # verify reverse error is IDENTICAL to forward error
    ident = np.allclose(err_r, -err_f)
    lhs=np.abs(dp_dt[interior])
    pw = np.abs(err_f)/np.maximum(lhs,1e-12)  # pointwise relative error
    # where is pointwise rel err large? check |LHS| there
    big = pw>0.10
    print(f"t={t}: err_r==(-err_f)? {ident}  |max abs err|={np.abs(err_f).max():.2e}")
    print(f"    pointwise rel>10%: count={big.sum()}/{len(pw)} ({100*big.mean():.1f}%), max={pw.max():.2f}")
    if big.sum()>0:
        print(f"    at those pts |LHS| range=[{lhs[big].min():.2e},{lhs[big].max():.2e}], p_t range=[{p_t[interior][big].min():.2e},{p_t[interior][big].max():.2e}]")
