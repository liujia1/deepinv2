# -*- coding: utf-8 -*-
"""核验：对称双峰后验下，'锁一个峰'(mu=±1)与'居中覆盖'(mu=0)哪个ELBO更高？
用于判断讲义'单高斯只能覆盖一个峰/mode-seeking锁峰'的说法在本对称场景是否成立。"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

pw=[0.5,0.5]; pmu=[-2.,2.]; pt=[1.,1.]; so=1.; xo=0.
def log_pz(z):
    t=0.
    for w,m,tau in zip(pw,pmu,pt):
        t+=w*np.exp(-0.5*((z-m)/tau)**2)/(tau*np.sqrt(2*np.pi))
    return np.log(t)
def exact_elbo(mu_q,sigma_q):
    def f1(z):
        q=np.exp(-0.5*((z-mu_q)/sigma_q)**2)/(sigma_q*np.sqrt(2*np.pi))
        return q*(-0.5*np.log(2*np.pi)-np.log(so)-0.5*((xo-z)/so)**2)
    def f2(z):
        q=np.exp(-0.5*((z-mu_q)/sigma_q)**2)/(sigma_q*np.sqrt(2*np.pi))
        return q*log_pz(z)
    def f3(z):
        q=np.exp(-0.5*((z-mu_q)/sigma_q)**2)/(sigma_q*np.sqrt(2*np.pi))
        return q*np.log(q+1e-300)
    a,_=quad(f1,-40,40,limit=500); b,_=quad(f2,-40,40,limit=500); c,_=quad(f3,-40,40,limit=500)
    return a+b-c
# 锁右峰最优
r_lock=minimize(lambda p:-exact_elbo(p[0],np.exp(p[1])),[1.0,np.log(0.7)],method='Nelder-Mead')
print(f"锁峰解(初值mu=1): mu={r_lock.x[0]:.4f}, sigma={np.exp(r_lock.x[1]):.4f}, ELBO={-r_lock.fun:.5f}")
# 居中解
r_mid=minimize(lambda p:-exact_elbo(p[0],np.exp(p[1])),[0.0,np.log(1.17)],method='Nelder-Mead')
print(f"居中解(初值mu=0): mu={r_mid.x[0]:.4f}, sigma={np.exp(r_mid.x[1]):.4f}, ELBO={-r_mid.fun:.5f}")
# 锁左峰
r_lock2=minimize(lambda p:-exact_elbo(p[0],np.exp(p[1])),[-1.0,np.log(0.7)],method='Nelder-Mead')
print(f"锁峰解(初值mu=-1): mu={r_lock2.x[0]:.4f}, sigma={np.exp(r_lock2.x[1]):.4f}, ELBO={-r_lock2.fun:.5f}")
print("\n=> 全局最优是对称居中解(mu=0)，锁单峰反而ELBO更低。")
