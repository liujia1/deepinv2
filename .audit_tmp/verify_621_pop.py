# -*- coding: utf-8 -*-
"""总体层面(数值积分)核查: ESM(θ)-ISM(θ) 是否严格为常数"""
import numpy as np

def gm_score(x):
    p1 = np.exp(-0.5 * (x + 2)**2) / np.sqrt(2*np.pi)
    p2 = np.exp(-0.5 * (x - 2)**2) / np.sqrt(2*np.pi)
    p = 0.5 * p1 + 0.5 * p2
    grad_p1 = -(x + 2) * p1
    grad_p2 = -(x - 2) * p2
    return (0.5 * grad_p1 + 0.5 * grad_p2) / p

def p_pdf(x):
    return 0.5*np.exp(-0.5*(x+2)**2)/np.sqrt(2*np.pi) + 0.5*np.exp(-0.5*(x-2)**2)/np.sqrt(2*np.pi)

xg = np.linspace(-15, 15, 200001)   # 20万点足够
dx = xg[1] - xg[0]
p = p_pdf(xg)
s = gm_score(xg)
print("∫p dx =", np.trapezoid(p, dx=dx))
print("∫x²p dx =", np.trapezoid(xg**2 * p, dx=dx))
print("∫x·s·p dx =", np.trapezoid(xg * s * p, dx=dx), "(理论 -1)")
print("0.5∫s²p dx =", 0.5*np.trapezoid(s**2 * p, dx=dx))
for t in [-1.5, -0.75, -0.2, 0.0, 0.5]:
    esm = np.trapezoid(0.5*(t*xg - s)**2 * p, dx=dx)
    ism = np.trapezoid((t + 0.5*(t*xg)**2) * p, dx=dx)
    print(f"θ={t:+.2f}: ESM={esm:.6f}, ISM={ism:.6f}, diff={esm-ism:.6f}")
