# -*- coding: utf-8 -*-
"""直接数值积分推导单样本方差，对比代码 theory 线"""
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import numpy as np
from numpy.polynomial.hermite_e import hermegauss

# 用 Gauss-Hermite(e) 求 E[g(z)], z~N(mu,s^2): z = mu + sqrt(2?)... 标准: weight e^{-x^2/2}
def gh_mean(mu, s, g, n=200):
    x, w = hermegauss(n)          # nodes for weight exp(-x^2/2)
    w = w / np.sqrt(2*np.pi)      # normalize to N(0,1)
    z = mu + s * x
    return np.sum(w * g(z))

def reinforce_var_single(mu, s):
    g2 = lambda z: (z**2 * (z-mu)/s**2)**2
    g1 = lambda z:  z**2 * (z-mu)/s**2
    return gh_mean(mu, s, g2) - gh_mean(mu, s, g1)**2

def reparam_var_single(mu, s):
    g2 = lambda z: (2*z)**2
    g1 = lambda z:  2*z
    return gh_mean(mu, s, g2) - gh_mean(mu, s, g1)**2

L = 50
print("验证: REINFORCE E[x]?", gh_mean(2.0,1.0, lambda z: z**2*(z-2)/1.0), "应=4")
print("REINFORCE 单样本方差 数值 vs 解析式(15σ²+14μ²+μ⁴/σ²):")
for s in [0.0316, 0.1, 0.5, 1.0, 2.0, 5.0]:
    num = reinforce_var_single(2.0, s)
    ana = 15*s**2 + 14*4 + 16/s**2
    code = (s**2 + 4/s**2)
    print(f"  σ={s:.4f}: 数值={num:.4f} 解析={ana:.4f} 代码theory单样本={code:.4f} 数值/代码={num/code:.2f}")

print("\nσ\t精确Var/L\t代码theory/L\t归档实测var\t精确/代码")
d = np.load(r"d:\works\deepinv\第9章 VAE与重参数化\第九章 配套实验\实验9.2-1\numerical_results_cache.npz")
sig = d['sigma_list']; rv = d['reinforce_vars']
for i in [0, 6, 12, 18, 24]:
    sv = sig[i]
    exact = reinforce_var_single(2.0, sv)/L
    code = (sv**2 + 4/sv**2)/L
    print(f"{sv:.4f}\t{exact:.4f}\t\t{code:.4f}\t\t{rv[i]:.4f}\t{exact/code:.2f}")
