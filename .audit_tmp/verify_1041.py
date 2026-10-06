# -*- coding: utf-8 -*-
"""独立验证实验10.4-1的表格数值：不复用其代码逻辑，用闭式解交叉验证"""
import numpy as np
import sys, io
if sys.stdout is None or not hasattr(sys.stdout, 'buffer'):
    pass

# ===== 表1: 离散→连续极限 =====
print("=== 表1: ab_L=(1-beta/L)^L vs exp(-beta), beta=1.0 ===")
for L in [5, 10, 50, 100, 500, 1000]:
    dis = (1 - 1.0/L)**L
    cont = np.exp(-1.0)
    err = abs(dis - cont)/cont
    print(f"L={L:5d}  dis={dis:.6f}  cont={cont:.6f}  rel_err={err:.6f}  rel_err_2sig={err:.4g}")

# ===== 表2: KL积分, 逐位复算代码公式 =====
print("\n=== 表2: 代码公式复算 (mean_sq - 1 - log_var) 的黎曼和 ===")
for L in [5, 10, 50, 100]:
    dt = 1.0/L
    a_step = 1 - dt
    total = 0.0
    mu2, s2 = 1.0, 0.25
    for t in range(1, L+1):
        ab = a_step**t
        mean_sq = ab*(mu2+s2) + (1-ab)
        log_var = np.log(ab*s2 + (1-ab))
        total += 0.5*(mean_sq - 1 - log_var)*dt
    print(f"L={L:5d}  KL_integral={total:.6f}")

# ===== 闭式解: ∫_0^1 0.5*(D e^{-βt} + βt - 1 - 0.5*ln(1-(1-s2) e^{-βt})) dt, D=mu2+s2=1.25
from scipy.integrate import quad
D, s2, beta = 1.25, 0.25, 1.0
def integrand(t):
    e = np.exp(-beta*t)
    return 0.5*(D*e + beta*t - 1 - 0.5*np.log(1-(1-s2)*e))
val, _ = quad(integrand, 0, 1)
print(f"\n闭式极限 (代码公式的L→∞) = {val:.6f}")

# ===== 真正意义的高斯边际KL: KL(N(sqrt(ab)mu, ab s2 + 1-ab) || N(0,1)) 逐样本平均 =====
np.random.seed(42)
x0 = np.vstack([np.random.randn(500,2)*0.5 + [1,0], np.random.randn(500,2)*0.5 + [-1,0]])
print("\n=== 真实边际KL(q(x_t)||N(0,I))的黎曼和 (2D GMM, 对样本平均) ===")
for L in [5, 10, 50, 100]:
    dt = 1.0/L
    total = 0.0
    for t in range(1, L+1):
        ab = (1-dt)**t
        # E_x~q(x_t)[ -log N(x;0,I) ] - E[log q] 难解析(混合非高斯), 用真高斯近似逐样本:
        # 对每个样本视为 N(sqrt(ab)x0_i, (1-ab)I) 的均值方向 —— 采用经验公式:
        # KL(N(m, vI)||N(0,I)) per-dim = 0.5*(||m||^2/v... ) 用 m=sqrt(ab)*x0_i, v=1-ab (逐样本条件分布的边际近似)
        m = np.sqrt(ab)*x0
        v = 1-ab
        kl_samples = 0.5*(np.sum(m**2, axis=1)/1 + 2*1 - 2 - np.log(v**2)).mean()
        # KL(N(m,I)||N(0,I)) = 0.5*(|m|^2 + d - d - d ln 1)?? 正确: = 0.5*(|m|^2) (d=2, 方差同为1) ——但边际方差是v不是1
        # 严格: KL(N(m, v I_d) || N(0, I_d)) = 0.5*( |m|^2 + d*v - d - d*ln v )
        kl = 0.5*(np.sum(m**2, axis=1) + 2*v - 2 - 2*np.log(v)).mean()
        total += kl*dt
    print(f"L={L:5d}  真实边际KL积分≈{total:.6f}")
