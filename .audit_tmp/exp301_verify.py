# -*- coding: utf-8 -*-
"""Verification for 实验3.0-1: reproduce numbers, check condition-number claim,
   and test the Chambolle-Pock extrapolation line."""
import numpy as np
from scipy.linalg import solve, circulant

rng = np.random.default_rng(20240715)
n = 200
x_true = np.zeros(n)
x_true[40:70] = 1.0
x_true[110:150] = -0.8

sigma_blur = 1.0
xx = np.arange(n)
kernel = np.exp(-0.5 * (xx - n // 2) ** 2 / sigma_blur ** 2)
kernel = np.roll(kernel, -n // 2)
kernel /= kernel.sum()
A_mat = circulant(kernel)

noise_level = 0.05
y = A_mat @ x_true + noise_level * rng.standard_normal(n)

# ---- condition number of A ----
svals = np.linalg.svd(A_mat, compute_uv=False)
cond_A = svals.max() / svals.min()
evals_AT = np.linalg.eigvalsh(A_mat.T @ A_mat)
L = evals_AT.max()
print(f"[cond] svd cond(A) = {cond_A:.2f}   (claim: ~70)")
print(f"[cond] sigma_max={svals.max():.4f} sigma_min={svals.min():.6f}")
print(f"[cond] L=||A^T A||_2 max eig = {L:.4f}")

x_naive = np.linalg.pinv(A_mat) @ y

lam_tik = 0.02
x_tik = solve(A_mat.T @ A_mat + lam_tik * np.eye(n), A_mat.T @ y)

def soft_threshold(v, mu):
    return np.sign(v) * np.maximum(np.abs(v) - mu, 0.0)

tau = 0.99 / L
mu_l1 = 0.05
x_l1 = np.zeros(n)
for _ in range(10000):
    grad = A_mat.T @ (A_mat @ x_l1 - y)
    x_l1 = soft_threshold(x_l1 - tau * grad, tau * mu_l1)

# ---- TV: as-written (buggy extrapolation) ----
D = (np.roll(np.eye(n), -1, axis=0) - np.eye(n))
Dt = D.T
lam_tv = 0.05
D_norm2 = 4.0
tau_tv = (-L / 2 + np.sqrt(L ** 2 / 4 + 4 * D_norm2)) / (2 * D_norm2)
tau_tv *= 0.99
sigma_tv = tau_tv

def run_tv(correct_extrap):
    x_tv = np.zeros(n); p_tv = np.zeros(n); x_bar = np.zeros(n)
    for _ in range(3000):
        p_tv = p_tv + sigma_tv * (D @ x_bar)
        p_tv = np.clip(p_tv, -lam_tv, lam_tv)
        x_new = x_tv - tau_tv * (A_mat.T @ (A_mat @ x_tv - y) + Dt @ p_tv)
        if correct_extrap:
            x_bar = 2 * x_new - x_tv   # proper CP extrapolation
            x_tv = x_new
        else:
            x_tv = x_new
            x_bar = 2 * x_tv - x_new   # as in script -> equals x_new
    return x_tv

x_tv_script = run_tv(False)
x_tv_correct = run_tv(True)

def rel_err(x):
    return np.linalg.norm(x - x_true) / np.linalg.norm(x_true)

def obj_tv(x):
    return 0.5*np.linalg.norm(A_mat@x - y)**2 + lam_tv*np.sum(np.abs(np.diff(x)))

print("\n[rel_err] script reproduction:")
print(f"  naive = {rel_err(x_naive):.4f}  (json 2.5229)")
print(f"  tik   = {rel_err(x_tik):.4f}  (json 0.2158)")
print(f"  l1    = {rel_err(x_l1):.4f}  (json 0.5714)")
print(f"  tv(script, x_bar=x_new) = {rel_err(x_tv_script):.4f}  (json 0.0537)")
print(f"  tv(correct extrap)      = {rel_err(x_tv_correct):.4f}")
print(f"  TV obj script={obj_tv(x_tv_script):.6f}  correct={obj_tv(x_tv_correct):.6f}")

# check x_bar truly degenerate in script version
x_tv = np.zeros(n); x_bar = np.zeros(n)
print(f"\n[extrap] in script: x_bar = 2*x_tv - x_new with x_tv=x_new -> x_bar==x_new (no extrapolation)")

# Verify TV is a genuine minimizer: small perturbation should raise objective
base = obj_tv(x_tv_script)
worse = 0
for _ in range(200):
    pert = x_tv_script + 0.01*rng.standard_normal(n)
    if obj_tv(pert) < base:
        worse += 1
print(f"[tv-opt] random perturbations lowering obj: {worse}/200 (0 => near minimizer)")

# L1 objective check
def obj_l1(x):
    return 0.5*np.linalg.norm(A_mat@x-y)**2 + mu_l1*np.linalg.norm(x,1)
base_l1 = obj_l1(x_l1)
worse_l1 = sum(1 for _ in range(200) if obj_l1(x_l1+0.01*rng.standard_normal(n)) < base_l1)
print(f"[l1-opt] perturbations lowering obj: {worse_l1}/200; obj={base_l1:.6f}")
print(f"[l1-sparsity] nonzero frac of x_l1 = {np.mean(np.abs(x_l1)>1e-6):.3f}; x_true nonzero frac={np.mean(x_true!=0):.3f}")
