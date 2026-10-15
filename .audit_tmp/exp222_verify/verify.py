# -*- coding: utf-8 -*-
import numpy as np

# ================= Part A reproduce =================
N, K, LAMBDA, SNR_DB = 50, 5, 0.6, 20
np.random.seed(42)

def make_sparse_signal(n, k, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    idx = rng.choice(n, size=k, replace=False)
    x[idx] = rng.uniform(-2.5, 2.5, size=k)
    return x, idx

x_true, support = make_sparse_signal(N, K, seed=7)
sig = np.linalg.norm(x_true) / (10 ** (SNR_DB / 20))
noise = np.random.normal(0, sig, size=N)
y = x_true + noise

def support_recovery(x_hat, x_true, tol=1e-6):
    true_sup = np.abs(x_true) > tol
    est_sup = np.abs(x_hat) > tol
    return np.mean(est_sup[true_sup])

x_ridge = y / (1.0 + LAMBDA)
x_lasso = np.sign(y) * np.maximum(np.abs(y) - LAMBDA, 0.0)

print("=== Part A ===")
print("x_true nonzero at:", sorted(support.tolist()))
print("x_true values:", np.round(x_true[support], 4))
print("y at true support:", np.round(y[support], 4))
print("Ridge MSE", np.mean((x_ridge-x_true)**2), "rec", support_recovery(x_ridge,x_true), "zero", int(np.sum(np.abs(x_ridge)<1e-6)))
print("LASSO MSE", np.mean((x_lasso-x_true)**2), "rec", support_recovery(x_lasso,x_true), "zero", int(np.sum(np.abs(x_lasso)<1e-6)))
# which true support killed by lasso?
killed = [i for i in support if abs(x_lasso[i])<1e-6]
print("LASSO killed true-support idx:", killed, "-> their y:", np.round(y[killed],4), "true x:", np.round(x_true[killed],4))
# ridge false positives count
print("Ridge nonzero count (should be 50):", int(np.sum(np.abs(x_ridge)>1e-6)))
print("LASSO nonzero count:", int(np.sum(np.abs(x_lasso)>1e-6)))

# ================= Part B reproduce =================
N_B, K_B, M_B, SNR_B, SEED_B = 256, 10, 80, 30, 7

def make_cs_problem(n, k, m, snr, seed=0):
    rng = np.random.default_rng(seed)
    alpha = np.zeros(n)
    idx = rng.choice(n, size=k, replace=False)
    alpha[idx] = rng.uniform(-3, 3, size=k)
    A = rng.standard_normal((m, n))
    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    sigma = np.linalg.norm(A @ alpha) / (10 ** (snr / 20))
    noise = rng.standard_normal(m) * sigma
    y = A @ alpha + noise
    return A, y, alpha, idx

A_b, y_b, alpha_true, support_b = make_cs_problem(N_B, K_B, M_B, SNR_B, seed=SEED_B)

def IHT(A, y, K_max, max_iter=500, tol=1e-8):
    m, n = A.shape
    At = A.T
    L = np.linalg.norm(A, 2) ** 2
    step = 1.0 / L
    alpha = np.zeros(n)
    for _ in range(max_iter):
        grad = At @ (y - A @ alpha)
        alpha_new = alpha + step * grad
        if K_max < alpha_new.size:
            thresh = np.sort(np.abs(alpha_new))[-K_max]
            alpha_new[np.abs(alpha_new) < thresh] = 0.0
        if np.linalg.norm(alpha_new - alpha) < tol:
            alpha = alpha_new; break
        alpha = alpha_new
    return alpha

def Reweighted_L1(A, y, n_outer=12, eps=1e-2, lam=0.1, max_inner=300):
    m, n = A.shape
    At = A.T
    L = np.linalg.norm(A, 2) ** 2
    step = 1.0 / L
    alpha = np.zeros(n)
    w = np.ones(n)
    for _ in range(n_outer):
        for _ in range(max_inner):
            grad = At @ (y - A @ alpha)
            soft = np.sign(alpha + step*grad)*np.maximum(np.abs(alpha+step*grad)-step*lam*w, 0.0)
            if np.linalg.norm(soft-alpha) < 1e-8:
                alpha = soft; break
            alpha = soft
        w = 1.0/(np.abs(alpha)+eps)
    return alpha

def OMP(A, y, K_max, tol=1e-6):
    m, n = A.shape
    residual = y.copy(); idx_set=[]; alpha=np.zeros(n)
    for _ in range(K_max):
        corr = A.T @ residual
        new_idx = int(np.argmax(np.abs(corr)))
        if new_idx in idx_set: break
        idx_set.append(new_idx)
        Asub = A[:, idx_set]
        coef, *_ = np.linalg.lstsq(Asub, y, rcond=None)
        residual = y - Asub @ coef
        if np.linalg.norm(residual) < tol: break
    alpha[idx_set] = coef
    return alpha

def rel_err(a,b): return np.linalg.norm(a-b)/np.linalg.norm(b)
def nnz(a): return int(np.sum(np.abs(a)>1e-6))

alpha_iht = IHT(A_b, y_b, K_B, max_iter=200)
alpha_rw = Reweighted_L1(A_b, y_b)
alpha_omp = OMP(A_b, y_b, K_B)

print("\n=== Part B ===")
print("true support:", sorted(support_b.tolist()))
print("true alpha values at support:", np.round(alpha_true[support_b],3))
for name,a in [('IHT',alpha_iht),('ReweightedL1',alpha_rw),('OMP',alpha_omp)]:
    rec_sup = set(np.where(np.abs(a)>1e-6)[0].tolist())
    hit = rec_sup & set(support_b.tolist())
    print(f"{name:14s} rel_err={rel_err(a,alpha_true):.4f} nnz={nnz(a)} 命中真支撑={len(hit)}/10 误报={len(rec_sup-set(support_b.tolist()))}")

# support magnitudes to judge "weak support killed"
print("\n排序后的真值幅度:", np.sort(np.abs(alpha_true[support_b])))
