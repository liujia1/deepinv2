"""diag2: 用 ASCII 输出，避免 GBK 编码问题"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np

n, L, sigma = 100, 0.1, 1.0
xg = np.linspace(0, 1, n)
x1, x2 = np.meshgrid(xg, xg)
Sigma = np.exp(-(x1 - x2) ** 2 / (2 * L ** 2))
w, V = np.linalg.eigh(Sigma)
print("eigen min=%.4e max=%.4f  cond=%.4e" % (w.min(), w.max(), w.max() / w.min()))
print("neg eigen count=%d, smallest abs=%d" % ((w < 0).sum(), 0) if False else "neg eigen count=%d" % (w < 0).sum())

I = np.eye(n)
print("\n[diag constancy] solve-based vs eigen-based")
for lam in [0.1, 1.0, 10.0]:
    Sp_solve = lam * np.linalg.solve(Sigma + lam * I, Sigma)
    Sp_eig = (V * (lam * w / (w + lam))) @ V.T
    d_s, d_e = np.diag(Sp_solve), np.diag(Sp_eig)
    print(" lam=%-4g: solve diag range=%.3e (%.4f..%.4f)  | eigen diag range=%.3e (%.6f..%.6f)"
          % (lam, d_s.max() - d_s.min(), d_s.min(), d_s.max(),
             d_e.max() - d_e.min(), d_e.min(), d_e.max()))
    print("         |Sp_solve - Sp_eig|_max = %.3e   (eigen解满足方程残差 %.3e)"
          % (np.abs(Sp_solve - Sp_eig).max(),
             np.abs((Sigma + lam * I) @ Sp_eig - lam * Sigma).max()))

print("\n[closed-form error source]")
Sig_inv = np.linalg.inv(Sigma)
Sp_direct = np.linalg.inv(Sig_inv + I / sigma ** 2)
Sp_formula = sigma ** 2 * np.linalg.solve(Sigma + sigma ** 2 * I, Sigma)
print(" max|direct-formula| = %.4e" % np.abs(Sp_direct - Sp_formula).max())
print(" |direct - eigen_ref| = %.4e   |formula - eigen_ref| = %.4e"
      % (np.abs(Sp_direct - Sp_formula * 0 + 0).max() * 0 + np.abs(Sp_direct - (V * (w / (w + 1))) @ V.T).max(),
         np.abs(Sp_formula - (V * (w / (w + 1))) @ V.T).max()))
print(" ||inv(Sig)@Sig - I||_max = %.4e ;  ||inv(Sig)||_max = %.4e" % (np.abs(Sig_inv @ Sigma - I).max(), np.abs(Sig_inv).max()))
print(" equation residual: direct=%.4e  formula=%.4e" % (
      np.abs((Sigma + sigma ** 2 * I) @ Sp_direct - sigma ** 2 * Sigma).max(),
      np.abs((Sigma + sigma ** 2 * I) @ Sp_formula - sigma ** 2 * Sigma).max()))
print(" diag(direct) range = %.4e ; diag(formula) range = %.4e" % (np.ptp(np.diag(Sp_direct)), np.ptp(np.diag(Sp_formula))))

# 用高精度（float64 + 特征分解已足够稳定）确认公式在数学上恒等
w_clipped = np.clip(w, 1e-30, None)
Sp_exact = (V * (w_clipped / (w_clipped + 1))) @ V.T
print(" eigen(clip) vs formula max err = %.4e" % np.abs(Sp_exact - Sp_formula).max())

print("\n[repeated-sampling calibration]  R=4000, both u and noise redrawn")
rng = np.random.default_rng(2026)
R = 4000
z = rng.standard_normal((R, n)); U = (z * np.sqrt(np.clip(w, 0, None))) @ V.T
E = rng.standard_normal((R, n)) * sigma
Y = U + E
for lam in [0.1, 1.0, 10.0]:
    a = w / (w + lam)
    MU = ((Y @ V) * a) @ V.T
    vpd = (V ** 2 * (lam * w / (w + lam))).sum(axis=1)
    cover = np.mean(np.abs(U - MU) < 1.96 * np.sqrt(vpd), axis=1)
    mse = np.mean((U - MU) ** 2, axis=1)
    print(" lam=%-4g: cover mean=%.3f std=%.3f  q10=%.2f q90=%.2f  | MSE=%.4f vs postvar=%.4f"
          % (lam, cover.mean(), cover.std(), np.quantile(cover, .1), np.quantile(cover, .9), mse.mean(), vpd.mean()))

print("\n[fixed u_true, redraw noise only]  (与脚本评价方式一致)")
np.random.seed(42)
u_true = np.random.multivariate_normal(np.zeros(n), Sigma)
_ = np.random.randn(n)  # 消耗脚本中噪声流位置（下面重新生成）
for lam in [0.1, 1.0, 10.0]:
    a = w / (w + lam)
    vpd = (V ** 2 * (lam * w / (w + lam))).sum(axis=1)
    cs = []
    rr = np.random.default_rng(7)
    for _ in range(4000):
        y1 = u_true + rr.standard_normal(n) * sigma
        mu1 = ((y1 @ V) * a) @ V.T
        cs.append(np.mean(np.abs(u_true - mu1) < 1.96 * np.sqrt(vpd)))
    cs = np.array(cs)
    print(" lam=%-4g: cover mean=%.3f std=%.3f  P(cover>0.999)=%.3f  P(cover<0.5)=%.3f  min=%.2f max=%.2f"
          % (lam, cs.mean(), cs.std(), (cs > 0.999).mean(), (cs < 0.5).mean(), cs.min(), cs.max()))

print("\n[script seed exact numbers]")
np.random.seed(42)
u_true = np.random.multivariate_normal(np.zeros(n), Sigma)
y = u_true + sigma * np.random.randn(n)
for lam in [0.1, 1.0, 10.0]:
    Sp = lam * np.linalg.solve(Sigma + lam * np.eye(n), Sigma)
    mu = np.linalg.solve(lam * np.eye(n) + Sigma, Sigma @ y)
    Sp_e = (V * (lam * w / (w + lam))) @ V.T
    cov_num = np.mean(np.abs(u_true - mu) < 1.96 * np.sqrt(np.diag(Sp)))
    cov_ideal = np.mean(np.abs(u_true - mu) < 1.96 * np.sqrt(np.diag(Sp_e)))
    print(" lam=%-4g: coverage(numeric diag)=%.3f  coverage(eigen diag)=%.3f  RMSE=%.3f sqrt(postvar)=%.3f  ratio=%.3f"
          % (lam, cov_num, cov_ideal, np.sqrt(np.mean((u_true - mu) ** 2)), np.sqrt(np.diag(Sp_e).mean()),
             np.sqrt(np.mean((u_true - mu) ** 2)) / np.sqrt(np.diag(Sp_e).mean())))
print(" u_true var=%.3f  ||u_true||=%.2f ; noise ||y-u||=%.2f" % (u_true.var(), np.linalg.norm(u_true), np.linalg.norm(y - u_true)))


