"""diag3: 验证讲义定性断言
A) 三个λ的实际重建 RMSE / PSNR（讲义称 λ=1.0 "重建质量最优"）
B) 重复抽样下 RMSE(λ) 曲线，检验 λ=σ²=1 是否平均最优（理论：后验均值是MMSE解）
C) "λ=0.1 重建曲线紧密跟随含噪观测" 的定量检验（与观测的贴合度 vs 与真值的贴合度）
D) 95%区间半宽 / 实际误差 的逐点比值（校准的局部视角）
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np

n, L, sigma = 100, 0.1, 1.0
xg = np.linspace(0, 1, n)
x1, x2 = np.meshgrid(xg, xg)
Sigma = np.exp(-(x1 - x2) ** 2 / (2 * L ** 2))
w, V = np.linalg.eigh(Sigma)
I = np.eye(n)
lams = [0.1, 1.0, 10.0]

np.random.seed(42)
u_true = np.random.multivariate_normal(np.zeros(n), Sigma)
y = u_true + sigma * np.random.randn(n)

print("[A] 单次实现（脚本种子 seed=42）的重建质量")
print("  lambda   RMSE(u_map,u_true)   PSNR_dB   ||u_map-y||/||y-u_true||   平均95%半宽")
for lam in lams:
    Sp = lam * np.linalg.solve(Sigma + lam * I, Sigma)
    mu = np.linalg.solve(lam * I + Sigma, Sigma @ y)
    rmse = np.sqrt(np.mean((u_true - mu) ** 2))
    psnr = 10 * np.log10(np.max(u_true) ** 2 / rmse ** 2) if np.max(u_true) > 0 else float('nan')
    half = 1.96 * np.sqrt(np.diag(Sp)).mean()
    print("  %-6g   %-20.4f %-9.2f %-26.4f %.4f" % (lam, rmse, psnr, np.linalg.norm(mu - y) / np.linalg.norm(y - u_true), half))
rmse_noise = np.sqrt(np.mean((u_true - y) ** 2))
print("  含噪基线 y 的 RMSE = %.4f ; 先验均值0的 RMSE = %.4f" % (rmse_noise, np.sqrt(np.mean(u_true ** 2))))

print("\n[B] 重复抽样 (R=3000, 同时抽 u 与噪声) 的 E[RMSE^2] 与覆盖率")
rng = np.random.default_rng(11)
R = 3000
U = (rng.standard_normal((R, n)) * np.sqrt(np.clip(w, 0, None))) @ V.T
Y = U + rng.standard_normal((R, n)) * sigma
sweep = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
for lam in sweep:
    a = w / (w + lam)
    MU = ((Y @ V) * a) @ V.T
    vpd = (V ** 2 * (lam * np.clip(w, 0, None) / (np.clip(w, 0, None) + lam))).sum(axis=1)
    mse = np.mean((U - MU) ** 2, axis=1).mean()
    cover = np.mean(np.abs(U - MU) < 1.96 * np.sqrt(vpd), axis=1).mean()
    tag = " <-- 脚本测试值" if lam in lams else ""
    print("  lam=%-7g E[RMSE]=%.4f   coverage=%.3f%s" % (lam, np.sqrt(mse), cover, tag))

print("\n[C] '紧密跟随含噪观测' 的定量检验（单次实现）")
for lam in lams:
    mu = np.linalg.solve(lam * I + Sigma, Sigma @ y)
    # 残差对观测的相对大小 & 高频含量（相邻点差分范数，衡量"毛刺"）
    rough = np.linalg.norm(np.diff(mu)) / np.sqrt(n - 1)
    print("  lam=%-6g ||u_map-y||=%.3f  ||u_map-u_true||=%.3f  粗糙度||Δu_map||=%.3f (真实信号 %.3f, 观测 %.3f)"
          % (lam, np.linalg.norm(mu - y), np.linalg.norm(mu - u_true), rough,
             np.linalg.norm(np.diff(u_true)) / np.sqrt(n - 1), np.linalg.norm(np.diff(y)) / np.sqrt(n - 1)))

print("\n[D] 局部校准：mean|误差|/(1.96*后验std) 的分布（应≈0.95 对应95%区间）")
for lam in lams:
    Sp = lam * np.linalg.solve(Sigma + lam * I, Sigma)
    mu = np.linalg.solve(lam * I + Sigma, Sigma @ y)
    ratio = np.abs(u_true - mu) / (1.96 * np.sqrt(np.diag(Sp)))
    print("  lam=%-6g 比值 mean=%.3f median=%.3f  P(ratio>1)=%.3f" % (lam, ratio.mean(), np.median(ratio), (ratio > 1).mean()))

print("\n[E] 理论核对：后验均值的MMSE最优性 & λ=σ²时 E[MSE]=tr(Σ_post)/n")
for lam in lams:
    Sp = lam * np.linalg.solve(Sigma + lam * I, Sigma)
    print("  lam=%-6g tr(Sigma_post)/n = %.4f   实测E[RMSE^2](重复抽样) 见[B]" % (lam, np.trace(Sp) / n))
