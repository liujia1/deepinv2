"""实验2.a-1 诊断脚本：验证四个疑点
1) Sigma 是否对称 Toeplitz -> Sigma_post 对角线是否恒定（讲义称"后验方差随位置变化反映空间结构"）
2) 闭式公式误差 4.64e-01 的真实来源（inv(Sigma) 病态 vs 公式本身错误）
3) 覆盖率的频率学校准：重复抽样下 λ=0.1/1/10 的覆盖率均值±std，检验 λ=σ²=1 是否≈95%
4) 单样本覆盖率 100% 是否只是单次实现的幸运抽样（有效独立维数很低）
"""
import numpy as np

n = 100
L = 0.1
x_grid = np.linspace(0, 1, n)
x1, x2 = np.meshgrid(x_grid, x_grid)
Sigma = np.exp(-(x1 - x2) ** 2 / (2 * L ** 2))
sigma = 1.0

print("=" * 70)
print("[1] Sigma 结构检查")
off = np.abs(Sigma - Sigma.T).max()
from scipy.linalg import toeplitz
col = np.exp(-(np.arange(n) * (1 / (n - 1))) ** 2 / (2 * L ** 2))
print(f"  对称性误差 |Sigma - Sigma^T|max = {off:.2e}")
print(f"  与 by-lag Toeplitz 重构误差 = {np.abs(Sigma - toeplitz(col)).max():.2e}")
print(f"  => Sigma 是对称 Toeplitz 矩阵（平稳核 + 均匀网格）")

eigvalsmethod = np.linalg.eigvalsh
evals = eigvalsmethod(Sigma)
print(f"  特征值范围: [{evals.min():.3e}, {evals.max():.3f}]")
print(f"  大于 1e-12 的特征值个数（数值秩）: {(evals > 1e-12).sum()}")
print(f"  大于 1e-8  的特征值个数: {(evals > 1e-8).sum()}")
print(f"  条件数(2范数) = {evals.max()/evals.min():.3e}")

print("\n  各 λ 下 Sigma_post 对角线统计：")
for lam in [0.1, 1.0, 10.0]:
    Sp = lam * np.linalg.solve(Sigma + lam * np.eye(n), Sigma)
    d = np.diag(Sp)
    print(f"  λ={lam:4}: diag min={d.min():.8f} max={d.max():.8f} 极差={d.max()-d.min():.3e} 相对变化={(d.max()-d.min())/d.mean():.3e}")

print("\n" + "=" * 70)
print("[2] 闭式公式误差来源分析")
Sig_inv = np.linalg.inv(Sigma)
Sp_direct = np.linalg.inv(Sig_inv + np.eye(n) / sigma ** 2)
Sp_formula = sigma ** 2 * np.linalg.solve(Sigma + sigma ** 2 * np.eye(n), Sigma)
err = np.abs(Sp_direct - Sp_formula).max()
print(f"  max|direct - formula| = {err:.3e}   （讲义值 4.64e-01）")
# 残差检验: 闭式解应满足 (Sigma + sigma^2 I) Sp = sigma^2 Sigma
res_formula = np.abs((Sigma + sigma ** 2 * np.eye(n)) @ Sp_formula - sigma ** 2 * Sigma).max()
res_direct = np.abs((Sigma + sigma ** 2 * np.eye(n)) @ Sp_direct - sigma ** 2 * Sigma).max()
print(f"  公式解的方程残差 ||(Σ+σ²I)Sp - σ²Σ||_max = {res_formula:.3e}")
print(f"  直接法解的方程残差                          = {res_direct:.3e}")
print(f"  ||inv(Σ)Σ - I||_max = {np.abs(Sig_inv @ Sigma - np.eye(n)).max():.3e}  (Σ⁻¹本身已不可靠)")
print(f"  ||Σ⁻¹||_max = {np.abs(Sig_inv).max():.3e}  （特征值倒数 ~1/{evals.min():.1e}）")

# 长精度：用 float128 重算直接法，看误差是否消失
try:
    Sld = Sigma.astype(np.longdouble)
    Ild = np.eye(n, dtype=np.longdouble)
    Si_ld = np.linalg.inv(Sld)
    Sp_d_ld = np.linalg.inv(Si_ld + Ild / np.longdouble(sigma ** 2))
    # 参考解：用特征分解（数值稳定）
    w, V = np.linalg.eigh(Sigma)
    Sp_ref = (V * (w / (w + sigma ** 2))) @ V.T
    print(f"  [float64 特征分解参考解] 公式解误差 = {np.abs(Sp_formula - Sp_ref).max():.3e}")
    print(f"  [float64 特征分解参考解] 直接法误差 = {np.abs(Sp_direct - Sp_ref).max():.3e}")
except Exception as e:
    print("  longdouble 跳过:", e)

tol_code = max(1e-10, np.linalg.cond(Sigma) * np.finfo(float).eps * 10)
print(f"  代码判据 tol = max(1e-10, cond*eps*10) = {tol_code:.3e} -> err<tol? {err < tol_code}")
print(f"  即代码实际走的是 ✓ 分支（声称一致），因为 tol 被 1e-10 下限卡住？ 不，err>tol → 走警告分支")
print(f"  实际打印：见 run1.log")

print("\n" + "=" * 70)
print("[3] 重复抽样频率学校准（用特征分解，数值稳定）")
w, V = np.linalg.eigh(Sigma)
rng = np.random.default_rng(2026)
R = 4000
lams = [0.1, 1.0, 10.0]
# 采样 u ~ N(0,Σ): u = V (sqrt(w) * z)
z = rng.standard_normal((R, n))
u = (z * np.sqrt(w)) @ V.T
eps = rng.standard_normal((R, n)) * sigma
Y = u + eps
for lam in lams:
    a = w / (w + lam)          # shrinkage per eigen-direction
    MU = ((Y @ V) * a) @ V.T    # posterior mean
    pv = lam * w / (w + lam)    # diag of posterior cov = sum_k V_k^2 * pv_k
    var_post_diag = (V ** 2 * pv).sum(axis=1)
    errv = u - MU
    cover = np.mean(np.abs(errv) < 1.96 * np.sqrt(var_post_diag), axis=1)
    mse = np.mean(errv ** 2, axis=1)
    print(f"  λ={lam:4}: 覆盖率 mean={cover.mean():.3f} ± {cover.std():.3f}"
          f"  (10-90分位 {np.quantile(cover,0.1):.2f}~{np.quantile(cover,0.9):.2f})"
          f" | MSE mean={mse.mean():.3f} vs 平均后验方差={var_post_diag.mean():.3f}")

print("\n  单次实现的覆盖率抽样波动（有效独立维数）:")
# 有效独立块数 ~ 区间长度/相关长度
print(f"  区间[0,1] / 相关长度L=0.1 ≈ 10 个等效独立区域")
for lam in lams:
    a = w / (w + lam)
    pv = lam * w / (w + lam)
    var_post_diag = (V ** 2 * pv).sum(axis=1)
    # 模拟"固定真实u、只换噪声" vs "换u换噪声"
    z1 = rng.standard_normal(n); u1 = (z1 * np.sqrt(w)) @ V.T
    coves = []
    for _ in range(2000):
        e = rng.standard_normal(n) * sigma
        y1 = u1 + e
        mu1 = ((y1 @ V) * a) @ V.T
        coves.append(np.mean(np.abs(u1 - mu1) < 1.96 * np.sqrt(var_post_diag)))
    coves = np.array(coves)
    print(f"  λ={lam:4} 固定u_true只换噪声: 覆盖率 mean={coves.mean():.3f} std={coves.std():.3f}"
          f" P(cover=100%)={np.mean(coves>0.999):.3f} P(cover<0.5)={np.mean(coves<0.5):.3f}")

print("\n" + "=" * 70)
print("[4] 复现脚本种子下的具体数值")
np.random.seed(42)
u_true = np.random.multivariate_normal(np.zeros(n), Sigma)
y = u_true + sigma * np.random.randn(n)
for lam in lams:
    Sp = lam * np.linalg.solve(Sigma + lam * np.eye(n), Sigma)
    mu = np.linalg.solve(lam * np.eye(n) + Sigma, Sigma @ y)
    cov = np.mean(np.abs(u_true - mu) < 1.96 * np.sqrt(np.diag(Sp)))
    rmse = np.sqrt(np.mean((u_true - mu) ** 2))
    print(f"  λ={lam:4}: 覆盖率={cov:.3f}  RMSE={rmse:.3f}  sqrt(平均后验方差)={np.sqrt(np.diag(Sp).mean()):.3f}"
          f"  ||u_map-y||/||noise||={np.linalg.norm(mu-y)/np.linalg.norm(y-u_true):.3f}")
print(f"  ||u_true|| = {np.linalg.norm(u_true):.3f}  (理论 sqrt(n)={np.sqrt(n):.3f})")
print(f"  u_true 的经验方差 = {u_true.var():.3f}  (先验期望1.0)")
