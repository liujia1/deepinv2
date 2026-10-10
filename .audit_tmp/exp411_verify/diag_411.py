"""
诊断脚本：实验4.1-1 复核
A. 步骤1 单次 MC 误差的抽样分布（讲义表格中的单点值是否代表典型量级）
B. 步骤2 斜率的可复现性（不同 seed 下斜率分布）
C. 步骤3 
   C1 似然缺归一化常数导致的系统性偏移（即使 M→∞ 也收敛不到 log p(y)）
   C2 固定 sigma=1.0 变维数 d：失败是否由"维数"引起
   C3 固定 d=50 变 sigma：失败是否由"噪声/信息量"引起
   C4 理论相对方差 = exp(KL(post||prior)) - 1，需要 M ~ exp(KL) 才可靠
"""
import numpy as np

print("=" * 70)
print("A. 步骤1：单次 MC(M=10000) 绝对误差的抽样分布")
print("=" * 70)
M = 10000
for d in [1, 2, 3, 4, 5]:
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(2000):
        s = rng.random((M, d))
        errs.append(abs(np.sum(s ** 2, axis=1).mean() - d / 3))
    errs = np.array(errs)
    theory = np.sqrt(4 * d / 45) / np.sqrt(M)  # SE
    print(f"d={d}: 单次|误差| 均值={errs.mean():.6f} 中位数={np.median(errs):.6f} "
          f"5%分位={np.percentile(errs,5):.6f} 95%分位={np.percentile(errs,95):.6f} | 理论SE={theory:.6f}")

print()
print("讲义表(d=1..5) MC误差: 0.001559, 0.001218, 0.005382, 0.005939, 0.000139")
print("本次运行(run1) MC误差: 0.001577, 0.001872, 0.001857, 0.005602, 0.001294")
for d, v in zip([1, 2, 3, 4, 5], [0.001559, 0.001218, 0.005382, 0.005939, 0.000139]):
    rng = np.random.default_rng(0)
    errs = np.array([abs(np.sum(rng.random((M, d)) ** 2, axis=1).mean() - d / 3) for _ in range(2000)])
    pct = (errs < v).mean() * 100
    print(f"  讲义 d={d}: 值={v:.6f} 位于抽样分布的 {pct:.1f}% 分位")

print()
print("=" * 70)
print("B. 步骤2：斜率的可复现性（同一设计，不同 seed，20 次重复）")
print("=" * 70)
M_values = np.array([100, 500, 1000, 2000, 5000, 10000, 20000, 50000])
for d in [1, 10, 100, 1000]:
    slopes = []
    for seed in range(30):
        rng = np.random.default_rng(1000 + seed)
        errs = []
        for M in M_values:
            e = [abs(np.sum(rng.random((M, d)) ** 2, axis=1).mean() - d / 3) for _ in range(20)]
            errs.append(np.mean(e))
        slopes.append(np.polyfit(np.log(M_values), np.log(errs), 1)[0])
    slopes = np.array(slopes)
    print(f"d={d}: 斜率 均值={slopes.mean():+.4f} 标准差={slopes.std():.4f} "
          f"范围=[{slopes.min():+.4f}, {slopes.max():+.4f}]")

print()
print("=" * 70)
print("C1. 步骤3：似然缺归一化常数造成的系统性偏移")
print("=" * 70)
d_demo, sigma, sigma_x = 50, 0.1, 1.0
rng3 = np.random.default_rng(456)
x_true = rng3.standard_normal(d_demo)
A = rng3.standard_normal((d_demo, d_demo)) / np.sqrt(d_demo)
y = A @ x_true + sigma * rng3.standard_normal(d_demo)

cov = sigma_x ** 2 * (A @ A.T) + sigma ** 2 * np.eye(d_demo)
_, logdet = np.linalg.slogdet(cov)
log_py_exact = -0.5 * (d_demo * np.log(2 * np.pi) + logdet + y @ np.linalg.solve(cov, y))
const = -0.5 * d_demo * np.log(2 * np.pi * sigma ** 2)   # 代码 log_likelihood 漏掉的常数
print(f"解析 log p(y) = {log_py_exact:.4f}")
print(f"代码 log_likelihood 缺失常数项 = {const:.4f} (每样本)")
print(f"=> 无偏估计量 E_prior[p_unnorm(y|x)] 的真值 = log p(y) + {-const:.4f} = {log_py_exact - const:.4f}")
print(f"=> 即使 M→∞，代码估计也应收敛到 {log_py_exact - const:.4f}，与 '解析解' 的固有偏差 = {-const:.4f}")

# 用超大 M 检验是否收敛到该值（用 log-sum-exp + 重要性检查：这里用解析的 E[exp(未归一化似然)]）
# E_{p(x)}[exp(-||y-Ax||^2/(2σ^2))] 可解析计算（高斯积分）
# = (2πσ^2)^{-d/2} * ... 用数值：直接算 log
Mhuge = 2_000_000
X = sigma_x * rng3.standard_normal((Mhuge, d_demo))
res = y[None, :] - X @ A.T
ll = -0.5 * np.sum(res ** 2, axis=1) / sigma ** 2
mx = ll.max()
est = np.log(np.mean(np.exp(ll - mx))) + mx
print(f"M={Mhuge} 代码式估计 = {est:.4f}（仍远离 {log_py_exact - const:.4f}，说明方差主导）")

print()
print("=" * 70)
print("C4. 理论：相对方差 = exp(KL(post||prior)) - 1，KL 随 d 与 1/σ 增长")
print("=" * 70)


def kl_post_prior(d, sig, sig_x, seed=7):
    rg = np.random.default_rng(seed)
    xt = rg.standard_normal(d)
    Am = rg.standard_normal((d, d)) / np.sqrt(d)
    yy = Am @ xt + sig * rg.standard_normal(d)
    post_cov = np.linalg.inv(np.eye(d) / sig_x ** 2 + Am.T @ Am / sig ** 2)
    post_mean = post_cov @ (Am.T @ yy / sig ** 2)
    # KL N(m,S) || N(0, sig_x^2 I)
    sgn, ld = np.linalg.slogdet(post_cov)
    return 0.5 * (np.trace(post_cov) / sig_x ** 2 + post_mean @ post_mean / sig_x ** 2 - d - ld + d * np.log(sig_x ** 2))


for d in [1, 5, 10, 20, 50, 100, 200]:
    print(f"  σ=0.1: d={d:4d} KL={kl_post_prior(d, 0.1, 1.0):8.2f} -> 需要 M ~ exp(KL)={np.exp(min(kl_post_prior(d,0.1,1.0),700)):.2e}")
for d in [1, 5, 10, 20, 50, 100, 200]:
    print(f"  σ=1.0: d={d:4d} KL={kl_post_prior(d, 1.0, 1.0):8.2f} -> 需要 M ~ exp(KL)={np.exp(min(kl_post_prior(d,1.0,1.0),700)):.2e}")

print()
print("=" * 70)
print("C2/C3. 控制实验：log p(y) 的绝对误差（5 次试验均值），变 d / 变 σ")
print("=" * 70)


def mc_log_py(d, sig, M, sig_x=1.0, n_trials=5, seed=42, with_const=False):
    rg = np.random.default_rng(seed)
    xt = rg.standard_normal(d)
    Am = rg.standard_normal((d, d)) / np.sqrt(d)
    yy = Am @ xt + sig * rg.standard_normal(d)
    cov = sig_x ** 2 * (Am @ Am.T) + sig ** 2 * np.eye(d)
    _, ld = np.linalg.slogdet(cov)
    exact = -0.5 * (d * np.log(2 * np.pi) + ld + yy @ np.linalg.solve(cov, yy))
    ests = []
    for _ in range(n_trials):
        Xs = sig_x * rg.standard_normal((M, d))
        ll = -0.5 * np.sum((yy[None, :] - Xs @ Am.T) ** 2, axis=1) / sig ** 2
        if with_const:
            ll = ll - 0.5 * d * np.log(2 * np.pi * sig ** 2)
        mx = ll.max()
        ests.append(np.log(np.mean(np.exp(ll - mx))) + mx)
    return exact, np.mean(ests), np.std(ests)


print("--- 固定 σ=1.0（中等信息量），变维数 ---")
for d in [1, 5, 10, 20, 50, 100, 200]:
    ex, e, s = mc_log_py(d, 1.0, 100000)
    print(f"  d={d:4d}: 解析={ex:10.2f} 估计={e:10.2f} 误差={abs(e-ex):9.2f} (试验std={s:.2f})")

print("--- 固定 d=50，变 σ ---")
for sig in [0.05, 0.1, 0.3, 0.5, 1.0, 2.0]:
    ex, e, s = mc_log_py(50, sig, 100000)
    print(f"  σ={sig:4.2f}: 解析={ex:10.2f} 估计={e:10.2f} 误差={abs(e-ex):9.2f} (试验std={s:.2f})")

print("--- 固定 σ=0.1，变维数（讲义场景的维数扫描）---")
for d in [1, 2, 5, 10, 20, 50]:
    ex, e, s = mc_log_py(d, 0.1, 100000)
    print(f"  d={d:4d}: 解析={ex:10.2f} 估计={e:10.2f} 误差={abs(e-ex):9.2f}")

print("--- 修正归一化常数后，讲义原场景 d=50, σ=0.1, M=100000 ---")
ex, e, s = mc_log_py(50, 0.1, 100000, with_const=True)
print(f"  含常数: 解析={ex:.4f} 估计={e:.4f} 误差={abs(e-ex):.4f}")
