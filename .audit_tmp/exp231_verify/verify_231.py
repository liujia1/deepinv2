"""实验2.3-1 独立验证脚本
核查项：
 A. Tikhonov "理论最优 λ=σ²/var_x" 是否真的最优（var_x用了E||x||²还是每坐标方差？网格搜索裁决）
 B. 贝叶斯去噪器 N_mc=2000 的 MC 误差/偏差：对比 N_mc=2e4/2e5 与 1D 数值积分的精确 E[x|y]
 C. 总体 MMSE 水平 vs 讲义 17.55%
 D. LASSO λ=0.08 敏感性（网格最优是多少）
 E. 多种子重跑：排序 identity>tikh>lasso>bayes 是否稳健；err_bayes 的随机波动 vs 方法间差距
 F. ESS 多种子分布：低 ESS 免责是否成立
"""
import numpy as np

sigma = 0.1

def sample_cross(n, rng):
    s = np.zeros((n, 2))
    mask = rng.random(n) < 0.5
    vals = rng.uniform(-1, 1, n)
    s[mask, 1] = vals[mask]
    s[~mask, 0] = vals[~mask]
    return s

def nrmse(x_est, x_true):
    return np.sqrt(np.mean(np.sum((x_est - x_true)**2, axis=1)) /
                   np.mean(np.sum(x_true**2, axis=1)))

def soft_thresh(v, l):
    return np.sign(v) * np.maximum(np.abs(v) - l, 0)

def bayes_mc(y, x_mc, sigma):
    diff = x_mc[None, :, :] - y[:, None, :]
    log_w = -np.sum(diff**2, axis=2) / (2 * sigma**2)
    log_w -= log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w)
    w /= w.sum(axis=1, keepdims=True)
    ess = 1.0 / np.sum(w**2, axis=1)
    return w @ x_mc, ess

def bayes_exact(y, sigma, n_grid=8001):
    """十字形先验的精确后验均值（1D 梯形积分，两臂完整似然，log 空间稳定）"""
    t = np.linspace(-1, 1, n_grid)
    dt = t[1] - t[0]
    # x轴臂: x=(t,0) → logw = -((y1-t)²+y2²)/2σ² ; y轴臂: x=(0,t) → -((y1)²+(y2-t)²)/2σ²
    lx = -((y[:, 0:1] - t[None, :])**2 + y[:, 1:2]**2) / (2 * sigma**2)   # (N,G)
    ly = -(y[:, 0:1]**2 + (y[:, 1:2] - t[None, :])**2) / (2 * sigma**2)
    m = np.maximum(lx.max(1), ly.max(1))[:, None]
    ex, ey = np.exp(lx - m), np.exp(ly - m)
    Ax = ex.sum(1) * dt
    Ay = ey.sum(1) * dt
    nux = (ex * t[None, :]).sum(1) * dt
    nuy = (ey * t[None, :]).sum(1) * dt
    Z = Ax + Ay
    return np.stack([nux / Z, nuy / Z], axis=1)

# ---------- 复现原实验 (seed42) ----------
rng = np.random.default_rng(42)
N = 1000
x = sample_cross(N, rng)
y = x + sigma * rng.standard_normal((N, 2))
x_mc = sample_cross(2000, rng)
idx = rng.choice(N, 200, replace=False)

var_total = np.mean(np.sum(x**2, axis=1))
var_coord = np.mean(x**2)  # 每坐标二阶矩
print(f"[A] E||x||^2={var_total:.4f}  每坐标二阶矩={var_coord:.4f}")
print(f"    代码 lambda=σ²/E||x||²={sigma**2/var_total:.4f};  若用每坐标方差 λ=σ²/v={sigma**2/var_coord:.4f}")

# Tikhonov 网格搜索（同一评估集）
grid = np.arange(0.005, 0.301, 0.005)
errs = [nrmse(y / (1 + lam), x) for lam in grid]
i_best = int(np.argmin(errs))
print(f"[A] Tikhonov 网格最优 λ={grid[i_best]:.3f} err={errs[i_best]*100:.2f}% | "
      f"λ=0.0299 err={nrmse(y/(1+0.029866), x)*100:.2f}% | λ=0.06 err={nrmse(y/1.06, x)*100:.2f}%")
# 理论 argmin（期望意义）: c*=v/(v+σ²)
c_star = var_coord / (var_coord + sigma**2)
print(f"[A] 期望最优 c*={c_star:.4f} -> λ*={1/c_star-1:.4f}")

# LASSO 网格
grid_l = np.arange(0.01, 0.251, 0.005)
errs_l = [nrmse(soft_thresh(y, l), x) for l in grid_l]
i_lb = int(np.argmin(errs_l))
print(f"[D] LASSO 网格最优 λ={grid_l[i_lb]:.3f} err={errs_l[i_lb]*100:.2f}% | λ=0.08 err={nrmse(soft_thresh(y,0.08), x)*100:.2f}%")

# Bayes MC 精度 & 精确解
xb2k = bayes_exact(y, sigma)
xb2k_mc, ess = bayes_mc(y, x_mc, sigma)
xb20k, _ = bayes_mc(y, sample_cross(20000, np.random.default_rng(1)), sigma)
xb200k, _ = bayes_mc(y, sample_cross(200000, np.random.default_rng(2)), sigma)
print(f"[B] err: MC2e3={nrmse(xb2k_mc, x)*100:.2f}%  MC2e4={nrmse(xb20k, x)*100:.2f}%  "
      f"MC2e5={nrmse(xb200k, x)*100:.2f}%  精确积分={nrmse(xb2k, x)*100:.2f}%")
print(f"[B] MC2e3 与精确解最大逐点偏差={np.abs(xb2k_mc - xb2k).max():.4f}")

# 总体 MMSE（大样本独立测试集 + 精确积分）
rngP = np.random.default_rng(123)
Np = 20000
xp = sample_cross(Np, rngP)
yp = xp + sigma * rngP.standard_normal((Np, 2))
print(f"[C] 总体 MMSE NRMSE(2e4独立样本,精确积分)={nrmse(bayes_exact(yp, sigma), xp)*100:.2f}%")
print(f"    总体 identity={nrmse(yp, xp)*100:.2f}%  tikh(λ=0.0299)={nrmse(yp/1.029866, xp)*100:.2f}%  "
      f"tikh(λ*)={nrmse(yp/1.06, xp)*100:.2f}%  lasso(0.08)={nrmse(soft_thresh(yp,0.08), xp)*100:.2f}%")

# ---------- E/F: 多种子重跑完整流程 ----------
print("\n[E/F] 20 个种子重跑（每种子: N=1000 训练+测试, N_mc=2000, oracle λ 按该种子数据计算）")
rows = []
for seed in range(20):
    r = np.random.default_rng(seed)
    xs = sample_cross(N, r)
    ys = xs + sigma * r.standard_normal((N, 2))
    vt = np.mean(np.sum(xs**2, axis=1))
    lam_t = sigma**2 / vt
    xmc = sample_cross(2000, r)
    _ = r.choice(N, 200, replace=False)  # 保持与主脚本相同的 rng 消耗顺序
    xb, es = bayes_mc(ys, xmc, sigma)
    e_id = nrmse(ys, xs); e_ti = nrmse(ys/(1+lam_t), xs)
    e_la = nrmse(soft_thresh(ys, 0.08), xs); e_ba = nrmse(xb, xs)
    # 该种子下真正最优的 tikhonov λ（每坐标）
    vc = np.mean(xs**2)
    e_ti_opt = nrmse(ys/(1+sigma**2/vc), xs)
    rows.append((e_id, e_ti, e_la, e_ba, e_ti_opt, es.mean(), es.min()))
rows = np.array(rows)
names = ['identity', 'tikh(codeλ)', 'lasso', 'bayes']
for j, nm in enumerate(names):
    print(f"    {nm:12s} mean={rows[:,j].mean()*100:6.2f}%  std={rows[:,j].std()*100:.2f}%  "
          f"range=[{rows[:,j].min()*100:.2f}, {rows[:,j].max()*100:.2f}]")
print(f"    tikh(λ*)  mean={rows[:,4].mean()*100:6.2f}%  (修正λ后的期望最优)")
order_ok = np.sum((rows[:,0] > rows[:,1]) & (rows[:,1] > rows[:,2]) & (rows[:,2] > rows[:,3]))
order_ok2 = np.sum((rows[:,0] > rows[:,4]) & (rows[:,4] > rows[:,2]) & (rows[:,2] > rows[:,3]))
print(f"    排序 id>tikh>lasso>bayes 成立比例: {order_ok}/20；用修正λ后: {order_ok2}/20")
gap = (rows[:,2] - rows[:,3])
print(f"    lasso-bayes 差距 mean={gap.mean()*100:.2f}%  min={gap.min()*100:.2f}%  (对比 bayes 跨种子std {rows[:,3].std()*100:.2f}%)")
print(f"    ESS mean 范围 [{rows[:,5].min():.0f}, {rows[:,5].max():.0f}],  ESS min 范围 [{rows[:,6].min():.0f}, {rows[:,6].max():.0f}]")
