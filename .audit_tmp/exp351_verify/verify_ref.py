"""独立验证 实验3.5-1：
1) 用ADMM（保留非光滑结构的正规算法）求全局最优作为参考真值
2) 用Powell（无导数）多起点交叉验证
3) 检查 L-BFGS-B 输出与讲义记录值 8.2699 / 8.2582 的KKT残差
4) 复算网格/随机最优，检查是否有更优网格点/并列
"""
import numpy as np
from scipy.optimize import minimize

A = np.array([[1,0,1,0],[0,1,0,1],[1,1,0,0],[0,0,1,1]], dtype=float)
target = np.array([2.0,2.0,6.0,7.0])
rhs = A @ target
alpha = 1.0
DH = np.array([[1,-1,0,0],[0,0,1,-1]], dtype=float)
DV = np.array([[1,0,-1,0],[0,1,0,-1]], dtype=float)
D = np.vstack([DH, DV])

def F(x):
    return 0.5*np.sum((A@x-rhs)**2) + alpha*(np.sum(np.abs(DH@x)) + np.sum(np.abs(DV@x)))

def grad_data(x):
    return A.T@(A@x-rhs)

# ---- ADMM: min 0.5||Ax-b||^2 + alpha||Dx||_1 ----
def admm(rho=1.0, iters=200000, tol=1e-13):
    M = A.T@A + rho*(D.T@D)
    Minv = np.linalg.inv(M)
    x = np.zeros(4); z = D@x; u = np.zeros(4)
    thr = alpha/rho
    for k in range(iters):
        x_new = Minv@(A.T@rhs + rho*(D.T@(z-u)))
        Dxu = D@x_new + u
        z_new = np.sign(Dxu)*np.maximum(np.abs(Dxu)-thr, 0.0)
        u = u + D@x_new - z_new
        dx = np.linalg.norm(x_new-x)
        x = x_new; z = z_new
        if dx < tol and k > 1000:
            break
    return x, F(x), k

x_admm, f_admm, k_admm = admm()
print(f"ADMM 全局最优: x = {np.round(x_admm,6)}  F = {f_admm:.8f}  (iters={k_admm})")

# 不同 rho 交叉验证
for rho in (0.1, 0.5, 2.0, 10.0):
    xa, fa, ka = admm(rho=rho, iters=100000)
    print(f"  rho={rho}: F = {fa:.8f}, x = {np.round(xa,6)}")

# ---- Powell 无导数多起点 ----
best = (np.inf, None)
rng = np.random.default_rng(0)
for i in range(60):
    x0 = rng.uniform(1, 7, 4)
    r = minimize(F, x0, method='Powell', options={'xtol':1e-12,'ftol':1e-14,'maxiter':100000})
    if r.fun < best[0]:
        best = (r.fun, r.x)
print(f"Powell 多起点最优: F = {best[0]:.8f}, x = {np.round(best[1],6)}")

# ---- KKT 检查：0 ∈ A^T(Ax-b) + alpha * D^T s, s ∈ ∂||·||_1(Dx) ----
def kkt_residual(x):
    g = grad_data(x)  # 数据项梯度
    # TV 次梯度集合: 对每个分量 d_i: s_i = sign(d_i) if d_i!=0 else [-1,1]
    # 检查是否存在 s_i∈[-1,1](仅0处) 使 g + alpha*D^T s = 0
    d = D@x
    # 分量为0的指标
    zero_idx = np.where(np.abs(d) < 1e-9)[0]
    fixed = np.sign(d)
    # 求 D^T s = -g/alpha 是否可对 zero_idx 自由取值 [t_i]∈[-1,1] 满足
    # D^T s = DH^T s_h + DV^T s_v；线性方程，未知 t_i ∈ [-1,1]
    # 写 D^T s = D^T s_fixed + sum_i t_i * (D^T e_i) → 解 4 方程
    Bfix = D.T@fixed
    free_cols = D.T[:, zero_idx]  # 4 x m
    resid_target = -g/alpha - Bfix
    if len(zero_idx) == 0:
        return np.linalg.norm(g + alpha*Bfix), None
    # 最小二乘解 t，并检查范围
    t, *_ = np.linalg.lstsq(free_cols, resid_target, rcond=None)
    in_range = np.all(np.abs(t) <= 1+1e-6)
    r = np.linalg.norm(free_cols@t - resid_target)
    return r, (t, in_range)

for name, x in [
    ("当前代码 L-BFGS-B 输出", np.array([2.5464,2.5464,6.0130,6.0130])),
    ("讲义/PNG 记录值", np.array([2.54,2.54,5.97,5.96])),
    ("ADMM 输出", x_admm),
    ("真解", target),
]:
    r, info = kkt_residual(x)
    print(f"{name}: x={np.round(x,4)}, F={F(x):.6f}, KKT残差={r:.2e}, 次梯度可解={info[1] if info else 'N/A(无零分量)'}")

# ---- 网格最优复核（向量化） ----
g = np.linspace(1,7,10)
X = np.array(np.meshgrid(g,g,g,g,indexing='ij')).reshape(4,-1).T
costs = 0.5*np.sum((X@A.T - rhs)**2, axis=1) + np.sum(np.abs(X@DH.T),axis=1) + np.sum(np.abs(X@DV.T),axis=1)
i = np.argmin(costs)
ties = np.where(costs < costs[i]+1e-12)[0]
print(f"网格最优: F={costs[i]:.4f}, x={X[i]}, 并列个数={len(ties)}")
for t in ties[:5]:
    print("   tie:", X[t])

# ---- 无约束(去掉[1,7]限制)网格能否更优? 边界检查 ----
print("网格搜索域 [1,7]^4，最优点是否在内部:", np.all((X[i]>1.01)&(X[i]<6.99)))

# ---- 收缩偏差量化 ----
for name, x in [("真解", target), ("网格", X[i]), ("随机(记录)", np.array([2.37,2.39,6.41,6.00])),
                ("ADMM最优", x_admm)]:
    tv = np.sum(np.abs(DH@x)) + np.sum(np.abs(DV@x))
    df = 0.5*np.sum((A@x-rhs)**2)
    print(f"{name}: TV={tv:.3f}, datafit={df:.4f}, F={tv+df:.4f}, x={np.round(x,3)}")
