import numpy as np
from skimage import data
from skimage.transform import resize
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from scipy.ndimage import gaussian_filter

np.random.seed(42)
n = 128
x = resize(data.camera(), (n, n))
DATA_RANGE = x.max() - x.min()
blur_sigma = 2.0
def A(u): return gaussian_filter(u, sigma=blur_sigma, mode='reflect')
def AT(u): return gaussian_filter(u, sigma=blur_sigma, mode='reflect')

noise_lev = 1e-1
y = A(x) + noise_lev * np.random.randn(n, n)
sigma = noise_lev; sigma_x = np.std(x)
lambda_Tikh = sigma**2/sigma_x**2

# --- 1) 闭式 MAP 解 (A != I): (A^T A + lam I)^{-1} A^T y 用共轭梯度精确求解 ---
from scipy.sparse.linalg import LinearOperator, cg
# A 作为稀疏线性算子作用于向量
def A_mat(v): return A(v.reshape(n,n)).ravel()
def AT_mat(v): return AT(v.reshape(n,n)).ravel()
LinOp = LinearOperator((n*n, n*n), matvec=A_mat, rmatvec=AT_mat, dtype=np.float64)
b = AT(y).ravel()
def Mop(v): return AT_mat(A_mat(v)) + lambda_Tikh*v
Mlin = LinearOperator((n*n,n*n), matvec=Mop)
sol, info = cg(LinOp@LinOp.rmatvec(None) if False else Mlin, b, maxiter=5000, atol=1e-12)
x_exact = sol.reshape(n,n)

# GD 解（复现讲义）
def solve_gd(y, lam, n_iter=500, lr=None):
    # 复用主脚本的 lr 估计
    rng = np.random.default_rng(0)
    v = rng.standard_normal((n,n)); v/=np.linalg.norm(v)
    for _ in range(20):
        v = AT(A(v)); v/=np.linalg.norm(v)
    A_norm = np.sqrt(max(np.dot(AT(A(v)).ravel(), v.ravel()),0))
    L = A_norm**2+lam
    if lr is None: lr = 0.9*(2.0/L)
    xh = AT(y).copy()
    for i in range(n_iter):
        grad = -AT(y-A(xh)) + lam*xh
        if np.linalg.norm(grad)/(np.linalg.norm(xh)+1e-8) < 1e-6: break
        xh = xh - lr*grad
    return xh
x_gd = solve_gd(y, lambda_Tikh)

def energy(xh, norm=True):
    if norm:
        return 0.5/sigma**2*np.sum((y-A(xh))**2)+0.5/sigma_x**2*np.sum(xh**2)
    return 0.5*np.sum((y-A(xh))**2)+0.5*lambda_Tikh*np.sum(xh**2)

print("=== A!=I 收敛性核查 ===")
print(f"闭式解(GD-free)归一化后验能量 = {energy(x_exact):.4f}")
print(f"脚本GD解归一化后验能量       = {energy(x_gd):.4f}")
print(f"闭式解 PSNR = {peak_signal_noise_ratio(x, x_exact, data_range=DATA_RANGE):.4f}")
print(f"脚本GD解 PSNR = {peak_signal_noise_ratio(x, x_gd, data_range=DATA_RANGE):.4f}")
print(f"两者像素差 ||x_gd-x_exact||/||x_exact|| = {np.linalg.norm(x_gd-x_exact)/np.linalg.norm(x_exact):.5f}")
print(f"残差 grad at GD解 归一化 = {np.linalg.norm(-AT(y-A(x_gd))+lambda_Tikh*x_gd)/np.linalg.norm(x_gd):.6f}")
print(f"残差 grad at 闭式解 归一化 = {np.linalg.norm(-AT(y-A(x_exact))+lambda_Tikh*x_exact)/np.linalg.norm(x_exact):.6f}")

# --- 2) 场景2 纯去噪: 关键基线 ---
print("\n=== A=I 纯去噪 核查 ===")
noise_lev2 = 1e-1
y_d = x + noise_lev2*np.random.randn(n,n)
psnr_noisy = peak_signal_noise_ratio(x, y_d, data_range=DATA_RANGE)
lambda_bayes = noise_lev2**2/sigma_x**2
print(f"图像均值 mean(x) = {x.mean():.4f}, std(x)={sigma_x:.4f}")
print(f"含噪观测 PSNR(基线) = {psnr_noisy:.4f} dB")
print(f"贝叶斯lambda={lambda_bayes:.4f} -> PSNR = {peak_signal_noise_ratio(x, y_d/(1+lambda_bayes), data_range=DATA_RANGE):.4f} dB")
lambdas = np.logspace(-4,1,30)
ps = [peak_signal_noise_ratio(x, y_d/(1+l), data_range=DATA_RANGE) for l in lambdas]
bi = int(np.argmax(ps))
print(f"扫描最优lambda={lambdas[bi]:.4f} -> PSNR = {ps[bi]:.4f} dB")
print(f"lambda->0 (不处理) PSNR = {peak_signal_noise_ratio(x, y_d, data_range=DATA_RANGE):.4f}")
# 理论最优(对零均值高斯先验，若假设成立)
# 真正对该图最优的标量收缩系数 c: min ||x - c*y||
c_opt = np.sum(x*y_d)/np.sum(y_d**2)
print(f"全局最优标量收缩 c*={c_opt:.4f} (对应lambda={1/c_opt-1:.4f}) -> PSNR={peak_signal_noise_ratio(x, c_opt*y_d, data_range=DATA_RANGE):.4f}")
print(f"贝叶lambda是否劣于不处理基线? bayes {peak_signal_noise_ratio(x, y_d/(1+lambda_bayes), data_range=DATA_RANGE):.3f} vs noisy {psnr_noisy:.3f}")
