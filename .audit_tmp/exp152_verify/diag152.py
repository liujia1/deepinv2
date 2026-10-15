# 诊断1.5-2：(a) 后验方差是否空间恒定 (b) λ峰值偏移是噪声运气还是系统性 (c) 曲线形状断言
import numpy as np, os, sys
sys.path.insert(0, '.chinese')
from chinese_font import setup_chinese_font
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

n = 32
phantom = resize(shepp_logan_phantom(), (n, n), order=3, preserve_range=True, anti_aliasing=True)
x = phantom / phantom.max()
x_vec = x.ravel()
N = n * n

def gaussian_psf(size, sigma):
    ax = np.concatenate((np.arange(0, size // 2), np.arange(-size // 2, 0)))
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return h / h.sum()

h = gaussian_psf(n, sigma=2.0)
H_fft = np.fft.fft2(h)
A = np.zeros((N, N))
for j in range(N):
    e_j = np.zeros(N)
    e_j[j] = 1.0
    A[:, j] = np.real(np.fft.ifft2(H_fft * np.fft.fft2(e_j.reshape(n, n)))).ravel()

sigma_noise = 0.05
sigma_prior = 0.5
lam = sigma_noise**2 / sigma_prior**2
AtA = A.T @ A

# ---- 复现 seed=42 的观测 ----
np.random.seed(42)
y42 = A @ x_vec + sigma_noise * np.random.randn(N)

# (a) 后验方差空间结构
M = AtA + lam * np.eye(N)
Sig = sigma_noise**2 * np.linalg.inv(M)
pv = np.diag(Sig).reshape(n, n)
print(f"(a) post_var: min={pv.min():.6e} max={pv.max():.6e} mean={pv.mean():.6e}")
print(f"    相对spread (max-min)/mean = {(pv.max()-pv.min())/pv.mean():.3e}")
# 讲义声称"边缘区域方差大，中心区域方差小"——检验边缘环带 vs 中心
edge = np.concatenate([pv[0,:], pv[-1,:], pv[:,0], pv[:,-1]])
center = pv[n//2-4:n//2+4, n//2-4:n//2+4]
print(f"    边缘均值={edge.mean():.6e} 中心均值={center.mean():.6e}")
# 理论：循环矩阵的逆对角恒定
# 用频域验证
Hk = H_fft.ravel()
pv_theory = sigma_noise**2 * np.mean(1.0/(np.abs(Hk)**2 + lam))  # 对角 = (1/N)Σ 1/(|H|²+λ) × σ²
print(f"    频域理论常数 = {pv_theory:.6e}")

# (b) λ 峰值偏移归因：多种子扫描
lambdas_sweep = np.logspace(-5, 1, 50)
def sweep_psnr(y):
    psnrs = []
    for l in lambdas_sweep:
        mu_l = np.linalg.solve(AtA + l*np.eye(N), A.T @ y)
        mse = np.mean((x - mu_l.reshape(n, n))**2)
        psnrs.append(10*np.log10(1.0/mse) if mse > 1e-12 else -np.inf)
    return np.array(psnrs)

p42 = sweep_psnr(y42)
print(f"\n(b) seed42: peak λ={lambdas_sweep[np.argmax(p42)]:.5f} PSNR={p42.max():.2f}, PSNR@lam=0.01: {np.interp(np.log10(lam), np.log10(lambdas_sweep), p42):.2f}")

rng = np.random.default_rng(0)
argmaxes, gaps = [], []
for s in range(30):
    eps = 0.05 * rng.standard_normal(N)
    ys = A @ x_vec + eps
    ps = sweep_psnr(ys)
    am = lambdas_sweep[np.argmax(ps)]
    argmaxes.append(am)
    # PSNR@0.01 与峰值的差
    v01 = np.interp(np.log10(lam), np.log10(lambdas_sweep), ps)
    gaps.append(ps.max() - v01)
argmaxes = np.array(argmaxes); gaps = np.array(gaps)
print(f"    30个独立噪声实现 argmax λ: 中位数={np.median(argmaxes):.4f} "
      f"[min={argmaxes.min():.4f}, max={argmaxes.max():.4f}]")
print(f"    argmax λ ≤ 0.01 的比例: {(argmaxes<=lam).mean():.0%}")
print(f"    峰值PSNR - PSNR@λ=0.01: 中位数={np.median(gaps):.3f} dB")

# (b2) 频率派期望MSE（固定 phantom x，对噪声取期望）——解析可算：
# x_hat = (AtA+λI)^{-1}A^T(Ax+e); err = (I-(AtA+λI)^{-1}AtA)x + (AtA+λI)^{-1}A^T e
# E||err||² = ||Bx||²/N + σ² tr(S A^T A S)/N, S=(AtA+λI)^{-1}  (对像素平均的MSE)
print("\n(b2) 频率派期望MSE曲线（固定phantom，对噪声平均）的最优λ:")
best_l, best_v = None, np.inf
evals, V = np.linalg.eigh(AtA)   # A^T A 对称，特征分解加速
VtAtx = V.T @ (A.T @ x_vec)
for l in lambdas_sweep:
    filt = evals/(evals + l)
    bias = V @ ((1-filt)*VtAtx)          # 偏差项 (I - S A^T A)x
    var = sigma_noise**2 * np.sum(evals/(evals+l)**2)  # σ² tr(S² A^TA)
    emse = (np.sum(bias**2) + var)/N
    if emse < best_v: best_v, best_l = emse, l
print(f"    频率派最优 λ={best_l:.5f}, MSE={best_v:.3e} (PSNR={10*np.log10(1/best_v):.2f} dB)")

# (b3) 贝叶斯期望MSE（先验 x~N(0,σx²I) 且噪声平均）——理论最优应为 λ*=σ²/σx²
# E||x_hat - x||² 对先验样本：用特征分解，x各向同性 → E[(V^T A^T A x)... ]
# MSE_i = [(1-filt_i)² σx² + σ² filt... ] 直接: 对特征方向 i, err_i = (1-filt_i)x_i + filt_i/e_i * e噪声分量
print("(b3) 贝叶斯期望MSE（x~先验 + 噪声）最优λ:")
fine = np.logspace(-4, 0.5, 200)
best_lb, best_vb = None, np.inf
for l in fine:
    filt = evals/(evals+l)
    emse = np.mean((1-filt)**2 * sigma_prior**2 + sigma_noise**2 * filt**2/np.maximum(evals,1e-300)*0 + sigma_noise**2/(evals+l)**2 * evals)
    # var项: σ² * evals/(evals+l)²
    if emse < best_vb: best_vb, best_lb = emse, l
print(f"    贝叶斯理论最优 λ={best_lb:.5f} (对比 σ²/σx²=0.01)")

# (c) 曲线单调性与左右断言
print(f"\n(c) PSNR@λ=1e-5={p42[0]:.2f}, @1e1={p42[-1]:.2f}, 峰值={p42.max():.2f}")
# λ=10 demo 的 PSNR
mu10 = np.linalg.solve(AtA + 10*np.eye(N), A.T @ y42)
print(f"    λ=10: PSNR={10*np.log10(1/np.mean((x-mu10.reshape(n,n))**2)):.2f} dB")
mu1e4 = np.linalg.solve(AtA + 1e-4*np.eye(N), A.T @ y42)
print(f"    λ=1e-4: PSNR={10*np.log10(1/np.mean((x-mu1e4.reshape(n,n))**2)):.2f} dB")
# 观测y本身的PSNR（算法效果基线）
print(f"    含噪观测y的PSNR={10*np.log10(1/np.mean((x-y.reshape(n,n))**2)):.2f} dB")
