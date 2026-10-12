# -*- coding: utf-8 -*-
"""复核实验3.1-1：重算所有数值并与 JSON/讲义表对比，检验断言。"""
import numpy as np
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio
from skimage.data import shepp_logan_phantom

np.random.seed(42)

n = 32
phantom = resize(shepp_logan_phantom(), (n, n), order=0, preserve_range=True, anti_aliasing=False)
x = phantom / phantom.max()
N = n * n

def gaussian_psf(size, sigma):
    ax = np.fft.fftfreq(size, d=1.0) * size
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return h / h.sum()

blur_sigma = 2.0
h = gaussian_psf(n, blur_sigma)
H_fft = np.fft.fft2(h)

A = np.zeros((N, N))
for j in range(N):
    e_j = np.zeros(N)
    e_j[j] = 1.0
    A[:, j] = np.real(np.fft.ifft2(H_fft * np.fft.fft2(e_j.reshape(n, n)))).ravel()

sigma_noise = 0.05
y = A @ x.ravel() + sigma_noise * np.random.randn(N)

sigma_prior = 0.5
lam = sigma_noise ** 2 / sigma_prior ** 2

AtA = A.T @ A
Aty = A.T @ y
Sigma_post = np.linalg.inv(AtA / sigma_noise ** 2 + np.eye(N) / sigma_prior ** 2)
mu_post = Sigma_post @ (Aty / sigma_noise ** 2)

# 检验闭式解是否等于 Tikhonov 公式 (AtA + lam I)^{-1} A^T y
x_tik = np.linalg.solve(AtA + lam * np.eye(N), Aty)
print(f"[闭式解一致性] ||mu_post - x_tikhonov|| = {np.linalg.norm(mu_post - x_tik):.3e}")

def posterior_energy(xv):
    d = 0.5 / sigma_noise**2 * np.linalg.norm(A @ xv - y)**2
    r = 0.5 / sigma_prior**2 * np.linalg.norm(xv)**2
    return d + r, d, r

# 梯度检验：MAP处能量梯度应≈0
grad_at_map = A.T @ (A @ mu_post - y) / sigma_noise**2 + mu_post / sigma_prior**2
print(f"[最优性] ||grad E(mu_post)|| = {np.linalg.norm(grad_at_map):.3e}")
# 全局最小能量
E_min = posterior_energy(mu_post)[0]

x_init_good = A.T @ y
x_init_bad = y
x_map = mu_post

rows = {}
for name, xv in [('观测初始', x_init_bad), ('转置初始', x_init_good), ('MAP', x_map), ('真解', x.ravel())]:
    E, d, r = posterior_energy(xv)
    p = peak_signal_noise_ratio(x, np.clip(xv.reshape(n, n), 0, 1))
    rows[name] = (d, r, E, p)
    print(f"{name:>8}: 数据项={d:8.2f} 正则项={r:8.2f} 能量={E:8.2f} PSNR={p:6.2f}")

# 检查MAP能量是否确为最小
print(f"\n[MAP能量最小?] E_map={rows['MAP'][2]:.2f} 是否 < 其它三个初始: ",
      all(rows['MAP'][2] < rows[k][2] for k in ['观测初始','转置初始','真解']))

# 检验PSNR"显著优于"断言
p_map = rows['MAP'][3]; p_bad = rows['观测初始'][3]; p_good = rows['转置初始'][3]
print(f"[PSNR断言] MAP {p_map:.2f} vs 观测 {p_bad:.2f} (Δ={p_map-p_bad:.2f}dB), vs 转置 {p_good:.2f} (Δ={p_map-p_good:.2f}dB)")

# 检验"转置初始质量较好"断言
print(f"[转置vs观测] 转置能量{rows['转置初始'][2]:.2f} vs 观测能量{rows['观测初始'][2]:.2f}; 转置PSNR{p_good:.2f} vs 观测PSNR{p_bad:.2f}")

# 对照讲义表与JSON
lecture = {
 '观测初始': (545.93, 53.04, 598.97, 14.64),
 '转置初始': (679.18, 43.59, 722.78, 14.60),
 'MAP': (441.16, 67.84, 509.00, 15.02),
 '真解': (490.25, 123.04, 613.28, None),
}
print("\n[与讲义表对比]")
for k, (d, r, E, p) in lecture.items():
    ad, ar, aE, ap = rows[k]
    print(f"  {k}: 数据项 表{d}↔算{ad:.2f} 正则项 表{r}↔算{ar:.2f} 能量 表{E}↔算{aE:.2f}" + (f" PSNR 表{p}↔算{ap:.2f}" if p else ""))

# 后验方差诊断：post_var 空间分布，检验"模糊核中心与边缘处不确定更高"
post_var = np.diag(Sigma_post).reshape(n, n)
print(f"\n[后验方差] min={post_var.min():.3e} max={post_var.max():.3e}")
# 中心 vs 边缘
edge = np.concatenate([post_var[0], post_var[-1], post_var[:,0], post_var[:,-1]])
center = post_var[n//2-3:n//2+3, n//2-3:n//2+3]
print(f"  边缘均值={edge.mean():.3e} 中心均值={center.mean():.3e}")
