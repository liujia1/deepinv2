"""实验3.2-1 诊断脚本：验证讲义关键断言
1) 精确特征值：L_true = max|H|^2 + lam, mu = min|H|^2 + lam, kappa
2) tau=1.8/L 的 J 曲线是否单调（讲义称"过冲-回拉锯齿"）
3) 收敛迭代数复算（阈值=初始差距的0.1%）
4) tau=2.1/L 曲线起点（讲义称"从1.32飙到10万"）
5) 理论最优固定步长 2/(L+mu) 对比
"""
import numpy as np
from skimage.transform import resize
from skimage.data import shepp_logan_phantom

n = 32; N = n*n
phantom = resize(shepp_logan_phantom(), (n, n), order=0, preserve_range=True, anti_aliasing=False)
x = phantom / phantom.max()

def gaussian_psf(size, sigma):
    ax = np.concatenate((np.arange(0, size//2), np.arange(-size//2, 0)))
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    return h/h.sum()

h = gaussian_psf(n, 3.0)
H_fft = np.fft.fft2(h)
sigma_noise = 0.05
np.random.seed(42)
# 复现 y 的生成顺序：先 x 无关，A_op(x)+noise
def A_op(v): return np.real(np.fft.ifft2(H_fft*np.fft.fft2(v.reshape(n,n)))).ravel()
y = A_op(x.ravel()) + sigma_noise*np.random.randn(N)
lam = 0.01

# 1) 精确谱（循环卷积在FFT下对角化）
eigs = np.abs(H_fft)**2  # A^T A 的特征值
L_true = eigs.max() + lam
mu = eigs.min() + lam
print(f"[1] 精确 L_true = {L_true:.6f} (幂迭代报告 1.0100)")
print(f"    mu = {mu:.6f}, kappa = L/mu = {L_true/mu:.1f}")

def objective(v):
    r = A_op(v) - y
    return 0.5*np.dot(r, r) + 0.5*lam*np.dot(v, v)
def grad(v):
    return np.real(np.fft.ifft2(np.conj(H_fft)*np.fft.fft2((A_op(v)-y).reshape(n,n)))).ravel() + lam*v

Y_fft = np.fft.fft2(y.reshape(n,n))
x_star = np.real(np.fft.ifft2(np.conj(H_fft)*Y_fft/(np.abs(H_fft)**2+lam))).ravel()
J_star = objective(x_star)
print(f"    J* = {J_star:.6f}")

def run(tau, n_iter):
    v = np.zeros(N); hist = []
    for k in range(n_iter):
        v = v - tau*grad(v)
        hist.append(objective(v))
    return np.array(hist)

J_init = objective(np.zeros(N))
print(f"[4] J(x_0) = {J_init:.4f}  (讲义发散图虚线 11.9；讲义文字称发散'从1.32飙起')")

for name, tau in [('1/L', 1.0/L_true), ('0.5/L', 0.5/L_true), ('1.8/L', 1.8/L_true), ('2.1/L', 2.1/L_true)]:
    hist = run(tau, 1000 if '2.1' not in name else 50)
    gap = hist - J_star
    inc = np.sum(np.diff(gap) > 1e-12)  # J 上升步数（真正的"振荡/过冲"证据）
    thresh = (J_init - J_star)*1e-3
    idx = np.where(gap < thresh)[0]
    first = idx[0]+1 if len(idx) else None
    # 去掉浮点噪声地板后的上升步数
    valid = gap > 1e-14
    inc_clean = np.sum(np.diff(gap[valid]) > 1e-14)
    print(f"[2/3] tau={name}: 迭代到阈值 {first} 次; J曲线上升步数(全量)={inc}, (排除噪声地板)={inc_clean}; 末值gap={gap[-1]:.3e}")

# 5) 理论最优固定步长
tau_cheb = 2.0/(L_true+mu)
hist = run(tau_cheb, 1000)
gap = hist - J_star
thresh = (J_init - J_star)*1e-3
idx = np.where(gap < thresh)[0]
print(f"[5] Chebyshev最优步长 2/(L+mu)={tau_cheb:.6f}={tau_cheb*L_true:.2f}/L: 到阈值迭代 {idx[0]+1 if len(idx) else None} 次")
