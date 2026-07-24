# -*- coding: utf-8 -*-
"""调试 forward / adjoint 算子的伴随性 + Lipschitz 估计"""
import numpy as np

np.random.seed(42)
n = 256
L = 4
m = n // L

# 高斯 PSF
s = 4
x_coords = np.concatenate((np.arange(0, n // 2), np.arange(-n // 2, 0)))
Y, X = np.meshgrid(x_coords, x_coords)
h = np.exp(-(X ** 2 + Y ** 2) / (2 * s ** 2))
h = h / np.sum(h)
H_fft = np.fft.fft2(np.fft.fftshift(h))


def blur(x):
    return np.real(np.fft.ifft2(H_fft * np.fft.fft2(x)))


def down_sampling(x):
    """4x4 块求和下采样（与 winter school lab 的 M_L 一致）"""
    out = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            out[i, j] = np.sum(x[L * i:L * i + L, L * j:L * j + L])
    return out


def forward(x):
    """前向算子 y = M A x，其中 M 是 4x4 块求和下采样，A 是循环卷积模糊"""
    return down_sampling(blur(x))


def adjoint(res_low):
    """伴随算子 (M∘A)^T = A^T ∘ M^T
    - M^T (res_low) = 最近邻上采样（每像素复制到 4x4 块，不除以 16）
    - A^T (z)      = 循环互相关（频域共轭）
    """
    up = np.repeat(np.repeat(res_low, L, axis=0), L, axis=1)
    return np.real(np.fft.ifft2(np.conj(H_fft) * np.fft.fft2(up)))


# 1. 验证伴随性 <Kx, y> = <x, K^T y>
np.random.seed(0)
x = np.random.randn(n, n)
y = np.random.randn(m, m)
lhs = np.sum(forward(x) * y)
rhs = np.sum(x * adjoint(y))
print(f"伴随性测试:")
print(f"  <Kx, y> = {lhs:.6f}")
print(f"  <x, K^T y> = {rhs:.6f}")
print(f"  相对误差: {abs(lhs - rhs) / max(abs(lhs), abs(rhs)):.2e}")

# 2. power iteration 估计 Lipschitz
np.random.seed(0)
v = np.random.randn(n, n)
v = v / np.linalg.norm(v)
for it in range(50):
    Kv = forward(v)
    KtKv = adjoint(Kv)
    norm = np.linalg.norm(KtKv)
    v = KtKv / norm
    if (it + 1) % 10 == 0:
        Kv = forward(v)
        KtKv = adjoint(Kv)
        eigval = np.sum(v * KtKv)
        print(f"  iter {it + 1}: ||K^T K v|| = {norm:.4f}, Rayleigh = {eigval:.4f}")

print(f"\nLipschitz 常数估计: {eigval:.4f}")
print(f"建议步长 tau = 0.9 / L = {0.9 / eigval:.4e}")

# 3. 测试 ISTA 一次迭代的步长
# 构造真信号（前向算子作用于真解）+ 噪声
np.random.seed(42)
gt = np.zeros((n, n))
N_mol = 80
margin = 5
for k in range(N_mol):
    i = np.random.randint(margin, n - margin)
    j = np.random.randint(margin, n - margin)
    gt[i, j] = 255.0

sigma_noise = 0.7
y = forward(gt) + sigma_noise * np.random.randn(m, m)
# K^T y
KtY = adjoint(y)
print(f"\n真信号驱动的 K^T y 统计量: max|K^T y| = {np.max(np.abs(KtY)):.4f}, mean|K^T y| = {np.mean(np.abs(KtY)):.4f}")
print(f"  95% 分位: {np.percentile(np.abs(KtY), 95):.4f}")
print(f"  99% 分位: {np.percentile(np.abs(KtY), 99):.4f}")

# ISTA 一次迭代
lmbda = 10
tau = 0.9 / eigval
# ISTA: z = x_k - tau * grad f(x_k) = x_k - tau * K^T(K x_k - y)
# k=0 时 x_0 = 0: z = 0 - tau * K^T(-y) = +tau * K^T y
z = -tau * gradient(np.zeros((n, n)))  # 正确符号：+tau * K^T y
x_new = np.sign(z) * np.maximum(0, np.abs(z) - tau * lmbda)
x_new = np.maximum(x_new, 0)
print(f"tau*lambda = {tau * lmbda:.4f}")
print(f"经过软阈值后 x_new 的非零像素数: {np.sum(x_new > 0)} / {n * n}")
print(f"x_new 最大值: {np.max(x_new):.4f}")
print(f"x_new 均值: {np.mean(x_new):.4f}")
