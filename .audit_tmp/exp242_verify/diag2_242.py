"""补充诊断：Tikhonov λ 向上扫描 + Wiener解的边界/内部再分解 + 理论最优λ"""
import numpy as np
import torch

torch.manual_seed(42)

def create_test_image(size=64):
    x = np.zeros((1, size, size))
    c = size // 2
    for i in range(size):
        for j in range(size):
            if abs(i - c) < size // 4 and abs(j - c) < size // 4:
                x[0, i, j] = 1.0
            elif abs(i - c) < size // 3 and abs(j - c) < size // 3:
                x[0, i, j] = 0.5
    return torch.tensor(x, dtype=torch.float32)

def create_blur_kernel(size=5, sigma=1.0):
    x = np.arange(size) - size // 2
    k1 = np.exp(-x**2 / (2 * sigma**2))
    k2 = np.outer(k1, k1)
    return torch.tensor(k2 / k2.sum(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def conv2d(x, k):
    return torch.nn.functional.conv2d(x, k, padding=k.shape[-1] // 2)

def psnr(x, xt):
    mse = torch.mean((x - xt)**2).item()
    return 10 * np.log10(1.0 / mse) if mse > 1e-12 else float('inf')

x_true = create_test_image(64)
kernel = create_blur_kernel(5, 1.0)
A = lambda x: conv2d(x, kernel)
A_T = lambda x: conv2d(x, kernel.flip(-1).flip(-2))
y = A(x_true) + 0.05 * torch.randn(x_true.shape)

def wiener(y, lam):
    h, w = y.shape[-2:]
    kp = torch.zeros(h, w)
    kp[:5, :5] = kernel.squeeze()
    kp = torch.roll(kp, (-2, -2), (-2, -1))
    H = torch.fft.fft2(kp)
    return torch.fft.ifft2(torch.conj(H) * torch.fft.fft2(y[0]) / (H.abs()**2 + lam)).real.unsqueeze(0)

print("Tikhonov λ 全扫描 (Wiener循环实现)，线性反演基准 PSNR={:.2f}".format(psnr(A_T(y), x_true)))
best = (None, -1)
for lam in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]:
    x = wiener(y, lam)
    p = psnr(x, x_true)
    if p > best[1]:
        best = (lam, p)
    print(f"  lam={lam:<7} PSNR={p:6.2f}")
print(f"  最优 lam={best[0]}, PSNR={best[1]:.2f} dB")

# 理论最优（oracle）：对每个频率用MMSE维纳滤波（已知信号谱）——教学上限参考
# CG 线性算子版扫描
def tikhonov_cg(y, lam, n=300):
    b = A_T(y)
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = (r * r).sum()
    for _ in range(n):
        Ap = A_T(A(p)) + lam * p
        alpha = rs / (p * Ap).sum()
        x += alpha * p
        r -= alpha * Ap
        rs2 = (r * r).sum()
        p = r + (rs2 / rs) * p
        rs = rs2
    return x

print("\nCG(真实线性算子A) λ 扫描:")
for lam in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
    print(f"  lam={lam:<6} PSNR={psnr(tikhonov_cg(y, lam), x_true):6.2f}")

# 边界/内部误差分解（lam=0.01 与最优lam）
for lam in [0.01, best[0]]:
    x = wiener(y, lam)
    err = (x - x_true) ** 2
    border = torch.zeros(64, 64, dtype=torch.bool)
    border[:6, :] = border[-6:, :] = border[:, :6] = border[:, -6:] = True
    print(f"\nlam={lam}: 边界带MSE={err[:, border].mean():.5f} 内部MSE={err[:, ~border].mean():.5f}")
