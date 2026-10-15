"""实验2.4-2 审查诊断脚本
裁决问题：
  B) Tikhonov 14.78dB 劣于线性反演 —— "循环卷积边界效应"免责是否成立？
     用真实线性算子 A 的 CG 闭式解（正确 Tikhonov）对照 + 边界/内部误差分解 + λ 扫描
  C) TV n_iter=100 "未完全收敛" —— 300/1000 步后 PSNR 如何变化？
  D) "PnP" 只对 A^T y 去噪一次，非迭代 PnP —— 真正的 PnP-GD 迭代能否更好？
  E) 多种子稳健性：排名是否依赖 seed=42？
  A) A_T 是否为 conv2d 零填充前向算子的真伴随？
"""
import numpy as np
import torch
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

torch.manual_seed(42)
device = torch.device('cpu')

def create_test_image(size=64):
    x = np.zeros((1, size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            if abs(i - center) < size // 4 and abs(j - center) < size // 4:
                x[0, i, j] = 1.0
            elif abs(i - center) < size // 3 and abs(j - center) < size // 3:
                x[0, i, j] = 0.5
    return torch.tensor(x, dtype=torch.float32)

def create_blur_kernel(size=5, sigma=1.0):
    x = np.arange(size) - size // 2
    k1 = np.exp(-x**2 / (2 * sigma**2))
    k2 = np.outer(k1, k1)
    k2 = k2 / k2.sum()
    return torch.tensor(k2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def conv2d(x, kernel, padding=None):
    if padding is None:
        padding = kernel.shape[-1] // 2
    return torch.nn.functional.conv2d(x, kernel, padding=padding)

def compute_psnr(x, x_true):
    mse = torch.mean((x - x_true)**2).item()
    return float('inf') if mse < 1e-12 else 10 * np.log10(1.0 / mse)

x_true = create_test_image(64)
kernel = create_blur_kernel(5, 1.0)
A = lambda x: conv2d(x, kernel)
A_T = lambda x: conv2d(x, kernel.flip(-1).flip(-2))
noise_level = 0.05

def make_y(seed):
    g = torch.Generator().manual_seed(seed)
    return A(x_true) + noise_level * torch.randn(x_true.shape, generator=g)

# ─────────────────────────────────────────────
# A) 伴随性检验 <Ax,z> vs <x,A_T z>
# ─────────────────────────────────────────────
print("=" * 60)
print("A) 伴随算子检验 (conv2d 零填充的转置是否等于 flip 卷积)")
g = torch.Generator().manual_seed(7)
z1 = torch.randn(1, 64, 64, generator=g)
z2 = torch.randn(1, 64, 64, generator=g)
lhs = torch.sum(A(z1) * z2).item()
rhs = torch.sum(z1 * A_T(z2)).item()
print(f"  <Ax,z>={lhs:.6f},  <x,A_T z>={rhs:.6f},  相对差={abs(lhs-rhs)/abs(lhs):.2e}")

# ─────────────────────────────────────────────
# B) Tikhonov：循环 Wiener vs 真实线性算子 CG 解
# ─────────────────────────────────────────────
def tikhonov_wiener(y, kernel, lam=0.01):
    h, w = y.shape[-2], y.shape[-1]
    kh, kw = kernel.shape[-2], kernel.shape[-1]
    kp = torch.zeros(h, w)
    kp[:kh, :kw] = kernel.squeeze()
    kp = torch.roll(kp, shifts=(-kh//2, -kw//2), dims=(-2, -1))
    H = torch.fft.fft2(kp)
    Y = torch.fft.fft2(y[0])
    Hc = torch.conj(H)
    return torch.fft.ifft2(Hc * Y / (H * Hc + lam)).real.unsqueeze(0)

def tikhonov_cg(y, lam, n_iter=200):
    """(A^T A + lam I) x = A^T y，用与讲义相同的 A（零填充线性卷积）"""
    b = A_T(y)
    x = torch.zeros_like(b)
    r = b - (A_T(A(x)) + lam * x)
    p = r.clone()
    rs = torch.sum(r * r)
    for _ in range(n_iter):
        Ap = A_T(A(p)) + lam * p
        alpha = rs / torch.sum(p * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r * r)
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x

y = make_y(42)
x_lin = A_T(y)
print("\n" + "=" * 60)
print("B) Tikhonov 免责裁决：循环Wiener vs 线性算子CG解")
print(f"  线性反演 PSNR={compute_psnr(x_lin, x_true):.2f} dB")
for lam in [0.01, 0.003, 0.001, 0.0003]:
    xw = tikhonov_wiener(y, kernel, lam)
    xc = tikhonov_cg(y, lam)
    print(f"  lam={lam:<7} Wiener(循环)={compute_psnr(xw, x_true):7.2f} dB   CG(线性A)={compute_psnr(xc, x_true):7.2f} dB")

# 边界 vs 内部误差分解（lam=0.01）
xw = tikhonov_wiener(y, kernel, 0.01)
err = (xw - x_true) ** 2
mask_border = torch.zeros(64, 64, dtype=torch.bool)
mask_border[:6, :] = True; mask_border[-6:, :] = True
mask_border[:, :6] = True; mask_border[:, -6:] = True
print(f"  Wiener(lam=0.01) 边界12像素带 MSE={err[:, mask_border].mean():.5f}, "
      f"内部 MSE={err[:, ~mask_border].mean():.5f}  (噪声方差={noise_level**2:.4f})")

# ─────────────────────────────────────────────
# C) TV 收敛性：100 vs 300 vs 1000 步
# ─────────────────────────────────────────────
def tv_deblur(y, A, A_T, lam=0.05, n_iter=100, seed=0, record=None):
    torch.manual_seed(seed)
    x = A_T(y).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.05)
    for it in range(n_iter):
        opt.zero_grad()
        data_term = 0.5 * torch.mean((A(x) - y) ** 2)
        dx = torch.diff(x, dim=1)
        dy = torch.diff(x, dim=2)
        tv = torch.mean(torch.sqrt(dx[:, :, :dy.shape[2]] ** 2 + dy[:, :dx.shape[1], :] ** 2 + 1e-10))
        loss = data_term + lam * tv
        loss.backward()
        opt.step()
        if record is not None and (it + 1) in record:
            record[it + 1] = (compute_psnr(x.detach(), x_true), loss.item())
    return x.detach()

print("\n" + "=" * 60)
print("C) TV 收敛性裁决 (lam=0.05, lr=0.05)")
rec = {50: None, 100: None, 200: None, 300: None, 1000: None}
tv_deblur(y, A, A_T, 0.05, 1000, record=rec)
for k, v in rec.items():
    print(f"  n_iter={k:<5} PSNR={v[0]:.2f} dB   loss={v[1]:.5f}")

# ─────────────────────────────────────────────
# D) 真正的迭代 PnP-GD
# ─────────────────────────────────────────────
def simple_denoiser(x, sigma_denoise=0.1):
    is_3d = x.dim() == 3
    if is_3d:
        x = x.unsqueeze(0)
    h, w = x.shape[-2], x.shape[-1]
    x_pad = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='reflect')
    patches = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            patches.append(x_pad[:, :, 1+di:1+di+h, 1+dj:1+dj+w])
    patches = torch.stack(patches, dim=0)
    diff_sq = (patches - x) ** 2
    weights = torch.exp(-diff_sq / (2 * sigma_denoise ** 2))
    weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-10)
    result = (weights * patches).sum(dim=0)
    return result.squeeze(0) if is_3d else result

def pnp_gd(y, step, sigma_d, n_iter):
    x = A_T(y)
    L_est = 1.0  # ||A^T A|| 谱范数近似（高斯核和=1）
    for _ in range(n_iter):
        grad = A_T(A(x) - y)
        x = simple_denoiser(x - step * grad, sigma_d)
    return x

print("\n" + "=" * 60)
print("D) 迭代 PnP-GD（讲义结论页声称的形式 x=D_tau(x - tau A^T(Ax-y))）")
print(f"  单次应用(现代码): PSNR={compute_psnr(simple_denoiser(x_lin, 0.2), x_true):.2f} dB, 线性反演={compute_psnr(x_lin, x_true):.2f} dB")
for step in [0.5, 1.0, 2.0]:
    for sd in [0.05, 0.1]:
        for ni in [50, 200]:
            xp = pnp_gd(y, step, sd, ni)
            print(f"  step={step}, sigma_d={sd}, n_iter={ni}: PSNR={compute_psnr(xp, x_true):.2f} dB")

# 去噪器权重均匀度检查：sigma=0.2 时是否退化为近似均值滤波
print("\n  simple_denoiser 权重形态检查 (对 x_lin):")
x3 = x_lin.unsqueeze(0)
xp3 = torch.nn.functional.pad(x3, (1,1,1,1), mode='reflect')
pat = torch.stack([xp3[:,:,1+di:1+di+64,1+dj:1+dj+64] for di in [-1,0,1] for dj in [-1,0,1]])
w = torch.exp(-(pat - x3)**2 / (2*0.2**2))
w = w / (w.sum(0, keepdim=True) + 1e-10)
print(f"    9个方向权重均值: {[round(float(w[i].mean()),4) for i in range(9)]}")
print(f"    权重标准差(空间平均): {float(w.std(dim=0).mean()):.4f}  → 接近均匀1/9={1/9:.4f}" )

# ─────────────────────────────────────────────
# E) 多种子稳健性
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("E) 多种子稳健性 (方法排名是否依赖 seed=42)")
for seed in [42, 0, 1, 7, 123]:
    ys = make_y(seed)
    xl = A_T(ys)
    p_lin = compute_psnr(xl, x_true)
    p_tik = compute_psnr(tikhonov_wiener(ys, kernel, 0.01), x_true)
    p_pnp = compute_psnr(simple_denoiser(xl, 0.2), x_true)
    p_tv = compute_psnr(tv_deblur(ys, A, A_T, 0.05, 100), x_true)
    print(f"  seed={seed:<4} 线性={p_lin:.2f}  Tik={p_tik:.2f}  TV={p_tv:.2f}  PnP={p_pnp:.2f}")
