# -*- coding: utf-8 -*-
"""诊断 16.3-1 的 cs_mri_ista：
   ① 末次迭代是 TV 梯度步，循环后没有再做 DC -> 最终解在已采样 k-space 上不再等于 y
   ② 命名：算法用"硬 DC 替换 + TV 一次梯度下降"，不是 ISTA（无软阈值/近端算子）
量化 ① 的实际影响：PSNR 与 DC 残差 ||M(Fx-y)||/||y||。
"""
import numpy as np
import torch
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr

np.random.seed(42)
torch.manual_seed(42)

n, R = 128, 4
phantom = resize(shepp_logan_phantom(), (n, n), order=0, preserve_range=True, anti_aliasing=False)
phantom = phantom / phantom.max()
x_true = torch.tensor(phantom, dtype=torch.float32)


def mask_variable_density(n, R, seed=42):
    torch.manual_seed(seed)
    n_sample = max(n // R, 1)
    i = torch.arange(n, dtype=torch.float32)
    dist = (i - n // 2).abs() / (n // 2)
    prob = (1 - dist ** 2) ** 1.5 + 0.02
    indices = torch.multinomial(prob / prob.sum(), n_sample, replacement=False)
    mask = torch.zeros(n)
    mask[indices] = 1
    mask[n // 2] = 1
    return mask


def mri_forward(x, mask):
    ks = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    return ks * mask.unsqueeze(1).expand(-1, x.shape[-1])


def cs_mri_ista(y, mask, n_iter=50, lam=0.01, final_dc=False, psnr_ref=None):
    mask_2d = mask.unsqueeze(1).expand(-1, y.shape[-1])
    x = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(y * mask_2d, dim=(-2, -1))))
    curve = []
    for k in range(n_iter):
        kspace_x = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        kspace_dc = mask_2d * y + (1 - mask_2d) * kspace_x
        x_dc = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(kspace_dc, dim=(-2, -1))))
        eps = 1e-8
        dx = torch.diff(x_dc, dim=1, prepend=x_dc[:, :1])
        dy = torch.diff(x_dc, dim=0, prepend=x_dc[:1, :])
        mag = torch.sqrt(dx ** 2 + dy ** 2 + eps)
        div_x = torch.diff(dx / mag, dim=1, append=(dx / mag)[:, -1:])
        div_y = torch.diff(dy / mag, dim=0, append=(dy / mag)[-1:, :])
        tv_grad = -(div_x + div_y)
        x = torch.clamp(x_dc - lam * tv_grad, 0, None)
        if psnr_ref is not None:
            curve.append(psnr(psnr_ref, x.numpy(), data_range=1.0))
    if final_dc:
        # 收尾：把最后一步 TV 更新破坏掉的数据一致性补回来
        ks = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        ks = mask_2d * y + (1 - mask_2d) * ks
        x = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(ks, dim=(-2, -1))))
        x = torch.clamp(x, 0, None)
        if psnr_ref is not None:
            curve.append(psnr(psnr_ref, x.numpy(), data_range=1.0))
    return x, curve


mask = mask_variable_density(n, R)
y = mri_forward(x_true, mask)


def dc_resid(x):
    """已采样 k-space 相对残差 ||M(Fx - y)|| / ||y||"""
    ks = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    m2 = mask.unsqueeze(1).expand(-1, n)
    return float((torch.linalg.norm(m2 * (ks - y)) / torch.linalg.norm(y)))


for tag, fd in (("原版（末次为 TV 步）", False), ("加最终 DC", True)):
    xa, ca = cs_mri_ista(y, mask, n_iter=50, lam=0.01, final_dc=fd, psnr_ref=phantom)
    print("%-22s PSNR=%.3f dB  DC残差=%.3e  迭代末PSNR=%.3f"
          % (tag, psnr(phantom, xa.numpy(), data_range=1.0), dc_resid(xa), ca[-1]))

# 额外：不同 lam 下破坏量
print("\n不同 lam 的末次 DC 残差（原版）：")
for lam in (0.005, 0.01, 0.02, 0.05):
    xa, _ = cs_mri_ista(y, mask, n_iter=50, lam=lam)
    xb, _ = cs_mri_ista(y, mask, n_iter=50, lam=lam, final_dc=True)
    print("  lam=%.3f  残差 %.3e -> %.3e ; PSNR %.3f -> %.3f"
          % (lam, dc_resid(xa), dc_resid(xb),
             psnr(phantom, xa.numpy(), data_range=1.0),
             psnr(phantom, xb.numpy(), data_range=1.0)))
