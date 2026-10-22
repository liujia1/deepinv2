# -*- coding: utf-8 -*-
"""检查 cs_mri_ista 里 torch.abs()（取幅度）对数据一致性的破坏：
   若改用 .real，DC 残差是否归零？PSNR 如何变化？
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


def mask_vd(n, R, seed=42):
    torch.manual_seed(seed)
    i = torch.arange(n, dtype=torch.float32)
    dist = (i - n // 2).abs() / (n // 2)
    prob = (1 - dist ** 2) ** 1.5 + 0.02
    idx = torch.multinomial(prob / prob.sum(), max(n // R, 1), replacement=False)
    m = torch.zeros(n); m[idx] = 1; m[n // 2] = 1
    return m


def fwd(x, m):
    return torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1)) * m.unsqueeze(1).expand(-1, x.shape[-1])


def recon(y, m, n_iter=50, lam=0.01, mode='abs'):
    m2 = m.unsqueeze(1).expand(-1, y.shape[-1])
    x = torch.fft.ifft2(torch.fft.ifftshift(y * m2, dim=(-2, -1)))
    x = torch.abs(x) if mode == 'abs' else x.real
    for _ in range(n_iter):
        ks = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        ks = m2 * y + (1 - m2) * ks
        xd = torch.fft.ifft2(torch.fft.ifftshift(ks, dim=(-2, -1)))
        xd = torch.abs(xd) if mode == 'abs' else xd.real
        dx = torch.diff(xd, dim=1, prepend=xd[:, :1])
        dy = torch.diff(xd, dim=0, prepend=xd[:1, :])
        mag = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        divx = torch.diff(dx / mag, dim=1, append=(dx / mag)[:, -1:])
        divy = torch.diff(dy / mag, dim=0, append=(dy / mag)[-1:, :])
        x = torch.clamp(xd + lam * (divx + divy), 0, None)
    # 最终 DC
    ks = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    ks = m2 * y + (1 - m2) * ks
    xf = torch.fft.ifft2(torch.fft.ifftshift(ks, dim=(-2, -1)))
    xf = torch.abs(xf) if mode == 'abs' else xf.real
    xf = torch.clamp(xf, 0, None)
    return x, xf


def dcres(x, y, m):
    ks = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    m2 = m.unsqueeze(1).expand(-1, n)
    return float(torch.linalg.norm(m2 * (ks - y)) / torch.linalg.norm(y))


m = mask_vd(n, R)
y = fwd(x_true, m)
for mode in ('abs', 'real'):
    xa, xf = recon(y, m, mode=mode)
    print("mode=%-5s  无最终DC: PSNR=%.3f 残差=%.2e | 有最终DC: PSNR=%.3f 残差=%.2e"
          % (mode, psnr(phantom, xa.numpy(), data_range=1.0), dcres(xa, y, m),
             psnr(phantom, xf.numpy(), data_range=1.0), dcres(xf, y, m)))
