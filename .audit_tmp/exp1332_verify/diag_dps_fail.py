# -*- coding: utf-8 -*-
"""13.3-2 DPS 子图失败诊断（CPU, deepinv 0.4.2）

检查两件事：
  A) DRUNet 的有效 sigma 范围：VP-SDE 的 sigma_t 会到 ~151，若 DRUNet 在
     大 sigma 下输出垃圾，则整个先验得分无效
  B) DPS 类在 minus_one_one=False / True、不同 weight 下的 PSNR
"""
import time
import numpy as np
import torch
import deepinv as dinv

dev = torch.device("cpu")
torch.manual_seed(0)

IMG = 64
try:
    url = dinv.utils.get_image_url("celeba_example.jpg")
    x = dinv.utils.load_url_image(url=url, img_size=IMG, resize_mode="resize").to(dev)
except Exception as e:  # pragma: no cover
    print("图片下载失败，改用合成图:", e)
    x = torch.rand(1, 3, IMG, IMG, device=dev) * 0.5 + 0.25
print("x:", tuple(x.shape), "range", float(x.min()), float(x.max()))

drunet = dinv.models.DRUNet(pretrained="download").to(dev).eval()

# ---------- A) DRUNet 在不同 sigma 下的去噪质量 ----------
print("\n=== A) DRUNet sigma 扫描（输入 [0,1]，加噪后直接送网络）===")
for s in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 50.0, 151.0]:
    g = torch.Generator().manual_seed(1)
    n = x + s * torch.randn(x.shape, generator=g)
    with torch.no_grad():
        out = drunet(n, torch.tensor(s))
    mse = ((out.clamp(0, 1) - x) ** 2).mean().item()
    print(f"  sigma={s:7.2f}  out[min,max]=({float(out.min()):6.2f},{float(out.max()):6.2f})"
          f"  out.mean={float(out.mean()):5.3f}  PSNR_vs_x={10 * np.log10(1 / (mse + 1e-12)):6.2f} dB")

# ---------- B) DPS 配置对比 ----------
k = torch.tensor(
    [[np.exp(-((i - 4) ** 2 + (j - 4) ** 2) / (2 * 2.0 ** 2)) for j in range(9)] for i in range(9)],
    dtype=torch.float32,
)
k = k / k.sum()
physics = dinv.physics.Blur(
    filter=k.view(1, 1, 9, 9), device=dev, padding="circular",
    noise_model=dinv.physics.GaussianNoise(sigma=0.02),
)
y = physics(x)
obs = 10 * np.log10(1 / (torch.mean((y - x) ** 2).item() + 1e-12))
print(f"\n观测 y PSNR = {obs:.2f} dB")

print("\n=== B) DPS 配置对比（num_steps=100, vp, alpha=1.0, 64x64）===")
for moo, w in [(False, 2.0), (False, 10.0), (True, 2.0), (False, 0.5)]:
    t0 = time.time()
    model = dinv.sampling.DPS(
        denoiser=drunet, schedule="vp", num_steps=100, weight=w, alpha=1.0,
        verbose=False, device=dev, dtype=torch.float32,
        rng=torch.Generator(device=dev), minus_one_one=moo,
    )
    out = model(y, physics, seed=42).detach().clamp(0, 1)
    mse = torch.mean((out - x) ** 2).item()
    res = torch.linalg.norm(physics.A(out) - y).item()
    print(f"  minus_one_one={moo!s:5s} weight={w:5.1f}  PSNR={10 * np.log10(1 / (mse + 1e-12)):6.2f} dB"
          f"  ||Ax-y||={res:.3f}  out[min,max]=({float(out.min()):.3f},{float(out.max()):.3f})"
          f"  [{time.time() - t0:.0f}s]")