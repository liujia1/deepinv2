# -*- coding: utf-8 -*-
"""13.3-2 DPS 子图失败诊断（二）：SDE/权重对比

假设：DRUNet 只在 sigma<~1 是有效 MMSE 去噪器（见 diag_dps_fail.py 扫描），
      而 DPS 默认的 VP-SDE 的 sigma_t 最大 ~151，先验得分在大部分轨迹上失效。
验证：在真实图上对比
  - VP-SDE（实验当前配置，sigma_max≈151）
  - VE-SDE sigma_max=10（官方 demo_diffusion_sde.py 的 DRUNet 配置）
  - VE-SDE sigma_max=1（贴合 DRUNet 有效范围）
以及 weight 与"无引导"对照组。
"""
import time
import numpy as np
import torch
import deepinv as dinv
from deepinv.sampling import (
    PosteriorDiffusion,
    VariancePreservingDiffusion,
    VarianceExplodingDiffusion,
    EulerSolver,
    DPSDataFidelity,
)

dev = torch.device("cpu")
torch.manual_seed(0)
IMG = 64

img_path = (r"C:\Users\admin\.cache\huggingface\hub\models--lllyasviel--sd-controlnet-depth"
            r"\snapshots\35e42a3ea49845b3c76f202f145f257b9fb1b7d4\images\stormtrooper.png")
from PIL import Image
im = Image.open(img_path).convert("RGB")
w, h = im.size
s = min(w, h)
im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((IMG, IMG), Image.LANCZOS)
x = torch.from_numpy(np.array(im)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
print("x:", tuple(x.shape), "range", float(x.min()), float(x.max()))

drunet = dinv.models.DRUNet(pretrained="download").to(dev).eval()

k = torch.tensor(
    [[np.exp(-((i - 4) ** 2 + (j - 4) ** 2) / (2 * 2.0 ** 2)) for j in range(9)] for i in range(9)],
    dtype=torch.float32,
)
k = k / k.sum()
physics = dinv.physics.Blur(filter=k.view(1, 1, 9, 9), device=dev, padding="circular",
                            noise_model=dinv.physics.GaussianNoise(sigma=0.02))
y = physics(x)
psnr = lambda a, b: 10 * np.log10(1 / (torch.mean((a - b) ** 2).item() + 1e-12))
print(f"观测 y  PSNR={psnr(y.clamp(0, 1), x):.2f} dB")

with torch.no_grad():
    base = drunet(y.clamp(0, 1), torch.tensor(0.08)).clamp(0, 1)
print(f"纯 DRUNet(y, 0.08) 基线  PSNR={psnr(base, x):.2f} dB  (无扩散过程的 PnP 参考)")


def run(tag, sde, weight, steps):
    ts = torch.linspace(1, 0.001, steps)
    solver = EulerSolver(timesteps=ts, rng=torch.Generator(device=dev))
    if weight > 0:
        fid = DPSDataFidelity(denoiser=drunet, weight=weight)
    else:
        from deepinv.optim import ZeroFidelity
        fid = ZeroFidelity()
    model = PosteriorDiffusion(data_fidelity=fid, denoiser=drunet, sde=sde, solver=solver,
                               dtype=torch.float32, device=dev, verbose=False)
    t0 = time.time()
    out = model(y, physics, seed=42).detach().clamp(0, 1)
    print(f"  {tag:38s} PSNR={psnr(out, x):6.2f} dB  ||Ax-y||={float(torch.linalg.norm(physics.A(out) - y)):6.3f}"
          f"  [{time.time() - t0:.0f}s]")


print("\n=== VP-SDE（实验当前配置）===")
run("VP weight=0 (纯先验/无引导)", VariancePreservingDiffusion(alpha=1.0, device=dev, dtype=torch.float32), 0.0, 100)
run("VP weight=0.5", VariancePreservingDiffusion(alpha=1.0, device=dev, dtype=torch.float32), 0.5, 100)
run("VP weight=2.0 (实验配置)", VariancePreservingDiffusion(alpha=1.0, device=dev, dtype=torch.float32), 2.0, 100)

print("\n=== VE-SDE ===")
run("VE sigma_max=10 weight=0", VarianceExplodingDiffusion(sigma_max=10.0, alpha=0.75, device=dev, dtype=torch.float32), 0.0, 100)
run("VE sigma_max=10 weight=0.3 (官方demo)", VarianceExplodingDiffusion(sigma_max=10.0, alpha=0.75, device=dev, dtype=torch.float32), 0.3, 100)
run("VE sigma_max=1 weight=0.3", VarianceExplodingDiffusion(sigma_max=1.0, alpha=0.75, device=dev, dtype=torch.float32), 0.3, 100)
run("VE sigma_max=0.5 weight=0.3", VarianceExplodingDiffusion(sigma_max=0.5, alpha=0.75, device=dev, dtype=torch.float32), 0.3, 100)