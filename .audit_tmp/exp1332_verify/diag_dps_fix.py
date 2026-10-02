# -*- coding: utf-8 -*-
"""13.3-2 DPS 子图修复验证：找出能出"正常图像"的配置

在同一张真实图 + 同一模糊核上跑：
  - 当前配置 VP weight=2.0
  - VP 降权 / VE-SDE（官方 demo_diffusion_sde.py 的 DRUNet 配方）
  - DiffPIR 作为上限对照（同一 DRUNet 先验）
输出 montage PNG 便于肉眼比对。
"""
import time
import numpy as np
import torch
import deepinv as dinv
from deepinv.sampling import (
    PosteriorDiffusion, VariancePreservingDiffusion, VarianceExplodingDiffusion,
    EulerSolver, DPSDataFidelity,
)

dev = torch.device("cpu")
torch.manual_seed(0)
IMG = 64

img_path = (r"C:\Users\admin\.cache\huggingface\hub\models--lllyasviel--sd-controlnet-depth"
            r"\snapshots\35e42a3ea49845b3c76f202f145f257b9fb1b7d4\images\stormtrooper.png")
from PIL import Image
im = Image.open(img_path).convert("RGB")
w0, h0 = im.size
s = min(w0, h0)
im = im.crop(((w0 - s) // 2, (h0 - s) // 2, (w0 + s) // 2, (h0 + s) // 2)).resize((IMG, IMG), Image.LANCZOS)
x = torch.from_numpy(np.array(im)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

drunet = dinv.models.DRUNet(pretrained="download").to(dev).eval()
k = torch.tensor([[np.exp(-((i - 4) ** 2 + (j - 4) ** 2) / (2 * 2.0 ** 2)) for j in range(9)] for i in range(9)],
                 dtype=torch.float32)
k = k / k.sum()
physics = dinv.physics.Blur(filter=k.view(1, 1, 9, 9), device=dev, padding="circular",
                            noise_model=dinv.physics.GaussianNoise(sigma=0.02))
y = physics(x)
psnr = lambda a, b: 10 * np.log10(1 / (torch.mean((a - b) ** 2).item() + 1e-12))
obs = psnr(y.clamp(0, 1), x)
print(f"观测 y PSNR={obs:.2f} dB")

panels = [("观测 y", y.clamp(0, 1)), ("真值 x", x)]


def run_dps(tag, sde, weight, steps):
    solver = EulerSolver(timesteps=torch.linspace(1, 0.001, steps), rng=torch.Generator(device=dev))
    model = PosteriorDiffusion(data_fidelity=DPSDataFidelity(denoiser=drunet, weight=weight),
                               denoiser=drunet, sde=sde, solver=solver,
                               dtype=torch.float32, device=dev, verbose=False)
    t0 = time.time()
    out = model(y, physics, seed=42).detach().clamp(0, 1)
    p = psnr(out, x)
    print(f"  {tag:34s} PSNR={p:6.2f} dB (Δobs={p - obs:+5.2f})  ||Ax-y||={float(torch.linalg.norm(physics.A(out) - y)):6.3f} [{time.time() - t0:.0f}s]")
    panels.append((tag, out))


run_dps("VP weight=2.0 (当前配置)", VariancePreservingDiffusion(alpha=1.0, device=dev, dtype=torch.float32), 2.0, 100)
run_dps("VP weight=0.5", VariancePreservingDiffusion(alpha=1.0, device=dev, dtype=torch.float32), 0.5, 100)
run_dps("VE sigma_max=10 w=0.3 (官方demo)", VarianceExplodingDiffusion(sigma_max=10.0, alpha=0.75, device=dev, dtype=torch.float32), 0.3, 200)
run_dps("VE sigma_max=10 w=1.0", VarianceExplodingDiffusion(sigma_max=10.0, alpha=0.75, device=dev, dtype=torch.float32), 1.0, 200)

diffpir = dinv.sampling.DiffPIR(model=drunet, data_fidelity=dinv.optim.L2(), sigma=0.02,
                                max_iter=100, zeta=0.3, lambda_=7.0, verbose=False, device=dev)
t0 = time.time()
out_dp = diffpir(y, physics, seed=42).detach().clamp(0, 1)
print(f"  {'DiffPIR (对照)':34s} PSNR={psnr(out_dp, x):6.2f} dB (Δobs={psnr(out_dp, x) - obs:+5.2f})"
      f"  ||Ax-y||={float(torch.linalg.norm(physics.A(out_dp) - y)):6.3f} [{time.time() - t0:.0f}s]")
panels.append(("DiffPIR (对照)", out_dp))

n = len(panels)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.4))
for ax, (t, v) in zip(axes, panels):
    ax.imshow(v[0].permute(1, 2, 0).numpy().clip(0, 1))
    ax.set_title(f"{t}\n{psnr(v, x):.2f} dB", fontsize=9)
    ax.axis("off")
plt.tight_layout()
out_png = r"D:\works\deepinv\.audit_tmp\exp1332_verify\dps_fix_montage.png"
plt.savefig(out_png, dpi=110)
print("montage:", out_png)