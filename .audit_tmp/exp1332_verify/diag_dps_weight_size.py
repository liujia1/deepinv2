# -*- coding: utf-8 -*-
"""验证假设：DPSDataFidelity 引导强度 ∝ weight*sqrt(C*H*W)，随图像尺寸增长

背景：GPU 重跑 13.3-2（256x256, VE sigma_max=10, w=0.3, 250步）后 DPS 子图全黑、
     PSNR=8.34。而同配置在 64x64 实测正常（16.60 dB）。
机制：noisy_datafidelity.py L217  out = sqrt(d(A(x0t),y) * N) * weight，
     d=L2Distance()=对 CHW 求和的 ||.||^2，N=y.numel()，
     故 loss = weight * ||Ax0t-y||_2 * sqrt(C*H*W)。
     等效引导强度：64x64 w=0.3 -> 0.3*110.9=33；256x256 w=0.3 -> 0.3*443=133
     （超过 64x64 实测崩溃阈值 w=1.0 -> 110，故 256 全黑）。
预测：256x256 下 w=0.075（=0.3*64/256，等效强度回到 33）应恢复正常图像。
"""
import time
import numpy as np
import torch
import deepinv as dinv
from PIL import Image
from deepinv.sampling import (
    PosteriorDiffusion, VarianceExplodingDiffusion, EulerSolver, DPSDataFidelity,
)

dev = torch.device("cpu")
torch.manual_seed(0)
IMG = 256

img_path = (r"C:\Users\admin\.cache\huggingface\hub\models--lllyasviel--sd-controlnet-depth"
            r"\snapshots\35e42a3ea49845b3c76f202f145f257b9fb1b7d4\images\stormtrooper.png")
im = Image.open(img_path).convert("RGB").resize((IMG, IMG), Image.LANCZOS)
x = torch.from_numpy(np.array(im)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

drunet = dinv.models.DRUNet(pretrained="download").to(dev).eval()
k = torch.tensor([[np.exp(-((i - 4) ** 2 + (j - 4) ** 2) / (2 * 2.0 ** 2)) for j in range(9)] for i in range(9)],
                 dtype=torch.float32)
k = k / k.sum()
physics = dinv.physics.Blur(filter=k.view(1, 1, 9, 9), device=dev, padding="circular",
                            noise_model=dinv.physics.GaussianNoise(sigma=0.02))
y = physics(x)
psnr = lambda a, b: 10 * np.log10(1 / (torch.mean((a - b) ** 2).item() + 1e-12))
print(f"观测 y PSNR={psnr(y.clamp(0, 1), x):.2f} dB   sqrt(C*H*W)={np.sqrt(3 * IMG * IMG):.1f}")

STEPS = 50
panels = [("观测 y", y.clamp(0, 1)), ("真值 x", x)]
for w in (0.3, 0.075, 0.15):
    solver = EulerSolver(timesteps=torch.linspace(1, 0.001, STEPS), rng=torch.Generator(device=dev))
    model = PosteriorDiffusion(data_fidelity=DPSDataFidelity(denoiser=drunet, weight=w),
                               denoiser=drunet,
                               sde=VarianceExplodingDiffusion(sigma_max=10.0, alpha=0.75, device=dev, dtype=torch.float32),
                               solver=solver, dtype=torch.float32, device=dev, verbose=False, minus_one_one=False)
    t0 = time.time()
    raw = model(y, physics, seed=42)
    raw = raw[0] if isinstance(raw, (tuple, list)) else raw
    out = raw.detach().clamp(0, 1)
    p = psnr(out, x)
    print(f"  w={w:5.3f}  raw[min,max]=({float(raw.min()):9.2f},{float(raw.max()):9.2f}) raw.mean={float(raw.mean()):8.3f}"
          f"  PSNR={p:6.2f} dB  ||Ax-y||={float(torch.linalg.norm(physics.A(out) - y)):7.3f}"
          f"  [{time.time() - t0:.0f}s, {STEPS}步]")
    panels.append((f"VE10 w={w}", out))

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
out_png = r"D:\works\deepinv\.audit_tmp\exp1332_verify\dps_weight_size_montage.png"
plt.savefig(out_png, dpi=110)
print("montage:", out_png)
