# -*- coding: utf-8 -*-
"""256x256 DPS 第三轮：clip + weight 阶梯 / ODE / warm-start

第二轮结论（diag_dps_smax.py）：
  - sigma_max=1/0.5 全部塌黑（先验在纯噪声输入上输出趋黑 + 低 sigma 区 g^2 极小
    -> 先验成硬吸引子，数据项无杠杆）
  - sigma_max=10 轨迹卡死（部分像素出界后回不来）
  - clip=(0,1) 能保住值域但 w=0.075 太弱（3.09 dB）
本轮测：
  G VE10 w=0.075 clip=(0,1)          (clip 基线，80 步看是否随步数改善)
  H VE10 w=0.3   clip=(0,1)          (clip + 中等引导)
  I VE10 w=1.0   clip=(0,1)          (clip + 强引导)
  J VE10 w=0.3   clip=(0,1) alpha=0  (ODE：无噪声注入，排除噪声累积)
  K VE(0.1) w=0.3 clip=(0,1) + x_init=A_dagger(y)+0.1noise
                                     (warm-start：整条轨迹在 DRUNet 全活区)
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
print(f"观测 y PSNR={psnr(y.clamp(0, 1), x):.2f} dB", flush=True)

STEPS = 80
panels = [("观测 y", y.clamp(0, 1)), ("真值 x", x)]


def run(tag, smax, weight, alpha=0.75, clip=(0.0, 1.0), x_init=None):
    solver = EulerSolver(timesteps=torch.linspace(1, 0.001, STEPS), rng=torch.Generator(device=dev))
    model = PosteriorDiffusion(
        data_fidelity=DPSDataFidelity(denoiser=drunet, weight=weight, clip=clip),
        denoiser=drunet,
        sde=VarianceExplodingDiffusion(sigma_max=smax, alpha=alpha, device=dev, dtype=torch.float32),
        solver=solver, dtype=torch.float32, device=dev, verbose=False, minus_one_one=False)
    t0 = time.time()
    kw = {}
    if x_init is not None:
        kw["x_init"] = x_init
    raw = model(y, physics, seed=42, **kw)
    raw = raw[0] if isinstance(raw, (tuple, list)) else raw
    out = raw.detach().clamp(0, 1)
    p = psnr(out, x)
    print(f"  {tag:36s} raw[min,max]=({float(raw.min()):7.2f},{float(raw.max()):7.2f})"
          f"  PSNR={p:6.2f} dB  ||Ax-y||={float(torch.linalg.norm(physics.A(out) - y)):7.3f}"
          f"  [{time.time() - t0:.0f}s, {STEPS}步]", flush=True)
    panels.append((tag, out))


print("\n=== 256x256, 80步 ===", flush=True)
run("G VE10 w=0.075 clip", 10.0, 0.075)
run("H VE10 w=0.3 clip", 10.0, 0.3)
run("I VE10 w=1.0 clip", 10.0, 1.0)
run("J VE10 w=0.3 clip ODE(a=0)", 10.0, 0.3, alpha=0.0)
g0 = physics.A_dagger(y).clamp(0, 1)
ginit = (g0 + 0.1 * torch.randn(g0.shape, generator=torch.Generator().manual_seed(7))).clamp(-0.2, 1.2)
run("K VE0.1 w=0.3 clip warm-start", 0.1, 0.3, x_init=ginit)

n = len(panels)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.4))
for ax, (t, v) in zip(axes, panels):
    ax.imshow(v[0].permute(1, 2, 0).numpy().clip(0, 1))
    ax.set_title(f"{t}\n{psnr(v, x):.2f} dB", fontsize=8)
    ax.axis("off")
plt.tight_layout()
out_png = r"D:\works\deepinv\.audit_tmp\exp1332_verify\dps_round3_montage.png"
plt.savefig(out_png, dpi=110)
print("montage:", out_png, flush=True)
