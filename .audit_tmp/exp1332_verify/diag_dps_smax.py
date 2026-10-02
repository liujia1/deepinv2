# -*- coding: utf-8 -*-
"""256x256 DPS 失败根因定界：σmax 是关键旋钮（把轨迹压进 DRUNet 有效区）

已证事实：
  - 256x256 + VE(sigma_max=10) 下 weight=0.3/0.15/0.075 全部失败
    （PSNR 5.63/8.39/7.30，输出被噪声淹没，等效引导强度与 64x64 成功案例相同）
  - GPU 上 256x256 + w=0.3 和 w=0.075 两次都是全黑（PSNR 都是 8.34=黑图对真值的确定值）
  - DiffPIR（同 DRUNet，sigma<=~1）在 256x256 上 25.9 dB，先验没问题

假设：失败根源不是 weight 而是 sigma_max=10。SDE 每步注入噪声 g(t)*sqrt(alpha*dt)，
去除噪声靠先验得分；DRUNet 只在 sigma<~0.3 是有效 MMSE 去噪器（64x64 sigma 扫描：
0.1->21.05dB, 0.5->16.85 且输出退化, >=20 输出越界），sigma_max=10 的轨迹约 90% 步数
先验死亡 -> 注入的噪声无人清理，越积越多 -> 最终被噪声淹没/发散。
预测：把整条轨迹压进 DRUNet 有效区（sigma_max=0.5 或 1.0）应恢复可用重建。
附加：测试 DPSDataFidelity clip=(0,1)（把 x0t 预测硬投影回合法图像范围）的稳定作用。
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

STEPS = 40
panels = [("观测 y", y.clamp(0, 1)), ("真值 x", x)]


def run(tag, smax, weight, clip=None, trace=False):
    solver = EulerSolver(timesteps=torch.linspace(1, 0.001, STEPS), rng=torch.Generator(device=dev))
    model = PosteriorDiffusion(
        data_fidelity=DPSDataFidelity(denoiser=drunet, weight=weight, clip=clip),
        denoiser=drunet,
        sde=VarianceExplodingDiffusion(sigma_max=smax, alpha=0.75, device=dev, dtype=torch.float32),
        solver=solver, dtype=torch.float32, device=dev, verbose=False, minus_one_one=False)
    t0 = time.time()
    if trace:
        raw, traj = model(y, physics, seed=42, get_trajectory=True)
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        ts = torch.linspace(1, 0.001, STEPS)
        for i in range(0, STEPS, 5):
            t = float(ts[i])
            sig = float(model.sde.sigma_t(t))
            ti = traj[i].detach()
            print(f"    step{i:3d} t={t:.3f} sigma={sig:7.3f} "
                  f"x_t[min,max]=({float(ti.min()):7.2f},{float(ti.max()):7.2f})", flush=True)
    else:
        raw = model(y, physics, seed=42)
        raw = raw[0] if isinstance(raw, (tuple, list)) else raw
    out = raw.detach().clamp(0, 1)
    p = psnr(out, x)
    print(f"  {tag:38s} raw[min,max]=({float(raw.min()):7.2f},{float(raw.max()):7.2f})"
          f"  PSNR={p:6.2f} dB  ||Ax-y||={float(torch.linalg.norm(physics.A(out) - y)):7.3f}"
          f"  [{time.time() - t0:.0f}s, {STEPS}步]", flush=True)
    panels.append((tag, out))


print("\n=== 256x256, 40步, alpha=0.75 ===", flush=True)
run("A sigma_max=10 w=0.075 (轨迹追踪)", 10.0, 0.075, trace=True)
run("B sigma_max=10 w=0.075 clip=(0,1)", 10.0, 0.075, clip=(0.0, 1.0))
run("C sigma_max=1  w=0.075", 1.0, 0.075)
run("D sigma_max=0.5 w=0.075", 0.5, 0.075)
run("E sigma_max=0.5 w=0.3", 0.5, 0.3)

print("\n=== 64x64 对照（同一脚本逻辑, 100步）===", flush=True)
IMG = 64
im2 = Image.open(img_path).convert("RGB").resize((IMG, IMG), Image.LANCZOS)
x2 = torch.from_numpy(np.array(im2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
y2 = physics(x2)
print(f"观测 y PSNR={psnr(y2.clamp(0, 1), x2):.2f} dB", flush=True)
solver = EulerSolver(timesteps=torch.linspace(1, 0.001, 100), rng=torch.Generator(device=dev))
m64 = PosteriorDiffusion(data_fidelity=DPSDataFidelity(denoiser=drunet, weight=0.075),
                         denoiser=drunet,
                         sde=VarianceExplodingDiffusion(sigma_max=10.0, alpha=0.75, device=dev, dtype=torch.float32),
                         solver=solver, dtype=torch.float32, device=dev, verbose=False, minus_one_one=False)
raw = m64(y2, physics, seed=42)
raw = raw[0] if isinstance(raw, (tuple, list)) else raw
out = raw.detach().clamp(0, 1)
print(f"  F 64x64 sigma_max=10 w=0.075     raw[min,max]=({float(raw.min()):7.2f},{float(raw.max()):7.2f})"
      f"  PSNR={psnr(out, x2):6.2f} dB", flush=True)
panels.append(("F 64x64 smax=10 w=0.075", out))

n = len(panels)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.4))
for ax, (t, v) in zip(axes, panels):
    ax.imshow(v[0].permute(1, 2, 0).numpy().clip(0, 1))
    ax.set_title(f"{t}\n{psnr(v, x if not t.startswith('F') else x2):.2f} dB", fontsize=8)
    ax.axis("off")
plt.tight_layout()
out_png = r"D:\works\deepinv\.audit_tmp\exp1332_verify\dps_smax_montage.png"
plt.savefig(out_png, dpi=110)
print("montage:", out_png, flush=True)
