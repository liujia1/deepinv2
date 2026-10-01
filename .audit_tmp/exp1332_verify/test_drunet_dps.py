# -*- coding: utf-8 -*-
"""审计用最小实验：验证 deepinv 0.4.0 中 DPS(PosteriorDiffusion+DPSDataFidelity)
是否可以直接使用 DRUNet（普通去噪器），以裁决脚本"关键结论4"的断言：
  'DPS 的 score-based 框架需要 score network（DiffUNet/ADM），不能用普通去噪器（DRUNet）'
CPU 小规模：64x64、5 步。
"""
import torch, numpy as np
import deepinv as dinv
from deepinv.sampling import PosteriorDiffusion, VarianceExplodingDiffusion, EulerSolver, DPSDataFidelity

torch.manual_seed(0)
dev = torch.device("cpu")

drunet = dinv.models.DRUNet(in_channels=3, out_channels=3, pretrained="download").to(dev).eval()
print("DRUNet loaded ok")

x = torch.rand(1, 3, 64, 64)
kernel = torch.ones(1, 1, 5, 5) / 25
physics = dinv.physics.Blur(filter=kernel, device=dev, noise_model=dinv.physics.GaussianNoise(sigma=0.02))
y = physics(x)

fid = DPSDataFidelity(denoiser=drunet, weight=0.5, clip=(0.0, 1.0))
sde = VarianceExplodingDiffusion(sigma_min=0.05, sigma_max=20.0, device=dev)
ts = torch.linspace(sde.T, 0.01, 6)
solver = EulerSolver(timesteps=ts)
model = PosteriorDiffusion(data_fidelity=fid, denoiser=drunet, sde=sde, solver=solver, device=dev)
out = model(y, physics, seed=42)
print("DPS-with-DRUNet output shape:", tuple(out.shape), "min/max:", float(out.min()), float(out.max()))
print("VERDICT: DPSDataFidelity/PosteriorDiffusion accepts DRUNet -> claim 'must be score network' is FALSE")
