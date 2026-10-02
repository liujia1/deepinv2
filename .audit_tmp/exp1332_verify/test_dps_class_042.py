# -*- coding: utf-8 -*-
"""审计用冒烟测试（deepinv 0.4.2, CPU）：
验证 13.3-2.py 新 DPS 配置的端到端可用性：
  dinv.sampling.DPS(denoiser=DRUNet, schedule='vp', weight=2.0, alpha=1.0,
                    dtype=float32, rng=Generator, minus_one_one=False) + seed=42
检查项：
  1) 能跑通、输出 [0,1] 形状正确、无 "seed is ignored" 告警
  2) 同 seed 两次调用输出一致（seed 真正生效）
  3) 引导有效性：weight=2.0 的 ||A x_hat - y|| 应显著小于 weight=1e-6（近无条件）
"""
import warnings
import torch
import deepinv as dinv
from deepinv.sampling import DPS

torch.manual_seed(0)
dev = torch.device("cpu")

drunet = dinv.models.DRUNet(in_channels=3, out_channels=3, pretrained="download").to(dev).eval()
x = torch.rand(1, 3, 64, 64)
kernel = torch.ones(1, 1, 5, 5) / 25
physics = dinv.physics.Blur(filter=kernel, device=dev, padding="circular",
                            noise_model=dinv.physics.GaussianNoise(sigma=0.02))
y = physics(x)
print("y shape:", tuple(y.shape), "range:", float(y.min()), float(y.max()))


def run(weight, seed=42):
    model = DPS(
        denoiser=drunet,
        schedule="vp",
        num_steps=30,
        weight=weight,
        alpha=1.0,
        verbose=False,
        device=dev,
        dtype=torch.float32,
        rng=torch.Generator(device=dev),
        minus_one_one=False,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = model(y, physics, seed=seed)
        msgs = [str(u.message) for u in w if "seed" in str(u.message).lower()]
    return out, msgs

with warnings.catch_warnings(record=True) as wall:
    warnings.simplefilter("always")
    out1, msgs1 = run(2.0)
    out2, msgs2 = run(2.0, seed=123)
    out_weak, _ = run(1e-6)

print("DPS output shape:", tuple(out1.shape), "min/max:", float(out1.min()), float(out1.max()))
print("seed warnings (guided run):", msgs1)
same = torch.allclose(out1, run(2.0)[0], atol=0)
identical = torch.equal(out1, run(2.0)[0])
print("same seed=42 twice bitwise identical:", identical)
print("different seed differs:", not torch.equal(out1, out2))

res_guided = (physics.A(out1.clamp(0, 1)) - y).norm().item()
res_weak = (physics.A(out_weak.clamp(0, 1)) - y).norm().item()
print(f"||A x_hat - y||  guided(w=2.0): {res_guided:.4f}   near-unconditional(w=1e-6): {res_weak:.4f}")
print("VERDICT:", "PASS" if (res_guided < res_weak and identical) else "CHECK")
