# -*- coding: utf-8 -*-
"""种子鲁棒性测试：zeta=0.3/0.7 各跑 4 个种子，量化缺口内填充的种子间方差。
回答：Colab(伪影,24.0dB) vs 本地 base(30.2dB) 的差异是 RNG 抽签还是配置性差异。
"""
import os
import time
import numpy as np
import torch
import deepinv as dinv
from deepinv.models import DRUNet
from deepinv.sampling import DiffPIR

OUT_DIR = r"d:\works\deepinv\.audit_tmp\exp1332_verify"
IMG_PATH = OUT_DIR + r"\celeba_example.jpg"
DRUNET_PATH = os.path.expanduser(
    r"~\.cache\torch\hub\checkpoints\drunet_deepinv_color_finetune_22k.pth")
device = torch.device("cpu")
IMG_SIZE = 64
HOLE = (slice(24, 40), slice(24, 40))

x_test = dinv.utils.load_image(IMG_PATH, img_size=IMG_SIZE, resize_mode="resize", device=device)
mask = torch.ones_like(x_test)
mask[..., HOLE[0], HOLE[1]] = 0.0
physics = dinv.physics.Inpainting(img_size=x_test.shape[1:], mask=mask, device=device)
y = physics(x_test)

drunet = DRUNet(pretrained=DRUNET_PATH).to(device).eval()
x_np = x_test[0].permute(1, 2, 0).numpy()
hole_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
hole_mask[HOLE[0], HOLE[1]] = True
vis_mask = ~hole_mask

print(f"{'配置':16s} {'seed':>5s} {'总PSNR':>8s} {'缺口内MSE':>10s} {'暗像素':>7s}")
for zeta in (0.3, 0.7):
    for seed in (0, 1, 2, 3):
        model = DiffPIR(model=drunet, data_fidelity=dinv.optim.L2(),
                        sigma=0.02, max_iter=100, zeta=zeta, lambda_=7.0,
                        verbose=False, device=device)
        xh = model(y, physics, seed=seed).detach().clamp(0, 1)[0].permute(1, 2, 0).numpy()
        mse_in = float(np.mean((xh - x_np)[hole_mask] ** 2))
        dark = float((xh.mean(axis=2)[hole_mask] < 0.25).mean())
        p = 10 * np.log10(1.0 / (np.mean((xh - x_np) ** 2) + 1e-10))
        print(f"zeta={zeta:<12} {seed:>5d} {p:8.2f} {mse_in:10.4f} {dark:7.1%}")
print("完成")
