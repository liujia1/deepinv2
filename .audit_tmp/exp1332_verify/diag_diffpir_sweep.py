# -*- coding: utf-8 -*-
"""DiffPIR 超参扫描（本地 CPU 验证）：检验"缺口内填充不受超参影响"的判断。

设计：与 13.3-2.py 完全同源——同图(celeba_example 64x64)、同缺口[24:40,24:40]、
同 Inpainting(无噪声)、同 DRUNet；只动 DiffPIR 的 sigma / zeta / lambda。
每组输出：总PSNR、缺口内MSE(vs真值)、缺口内暗像素占比(<0.25)、缺口外MSE。
基线(baseline)应复现 24.0dB / 缺口内~0.061。
"""
import os
import time
import numpy as np
import torch
import deepinv as dinv
from deepinv.models import DRUNet
from deepinv.sampling import DiffPIR
from PIL import Image

OUT_DIR = r"d:\works\deepinv\.audit_tmp\exp1332_verify"
IMG_PATH = OUT_DIR + r"\celeba_example.jpg"   # hf-mirror.com 下载，与 Colab 同一 HF 文件
DRUNET_PATH = os.path.expanduser(
    r"~\.cache\torch\hub\checkpoints\drunet_deepinv_color_finetune_22k.pth")
device = torch.device("cpu")
IMG_SIZE = 64
HOLE = (slice(24, 40), slice(24, 40))

# ---- 与 13.3-2.py 逐字同源的数据构造 ----
# load_image 即 load_url_image 内部调用的同一函数，本地路径等价复刻
x_test = dinv.utils.load_image(
    IMG_PATH, img_size=IMG_SIZE, resize_mode="resize", device=device
)
mask = torch.ones_like(x_test)
mask[..., HOLE[0], HOLE[1]] = 0.0
physics = dinv.physics.Inpainting(img_size=x_test.shape[1:], mask=mask, device=device)
y = physics(x_test)
mse_y = torch.mean((y - x_test) ** 2).item()
print(f"观测y PSNR: {10 * np.log10(1.0 / (mse_y + 1e-10)):.2f} dB  (脚本实测15.89)")

print("正在加载 DRUNet（本地缓存权重，离线）...")
t0 = time.time()
drunet = DRUNet(pretrained=DRUNET_PATH).to(device).eval()
print(f"DRUNet加载完成 [{time.time() - t0:.0f}s]")

x_np = x_test[0].permute(1, 2, 0).numpy()
hole_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
hole_mask[HOLE[0], HOLE[1]] = True
vis_mask = ~hole_mask


def psnr(a, b):
    return 10 * np.log10(1.0 / (np.mean((a - b) ** 2) + 1e-10))


CONFIGS = [
    dict(tag="base            sigma=0.02 zeta=0.3 lam=7  it=100", sigma=0.02, zeta=0.3, lambda_=7.0, max_iter=100),
    dict(tag="zeta=0.1        sigma=0.02 zeta=0.1 lam=7  it=100", sigma=0.02, zeta=0.1, lambda_=7.0, max_iter=100),
    dict(tag="zeta=0.7        sigma=0.02 zeta=0.7 lam=7  it=100", sigma=0.02, zeta=0.7, lambda_=7.0, max_iter=100),
    dict(tag="sigma=0.1       sigma=0.1  zeta=0.3 lam=7  it=100", sigma=0.1, zeta=0.3, lambda_=7.0, max_iter=100),
    dict(tag="sigma=0.4       sigma=0.4  zeta=0.3 lam=7  it=100", sigma=0.4, zeta=0.3, lambda_=7.0, max_iter=100),
    dict(tag="lambda=25       sigma=0.02 zeta=0.3 lam=25 it=100", sigma=0.02, zeta=0.3, lambda_=25.0, max_iter=100),
]

results = []
recons = []
print("\n================ 扫描开始 ================")
for cfg in CONFIGS:
    t0 = time.time()
    model = DiffPIR(
        model=drunet,
        data_fidelity=dinv.optim.L2(),
        sigma=cfg["sigma"],
        max_iter=cfg["max_iter"],
        zeta=cfg["zeta"],
        lambda_=cfg["lambda_"],
        verbose=False,
        device=device,
    )
    x_hat = model(y, physics, seed=42)
    dt = time.time() - t0
    xh = x_hat.detach().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    mse_in = float(np.mean((xh - x_np)[hole_mask] ** 2))
    mse_out = float(np.mean((xh - x_np)[vis_mask] ** 2))
    dark_ratio = float((xh.mean(axis=2)[hole_mask] < 0.25).mean())
    p_total = psnr(xh, x_np)
    results.append((cfg["tag"], p_total, mse_in, mse_out, dark_ratio, dt))
    recons.append(xh)
    print(f"{cfg['tag']}  总PSNR={p_total:6.2f}  缺口内MSE={mse_in:.4f}  "
          f"暗像素占比={dark_ratio:5.1%}  缺口外MSE={mse_out:.5f}  [{dt:.0f}s]")

print("\n================ 基线校验 ================")
print("Colab 实测基线: 总PSNR=24.0, 缺口内MSE~0.061")
print("若本地 base 组明显偏离，说明数据/权重链路不一致，扫描结论不可比。")

# ---- 拼接对比条带：每行 = y | 基线 | 各配置重建 ----
scale = 4
tiles = [y[0].permute(1, 2, 0).numpy()] + recons
labels = ["y(obs)"] + [c["tag"].split()[0] for c in CONFIGS]
canvas = np.ones(((IMG_SIZE + 14) * len(tiles), (IMG_SIZE + 8) * scale // scale * 1, 3))  # placeholder
from PIL import ImageDraw
tile_imgs = [np.asarray(Image.fromarray((np.clip(t, 0, 1) * 255).astype(np.uint8)).resize((IMG_SIZE * scale, IMG_SIZE * scale), Image.NEAREST), dtype=np.float64) / 255 for t in tiles]
W = max(t.shape[1] for t in tile_imgs) + 90
Hc = sum(t.shape[0] + 16 for t in tile_imgs)
canvas = np.ones((Hc, W, 3))
ycur = 4
img_pil = Image.new("RGB", (W, Hc), (255, 255, 255))
draw = ImageDraw.Draw(img_pil)
for lab, t in zip(labels, tile_imgs):
    img_pil.paste(Image.fromarray((np.clip(t, 0, 1) * 255).astype(np.uint8)), (86, ycur))
    draw.text((4, ycur + IMG_SIZE * scale // 2 - 4), lab[:12], fill=(0, 0, 0))
    ycur += t.shape[0] + 16
out_png = f"{OUT_DIR}\\diffpir_sweep_montage.png"
img_pil.save(out_png)
print(f"\n对比图已保存: {out_png}  （从上到下: {', '.join(labels)}）")
print("完成")
