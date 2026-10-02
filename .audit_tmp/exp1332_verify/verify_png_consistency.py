# -*- coding: utf-8 -*-
"""DPS 面板可见区一致性深度验证（修正版）。

已知：无测量噪声 + Inpainting -> 缺口外 y == x_test 逐像素相等。
故 MSE(面板2, 面板1) 在缺口外 == MSE(x_hat_dps, x_test) 在缺口外，
可直接从 PNG 量化 DPS 的"可见区改写"程度（DiffPIR 作提取标定）。
"""
import numpy as np
from PIL import Image

PNG = r"d:\works\deepinv\第13章 条件生成与逆问题求解\第十三章 配套实验\实验13.3-2\DPSvsDiffPIR对比.png"
OUT_DIR = r"d:\works\deepinv\.audit_tmp\exp1332_verify"

img = np.asarray(Image.open(PNG).convert("RGB"), dtype=np.float64) / 255.0
H, W, _ = img.shape

# ---- 面板提取（同前，已验证可信）----
col_density = (img.min(axis=2) < 0.95).mean(axis=0)
groups, start = [], None
for j in range(W):
    dense = col_density[j] > 0.3
    if dense and start is None:
        start = j
    elif not dense and start is not None:
        if j - start >= 0.1 * W:
            groups.append((start, j))
        start = None
if start is not None and W - start >= 0.1 * W:
    groups.append((start, W))
assert len(groups) == 3
panels_full = []   # 原分辨率面板（用于白像素检测）
panels = []
for c0, c1 in groups:
    sub = img[:, c0:c1]
    row_density = (sub.min(axis=2) < 0.95).mean(axis=1)
    rows = np.where(row_density > 0.5)[0]
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    panels_full.append(img[r0:r1, c0:c1])
    to64 = np.asarray(Image.fromarray(
        (np.clip(img[r0:r1, c0:c1], 0, 1) * 255).astype(np.uint8)
    ).resize((64, 64), Image.BILINEAR), dtype=np.float64) / 255.0
    panels.append(to64)
y_img, dps, pir = panels

mask = np.ones((64, 64), dtype=bool)
mask[24:40, 24:40] = False  # True=可见区

def m(a, b, msk=None):
    d = (a - b) ** 2
    return float(d.mean()) if msk is None else float(d[msk].mean())

print("== 总体分解（64x64, 已由 DiffPIR 缺口外=0.00016 标定）==")
print(f"DPS     缺口外 MSE = {m(dps, y_img, mask):.5f}   -> 缺口外PSNR = {10*np.log10(1/m(dps, y_img, mask)):.1f} dB")
print(f"DiffPIR 缺口外 MSE = {m(pir, y_img, mask):.5f}")
tot_out = m(dps, y_img, mask) * mask.sum() / 4096
tot_all = 10 ** (-19.76 / 10)
print(f"DPS 缺口外对总MSE贡献: {tot_out/tot_all*100:.0f}%   (总MSE={tot_all:.5f})")
print(f"DPS vs DiffPIR 缺口内填充差异 MSE = {m(dps, pir, ~mask):.5f}")

# ---- 分带定位（缺口外）----
print("\n== 分带 MSE（缺口外像素）==")
bands = [("额发 0:24", slice(0, 24)), ("眼-缺口带 24:40", slice(24, 40)), ("鼻嘴下巴 40:64", slice(40, 64))]
for name, rs in bands:
    band = np.zeros_like(mask); band[rs] = mask[rs]
    print(f"  {name:14s}: DPS={m(dps, y_img, band):.5f}  DiffPIR={m(pir, y_img, band):.5f}")

print("\n== DPS 缺口外逐行 MSE top-8 ==")
sq = ((dps - y_img) ** 2).mean(axis=2)          # (64,64)
row_mse = (sq * mask).sum(axis=1) / mask.sum(axis=1)
for r in np.argsort(row_mse)[::-1][:8]:
    print(f"  row {r:2d}: {row_mse[r]:.4f}")

# ---- 原分辨率白像素检测（露齿判定，不做降采样）----
print("\n== 原分辨率亮白像素（min通道>0.82）==")
for name, p in (("y观测", panels_full[0]), ("DPS", panels_full[1]), ("DiffPIR", panels_full[2])):
    bright = p.min(axis=2) > 0.82
    frac = bright.mean()
    if bright.any():
        rr = np.where(bright.any(axis=1))[0]; cc = np.where(bright.any(axis=0))[0]
        # 换算回 64 坐标
        box = (rr.min() / p.shape[0] * 64, rr.max() / p.shape[0] * 64,
               cc.min() / p.shape[1] * 64, cc.max() / p.shape[1] * 64)
        print(f"  {name:6s}: {frac*100:5.2f}%  位置 rows[{box[0]:.0f}:{box[1]:.0f}] cols[{box[2]:.0f}:{box[3]:.0f}]")
    else:
        print(f"  {name:6s}: {frac*100:5.2f}%")

# ---- 放大对比条带：自动取漂移最大行的邻域 ----
r_top = int(np.argsort(row_mse)[::-1][0])
r0, r1 = max(0, r_top - 8), min(64, r_top + 8)
scale = 8
def strip(arrs, r0, r1, c0, c1, fname):
    h, w = r1 - r0, c1 - c0
    canvas = np.ones((h * scale, 3 * w * scale + 40, 3))
    for i, a in enumerate(arrs):
        big = np.asarray(Image.fromarray(
            (np.clip(a[r0:r1, c0:c1], 0, 1) * 255).astype(np.uint8)
        ).resize((w * scale, h * scale), Image.NEAREST), dtype=np.float64) / 255
        canvas[:, i * (w * scale + 20): i * (w * scale + 20) + w * scale] = big
    path = f"{OUT_DIR}\\{fname}"
    Image.fromarray((canvas * 255).astype(np.uint8)).save(path)
    print(f"已保存: {path}  (rows{r0}:{r1} cols{c0}:{c1}, y|DPS|DiffPIR)")

strip(panels, r0, r1, 8, 56, f"drift_row{r0}_{r1}_zoom.png")
strip(panels, 14, 32, 8, 56, "eye_zoom_y_DPS_DiffPIR.png")
print("完成")
