# -*- coding: utf-8 -*-
"""定位 DiffPIR 面板伪影：右眼变黑是缺口内填充错误还是缺口外漂移？

已知（无噪声+Inpainting）：缺口外 y == x_test 逐像素相等。
- 若伪影在缺口外 -> DiffPIR 数据步失守（不应发生，可见区MSE=0.00016）
- 若伪影在缺口内 -> 填充错误：DRUNet 先验把"眼睛内半部"补成了头发/阴影
另查：缺口边界接缝（环带 MSE）、双目区域逐一分解。
"""
import numpy as np
from PIL import Image

PNG = r"d:\works\deepinv\第13章 条件生成与逆问题求解\第十三章 配套实验\实验13.3-2\DPSvsDiffPIR对比.png"
OUT_DIR = r"d:\works\deepinv\.audit_tmp\exp1332_verify"

img = np.asarray(Image.open(PNG).convert("RGB"), dtype=np.float64) / 255.0
H, W, _ = img.shape
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
panels = []
for c0, c1 in groups:
    sub = img[:, c0:c1]
    rows = np.where((sub.min(axis=2) < 0.95).mean(axis=1) > 0.5)[0]
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    panels.append(np.asarray(Image.fromarray(
        (np.clip(img[r0:r1, c0:c1], 0, 1) * 255).astype(np.uint8)
    ).resize((64, 64), Image.BILINEAR), dtype=np.float64) / 255.0)
y_img, dps, pir = panels

mask = np.ones((64, 64), dtype=bool); mask[24:40, 24:40] = False  # True=可见区

def m(a, b, msk=None):
    d = (a - b) ** 2
    return float(d.mean()) if msk is None else float(d[msk].mean())

print("== 双目/五官区域分解（每格: 缺口外MSE | 缺口内MSE）==")
regions = [("左眼 rows18:32 cols10:30", slice(18,32), slice(10,30)),
           ("右眼 rows18:32 cols34:54", slice(18,32), slice(34,54)),
           ("鼻(全在缺口) rows28:42 cols24:40", slice(28,42), slice(24,40)),
           ("嘴 rows42:56 cols18:46", slice(42,56), slice(18,46))]
for name, rs, cs in regions:
    box = np.zeros_like(mask); box[rs, cs] = True
    out = box & mask; inn = box & ~mask
    o_d = m(dps, y_img, out) if out.any() else float('nan')
    i_d = m(dps, y_img, inn) if inn.any() else float('nan')
    o_p = m(pir, y_img, out) if out.any() else float('nan')
    i_p = m(pir, y_img, inn) if inn.any() else float('nan')
    n_in = int(inn.sum())
    print(f"  {name:28s}: DPS {o_d:.5f}|{i_d:.5f}   DiffPIR {o_p:.5f}|{i_p:.5f}  (缺口内像素{n_in})")

print("\n== 缺口边界接缝检查（缺口外1px环带, vs y）==")
ring = np.zeros_like(mask)
ring[22:24, 22:42] = mask[22:24, 22:42]; ring[40:42, 22:42] = mask[40:42, 22:42]
ring[22:42, 22:24] = mask[22:42, 22:24]; ring[22:42, 40:42] = mask[22:42, 40:42]
print(f"  DPS={m(dps, y_img, ring):.5f}  DiffPIR={m(pir, y_img, ring):.5f}  (环带像素{int(ring.sum())})")

print("\n== DiffPIR 缺口内填充 vs DPS 缺口内填充 ==")
print(f"  两法填充差异 MSE = {m(pir, dps, ~mask):.5f}（DPS填充被实测验证为优）")

print("\n== DiffPIR 缺口外逐行/逐列 MSE top-6 ==")
sq = ((pir - y_img) ** 2).mean(axis=2)
row_mse = (sq * mask).sum(axis=1) / mask.sum(axis=1)
col_mse = (sq * mask).sum(axis=0) / mask.sum(axis=0)
for r in np.argsort(row_mse)[::-1][:6]:
    print(f"  row {r:2d}: {row_mse[r]:.5f}")
for c in np.argsort(col_mse)[::-1][:6]:
    print(f"  col {c:2d}: {col_mse[c]:.5f}")

# ---- 放大对比：双目区 + 缺口整体 ----
scale = 8
def strip(arrs, r0, r1, c0, c1, fname):
    h, w = r1 - r0, c1 - c0
    canvas = np.ones((h * scale, 3 * w * scale + 40, 3))
    for i, a in enumerate(arrs):
        big = np.asarray(Image.fromarray(
            (np.clip(a[r0:r1, c0:c1], 0, 1) * 255).astype(np.uint8)
        ).resize((w * scale, h * scale), Image.NEAREST), dtype=np.float64) / 255
        canvas[:, i * (w * scale + 20): i * (w * scale + 20) + w * scale] = big
    Image.fromarray((canvas * 255).astype(np.uint8)).save(f"{OUT_DIR}\\{fname}")
    print(f"已保存: {OUT_DIR}\\{fname}")

strip([y_img, dps, pir], 14, 34, 8, 56, "diffpir_eyes_zoom.png")   # 双目区
strip([y_img, dps, pir], 20, 44, 16, 48, "diffpir_hole_zoom.png")  # 缺口及邻域
print("完成")
