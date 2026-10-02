# -*- coding: utf-8 -*-
"""DiffPIR 缺口内填充的亮度地图（ASCII）：精确定位暗块位置。
复用 PNG 面板提取逻辑，输出 DPS 与 DiffPIR 在缺口 [24:40,24:40] 及邻域的
2x2 块平均亮度图（' .:-=+*#%@' 10 级），并给出缺口内亮度统计。
"""
import numpy as np
from PIL import Image

PNG = r"d:\works\deepinv\第13章 条件生成与逆问题求解\第十三章 配套实验\实验13.3-2\DPSvsDiffPIR对比.png"
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

gray = lambda a: a.mean(axis=2)
CH = " .:-=+*#%@"  # 0(黑) -> 9(白)

def blockmap(a, r0, r1, c0, c1, bs=2):
    g = gray(a)
    lines = []
    for r in range(r0, r1, bs):
        line = ""
        for c in range(c0, c1, bs):
            v = g[r:r+bs, c:c+bs].mean()
            line += CH[min(9, int(v * 10))]
        lines.append(line)
    return lines

R0, R1, C0, C1 = 20, 44, 20, 44  # 缺口及 4px 邻域
bd, bp = blockmap(dps, R0, R1, C0, C1), blockmap(pir, R0, R1, C0, C1)
by = blockmap(y_img, R0, R1, C0, C1)
print(f"缺口区域 rows{R0}:{R1} cols{C0}:{C1}，每格 2x2 块平均灰度（' '黑→'@'白）")
print("      [观测y(缺口内为黑)]        [DPS填充]                [DiffPIR填充]")
for i, (ly, ld, lp) in enumerate(zip(by, bd, bp)):
    print(f"  r{R0+2*i:02d} |{ly}|   |{ld}|   |{lp}|")

hole = np.zeros((64, 64), dtype=bool); hole[24:40, 24:40] = True
g_d, g_p = gray(dps)[hole], gray(pir)[hole]
print(f"\n缺口内灰度统计:  DPS  mean={g_d.mean():.3f} min={g_d.min():.3f}")
print(f"                DiffPIR mean={g_p.mean():.3f} min={g_p.min():.3f}")
dark = g_p < 0.25
print(f"DiffPIR 缺口内暗像素(<0.25)占比: {dark.mean():.2%}   DPS: {(g_d<0.25).mean():.2%}")
print(f"DiffPIR 缺口内最暗块(4x4):")
g2 = gray(pir)
best = None
for r in range(24, 37):
    for c in range(24, 37):
        v = g2[r:r+4, c:c+4].mean()
        if best is None or v < best[0]:
            best = (v, r, c)
print(f"  位置 rows{best[1]}:{best[1]+4} cols{best[2]}:{best[2]+4}  灰度={best[0]:.3f}")
print(f"DPS 同位置灰度={gray(dps)[best[1]:best[1]+4, best[2]:best[2]+4].mean():.3f}")
