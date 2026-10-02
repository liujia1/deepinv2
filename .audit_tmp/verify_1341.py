# -*- coding: utf-8 -*-
"""验证 CFG引导对比.png 中 guidance_scale=7.5 面板 与 文生图示例.png 是否同源（像素一致性）"""
import numpy as np
from PIL import Image

comp = Image.open(r"d:\works\deepinv\第13章 条件生成与逆问题求解\第十三章 配套实验\实验13.4-1\CFG引导对比.png").convert("RGB")
solo = Image.open(r"d:\works\deepinv\第13章 条件生成与逆问题求解\第十三章 配套实验\实验13.4-1\文生图示例.png").convert("RGB")
print("composite size:", comp.size, "solo size:", solo.size)

a = np.asarray(comp)
gray = a.mean(axis=2)
# 先行方向定位面板行区域（排除标题文字行）
row_nonwhite = (gray < 240).any(axis=1)
rsegs = []
in_seg = False
for i, v in enumerate(row_nonwhite):
    if v and not in_seg:
        start = i; in_seg = True
    elif not v and in_seg:
        rsegs.append((start, i)); in_seg = False
if in_seg: rsegs.append((start, len(row_nonwhite)))
rsegs = [rg for rg in rsegs if rg[1]-rg[0] > 200]
print("wide row segments:", rsegs)

# 只在面板行区域内按列切分，避免标题文字粘连
y0, y1 = rsegs[0]
gband = gray[y0:y1, :]
col_nonwhite = (gband < 240).any(axis=0)
segs = []
in_seg = False
for i, v in enumerate(col_nonwhite):
    if v and not in_seg:
        start = i; in_seg = True
    elif not v and in_seg:
        segs.append((start, i)); in_seg = False
if in_seg: segs.append((start, len(col_nonwhite)))
segs = [sg for sg in segs if sg[1]-sg[0] > 200]
print("wide column segments:", segs)

if len(segs) >= 3:
    x0, x1 = segs[2]  # 第3个面板 gs=7.5
    panel = a[y0:y1, x0:x1]
    p_img = Image.fromarray(panel).resize((solo.size[0], solo.size[1]))
    p = np.asarray(p_img).astype(float)
    s = np.asarray(solo).astype(float)
    diff = np.abs(p - s)
    print("panel crop shape:", panel.shape)
    print("mean abs diff:", diff.mean(), "max:", diff.max())
    print("pct pixels within 8/255:", (diff.max(axis=2) <= 8).mean())
