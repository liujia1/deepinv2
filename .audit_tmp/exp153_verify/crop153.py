# -*- coding: utf-8 -*-
"""裁剪 1.5-3 归档/重跑 PNG 的标题差异区域，放大并排放，便于目视。"""
import io, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, 'exp153_rerun', 'png_before.png')
NEW = r'd:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.5-3\实验1_5_3_似然等于最小二乘.png'
box = (700, 0, 1000, 55)
out = os.path.join(HERE, 'title_diff_153.png')
ims = []
for p in (OLD, NEW):
    im = Image.open(p).convert('RGB').crop(box)
    ims.append(im)
W = max(i.width for i in ims)
H = sum(i.height for i in ims) + 10
canvas = Image.new('RGB', (W, H), 'white')
y = 0
for im in ims:
    canvas.paste(im, (0, y))
    y += im.height + 10
canvas = canvas.resize((canvas.width * 4, canvas.height * 4), Image.LANCZOS)
canvas.save(out)
print('saved', out, canvas.size)

a = np.asarray(Image.open(OLD).convert('RGB'), dtype=np.int16)
b = np.asarray(Image.open(NEW).convert('RGB'), dtype=np.int16)
d = (np.abs(a - b).sum(axis=2) > 0)
sub = d[15:45, 760:790]
print('差异掩码在该块的行数分布:')
for r in range(sub.shape[0]):
    cols = np.nonzero(sub[r])[0]
    if len(cols):
        print('  y=%d  x=%s' % (15 + r, list(760 + cols)))
