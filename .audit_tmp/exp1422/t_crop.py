"""裁剪并放大步骤5图 (a) 左边缘，确认 ylabel 是否真的缺字形。"""
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=False)
from PIL import Image

p = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.2-2\步骤5_边缘双峰诊断.png'
im = Image.open(p)
print('size', im.size, im.mode)
# 左边缘 + 中部高度
box = (40, 240, 140, 400)
crop = im.crop(box).resize(((box[2]-box[0])*5, (box[3]-box[1])*5), Image.LANCZOS)
crop.save(r'h:\deepinv\deepinv2\.audit_tmp\exp1422\crop_ylabel.png')
print('saved crop')

# 顺带统计：整幅图是否为 RGBA / 白底
import numpy as np
a = np.asarray(im.convert('RGB'))
print('unique colors (top5):', np.unique(a.reshape(-1,3), axis=0).shape)
