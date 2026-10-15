# -*- coding: utf-8 -*-
"""判断 1.5-3 归档 PNG 与当前重跑 PNG 的差异层次：元数据 vs 像素内容。"""
import io, os, struct, zlib
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, 'exp153_rerun', 'png_before.png')          # 归档(HEAD)副本
NEW = r'd:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.5-3\实验1_5_3_似然等于最小二乘.png'


def chunks(p):
    d = io.open(p, 'rb').read()
    assert d[:8] == b'\x89PNG\r\n\x1a\n'
    i, out = 8, []
    while i < len(d):
        ln = struct.unpack('>I', d[i:i + 4])[0]
        typ = d[i + 4:i + 8].decode('latin1')
        body = d[i + 8:i + 8 + ln]
        out.append((typ, body))
        i += 12 + ln
    return out


for tag, p in (('OLD(归档)', OLD), ('NEW(重跑)', NEW)):
    cs = chunks(p)
    print('%s size=%d  chunks=%s' % (tag, os.path.getsize(p), [c[0] for c in cs]))
    for typ, body in cs:
        if typ in ('tEXt', 'iTXt', 'zTXt'):
            print('   %s: %s' % (typ, body[:200]))

co = dict((t, b) for t, b in chunks(OLD))
cn = dict((t, b) for t, b in chunks(NEW))
for t in set(co) | set(cn):
    if co.get(t) != cn.get(t):
        print('CHUNK 差异: %s (old len=%s new len=%s)' % (t, len(co.get(t, b'')), len(cn.get(t, b''))))

a = np.asarray(Image.open(OLD).convert('RGB'), dtype=np.int16)
b = np.asarray(Image.open(NEW).convert('RGB'), dtype=np.int16)
print('shape old=%s new=%s' % (a.shape, b.shape))
if a.shape == b.shape:
    d = np.abs(a - b)
    n_diff = int((d.sum(axis=2) > 0).sum())
    print('像素不同的点数 = %d / %d  (%.4f%%)  最大通道差 = %d'
          % (n_diff, a.shape[0] * a.shape[1], 100.0 * n_diff / (a.shape[0] * a.shape[1]), d.max()))
    if n_diff:
        ys, xs = np.nonzero(d.sum(axis=2) > 0)
        print('  差异分布: y[%d..%d] x[%d..%d]' % (ys.min(), ys.max(), xs.min(), xs.max()))
