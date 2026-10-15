# -*- coding: utf-8 -*-
import io, sys
sys.stdout.reconfigure(encoding='utf-8')
P = r'd:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\2.3 先验的质量：MMSE vs MAP估计器.md'
s = io.open(P, encoding='utf-8').read()
for key in ['2.3-3.py', '取若干', '求解方式', '收缩最温和', '不同先验对应不同去噪器']:
    i = 0
    while True:
        i = s.find(key, i)
        if i < 0:
            break
        line_start = s.rfind('\n', 0, i) + 1
        line_end = s.find('\n', i)
        print('%-14s @%d: %s' % (key, i, repr(s[line_start:line_end])[:260]))
        i += 1
    print('-' * 40)
