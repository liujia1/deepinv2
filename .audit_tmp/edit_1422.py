# -*- coding: utf-8 -*-
"""14.2-2：results_summary.json 的 output_files 不再存绝对路径（只存文件名）。"""
import json

PY = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.2-2\14.2-2.py'
JS = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.2-2\results_summary.json'

s = open(PY, encoding='utf-8').read()
o = "'output_files': [out_loss, out_samp, out_stat, out_shape],"
n = "'output_files': [os.path.basename(p) for p in (out_loss, out_samp, out_stat, out_shape)],  # 只存文件名，跨机器有效"
print('py count', s.count(o))
assert s.count(o) == 1
open(PY, 'w', encoding='utf-8').write(s.replace(o, n))

d = json.load(open(JS, encoding='utf-8'))
import os
print('old:', d['output_files'])
d['output_files'] = [os.path.basename(p) for p in d['output_files']]
print('new:', d['output_files'])
with open(JS, 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=2)
print('OK')
