# -*- coding: utf-8 -*-
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
p = r'D:\works\deepinv\.audit_tmp\exp1041_verify\final_run.log'
print('size:', os.path.getsize(p))
raw = open(p, 'rb').read()
for enc in ('utf-8', 'utf-16', 'gbk'):
    try:
        txt = raw.decode(enc)
        used = enc
        break
    except Exception:
        continue
else:
    txt = raw.decode('utf-8', errors='replace'); used = 'utf-8(replace)'
print('decoded as:', used)
for l in txt.splitlines():
    if any(k in l for k in ['KL', '闭式', 'Cosine', 'C1', 'C2', 'C3', 'C4', '保存', 'Traceback', 'Error', '0.4', '0.5', '字体']):
        print(l)
