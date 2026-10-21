# -*- coding: utf-8 -*-
"""14.3-1 步骤2：补 per_iter 单位与「中位数不可乘」的口径说明。"""
P = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\14.3 Flow Matching.md'
s = open(P, encoding='utf-8').read()
subs = [
    ('Sinkhorn **每迭代时间** 0.8→6.9 ms（$\\varepsilon$=0.1）：',
     'Sinkhorn **每迭代时间** 0.8→6.9 ms（$\\varepsilon$=0.1；json 字段 `per_iter_us` 以微秒存档）：'),
    ('- 但总时间（迭代数×每迭代时间）在本实现下仍远高于匈牙利',
     '- 但总时间（≈迭代数×每迭代时间；`total_ms` 与 `per_iter_us` 各是 3 次计时的中位数，二者乘积只在量级意义上等于实测总墙钟）在本实现下仍远高于匈牙利'),
]
for o, n in subs:
    c = s.count(o)
    print('count', c, '|', o[:40])
    assert c == 1, 'COUNT NOT 1'
    s = s.replace(o, n)
open(P, 'w', encoding='utf-8').write(s)
print('ctrl:', [hex(ord(c)) for c in s if ord(c) < 32 and c not in '\n\r\t'])
print('OK')
