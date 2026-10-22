# -*- coding: utf-8 -*-
"""14.4 正文：14.4-3 重跑后的曲率/loss 与 json 对齐，补设备行。"""
P = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\14.4 Rectified Flow.md'
s = open(P, encoding='utf-8').read()
subs = [
    ('| 2-RF | 0.000089 | 99.9% |', '| 2-RF | 0.000087 | 99.9% |'),
    ('| 3-RF | 0.000040 | 55.1% |', '| 3-RF | 0.000040 | 54.0% |'),
    ('同时训练loss从1-RF的最终1.865断崖式跌至2-RF的0.00098（约三个数量级）',
     '同时训练loss从1-RF的最终1.8655断崖式跌至2-RF的0.000981（约三个数量级；对应 json 的 `final_loss_1rf`/`final_loss_2rf`，完整曲线见 `loss_curve_*`）'),
    ('| 训练轮数 | 每个模型300 epochs |',
     '| 训练轮数 | 每个模型300 epochs |\n| 运行设备 | 固定 CPU（`device=\'cpu\'`，实验只用到 2D 点云；固定设备避免有卡机器换到 GPU 后数值漂移） |'),
]
for o, n in subs:
    c = s.count(o)
    print('count', c, '|', o[:46])
    assert c == 1, 'COUNT NOT 1'
    s = s.replace(o, n)
open(P, 'w', encoding='utf-8').write(s)
print('ctrl:', [hex(ord(c)) for c in s if ord(c) < 32 and c not in '\n\r\t'])
print('OK')
