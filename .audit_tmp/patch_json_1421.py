# -*- coding: utf-8 -*-
"""把已从 checkpoint 核实的分位数写入 14.2-1 json（与脚本内 _loss_jitter 同口径）。"""
import json
JS = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.2-1\results_summary.json'
d = json.load(open(JS, encoding='utf-8'))
d['loss_jitter_cnll'] = 0.0212
d['loss_jitter_fm'] = 0.0452
d['loss_jitter_note'] = '后300 epoch一次线性去趋势后的残差std（抖动指标，非绝对值）'
with open(JS, 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=2)
print('keys:', list(d.keys()))
