# -*- coding: utf-8 -*-
"""核实 14.2 正文 L180：后300 epoch 去趋势残差 std（FM≈0.046 vs CNLL≈0.020）是否成立。
口径：对最后 300 个 epoch 的 loss 序列做一次线性去趋势（polyfit deg=1），取残差 std。
"""
import torch
import numpy as np

D = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.2-1'

def detrended_std(losses, tail=300):
    y = np.asarray(losses, dtype=np.float64)
    print('   len=%d, first=%.6f, last=%.6f' % (len(y), y[0], y[-1]))
    seg = y[-tail:]
    x = np.arange(len(seg))
    coef = np.polyfit(x, seg, 1)
    resid = seg - np.polyval(coef, x)
    print('   slope=%.3e, mean=%.6f, resid_std=%.6f' % (coef[0], seg.mean(), resid.std()))
    return resid.std()

for tag, fn in [('CNLL', 'cnll_final.pth'), ('FM', 'fm_final.pth')]:
    ck = torch.load(f'{D}\\{fn}', map_location='cpu', weights_only=False)
    L = ck.get('train_losses', [])
    print(tag, 'n=%d' % len(L))
    detrended_std(L)

# 备选口径：二次去趋势 / 只看相对波动
print('\n--- 备选口径：原始 std（不去趋势）---')
for tag, fn in [('CNLL', 'cnll_final.pth'), ('FM', 'fm_final.pth')]:
    ck = torch.load(f'{D}\\{fn}', map_location='cpu', weights_only=False)
    y = np.asarray(ck['train_losses'], dtype=np.float64)[-300:]
    print(tag, 'raw_std=%.6f, cv=%.3f%%' % (y.std(), 100 * y.std() / y.mean()))
