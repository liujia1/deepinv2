# -*- coding: utf-8 -*-
"""1.5-1 取证：验证归档可复现性 + 量化"前三个后验均值都停在 0.5"是巧合还是必然。"""
import io, os, json
import numpy as np
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = r'd:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.5-1'
out = io.open(os.path.join(HERE, 'out_151.txt'), 'w', encoding='utf-8')


def w(s):
    out.write(str(s) + '\n')


theta, a0, b0 = 0.7, 1.0, 1.0
ns = [0, 2, 10, 50, 500]

# (1) 复现归档：独立生成 500 次抛掷
rng = np.random.default_rng(42)
flips = (rng.random(500) < theta).astype(int)
w('(1) seed42 复现: heads = %s' % [int(flips[:n].sum()) for n in ns])
j = json.load(io.open(os.path.join(EXP, 'results_summary.json'), encoding='utf-8'))
w('    JSON heads  = %s' % [d['heads'] for d in j['posterior']])
w('    JSON mean   = %s' % [round(d['mean'], 6) for d in j['posterior']])
w('    一致: %s' % ([int(flips[:n].sum()) for n in ns] == [d['heads'] for d in j['posterior']]))

# (2) 解析概率：三点均值同时停在 0.5 需要 前2次恰1正 且 前10次恰5正
p2 = comb(2, 1) * theta ** 1 * (1 - theta) ** 1
p8 = comb(8, 4) * theta ** 4 * (1 - theta) ** 4
p_joint = p2 * p8
w('')
w('(2) 解析概率 P(前2次1正)=%.4f  P(后8次再4正)=%.4f  联合=%.4f (约 %.1f%%)'
  % (p2, p8, p_joint, 100 * p_joint))
w('    即"n=0,2,10 三条后验全对称于 0.5"并非必然，本实现约 1/17 概率撞上')

# (3) 蒙特卡洛核对 + 各 n 下后验均值的分布
M = 40000
big = (np.random.default_rng(0).random((M, 500)) < theta).astype(np.int8)
cum = np.cumsum(big, axis=1)
w('')
w('(3) %d 次重演的后验均值分布（含先验 Beta(1,1)）:' % M)
for n in ns:
    h = cum[:, n - 1] if n > 0 else np.zeros(M, dtype=int)
    m = (a0 + h) / (a0 + b0 + n)
    w('    n=%-4d 均值: 期望=%.4f  中位=%.4f  P(均值==0.5)=%.4f  P(均值<0.5)=%.4f  范围[%.3f,%.3f]'
      % (n, m.mean(), np.median(m), float(np.mean(m == 0.5)), float(np.mean(m < 0.5)), m.min(), m.max()))

# (4) 先验话语权的定量衰减：后验均值 - 频率估计 = (n/2 - heads)/(n(n+2)) 之类
w('')
w('(4) 先验拉力的量级：Beta(1,1) 相当于 2 个"伪观测"（各 1）')
for n in ns[1:]:
    w('    n=%-4d 先验权重占比 = 2/(2+n) = %.4f' % (n, 2.0 / (2 + n)))
out.close()
print('ok')
