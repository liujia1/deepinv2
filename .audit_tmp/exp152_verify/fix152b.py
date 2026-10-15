# -*- coding: utf-8 -*-
import io
PY = r'd:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.5-2\1.5-2.py'
MD = r'd:\works\deepinv\第1章 逆问题与贝叶斯推断\1.5 贝叶斯框架：从似然到后验.md'
jobs = [
    (PY, [(u'最优 λ（下方特征分解计算）≈ 0.083', u'最优 λ（下方特征分解计算）≈ 0.074')]),
    (MD, [(u'| 该 phantom 的频率派最优 $\\lambda$（期望 MSE） | $\\approx 0.083$ |',
           u'| 该 phantom 的频率派最优 $\\lambda$（期望 MSE） | $\\approx 0.074$ |'),
          (u'频率派期望 MSE 最优 $\\lambda \\approx 0.083$',
           u'频率派期望 MSE 最优 $\\lambda \\approx 0.074$'),
          (u'相对 spread $\\sim 4\\times 10^{-15}$',
           u'相对 spread $= 7.8\\times 10^{-15}$')]),
]
for p, pairs in jobs:
    s = io.open(p, encoding='utf-8').read()
    for o, nw in pairs:
        assert s.count(o) == 1, (p, o, s.count(o))
        s = s.replace(o, nw)
    io.open(p, 'w', encoding='utf-8').write(s)
    print('fixed', p)
