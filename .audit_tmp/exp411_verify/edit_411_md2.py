# -*- coding: utf-8 -*-
"""精确校正两处比值表述。"""
import io

p = r'd:\works\deepinv\第4章 MCMC与ULA算法\4.1 Monte Carlo方法：从积分到采样.md'
t = io.open(p, encoding='utf-8').read()

pairs = [
    (r'''MC 耗时始终 $\sim 10^{-3}$ s 量级。''',
     r'''MC 每次重复耗时 $2\times10^{-4}$–$8\times10^{-4}$ s，与维数基本无关。'''),
    (r'''对应理论 0.0030/0.0042/0.0052/0.0060/0.0067，比值 0.80–0.86（50 次重复取均值后再取绝对值，系统性略低于 $\mathrm{SE}$，符合''',
     r'''对应理论 0.0030/0.0042/0.0052/0.0060/0.0067，比值 0.75–0.84（50 次重复取均值后再取绝对值，故系统性略低于 $\mathrm{SE}$，符合'''),
]
for old, new in pairs:
    assert t.count(old) == 1, (t.count(old), old[:40])
    t = t.replace(old, new)

io.open(p, 'w', encoding='utf-8', newline='\n').write(t)
lines = t.split('\n')
print('lines:', len(lines))
print('dbl bs:', [i + 1 for i, l in enumerate(lines) if '\\\\' in l])
print('has 10^{-3} s 量级:', 's 量级' in t)
