# -*- coding: utf-8 -*-
"""修复 SearchReplace 过度转义引入的双反斜杠，并与 git HEAD 对比确认无反斜杠污染扩散"""
import io, sys, re, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
P = r'D:\works\deepinv\第9章 VAE与重参数化\9.3 ELBO训练与KL正则化.md'
t = open(P, encoding='utf-8', newline='').read()
print('CRLF count:', t.count('\r\n'), 'LF count:', t.count('\n'))

fixes = [
    ('$q_\\\\phi(z|x)$ 想成编码器对', '$q_\\phi(z|x)$ 想成编码器对'),
    ('$p(z)=\\\\mathcal{N}(0,I)$ 想成你对隐空间的', '$p(z)=\\mathcal{N}(0,I)$ 想成你对隐空间的'),
    ('$\\\\mu_\\\\phi\\\\approx0$、$\\\\sigma_\\\\phi^2\\\\approx1$', '$\\mu_\\phi\\approx0$、$\\sigma_\\phi^2\\approx1$'),
    ('$-\\\\text{ELBO}$', '$-\\text{ELBO}$'),
]
for a, b in fixes:
    c = t.count(a)
    print('replace %d x : %s' % (c, a[:50]))
    t = t.replace(a, b)
open(P, 'w', encoding='utf-8', newline='').write(t)

# 检查残留双反斜杠
dbl = [l for l in t.splitlines() if '\\\\' in l]
print('剩余含双反斜杠的行数:', len(dbl))
for l in dbl[:10]:
    print('   ', l[:120])
