# -*- coding: utf-8 -*-
"""14.3-1 L198：消除「很快…，但…」的双重转折。"""
P = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\14.3 Flow Matching.md'
s = open(P, encoding='utf-8').read()
o = '见 14.3-1 步骤2），但 $\\varepsilon>0$ 引入'
n = '见 14.3-1 步骤2）；代价是 $\\varepsilon>0$ 引入'
print('count', s.count(o))
assert s.count(o) == 1
s = s.replace(o, n)
open(P, 'w', encoding='utf-8').write(s)
print('ctrl:', [hex(ord(c)) for c in s if ord(c) < 32 and c not in '\n\r\t'])
print('nabla:', s.count('\\nabla'), 'varepsilon:', s.count('\\varepsilon'))
print('OK')
