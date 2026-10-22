# -*- coding: utf-8 -*-
P = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\14.3 Flow Matching.md'
s = open(P, encoding='utf-8').read()
a = '即"几十倍"这个量级'
b = '即“几十倍”这个量级'
print('found', s.count(a))
s = s.replace(a, b)
open(P, 'w', encoding='utf-8').write(s)
