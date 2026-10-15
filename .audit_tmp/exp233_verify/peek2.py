# -*- coding: utf-8 -*-
import io, sys
sys.stdout.reconfigure(encoding='utf-8')
P = r'd:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\2.3 先验的质量：MMSE vs MAP估计器.md'
s = io.open(P, encoding='utf-8').read()
i = s.find('正向映射')
print([hex(ord(c)) for c in s[i-3:i+6]])
j = s.find('死区')
print([hex(ord(c)) for c in s[j-3:j+4]])
print('ASCII双引号数量:', s.count('"'), ' 左花引号:', s.count('\u201c'), ' 右花引号:', s.count('\u201d'))
