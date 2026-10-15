# -*- coding: utf-8 -*-
import io
PY = r'd:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.5-2\1.5-2.py'
s = io.open(PY, encoding='utf-8').read()
old = u"周期卷积+白先验 ⇒ 各像素恒为"
new = u"周期卷积+白先验 → 各像素恒为"
assert s.count(old) == 1, s.count(old)
io.open(PY, 'w', encoding='utf-8').write(s.replace(old, new))
print('fixed arrow')
