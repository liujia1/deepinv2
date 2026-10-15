# -*- coding: utf-8 -*-
import io
p = r"d:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\2.2 经典先验族.md"
t = io.open(p, encoding="utf-8").read()
bs = chr(92)
nl = chr(10)
broken = nl + "abla"
good = bs + "nabla"
n = t.count(broken)
t = t.replace(broken, good)
io.open(p, "w", encoding="utf-8").write(t)
# re-verify
t2 = io.open(p, encoding="utf-8").read()
print("repaired:", n)
print("remaining broken:", t2.count(broken))
print("good \\nabla now:", t2.count(good))
print("stray abla now:", t2.count("abla"))
