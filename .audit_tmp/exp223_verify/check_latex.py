# -*- coding: utf-8 -*-
import io
p = r"d:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\2.2 经典先验族.md"
t = io.open(p, encoding="utf-8").read()
bs = chr(92)
nl = chr(10)
print("good \\nabla count:", t.count(bs + "nabla"))
print("broken (newline+abla) count:", t.count(nl + "abla"))
print("stray 'abla' total:", t.count("abla"))
# show any broken context
idx = 0
bad = 0
while True:
    i = t.find(nl + "abla", idx)
    if i < 0:
        break
    bad += 1
    print("---- broken at", i, "----")
    print(repr(t[i-30:i+20]))
    idx = i + 1
print("total broken occurrences found:", bad)
