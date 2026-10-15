# -*- coding: utf-8 -*-
import io
p = r"d:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\2.2 经典先验族.md"
txt = io.open(p, encoding="utf-8").read()
# count full-width vs ascii double quotes
print("full-width left  \u201c :", txt.count("\u201c"))
print("full-width right \u201d :", txt.count("\u201d"))
print("ascii double quote \"   :", txt.count('"'))
# show ascii-quote contexts (to see if they are mine)
import re
for i, line in enumerate(txt.splitlines(), 1):
    if '"' in line:
        print(i, repr(line[:80]))
