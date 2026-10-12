# -*- coding: utf-8 -*-
import io
p = r"d:\works\deepinv\第3章 从MAP到后验探索\3.6 收敛性分析与正则化参数选择.md"
with io.open(p, "r", encoding="utf-8") as f:
    s = f.read()

# broken fragment: "t^{" + newline + "u/2}"  -> restore to "t^{\nu/2}"
broken = "t^{\nu/2}"        # \n here is a real newline
correct = "t^{\\nu/2}"
cnt = s.count(broken)
s = s.replace(broken, correct)
with io.open(p, "w", encoding="utf-8") as f:
    f.write(s)
print("replaced", cnt, "occurrence(s)")
# verify no stray newline+u remains in that paragraph
with io.open(p, "r", encoding="utf-8") as f:
    t = f.read()
print("still broken?", "t^{\nu/2}" in t)
idx = t.find("收敛速率与源条件")
print(t[idx:idx+230])
