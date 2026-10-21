# -*- coding: utf-8 -*-
"""修复附录13A末尾被误写成换行的 \\nabla（SearchReplace 转义副作用）。"""
import re

p = r"h:\deepinv\deepinv2\第13章 条件生成与逆问题求解\附录13A DPS算法的完整推导.md"
s = open(p, encoding="utf-8").read()

new = (r"$$\underbrace{\nabla\log p(x_t|y) = \nabla\log p(x_t) + "
       r"\nabla\log p(y|x_t)}_{\text{后验得分分解}} "
       r"\xrightarrow{p(y|x_t) \approx p(y|\hat{x}_{0|t})} "
       r"\underbrace{\nabla\log p(x_t|y) \approx s_\theta + \zeta \cdot "
       r"\frac{A^\top(y - A\hat{x}_0)}{\sigma_y^2}}_{\text{DPS近似}}$$")

pat = re.compile(r"\$\$\\underbrace\{[\s\S]*?\}_\{\\text\{DPS近似\}\}\$\$")
s2, n = pat.subn(lambda m: new, s)
print("replacements:", n)
if n == 1:
    open(p, "w", encoding="utf-8", newline="\n").write(s2)
    print("written")
else:
    print("aborted, unexpected match count")
