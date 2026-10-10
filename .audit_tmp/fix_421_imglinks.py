# -*- coding: utf-8 -*-
"""修复 4.2.md 中实验配图链接：补回目录名中的空格并用尖括号包裹（对齐第三章可用写法）。
只替换 `](./第四章配套实验/....png)` -> `](<./第四章 配套实验/....png>)`，不动其它内容。
"""
import re
import io

path = r"d:/works/deepinv/第4章 MCMC与ULA算法/4.2 Metropolis-Hastings算法.md"
with io.open(path, "r", encoding="utf-8") as f:
    text = f.read()

pattern = re.compile(r"\]\(\./第四章配套实验/([^)]+\.png)\)")
new_text, n = pattern.subn(r"](<./第四章 配套实验/\1>)", text)

with io.open(path, "w", encoding="utf-8") as f:
    f.write(new_text)

print(f"替换链接数: {n}")
# 校验：不应再有旧的无空格形式，且新形式数量一致
leftover = len(re.findall(r"\]\(\./第四章配套实验/", new_text))
fixed = len(re.findall(r"\]\(<\./第四章 配套实验/[^)]+\.png>\)", new_text))
print(f"残留旧形式: {leftover}, 新形式(带尖括号): {fixed}")
