# -*- coding: utf-8 -*-
import io, sys
PATH = r"d:\works\deepinv\第6章 得分匹配：从去噪中学习得分\6.2 显式得分匹配（ESM）与隐式得分匹配（ISM）.md"
with io.open(PATH, 'r', encoding='utf-8') as f:
    text = f.read()
old = "两条曲线形状不同，但最优解都约"
new = "两条曲线各自减去最小值后几乎完全重合——这本身就是“差一个常数”最直接的证据；最优解都约"
assert text.count(old) == 1, f"匹配次数={text.count(old)}"
text = text.replace(old, new)
with io.open(PATH, 'w', encoding='utf-8') as f:
    f.write(text)
print("[OK] 已修正")
