# -*- coding: utf-8 -*-
"""修复 14.5 md 中被换行符破坏的 \nabla (SearchReplace 二次转义 \n 所致)"""
import io

path = r'd:\works\deepinv\第14章 Flow Matching与最优传输\14.5 前沿组合：SD3-Flux = Rectified Flow + DiT.md'
with io.open(path, 'r', encoding='utf-8') as f:
    s = f.read()

# 损坏形态: 原本的 "\nabla" 变成了 换行+"abla"
broken1 = "a_t \nabla \\log p_t + b_t x"          # a_t ⏎abla \log...
fixed1 = "a_t \\nabla \\log p_t + b_t x"
broken2 = "\u8d1d\u53f6\u65af\u5206\u89e3 $\nabla\\log p_t(x \\mid y) = \nabla\\log p_t(x) + \nabla\\log p_t(y \\mid x)$"
fixed2 = "\u8d1d\u53f6\u65af\u5206\u89e3 $\\nabla\\log p_t(x \\mid y) = \\nabla\\log p_t(x) + \\nabla\\log p_t(y \\mid x)$"

n1 = s.count(broken1)
n2 = s.count(broken2)
s = s.replace(broken1, fixed1).replace(broken2, fixed2)

with io.open(path, 'w', encoding='utf-8', newline='') as f:
    f.write(s)
print(f"fixed broken1 x{n1}, broken2 x{n2}")

# 校验: 不应再有 行首"abla" 的残留
with io.open(path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
bad = [i + 1 for i, l in enumerate(lines) if l.startswith('abla')]
print("残留行首abla:", bad if bad else "无")
ctx = [l for l in lines if '14.3.4' in l]
print("修复后该行片段:", ctx[0][:120] if ctx else "N/A")
