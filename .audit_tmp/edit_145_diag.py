# -*- coding: utf-8 -*-
"""把 14.5 诊断段的实测值改为脚本日志里的真实数字，并再次修复 \\nabla 行。"""
import re

P = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\14.5 前沿组合：SD3-Flux = Rectified Flow + DiT.md'
with open(P, encoding='utf-8') as f:
    text = f.read()

old = ('Logit-Normal在端点区域采样概率约为均匀采样的14%（$m=0,s=1$ 解析值 '
       '$\\Phi(\\text{logit}(0.1))=\\Phi(-2.197)\\approx1.4\\%$，脚本 10 万次实测 1.42%），导致：')
new = ('Logit-Normal在端点区域采样概率约为均匀采样的14%（$m=0,s=1$ 解析值 '
       '$\\Phi(\\text{logit}(0.1))=\\Phi(-2.197)\\approx1.40\\%$，脚本 10 万次实测 '
       '$t<0.1$ 为 1.38%、$t>0.9$ 为 1.39%），导致：')
n = text.count(old)
text = text.replace(old, new, 1)

# 兜底：修复可能再次被转义破坏的 \nabla 段
pat = re.compile(r'\$\\varnothing\$ 是\*\*空标记\*\*.*?abla\\log p_t\(y \\mid x\)\$：', re.S)
m = pat.search(text)
if m:
    fixed = (
        '$\\varnothing$ 是**空标记**（表示"无条件"），$w \\geq 1$ 是引导权重。'
        '为什么这个线性组合式成立？回想 14.3.4 节的三角转换：高斯路径下向量场与得分线性相关，'
        '$v_t = a_t \\nabla \\log p_t + b_t x$。条件场与无条件场用的是**同一条概率路径**'
        '（同样的 $a_t, b_t$），差别只在得分项。于是由贝叶斯分解 '
        '$\\nabla\\log p_t(x \\mid y) = \\nabla\\log p_t(x) + \\nabla\\log p_t(y \\mid x)$：'
    )
    text = text.replace(m.group(0), fixed, 1)

with open(P, 'w', encoding='utf-8') as f:
    f.write(text)

print('diag replaced:', n)
print('nabla broken fixed:', bool(m))
print('1.42 remains:', text.count('1.42%'))
bad = [(i, hex(ord(c))) for i, c in enumerate(text) if ord(c) < 32 and c not in '\n\t\r']
print('control chars:', bad)
