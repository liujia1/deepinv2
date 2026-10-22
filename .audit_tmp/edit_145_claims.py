# -*- coding: utf-8 -*-
"""14.5 核心知识点段与实测结论对齐（避免“更优”“直接解释”的过强表述）。"""
import re

P = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\14.5 前沿组合：SD3-Flux = Rectified Flow + DiT.md'
with open(P, encoding='utf-8') as f:
    text = f.read()

subs = [
    ('**为什么Logit-Normal在少步采样下更优？**',
     '**为什么Logit-Normal 有理由在少步采样下更优？**'),
    ('- 这直接解释了SD3在60种轨迹对比中"Rectified Flow + Logit-Normal"最优的结论',
     '- 这是 SD3 大规模对比中"Rectified Flow + Logit-Normal"最优的机理层面解释；'
     '本实验在 MNIST 上只测到同向的 2%~3% 改善，量级上并不足以复现该结论'),
    ('- Logit-Normal在端点采样不足，导致1步采样改善有限甚至略差',
     '- Logit-Normal在端点采样不足，1步采样不但没有改善，反而明显更差（-15.2%±0.2%）'),
]
for old, new in subs:
    c = text.count(old)
    print('replace', c, '->', old[:28])
    text = text.replace(old, new, 1)

# 兜底：修复可能再次被转义破坏的 \nabla 段
pat = re.compile(r'\$\\varnothing\$ 是\*\*空标记\*\*.*?abla\\log p_t\(y \\mid x\)\$：', re.S)
m = pat.search(text)
if m and ('\n' in m.group(0) or '\x0b' in m.group(0)):
    fixed = (
        '$\\varnothing$ 是**空标记**（表示"无条件"），$w \\geq 1$ 是引导权重。'
        '为什么这个线性组合式成立？回想 14.3.4 节的三角转换：高斯路径下向量场与得分线性相关，'
        '$v_t = a_t \\nabla \\log p_t + b_t x$。条件场与无条件场用的是**同一条概率路径**'
        '（同样的 $a_t, b_t$），差别只在得分项。于是由贝叶斯分解 '
        '$\\nabla\\log p_t(x \\mid y) = \\nabla\\log p_t(x) + \\nabla\\log p_t(y \\mid x)$：'
    )
    text = text.replace(m.group(0), fixed, 1)
    print('nabla block repaired')

with open(P, 'w', encoding='utf-8') as f:
    f.write(text)

bad = [(i, hex(ord(c))) for i, c in enumerate(text) if ord(c) < 32 and c not in '\n\t\r']
print('control chars:', bad)
print('total lines:', text.count('\n') + 1)
