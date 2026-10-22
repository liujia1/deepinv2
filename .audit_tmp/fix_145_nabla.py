# -*- coding: utf-8 -*-
"""修复 14.5 正文中因编辑工具转义而断裂的 \\nabla 行，并扫描全文控制字符。"""
import re

P = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\14.5 前沿组合：SD3-Flux = Rectified Flow + DiT.md'

with open(P, encoding='utf-8') as f:
    text = f.read()

# 定位受损段：从 "$\varnothing$ 是**空标记**" 到 "abla\log p_t(y \mid x)$："
pat = re.compile(r'\$\\varnothing\$ 是\*\*空标记\*\*.*?abla\\log p_t\(y \\mid x\)\$：', re.S)
m = pat.search(text)
print('found broken block:', bool(m))
if m:
    old = m.group(0)
    print('broken len:', len(old))
    print('control chars:', [(i, hex(ord(c))) for i, c in enumerate(old) if ord(c) < 32 and c != '\n'])
    new = (
        '$\\varnothing$ 是**空标记**（表示"无条件"），$w \\geq 1$ 是引导权重。'
        '为什么这个线性组合式成立？回想 14.3.4 节的三角转换：高斯路径下向量场与得分线性相关，'
        '$v_t = a_t \\nabla \\log p_t + b_t x$。条件场与无条件场用的是**同一条概率路径**'
        '（同样的 $a_t, b_t$），差别只在得分项。于是由贝叶斯分解 '
        '$\\nabla\\log p_t(x \\mid y) = \\nabla\\log p_t(x) + \\nabla\\log p_t(y \\mid x)$：'
    )
    text = text.replace(old, new, 1)

with open(P, 'w', encoding='utf-8') as f:
    f.write(text)

# 全文扫描：除 \n \r \t 之外的控制字符，以及"整行以 abla 开头"这类断裂迹象
bad = [(ln, hex(ord(c))) for ln, line in enumerate(text.split('\n'), 1)
       for c in line if ord(c) < 32 and c not in '\t\r']
print('remaining control chars:', bad)
broken_lines = [ln for ln, line in enumerate(text.split('\n'), 1)
                if line.lstrip().startswith('abla') or line.rstrip().endswith('\\')]
print('suspicious lines:', broken_lines)
print('nabla count:', text.count('\\nabla'))
