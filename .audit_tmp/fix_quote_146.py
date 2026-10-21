# -*- coding: utf-8 -*-
"""把本次新写入 14.6 正文的半角引号统一为中文引号。"""
P = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\14.6 离散扩散与掩码语言模型：从向量场到速率矩阵.md'
s = open(P, encoding='utf-8').read()
pairs = [
    ('分母为"更新前处于实词状态"的步', '分母为“更新前处于实词状态”的步'),
    ('以"更新前为实词的位置被改写率"', '以“更新前为实词的位置被改写率”'),
    ('近乎"落子无悔"', '近乎“落子无悔”'),
    ('"碰巧成词的噪声"', '“碰巧成词的噪声”'),
    ('任务更"对口"', '任务更“对口”'),
]
for o, n in pairs:
    c = s.count(o)
    print(c, '|', o[:26])
    assert c == 1, 'COUNT NOT 1'
    s = s.replace(o, n)
open(P, 'w', encoding='utf-8').write(s)
print('done')
