# -*- coding: utf-8 -*-
"""统一 1.5 md 的引号风格：该文件正文用半角 U+0022 引号，
把先前 patch 写入的全角 “ ” 归一回半角（只针对本次新增的具体片段，不动其他内容）。"""
import io

P = r'd:\works\deepinv\第1章 逆问题与贝叶斯推断\1.5 贝叶斯框架：从似然到后验.md'
src = io.open(P, encoding='utf-8').read()

FRAGS = [
    '\u201c只差常数因子\u201d',
    '\u201c数值验证\u201d',
    '\u201c实测 $-\\ln p$ 与 $\\frac{1}{2\\sigma^2}\\|y-Ax\\|^2$ 的最大相对偏差\u201d',
    '\u201c代数等价成立\u201d',
    '\u201c最大化似然 $=$ 最小二乘\u201d',
]
for f in FRAGS:
    n = src.count(f)
    assert n == 1, '命中 %d 次: %r' % (n, f[:30])
    src = src.replace(f, f.replace('\u201c', '"').replace('\u201d', '"'))

io.open(P, 'w', encoding='utf-8', newline='').write(src)
print('unified %d quotes' % len(FRAGS))
