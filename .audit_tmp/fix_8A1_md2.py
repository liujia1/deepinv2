# -*- coding: utf-8 -*-
"""附录8A 实验段细节修正（次梯度记号、mathtext 内中文、多余空行）"""
import io

MD = r"d:\works\deepinv\第8章 变分推断与ELBO\附录8A Fenchel共轭与ELBO的优化理论根基.md"
with io.open(MD, 'r', encoding='utf-8') as fh:
    txt = fh.read()

pairs = [
    (r'切点即 $x^*\in\partial f^{-1}(y)$',
     r'切点满足 $y \in \partial f(x^*)$（次梯度取等条件）'),
    (r'两配置均满足 $\sup_{q\in\text{单高斯}}\text{ELBO} \le \log p(x)$',
     r'两配置均满足单高斯族上的 $\sup_q \text{ELBO}(q) \le \log p(x)$'),
    ('（含全部实测数值，seed=42 可逐位复现）。\n\n\n---',
     '（含全部实测数值，seed=42 可逐位复现）。\n\n---'),
]
for old, new in pairs:
    assert txt.count(old) == 1, (old[:40], txt.count(old))
    txt = txt.replace(old, new)

with io.open(MD, 'w', encoding='utf-8') as fh:
    fh.write(txt)
print('OK，共完成', len(pairs), '处修正')
for kw in ['\\partial f(x^*)', '\\sup_q \\text{ELBO}(q) \\le', '次梯度取等条件']:
    print('核查', repr(kw), '出现次数:', txt.count(kw))
