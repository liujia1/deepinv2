# -*- coding: utf-8 -*-
"""从 17.6 md 中提取"动手实验"代码块实跑，验证文档里的每个数值引用"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import re, io, contextlib

MD = r'h:\deepinv\deepinv2\第17章 自监督学习与等变架构\17.6 测量一致性与等变架构.md'
src = open(MD, encoding='utf-8').read()
blocks = re.findall(r'```python\n(.*?)```', src, flags=re.S)
demo = [b for b in blocks if '17.6 动手实验' in b]
assert len(demo) == 1, f'找到 {len(demo)} 个 demo 块'
code = demo[0]
print('--- 执行 md 代码块 ---')
exec(compile(code, '<17.6_demo>', 'exec'), {'__name__': '__main__'})
