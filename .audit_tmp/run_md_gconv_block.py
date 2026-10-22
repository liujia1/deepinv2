# -*- coding: utf-8 -*-
"""抽取 17.6 md 中新增的 p4 群卷积代码块并实跑，确认输出与文中标注一致"""
import sys, re
sys.stdout.reconfigure(encoding='utf-8')

P = r'h:\deepinv\deepinv2\第17章 自监督学习与等变架构\17.6 测量一致性与等变架构.md'
md = open(P, encoding='utf-8').read()
blocks = re.findall(r'```python\n(.*?)```', md, flags=re.S)
target = [b for b in blocks if 'gconv' in b]
print(f'md 中 python 代码块共 {len(blocks)} 个，含 gconv 的 {len(target)} 个')
assert len(target) == 1, '应恰好 1 个群卷积代码块'
print('=== 执行代码块 ===')
exec(compile(target[0], 'md_gconv', 'exec'), {})
print('=== 期望: 群卷积误差=0.0e+00  标准卷积误差=1.6e+01 ===')
