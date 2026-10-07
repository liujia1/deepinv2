# -*- coding: utf-8 -*-
import io

MD = r'd:/works/deepinv/第7章 扩散模型：SDE视角/7.4 概率流ODE：随机采样的确定性等价.md'
PY = r'd:/works/deepinv/第7章 扩散模型：SDE视角/第七章 配套实验/实验7.4-1/7.4-1.py'

# #2 markdown 步骤2 措辞（避免把"等价性验证"错挂到只画解析边际的步骤2上）
md_old = '在多个时间点可视化边际分布的演化，验证逆向SDE与PF-ODE产生完全相同的 $p_t(x)$。'
md_new = '在多个时间点可视化边际分布的演化，给出逆向SDE与PF-ODE共同依赖的解析边际 $p_t(x)$（其等价性由步骤1解析证明、步骤3数值验证），为步骤3提供对照基准。'

# #3 docstring 自相矛盾（步骤3本就是数值采样）
py_old = '  - ★ 原创设计：解析验证而非数值采样验证'
py_new = '  - ★ 原创设计：解析推导与数值采样验证并重（步骤2解析、步骤3数值）'

m = io.open(MD, encoding='utf-8').read()
assert m.count(md_old) == 1, ('md_old', m.count(md_old))
m2 = m.replace(md_old, md_new)
io.open(MD, 'w', encoding='utf-8').write(m2)

p = io.open(PY, encoding='utf-8').read()
assert p.count(py_old) == 1, ('py_old', p.count(py_old))
p2 = p.replace(py_old, py_new)
io.open(PY, 'w', encoding='utf-8').write(p2)

# 双向核查
print('md nabla_ok', m2.count(chr(92) + 'nabla'), 'bad', m2.count(chr(10) + 'abla'))
print('md applied', md_new in m2, 'old_gone', md_old not in m2)
print('py applied', py_new in p2, 'old_gone', py_old not in p2)
print('py py_new', repr(py_new))
