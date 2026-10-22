# -*- coding: utf-8 -*-
"""14.3-1：让实验目的/知识点与实测结论自洽（本 CPU 教学实现下 Sinkhorn 墙钟更慢）。"""
P = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\14.3 Flow Matching.md'
s = open(P, encoding='utf-8').read()
subs = []

# 1) Gondzio 对比框：给「快」加实现前提
subs.append((
    '> - **Sinkhorn**：快、可微、可 GPU，但 $\\varepsilon>0$ 引入**熵偏差**',
    '> - **Sinkhorn**：可微、可 GPU 批量、成熟库的 GPU 实现很快（但本实验的纯 NumPy 教学实现在 CPU 上反而慢于 scipy 匈牙利，见 14.3-1 步骤2），但 $\\varepsilon>0$ 引入**熵偏差**',
))

# 2) 实验目的 1：验证 -> 量化
subs.append((
    '1. **验证Sinkhorn算法的精度**：对比不同正则化参数$\\varepsilon$下的近似精度与迭代次数',
    '1. **量化Sinkhorn算法的精度**：对比不同正则化参数$\\varepsilon$下的近似精度与迭代次数',
))

# 3) 实验目的 2：Sinkhorn「快」与实测反向
subs.append((
    '2. **揭示精度-效率权衡**：匈牙利算法精确但慢，Sinkhorn算法快但需调参',
    '2. **揭示精度-效率权衡**：匈牙利算法精确、渐近 $O(n^3)$；Sinkhorn 可微分、可 GPU 批量并行、$\\varepsilon$ 是可调的精度旋钮——但本 CPU 教学实现下其墙钟时间反而慢约三个数量级（纯 NumPy 常数开销，见步骤2）',
))

# 4) 实验目的 3：「更稳定」含糊，改为增长率对比
subs.append((
    '3. **验证扩展性**：随样本量增长，匈牙利算法时间急剧增加，Sinkhorn算法更稳定',
    '3. **对比扩展性趋势**：随样本量增长，匈牙利总时间超线性增长（渐近 $O(n^3)$），Sinkhorn 每迭代时间渐近 $O(n^2)$——增长率更慢，但本实现下绝对时间仍高得多',
))

# 5) 核心知识点：给「效率平衡」加实现前提
subs.append((
    '- 匈牙利算法精确但慢，限制了训练速度\n- Sinkhorn算法提供了精度与效率的实用平衡',
    '- 匈牙利算法精确但渐近 $O(n^3)$，batch 大时不可扩展；本 CPU 教学实现因 scipy 高度优化而仍比纯 NumPy Sinkhorn 快，生产中 GPU 批量求解时这个排序会反转\n- Sinkhorn 的「效率」优势建立在 GPU 并行实现上：用 $\\varepsilon$ 换可微性、可批量性与可控的精度损失（本实验步骤2 是这一前提的反例——纯 NumPy + CPU 下总时间慢约三个数量级）',
))

for old, new in subs:
    c = s.count(old)
    print('count', c, '|', old.strip()[:56])
    assert c == 1, 'COUNT NOT 1'
    s = s.replace(old, new)

open(P, 'w', encoding='utf-8').write(s)
print('OK')
