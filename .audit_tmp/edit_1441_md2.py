# -*- coding: utf-8 -*-
"""14.4 正文两处旧结论收尾：距离指标在 CRN 口径下与多样性同向，
「会倒果为因/会得出反向结论」的绝对化表述需改为「绝对值不可当质量分」。"""
import io
MD = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\14.4 Rectified Flow.md'
s = io.open(MD, encoding='utf-8').read()

o1 = "- 2-RF的1步采样质量优于1-RF的1步采样（Reflow拉直了路径；注意：最近邻距离指标会倒果为因，需看多样性与图像）"
n1 = "- 2-RF的1步采样质量优于1-RF的1步采样（Reflow拉直了路径；注意：最近邻距离的绝对值不可当质量分——真实样本基线反而最差，需与多样性、图像联合判断）"
print('c1', s.count(o1)); assert s.count(o1) == 1
s = s.replace(o1, n1)

o2 = "- **高维上需要选对指标**：2D曲率可以直接算，MNIST上则需要多样性/视觉联合判断，单一距离指标会得出反向结论"
n2 = "- **高维上需要选对指标**：2D曲率可以直接算，MNIST上则需要多样性/视觉联合判断；单一最近邻距离不仅绝对值不可信（真实基线反而最大），早期未固定采样噪声的版本里甚至给出过反向排序"
print('c2', s.count(o2)); assert s.count(o2) == 1
s = s.replace(o2, n2)

io.open(MD, 'w', encoding='utf-8').write(s)
print('OK')
