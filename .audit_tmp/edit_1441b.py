# -*- coding: utf-8 -*-
"""14.4-1 脚本打印逻辑修正：CRN 后 2-RF 1步距离(5.747)已小于 1-RF 1步(5.834)，
旧代码无条件打印"2-RF 距离更大是指标缺陷"，与事实相反。改为按实测比较。"""
PY = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.4-1\14.4-1.py'
s = open(PY, encoding='utf-8').read()

o1 = """if div_2rf_1 > 3 * max(div_1rf_1, 1e-6):
    print(f"  → Reflow 生效：2-RF 1步消除了1-RF 1步的模式塌缩（多样性 {div_1rf_1:.2f} → {div_2rf_1:.2f}）")
    print(f"  → 2-RF 距离({dist_2rf_1step:.4f})大于1-RF 1步({dist_1rf_1step:.4f})是指标缺陷而非质量退化：")
    print(f"     模糊低能量样本天然离数据云中心更近，连真实留出样本基线也只有 {dist_real:.4f}")
else:"""
n1 = """if div_2rf_1 > 3 * max(div_1rf_1, 1e-6):
    print(f"  → Reflow 生效：2-RF 1步消除了1-RF 1步的模式塌缩（多样性 {div_1rf_1:.2f} → {div_2rf_1:.2f}）")
    if dist_2rf_1step < dist_1rf_1step:
        print(f"  → 2-RF 距离({dist_2rf_1step:.4f})同时小于1-RF 1步({dist_1rf_1step:.4f})：CRN下两指标同向，Reflow在两个维度都改善")
    else:
        print(f"  → 2-RF 距离({dist_2rf_1step:.4f})反而大于1-RF 1步({dist_1rf_1step:.4f})：这是距离指标的局限，")
        print(f"     模糊低能量样本天然离数据云中心更近；且真实留出样本基线本身也有 {dist_real:.4f}")
else:"""
print('c1', s.count(o1)); assert s.count(o1) == 1
s = s.replace(o1, n1)

o2 = """   - 注意：最近邻距离偏袒塌缩到均值的模糊样本，需结合多样性判断1步质量"""
n2 = """   - 注意：最近邻距离的绝对值不可当"质量分"读（本次真实留出样本基线反而是全场最大），
     判断1步质量应以多样性+图像为准
   - CRN：四组采样共用 seed=2025 的同一批噪声，组间差异只来自"模型×步数\""""
print('c2', s.count(o2)); assert s.count(o2) == 1
s = s.replace(o2, n2)

o3 = """   - 量化指标为Reflow效果提供了客观评价依据"""
n3 = """   - 量化指标（多样性×近邻距离×真实基线）互证，为Reflow效果提供评价依据"""
print('c3', s.count(o3)); assert s.count(o3) == 1
s = s.replace(o3, n3)

open(PY, 'w', encoding='utf-8').write(s)
print('OK')
