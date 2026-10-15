# -*- coding: utf-8 -*-
"""实验2.4-1 讲义补充编辑：图3图注说明含噪输入 PSNR 与表 18.50 的差异来源"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

P = r"d:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\2.4 从显式先验到隐式先验.md"
with open(P, encoding='utf-8', newline='') as f:
    s = f.read()

edits = [
    (r"*图 3：上排从左到右为含噪输入 $y$、第 2/5/10 步 unrolled 去噪结果 $x_t$、干净真值 $x^*$；",
     r"*图 3：上排从左到右为含噪输入 $y$（独立抽样，PSNR 18.36 dB，与表中 18.50 dB 的差异来自图 2 / 图 3 两次独立的噪声实现）、第 2/5/10 步 unrolled 去噪结果 $x_t$、干净真值 $x^*$；"),
]

for old, new in edits:
    c = s.count(old)
    assert c == 1, f"匹配数={c}，非唯一：{old[:50]}..."
    s = s.replace(old, new)

with open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(s)
print(f"OK: {len(edits)} 处编辑完成")
