# -*- coding: utf-8 -*-
import io

path = r"d:/works/deepinv/第3章 从MAP到后验探索/3.6 收敛性分析与正则化参数选择.md"
with io.open(path, "r", encoding="utf-8") as f:
    txt = f.read()

old = "1. **EB 与 oracle 一致**：本种子下 $\\hat\\alpha=0.0351$ 与 PSNR 最优完全相同——无需真解仅凭 $y$ 选到最优；"

new = ("1. **EB 与 oracle 高度接近（本例网格上重合，但非恒等）**：本种子下 $\\hat\\alpha=0.0351$ 与 PSNR 最优落在**同一网格点**，"
       "看似\"完全相同\"。需澄清这是**离散化+种子**的巧合而非固有属性：连续意义上的两个峰值其实不同"
       "（边际似然峰 $\\approx0.0355$、PSNR 峰 $\\approx0.0368$，差约 4%），只因 100 点 log 网格相邻点比约 1.15× 而被舍入到同一点；"
       "换种子后二者常分离（实测 seed 42–51 仅 2/10 落在同一网格点，个别相差约 2.7×）。"
       "正确结论是：**经验贝叶斯无需真解即可把 $\\alpha$ 选到接近重建最优**（与\"预期结果：接近\"一致），"
       "但它是 oracle 的**近似**，不能宣称为恒等；")

cnt = txt.count(old)
print("occurrences of old:", cnt)
assert cnt == 1, "expected exactly 1 match"
txt = txt.replace(old, new)

with io.open(path, "w", encoding="utf-8") as f:
    f.write(txt)
print("done")
