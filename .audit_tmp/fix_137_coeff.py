# -*- coding: utf-8 -*-
"""修复 13.7 小结 line 34：DPS 似然梯度系数 1/sigma^2 -> 1/(2*sigma_y^2)。"""
import io

path = r"h:\deepinv\deepinv2\第13章 条件生成与逆问题求解\13.7 本章小结：条件扩散等于逆问题求解——全书闭环.md"

old = r"$$\text{DPS 的核心近似}:\quad \nabla\log p(y|x_t) \approx -\frac{1}{\sigma^2}\,\nabla_{x_t}\|y - A\hat x_{0|t}\|^2$$"
new = (r"$$\text{DPS 的核心近似}:\quad \nabla\log p(y|x_t) \approx "
       r"-\frac{1}{2\sigma_y^2}\,\nabla_{x_t}\|y - A\hat x_{0|t}\|^2$$"
       "\n\n"
       r"> 注：$\zeta$ 为实践中替代理论系数 $1/(2\sigma_y^2)$ 的可调经验参数（见 13.3.2）。")

with io.open(path, "r", encoding="utf-8") as f:
    text = f.read()

cnt = text.count(old)
print("occurrences:", cnt)
if cnt == 1:
    text = text.replace(old, new)
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("replaced: 1")
else:
    print("no change")
