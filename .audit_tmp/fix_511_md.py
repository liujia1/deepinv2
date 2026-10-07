import io

path = r"d:\works\deepinv\第5章 朗之万动力学与得分函数\5.1 从MCMC到Langevin SDE.md"
with io.open(path, encoding="utf-8") as f:
    t = f.read()

# 被 SearchReplace 的 \n 转义损坏：\nabla 变成 换行+"abla"
broken = ("$$\\frac{\\partial \\rho_t}{\\partial t} = \n"
          "abla\\cdot(\\rho_t\\,\n"
          "abla U) = \n"
          "abla\\cdot(\\rho_t\\,\n"
          "abla(-\\log p))$$")
fixed = ("$$\\frac{\\partial \\rho_t}{\\partial t} = \\nabla\\cdot(\\rho_t\\,"
         "\\nabla U) = \\nabla\\cdot(\\rho_t\\,\\nabla(-\\log p))$$")

assert broken in t, "未找到损坏块"
t = t.replace(broken, fixed)

# 双向核查：不应残留任何 换行+abla 或 换行+oindent 之类的转义损坏
import re
leftover = re.findall(r"\n(abla|oindent|eq|abla)", t)
print("leftover corruptions:", leftover)

with io.open(path, "w", encoding="utf-8") as f:
    f.write(t)
print("repaired OK")
