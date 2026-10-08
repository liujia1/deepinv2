# -*- coding: utf-8 -*-
import io

path = r"d:\works\deepinv\第5章 朗之万动力学与得分函数\5.6 近似理论与收敛保证.md"
with io.open(path, "r", encoding="utf-8") as f:
    text = f.read()

broken = (
    "**证明**：$\n"
    "abla\\log p_\\lambda=-\n"
    "abla f-\n"
    "abla g_\\lambda$；$\n"
    "abla f$ 是 $L_f$-Lipschitz；$\n"
    "abla g_\\lambda$ 是 $1/\\lambda$-Lipschitz；故 $\n"
    "abla\\log p_\\lambda$ 的 Lipschitz 常数 $\\le L_f+1/\\lambda$。"
)

fixed = (
    "**证明**：$\\nabla\\log p_\\lambda=-\\nabla f-\\nabla g_\\lambda$；"
    "$\\nabla f$ 是 $L_f$-Lipschitz；$\\nabla g_\\lambda$ 是 $1/\\lambda$-Lipschitz；"
    "故 $\\nabla\\log p_\\lambda$ 的 Lipschitz 常数 $\\le L_f+1/\\lambda$。"
)

cnt = text.count(broken)
print("found broken block:", cnt)
if cnt == 1:
    text = text.replace(broken, fixed)
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("repaired OK")
else:
    print("NOT auto-repaired; inspect manually")

# verify no stray lines starting with 'abla'
bad = [i+1 for i, ln in enumerate(text.splitlines()) if ln.startswith("abla")]
print("remaining lines starting with 'abla':", bad)
