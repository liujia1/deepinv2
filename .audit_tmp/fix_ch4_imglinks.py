# -*- coding: utf-8 -*-
"""修复第4章 4.3/4.4/4.5/4.7.md 的配图坏链：
`](./第四章配套实验/<path>)` -> `](<./第四章 配套实验/<path>>)`
修复前先核验目标文件在带空格路径下确实存在，避免误改真正缺失的图。
"""
import re
import io
import os

base = r"d:/works/deepinv/第4章 MCMC与ULA算法"
files = [
    "4.3 ULA：Langevin采样的Euler离散.md",
    "4.4 MYULA：近端ULA.md",
    "4.5 Gibbs采样：利用条件结构的采样.md",
    "4.7 收敛诊断与不确定性量化.md",
]

pat = re.compile(r"\]\(\./第四章配套实验/([^)]+)\)")

total_fixed = 0
for fn in files:
    path = os.path.join(base, fn)
    with io.open(path, "r", encoding="utf-8") as f:
        text = f.read()
    targets = pat.findall(text)
    print(f"\n=== {fn} : 发现 {len(targets)} 条坏链 ===")
    ok_targets = []
    for t in targets:
        corrected = os.path.join(base, "第四章 配套实验", *t.split("/"))
        exists = os.path.isfile(corrected)
        print(f"  [{'OK' if exists else 'MISSING'}] {t}")
        if exists:
            ok_targets.append(t)
    # 仅替换目标确实存在的链接
    new_text = pat.sub(lambda m: m.group(0) if m.group(1) not in ok_targets
                       else "](<./第四章 配套实验/" + m.group(1) + ">)", text)
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(new_text)
    leftover = len(pat.findall(new_text))
    fixed = len(re.findall(r"\]\(<\./第四章 配套实验/[^)]+>\)", new_text))
    total_fixed += len(ok_targets)
    print(f"  -> 已修 {len(ok_targets)} 条, 残留旧形式 {leftover}, 现带尖括号 {fixed}")

print(f"\n合计修复: {total_fixed} 条")
