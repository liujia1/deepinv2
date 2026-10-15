# -*- coding: utf-8 -*-
import io, re
p = r"d:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\2.2 经典先验族.md"
t = io.open(p, encoding="utf-8").read()

# 1. literal control chars that should never appear mid-content
print("TAB count:", t.count(chr(9)))
print("CR count:", t.count(chr(13)))

# 2. fragments that indicate a broken \command (newline + leftover letters)
frag = ["abla","imes","eta","ightarrow","o ","eq ","u ","otin","ewline","abla",
        "ot ","orm","ert","otag","on","olimits","abla"]
for f in sorted(set(frag)):
    c = t.count(chr(10)+f)
    if c:
        print(f"suspicious newline+{f!r}: {c}")

# 3. verify expected intact commands still present
for cmd in ["\\nabla","\\times","\\theta","\\rightarrow","\\lambda","\\alpha","\\sigma","\\ell","\\varepsilon"]:
    print("intact", cmd, ":", t.count(cmd))
