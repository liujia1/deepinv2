# -*- coding: utf-8 -*-
"""从 16.7 md 中抽取 python 代码块并原样执行，检查修复后的 PSNR 输出。"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.show = lambda *a, **k: None  # 无头模式下 show 不阻塞

MD = r"h:\deepinv\deepinv2\第16章 CT：MRI重建\16.7 章末代码：用skimage复现CT不适定.md"
src = open(MD, encoding="utf-8").read()
blocks = re.findall(r"```python\n(.*?)```", src, re.S)
print("### 抽取到 python 代码块数:", len(blocks))
code = "\n".join(blocks)

g = {"__name__": "__main__"}
exec(compile(code, "16.7_md_block", "exec"), g)

# 事后核查：确认没有出现 clip 后全饱和
print("\n### 事后核查")
print("rec_bp range:", float(g["rec_bp"].min()), float(g["rec_bp"].max()))
import numpy as np
frac_one = float((g["rec_bp"] >= 0.999).mean())
print("frac(rec_bp >= 0.999):", frac_one, "  (修复前应为 1.0)")
print("BP/FBP PSNR in fig6:",
      g["psnr"](g["x_true"], g["rec_bp"]),
      g["psnr"](g["x_true"], g["rec_fbp"]))
