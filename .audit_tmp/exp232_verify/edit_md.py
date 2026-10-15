# -*- coding: utf-8 -*-
"""Safely edit the 2.3-2 lecture section (LaTeX-heavy). Assert every replacement hits."""
import io, sys

MD = r"d:/works/deepinv/第2章 先验：贝叶斯推断的灵魂/2.3 先验的质量：MMSE vs MAP估计器.md"

with io.open(MD, "r", encoding="utf-8") as f:
    txt = f.read()

edits = []

# ---- 1) 输入表：修正 (b) 的框架表述（直接构造后验，非先验×似然），并转义 |x| 竖线 ----
old_table = (
"| 参数 | 图 2-3（偏斜后验） | 图 2-4（双峰后验） |\n"
"|------|-------------------|-------------------|\n"
"| 先验 | Laplace $p(x)\\propto e^{-|x|/b}$，$b=0.8$ | 二值混合 $0.6\\,\\mathcal{N}(0,0.15^2)+0.4\\,\\mathcal{N}(1,0.15^2)$ |\n"
"| 似然 | 高斯 $\\mathcal{N}(y;x,\\sigma^2)$，$\\sigma=1.0$ | 高斯 $\\mathcal{N}(y;x,\\sigma^2)$，$\\sigma=0.15$ |\n"
"| 观测 $y$ | $2.0$ | $0.4$（混合观测） |"
)
new_table = (
"| 参数 | 图 2-3（偏斜后验） | 图 2-4（双峰后验） |\n"
"|------|-------------------|-------------------|\n"
"| 后验来源 | 由先验 × 似然推得 | 直接构造的双峰混合密度（仅示意多峰情形，非由先验 × 似然推得） |\n"
"| 先验 | Laplace $p(x)\\propto e^{-\\vert x\\vert/b}$，$b=0.8$ | ——（不显式设定） |\n"
"| 似然 / 观测 | 高斯 $\\mathcal{N}(y;x,\\sigma^2)$，$\\sigma=1.0$，$y=2.0$ | ——（无单独观测输入） |\n"
"| 后验密度 | $\\propto e^{-\\vert x\\vert/b-(x-y)^2/2\\sigma^2}$ | $0.6\\,\\mathcal{N}(0,0.15^2)+0.4\\,\\mathcal{N}(1,0.15^2)$ |"
)
edits.append((old_table, new_table))

# ---- 2) 步骤1 数值结果：MAP 用解析值 0.750，补软阈值公式与网格对照，分歧 0.251→0.249 ----
old_res = (
"- MAP = 0.748（偏向先验峰值/零点）\n"
"- MMSE = 0.999（被右侧尾部质量拉远）\n"
"- 分歧 ≈ 0.251"
)
new_res = (
"- MAP = 0.750（解析软阈值 $x^\\star=\\mathrm{sign}(y)\\max(\\vert y\\vert-\\sigma^2/b,\\,0)$，偏向先验峰值/零点；网格 argmax 近似 0.748）\n"
"- MMSE = 0.999（后验均值，被右侧尾部质量拉远）\n"
"- 分歧 ≈ 0.249"
)
edits.append((old_res, new_res))

# ---- 3) 步骤2：说明 (b) 为直接构造的后验 ----
old_b = "两个峰分别对应\"背景\"（像素值=0）和\"前景\"（像素值=1）两种判断。数值结果："
new_b = ("两个峰分别对应\"背景\"（像素值=0）和\"前景\"（像素值=1）两种判断"
         "（此处直接给定双峰后验密度以示意多峰情形，两峰权重 0.6/0.4 即后验混合权重）。数值结果：")
edits.append((old_b, new_b))

# ---- 4) 核心结论：分歧 0.251→0.249 ----
old_c = "- 偏斜单峰：均值偏离众数，MMSE 比 MAP 更温和（分歧 ≈ 0.251）。"
new_c = "- 偏斜单峰：均值偏离众数，MMSE 比 MAP 更温和（分歧 ≈ 0.249）。"
edits.append((old_c, new_c))

for i, (o, n) in enumerate(edits, 1):
    cnt = txt.count(o)
    if cnt != 1:
        print(f"[EDIT {i}] FAIL: match count = {cnt}")
        sys.exit(1)
    txt = txt.replace(o, n)
    print(f"[EDIT {i}] OK")

with io.open(MD, "w", encoding="utf-8") as f:
    f.write(txt)
print("ALL EDITS APPLIED")
