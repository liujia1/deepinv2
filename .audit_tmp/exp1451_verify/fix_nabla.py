# -*- coding: utf-8 -*-
"""修复被编辑工具拆断的 \nabla：文件中 '\n'+'abla' 实为换行+abla，应还原为 \nabla"""
import io

p = r"d:\works\deepinv\第14章 Flow Matching与最优传输\14.5 前沿组合：SD3-Flux = Rectified Flow + DiT.md"
with io.open(p, 'r', encoding='utf-8') as f:
    s = f.read()

before = s.count("\nabla")  # 换行+abla 的损坏计数
# 损坏形态：真实换行后紧跟 "abla"。还原为 "\nabla"（反斜杠+nabla）
s = s.replace("\nabla", "\\nabla")
# 修正可能引入的双反斜杠误伤：确保不误伤正常换行（仅 abla 前才是 nabla）
with io.open(p, 'w', encoding='utf-8') as f:
    f.write(s)
print("修复的 \\nabla 断行数:", before)

# 校验
with io.open(p, 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i in range(233, 243):
    print(i + 1, lines[i].rstrip("\n"))
