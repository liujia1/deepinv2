# -*- coding: utf-8 -*-
"""全文扫描：检测被编辑工具拆断的 LaTeX 命令（\t \n \b \f \r 开头的合法JSON转义）
损坏形态：反斜杠命令的首个转义被解码成控制字符(TAB/LF/BS/FF/CR)，命令余部散落到行首/行内。
"""
import io, re
p = r"d:\works\deepinv\第14章 Flow Matching与最优传输\14.5 前沿组合：SD3-Flux = Rectified Flow + DiT.md"
with io.open(p, 'r', encoding='utf-8') as f:
    txt = f.read()

# 1) 残留控制字符（正常 md 不应有 TAB/FF/BS/CR）
for name, ch in [('TAB', '\t'), ('FF', '\f'), ('BS', '\x08'), ('CR', '\r')]:
    n = txt.count(ch)
    if n:
        print(f"[控制字符] {name} x{n}")

# 2) 行首出现疑似被拆断的命令残部（换行后紧跟这些片段）
suspects = ['abla', 'heta', 'imes', 'ext', 'o ', 'o$', 'rac', 'ight', 'ar', 'u ', 'u$']
lines = txt.split('\n')
for i, ln in enumerate(lines):
    for sp in suspects:
        if ln.startswith(sp):
            print(f"[行首残部] L{i+1}: {ln[:40]!r}")
            break

# 3) 数学块内出现裸换行（$...$ 跨行且中间无配对）粗检：统计 $ 计数奇偶
print("全文 $ 计数:", txt.count('$'), "(偶数=配对OK)")
print("扫描完成：若无上面输出则全文LaTeX命令完好")
