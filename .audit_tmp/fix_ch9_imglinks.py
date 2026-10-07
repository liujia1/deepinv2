# -*- coding: utf-8 -*-
"""修复第9章 md 中图片相对路径缺少空格的问题（实际目录名为 '第九章 配套实验'）
统一用 %20 转义（与第13章及以后章节的写法一致，避免空格被当成链接分隔符）
"""
import io, sys, glob, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
for f in sorted(glob.glob(r'D:\works\deepinv\第9章 VAE与重参数化\*.md')):
    t = open(f, encoding='utf-8', newline='').read()
    n1 = t.count('./第九章配套实验/')
    if n1:
        t = t.replace('./第九章配套实验/', './第九章%20配套实验/')
        open(f, 'w', encoding='utf-8', newline='').write(t)
    print(os.path.basename(f), 'fixed links:', n1)

# 校验
import re, urllib.parse
for f in sorted(glob.glob(r'D:\works\deepinv\第9章 VAE与重参数化\*.md')):
    d = os.path.dirname(f)
    for l in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', open(f, encoding='utf-8').read()):
        p = os.path.normpath(os.path.join(d, urllib.parse.unquote(l)))
        print(('OK  ' if os.path.exists(p) else 'BROKEN'), os.path.basename(f), '->', l)
