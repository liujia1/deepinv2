# -*- coding: utf-8 -*-
"""编辑后完整性检查: 反斜杠命令是否被拆分/丢失, 行数是否异常"""
import re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

files = [
    r'd:/works/deepinv/第14章 Flow Matching与最优传输/14.3 Flow Matching.md',
    r'd:/works/deepinv/第14章 Flow Matching与最优传输/第14章 Flow Matching与最优传输-提纲.md',
]
for f in files:
    s = open(f, encoding='utf-8').read()
    lines = s.split('\n')
    # 1) 可疑断裂: 行尾孤立的反斜杠前缀命令片段(如 \nabla 被拆成 "\n" + "abla")
    broken = [i + 1 for i, l in enumerate(lines)
              if re.search(r'\\(nabla|abla|alpha|eta|igma|hi|o)\s*$', l) or l.rstrip().endswith('\\n')]
    # 2) 双反斜杠命令中混入空格断裂, 如 "\ nabla"
    sp = re.findall(r'\\ \\?[a-z]{2,}', s)
    # 3) 数学块配对
    n_dollar = s.count('$$')
    print(f"{f.split('/')[-1]}: 行数={len(lines)} $$配对={n_dollar} (偶数OK={n_dollar%2==0}) 可疑断行={broken} 空格断裂={sp[:5]}")
