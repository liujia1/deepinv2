# -*- coding: utf-8 -*-
"""提取实验6.1-1脚本在历史各提交中的版本，对比关键绘图/数值段"""
import subprocess, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def git(*args):
    return subprocess.run(['git'] + list(args), capture_output=True).stdout.decode('utf-8', errors='replace')

commits = ['2d212a8', '9aac748', 'faefddd', 'a87d11e']
# 用 --name-only 输出的历史路径（旧路径），需逐个尝试
paths_old = [
    '第6章 得分匹配：从去噪中学习得分/第六章配套实现/6.1-1/6.1-1.py',
    '第6章 得分匹配：从去噪中学习得分/第六章配套实现/实验6.1-1/6.1-1.py',
    '第6章 得分匹配：从去噪中学习得分/第六章 配套实现/实验6.1-1/6.1-1.py',
]

def show(commit):
    for p in paths_old:
        out = git('show', f'{commit}:{p}')
        if out.strip():
            return out
    return ''

prev = None
for c in commits:
    src = show(c)
    print('=' * 20, c, 'len=', len(src))
    if not src:
        print('  !! not found at any known path')
        continue
    fn = f'.audit_tmp/611_src_{c}.py'
    io.open(fn, 'w', encoding='utf-8').write(src)
    if prev is not None:
        import difflib
        d = list(difflib.unified_diff(prev.splitlines(), src.splitlines(), lineterm='', n=1))
        for l in d[:80]:
            print(l)
        print('  (diff lines total:', len(d), ')')
    prev = src
