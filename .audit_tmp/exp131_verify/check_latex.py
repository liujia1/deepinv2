# -*- coding: utf-8 -*-
"""校验 md 与 py 的 LaTeX 反斜杠命令未被编辑破坏：与 git HEAD 版本对比命令频次。"""
import io, re, subprocess, collections

def cmds(t):
    return collections.Counter(re.findall(r'\\[a-zA-Z]+', t))

for repo, work in [
    (r'第1章 逆问题与贝叶斯推断/1.3 不适定性：为什么逆问题难？.md',
     r'第1章 逆问题与贝叶斯推断\1.3 不适定性：为什么逆问题难？.md'),
    (r'第1章 逆问题与贝叶斯推断/第一章 配套实验/实验1.3-1/1.3-1.py',
     r'第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.3-1\1.3-1.py'),
]:
    head = subprocess.run(['git', 'show', f'HEAD:{repo}'], capture_output=True)
    old = head.stdout.decode('utf-8')
    new = io.open(work, encoding='utf-8').read()
    co, cn = cmds(old), cmds(new)
    removed = {k: co[k] - cn.get(k, 0) for k in co if co[k] > cn.get(k, 0)}
    added = {k: cn[k] - co.get(k, 0) for k in cn if cn[k] > co.get(k, 0)}
    print(f'--- {work}')
    print(f'  HEAD 命令数 {sum(co.values())} -> 工作区 {sum(cn.values())}')
    print(f'  减少: {removed}')
    print(f'  增加: {added}')
    # 孤立反斜杠（后面既不是字母也不是常见符号）通常是转义损坏的痕迹
    bad = re.findall(r'\\(?![a-zA-Z\\_{}$\s^])', new)
    print(f'  可疑孤立反斜杠 {len(bad)} 处: {set(bad)}')
