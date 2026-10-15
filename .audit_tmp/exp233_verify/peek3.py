# -*- coding: utf-8 -*-
import io, sys
sys.stdout.reconfigure(encoding='utf-8')
sc = io.open(r'd:\works\deepinv\.audit_tmp\exp233_verify\edit_233_md.py', encoding='utf-8').read()
md = io.open(r'd:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\2.3 先验的质量：MMSE vs MAP估计器.md', encoding='utf-8').read()
BS = chr(92)
for ln in sc.splitlines():
    if ln.strip().startswith("L('实验 `2.3-3.py` 计算并绘制"):
        mine = ln.strip()[3:-3].replace('[[', BS)
        break
else:
    mine = None
target = md[md.find('实验 `2.3-3.py` 计算并绘制'):]
target = target[:target.find('\n')]
print('mine  :', repr(mine))
print('target:', repr(target))
print('equal:', mine == target)
if mine:
    for k, (a, b) in enumerate(zip(mine, target)):
        if a != b:
            print('first diff @', k, repr(a), repr(b))
            print('mine ctx  :', repr(mine[max(0, k-20):k+20]))
            print('targ ctx  :', repr(target[max(0, k-20):k+20]))
            break
