# -*- coding: utf-8 -*-
"""验证 17.6-1.py 修改后的 Step4 结论段：抽取源文件中真实代码执行，
   用 results_summary.json 的实测相对误差作为输入，检查打印自洽。"""
import sys, os, json, re
sys.stdout.reconfigure(encoding='utf-8')

BASE = r'h:\deepinv\deepinv2\第17章 自监督学习与等变架构\第十七章 配套实验\实验17.6-1'
src = open(os.path.join(BASE, '17.6-1.py'), encoding='utf-8').read()

# 抽取 "结论引用实测数值" 到 "总结" 之间的真实代码
i0 = src.index('# 结论引用实测数值')
i1 = src.index('# 总结', i0)
block = src[i0:i1]
print('=== 被执行的源文件代码段（前3行）===')
for line in block.splitlines()[:3]:
    print(' ', line)

j = json.load(open(os.path.join(BASE, 'results_summary.json'), encoding='utf-8'))
rel = j['等变性验证']['相对误差']
sampling_modes = ['vertical', 'random', 'cartesian']
rel_results = {}
for k, v in rel.items():
    op, t = k.rsplit('_', 1)
    rel_results[(op, t)] = v

ns = {'rel_results': rel_results, 'sampling_modes': sampling_modes}
exec(block, ns)
print('\n=== 执行完毕（以上为文件真实输出的结论段）===')
