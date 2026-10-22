# -*- coding: utf-8 -*-
"""验证 17.4-1.py 修改后的 Step5 分项报告与 concl4：
   抽取源文件真实代码段执行，注入 results_summary.json 的实测 tweedie_results。"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')

BASE = r'h:\deepinv\deepinv2\第17章 自监督学习与等变架构\第十七章 配套实验\实验17.4-1'
src = open(os.path.join(BASE, '17.4-1.py'), encoding='utf-8').read()
j = json.load(open(os.path.join(BASE, 'results_summary.json'), encoding='utf-8'))
tw = j['SURE→Tweedie验证']

# 1) Step5 打印段：从 "Tweedie闭环验证 (修复版)" 到 "# 增强的Tweedie验证可视化"
i0 = src.index('  Tweedie闭环验证 (修复版)')
i0 = src.rindex('print(', 0, i0)
i1 = src.index('# 增强的Tweedie验证可视化')
block = src[i0:i1]
print('=== 执行 Step5 打印段（源文件真实代码）===')
exec(compile(block, 'step5_print', 'exec'), {'tweedie_results': tw})

# 2) concl4 段：从 "# 4. SURE → Tweedie" 到 "# 5. 理论闭环"
j0 = src.index('# 4. SURE → Tweedie')
j1 = src.index('# 5. 理论闭环')
block2 = src[j0:j1]
ns = {'tweedie_results': tw}
print('\n=== 执行 concl4 段 ===')
exec(compile(block2, 'concl4', 'exec'), ns)
print('concl4 =', ns['concl4'])
