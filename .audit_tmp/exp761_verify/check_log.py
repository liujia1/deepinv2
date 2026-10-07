# -*- coding: utf-8 -*-
import re, sys, io, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
r = subprocess.run([sys.executable, r'd:\works\deepinv\.audit_tmp\exp761_verify\7.6-1.py'],
                   capture_output=True)
txt = r.stdout.decode('utf-8', errors='replace')
open(r'd:\works\deepinv\.audit_tmp\exp761_verify\run3.log', 'w', encoding='utf-8').write(txt)
for l in txt.splitlines():
    if re.search(r'注：|剔除|改善|最优|不敌|收敛|定性|抽样|5步|std=|Bug', l):
        print(l)
