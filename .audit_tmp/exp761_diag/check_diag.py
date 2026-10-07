# -*- coding: utf-8 -*-
import re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
raw = open(r'd:\works\deepinv\.audit_tmp\exp761_diag\diag_run3.log', 'rb').read()
for enc in ('utf-16', 'utf-8', 'gbk'):
    try:
        txt = raw.decode(enc); print(f'--- decoded as {enc} ---'); break
    except Exception:
        continue
lines = [l for l in txt.splitlines() if re.search(r'反而最窄|回平衡|居中|最好|最差|教学意义|无系统差异|内部区间', l)]
open(r'd:\works\deepinv\.audit_tmp\exp761_diag\check_diag_out.txt', 'w', encoding='utf-8').write(f'--- decoded as {enc} ---\n' + '\n'.join(lines))
print('done', len(lines))
