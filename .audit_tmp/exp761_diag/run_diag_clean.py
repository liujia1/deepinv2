# -*- coding: utf-8 -*-
import re, subprocess, sys
r = subprocess.run([sys.executable, r'd:\works\deepinv\.audit_tmp\exp761_diag\7.6-1_diag.py'], capture_output=True)
txt = r.stdout.decode('utf-8', errors='replace')
open(r'd:\works\deepinv\.audit_tmp\exp761_diag\diag_clean.log', 'w', encoding='utf-8').write(txt)
lines = [l for l in txt.splitlines() if re.search(r'反而最窄|回平衡|居中|最好|最差|教学意义|无系统差异|内部区间|注：', l)]
open(r'd:\works\deepinv\.audit_tmp\exp761_diag\check_diag_out.txt', 'w', encoding='utf-8').write(f'exit={r.returncode}\n' + '\n'.join(lines))
print('done', len(lines))
