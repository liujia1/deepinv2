# -*- coding: utf-8 -*-
src = r'd:\works\deepinv\.audit_tmp\exp761_diag\7.6-1.py'
txt = open(src, encoding='utf-8').read()
assert 'DIAGNOSE = False' in txt
txt = txt.replace('DIAGNOSE = False', 'DIAGNOSE = True')
open(r'd:\works\deepinv\.audit_tmp\exp761_diag\7.6-1_diag.py', 'w', encoding='utf-8').write(txt)
print('patched ok')
