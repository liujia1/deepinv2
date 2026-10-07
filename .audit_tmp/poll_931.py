# -*- coding: utf-8 -*-
import io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
paths = {
    'MAIN': r'.audit_tmp/exp931_verify/out.log',
    'SMOKE': r'.audit_tmp/exp931_smoke/out.log',
    'CTRL': r'.audit_tmp/exp931_control/run.log',
}
keys = ('Epoch', '探针', '快速', 'Active', 'ep ', '完成', 'Traceback', 'Error', 'error')
for tag, p in paths.items():
    if not os.path.exists(p):
        print('===', tag, '(no log yet)')
        continue
    t = open(p, encoding='utf-8', errors='replace').read().splitlines()
    ls = [l for l in t if any(k in l for k in keys)]
    print('===', tag, 'lines=%d' % len(ls))
    print(chr(10).join(ls[-8:]))
    err = p.replace('out.log', 'err.log')
    if os.path.exists(err):
        e = open(err, encoding='utf-8', errors='replace').read().strip().splitlines()
        e = [l for l in e if 'Traceback' in l or 'Error' in l]
        if e:
            print('   !! stderr:', e[-3:])
