# -*- coding: utf-8 -*-
"""Chapter-1 remaining experiments: static scan for shared-RNG-stream coupling."""
import io, os, re, json

ROOT = r'd:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验'
EXPS = ['实验1.2-2', '实验1.4-1', '实验1.4-2', '实验1.5-1', '实验1.5-2', '实验1.5-3', '附录1A']
RNG = re.compile(r'np\.random\.(randn|rand|normal|choice|randint|permutation)\b|np\.random\.random\(')

for e in EXPS:
    d = os.path.join(ROOT, e)
    print('=' * 70)
    files = sorted(os.listdir(d))
    print('[%s] %s' % (e, [f for f in files if not f.startswith('.')]))
    for py in [f for f in files if f.endswith('.py')]:
        p = os.path.join(d, py)
        src = io.open(p, encoding='utf-8').read()
        lines = src.splitlines()
        print('  --- %s: %d lines, seeds=%d' % (py, len(lines),
              len(re.findall(r'np\.random\.seed|default_rng|RandomState', src))))
        infor = None
        hits = []
        for i, ln in enumerate(lines, 1):
            m = re.match(r'^(\s*)for\s+.*:\s*(#.*)?$', ln)
            if m:
                infor = (len(m.group(1)), i)
                continue
            if infor:
                cur = len(ln) - len(ln.lstrip()) if ln.strip() else 999
                if ln.strip() and cur <= infor[0]:
                    infor = None
                elif RNG.search(ln):
                    hits.append((i, ln.strip()[:90]))
        if hits:
            print('    [!] RNG consumed inside for-loops (%d):' % len(hits))
            for i, t in hits[:6]:
                print('       L%d: %s' % (i, t))
        else:
            print('    no RNG consumption inside sweep loops')
        r6 = re.findall(r'round\([^,]+,\s*(\d+)\)', src)
        if r6:
            print('    round() precisions in file: %s' % r6[:10])
    jf = os.path.join(d, 'results_summary.json')
    if os.path.exists(jf):
        t = io.open(jf, encoding='utf-8').read()
        print('  JSON %d chars; keys=%s' % (len(t), list(json.loads(t).keys())[:16]))
        if re.search(r':\s*0\.0[,}\n]', t):
            print('    [!] JSON contains literal 0.0 values (possible precision loss)')
    else:
        print('    [!] no results_summary.json')
