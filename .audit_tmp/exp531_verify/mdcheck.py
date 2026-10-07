import glob
import os
import re

cands = []
for root in ['d:\\works\\deepinv', os.path.abspath('.')]:
    cands += [str(x) for x in __import__('pathlib').Path(root).rglob('5.3*.md')]
    if cands:
        break
p = cands[0]
t = open(p, encoding='utf-8').read()
L = t.splitlines()
print('FILE', p)
print('lines', len(L))
print('replacement_char_count', t.count('\ufffd'))
print('last_line', repr(L[-1])[:100])
print('--- image lines ---')
for i, l in enumerate(L):
    if 'png' in l:
        print(i + 1, l)
print('--- latex commands used ---')
print(sorted(set(re.findall(r'\\[a-zA-Z]+', t))))
print('--- suspicious backslash-nonalpha ---')
for i, l in enumerate(L):
    for m in re.finditer(r'\\[^a-zA-Z\\]', l):
        print(i + 1, repr(l[max(0, m.start() - 20):m.start() + 20]))
