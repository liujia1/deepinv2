import io

p = '4.1-1.py'
lines = io.open(p, encoding='utf-8').read().split('\n')
n = 0
for i, ln in enumerate(lines):
    st = ln.strip()
    if ('\u2082' in ln) and st.startswith('f"'):
        lines[i] = ln.replace('\u2082', '2')
        n += 1
io.open(p, 'w', encoding='utf-8').write('\n'.join(lines))
print('fixed lines:', n)
for i, ln in enumerate(lines, 1):
    st = ln.strip()
    if '\u2082' in ln and st.startswith('f"'):
        print('REMAIN', i, ln)
