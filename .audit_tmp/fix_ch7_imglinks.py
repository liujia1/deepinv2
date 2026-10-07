# -*- coding: utf-8 -*-
import io, os, re, glob, sys

CH = r'd:/works/deepinv/第7章 扩散模型：SDE视角'
mds = sorted(glob.glob(os.path.join(CH, '*.md')))

OLD = '配套实验/7.'
NEW = '配套实验/实验7.'

# 匹配 markdown 图片链接: ![...](path)
img_re = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')

changed_files = []
for md in mds:
    s = io.open(md, encoding='utf-8').read()
    if OLD in s:
        s2 = s.replace(OLD, NEW)
        io.open(md, 'w', encoding='utf-8').write(s2)
        changed_files.append((os.path.basename(md), s.count(OLD)))

print('=== 修改的文件(次数) ===')
for f, c in changed_files:
    print(f'  {f} : {c}')

print('\n=== 校验全部配图链接是否存在 ===')
missing = 0
total = 0
for md in mds:
    base = os.path.dirname(md)
    s = io.open(md, encoding='utf-8').read()
    for m in img_re.finditer(s):
        link = m.group(1)
        if not link.startswith('./') and not link.startswith('#'):
            continue
        rel = link.replace('%20', ' ')
        if rel.startswith('./'):
            rel = rel[2:]
        path = os.path.normpath(os.path.join(base, rel))
        total += 1
        if not os.path.isfile(path):
            missing += 1
            print(f'  [缺失] {os.path.basename(md)} -> {link}')

print(f'\n共检查 {total} 条本地图片链接, 缺失 {missing} 条')
