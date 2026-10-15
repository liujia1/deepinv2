# -*- coding: utf-8 -*-
"""两项通用检查：
(1) 每个实验脚本里的标题/自述编号 与 所在目录名 是否一致（抓"图内编号写错且未重跑"）
(2) git 历史：1.5-3.py 标题行何时改动，归档 PNG 是否晚于该改动
"""
import io, os, re, subprocess

ROOT = r'd:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验'
out = io.open(r'd:\works\deepinv\.audit_tmp\exp153_verify\out_title_scan.txt', 'w', encoding='utf-8')


def w(s):
    out.write(s + '\n')


w('=== (1) 脚本内编号自述 vs 目录名 ===')
for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, d)
    if not os.path.isdir(p) or d.startswith('.'):
        continue
    for f in sorted(os.listdir(p)):
        if not f.endswith('.py'):
            continue
        src = io.open(os.path.join(p, f), encoding='utf-8').read()
        # 抓 suptitle / set_title / print 头部里出现的 "实验X.Y-Z" 或 "X.Y-Z" 编号
        ids = set(re.findall(r'(?:实验|Experiment)\s*(\d+\.\d+-\d+|\d+[a-zA-Z]-\d+)', src))
        self_id = re.match(r'(\d+\.\d+-\d+|\d+\.[a-zA-Z]-\d+)', f)
        expect = self_id.group(1) if self_id else d
        w('[%s] 文件=%s  期望编号=%s  文中出现的实验编号=%s' % (d, f, expect, sorted(ids)))
        for i in sorted(ids):
            if i.replace(' ', '') != expect.replace(' ', ''):
                w('    [!] 文中出现 %s，与自身编号 %s 不一致（可能是从别的脚本拷来的标题）' % (i, expect))
        # suptitle 具体文本
        for m in re.finditer(r'suptitle\((.{0,120})', src):
            w('    suptitle: %s' % m.group(1).replace('\n', ' ')[:120])

w('')
w('=== (2) 1.5-3 标题行与 PNG 的提交历史 ===')
repo = '第1章 逆问题与贝叶斯推断/第一章 配套实验/实验1.5-3'
for path in (repo + '/1.5-3.py', repo + '/实验1_5_3_似然等于最小二乘.png'):
    r = subprocess.run(['git', 'log', '--format=%h %ad %s', '--date=short', '--', path],
                       capture_output=True, cwd=r'd:\works\deepinv')
    w('--- %s' % path)
    w(r.stdout.decode('utf-8', 'replace').strip()[:800])

r = subprocess.run(['git', 'log', '-p', '--format=COMMIT %h %ad', '--date=short', '-S', '实验1.5-3：高斯似然',
                    '--', repo + '/1.5-3.py'], capture_output=True, cwd=r'd:\works\deepinv')
w('--- 引入 "实验1.5-3：高斯似然" 字样的提交 ---')
txt = r.stdout.decode('utf-8', 'replace')
w('\n'.join([l for l in txt.splitlines() if l.startswith('COMMIT') or l.startswith('+') or l.startswith('-')][:40]))
out.close()
print('ok')
