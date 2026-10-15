# -*- coding: utf-8 -*-
"""Compare LaTeX command frequency: working tree vs git HEAD.
Usage: python check_latex_generic.py <repo/relative/path.md> ...
"""
import io
import os
import re
import sys
import subprocess
import collections

ROOT = 'd:/works/deepinv'
CMD = re.compile(r'\\\\[a-zA-Z]+')


def cmds(t):
    return collections.Counter(re.findall(r'\\[a-zA-Z]+', t))


for repo in sys.argv[1:]:
    head = subprocess.run(['git', '-c', 'core.quotepath=false', 'show', 'HEAD:' + repo],
                          capture_output=True, cwd=ROOT)
    if head.returncode:
        print('!! cannot read HEAD:' + repo)
        continue
    old = head.stdout.decode('utf-8')
    new = io.open(os.path.join(ROOT, repo), encoding='utf-8').read()
    co, cn = cmds(old), cmds(new)
    removed = dict((k, co[k] - cn.get(k, 0)) for k in co if co[k] > cn.get(k, 0))
    added = dict((k, cn[k] - co.get(k, 0)) for k in cn if cn[k] > co.get(k, 0))
    print('--- ' + repo)
    print('  HEAD cmds %d -> worktree %d' % (sum(co.values()), sum(cn.values())))
    print('  removed (should be empty): %s' % removed)
    print('  added: %s' % added)
    bad = re.findall(r'\\(?![a-zA-Z\\_{}$\s^.,!|;&])', new)
    print('  suspicious lone backslash %d: %s' % (len(bad), sorted(set(bad))))
