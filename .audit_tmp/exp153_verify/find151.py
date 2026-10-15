# -*- coding: utf-8 -*-
"""定位 1.5-1 结论段的精确字符内容（排查引号/破折号等不可见差异）。"""
import io

P = r'd:\works\deepinv\第1章 逆问题与贝叶斯推断\1.5 贝叶斯框架：从似然到后验.md'
lines = io.open(P, encoding='utf-8').read().splitlines()
out = io.open(r'd:\works\deepinv\.audit_tmp\exp153_verify\out_repr151.txt', 'w', encoding='utf-8')
for i, ln in enumerate(lines, 1):
    if '五条曲线' in ln or '一路逼近' in ln:
        out.write('L%d repr:\n%r\n\n' % (i, ln))
        out.write('逐字符: ' + ' '.join('%s(U+%04X)' % (c, ord(c)) for c in ln[:60]) + '\n\n')
out.close()
print('ok')
