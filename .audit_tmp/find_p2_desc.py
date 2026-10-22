# -*- coding: utf-8 -*-
"""从 transcript 中定位上一轮报告的 P2 项原始描述"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
p = r'C:\Users\liuji\.qoder-cn\cache\projects\deepinv2-5dfe434\conversation-history\88ae662c\88ae662c.jsonl'

hits = []
for i, line in enumerate(open(p, encoding='utf-8')):
    try:
        o = json.loads(line)
    except Exception:
        continue
    if o.get('role') != 'assistant':
        continue
    txt = ''.join(c.get('text', '') for c in o['message'].get('content', []) if isinstance(c, dict))
    if ('P2' in txt or 'p2' in txt) and ('G=1' in txt or 'Tweedie' in txt or '覆盖度' in txt):
        hits.append((i, len(txt), txt))

print(f'命中 {len(hits)} 条 assistant 消息')
for i, n, txt in hits[-3:]:
    print(f'\n===== line {i} (len {n}) =====')
    # 只打印包含 P2 的段落
    k = txt.find('P2')
    if k < 0:
        k = txt.find('p2')
    print(txt[max(0, k-200): k+2600])
