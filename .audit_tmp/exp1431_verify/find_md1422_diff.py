# -*- coding: utf-8 -*-
"""定位用户对 14.2 CNF.md 的 15:15 批次改动: 对比最近两个历史快照及当前文件"""
import json, glob, os, io, sys, difflib, urllib.parse, datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

MD = r'd:\works\deepinv\第14章 Flow Matching与最优传输\14.2 连续归一化流(CNF).md'

for x in glob.glob(r'C:\Users\admin\AppData\Roaming\Qoder*\User\History\**\entries.json', recursive=True):
    try:
        d = json.load(open(x, encoding='utf-8'))
    except Exception:
        continue
    r = urllib.parse.unquote(d.get('resource', ''))
    if not r.endswith('14.2 连续归一化流(CNF).md'):
        continue
    es = sorted(d['entries'], key=lambda e: e['timestamp'])
    for e in es:
        fp = os.path.join(os.path.dirname(x), e['id'])
        print(e['id'], datetime.datetime.fromtimestamp(e['timestamp'] / 1000), os.path.getsize(fp) if os.path.exists(fp) else -1)
    # 找 15:15 之前最后一个快照 与 当前文件对比
    cut = datetime.datetime(2026, 9, 30, 15, 0).timestamp() * 1000
    before = [e for e in es if e['timestamp'] < cut]
    if not before:
        break
    a = open(os.path.join(os.path.dirname(x), before[-1]['id']), encoding='utf-8').read().splitlines()
    cur = open(MD, encoding='utf-8').read().splitlines()
    dd = list(difflib.unified_diff(a, cur, 'before_15', 'current', lineterm='', n=4))
    print('=== diff lines:', len(dd))
    print('\n'.join(dd))
    break
