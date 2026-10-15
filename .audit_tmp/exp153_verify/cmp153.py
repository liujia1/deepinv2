# -*- coding: utf-8 -*-
"""比较 1.5-3 重跑产物与归档产物（避免 stdout 编码问题，结果写文件）。"""
import io, os, json, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = r'd:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.5-3'
TMP = os.path.join(HERE, 'exp153_rerun')
out = io.open(os.path.join(HERE, 'out153_cmp.txt'), 'w', encoding='utf-8')


def md5(p):
    return hashlib.md5(io.open(p, 'rb').read()).hexdigest()


def w(s):
    out.write(s + '\n')


pj_new = os.path.join(TMP, 'results_summary.json')
pj_old = os.path.join(EXP, 'results_summary.json')
w('new json exists: %s' % os.path.exists(pj_new))
a = json.load(io.open(pj_new, encoding='utf-8'))
b = json.load(io.open(pj_old, encoding='utf-8'))
w('JSON 逐位一致: %s' % (a == b))
for k in b:
    if a.get(k) != b[k]:
        w('  差异 %s: 归档=%r 新=%r' % (k, b[k], a.get(k)))

pn = os.path.join(TMP, '实验1_5_3_似然等于最小二乘.png')
po = os.path.join(EXP, '实验1_5_3_似然等于最小二乘.png')
w('PNG 新 size=%d md5=%s' % (os.path.getsize(pn), md5(pn)[:12]))
w('PNG 旧 size=%d md5=%s' % (os.path.getsize(po), md5(po)[:12]))
w('PNG 一致: %s' % (md5(pn) == md5(po)))

# 重跑 stdout 的关键段落
log = io.open(os.path.join(TMP, 'run.log'), encoding='utf-8', errors='replace').read() if os.path.exists(os.path.join(TMP, 'run.log')) else ''
w('--- run.log tail ---')
w('\n'.join(log.splitlines()[-25:]))
out.close()
print('ok')
