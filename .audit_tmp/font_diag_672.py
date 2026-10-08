# -*- coding: utf-8 -*-
"""诊断 Colab 中文乱码根因：字体URL可达性 / 本地ttf注册名 / Colab候选命中情况"""
import os, urllib.request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

TTF = r'D:\works\deepinv\第6章 得分匹配：从去噪中学习得分\第六章 配套实现\实验6.7-2\.chinese\NotoSansSC-Regular.ttf'

print('=== 1. 下载URL可达性 ===')
url = 'https://github.com/jsntn/webfonts/raw/master/NotoSansSC-Regular.ttf'
try:
    req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=25) as r:
        print('OK', r.status, r.headers.get('Content-Length'))
except Exception as e:
    print('FAILED:', type(e).__name__, e)

print('=== 2. ttf 文件信息 ===')
print('exists', os.path.exists(TTF), os.path.getsize(TTF) if os.path.exists(TTF) else '-')
fp = matplotlib.font_manager.FontProperties(fname=TTF)
print('family name from file =', fp.get_name())
fontManager.addfont(TTF)
names = sorted({f.name for f in fontManager.ttflist if 'Noto' in f.name or 'SimHei' in f.name})
print('Noto/SimHei families in registry:', names[:30])
print('name registered?', fp.get_name() in {f.name for f in fontManager.ttflist})

print('=== 3. 模拟Colab候选列表命中(本地环境) ===')
cands = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans CJK',
         'Source Han Sans SC', 'AR PL UMing CN', 'SimHei']
avail = {f.name for f in fontManager.ttflist}
for c in cands:
    print(' ', c, '->', c in avail)
