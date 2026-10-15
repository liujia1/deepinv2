# -*- coding: utf-8 -*-
"""终检：md 表格与 results_summary.json 是否逐项一致"""
import json, re

md = open(r'd:/works/deepinv/第1章 逆问题与贝叶斯推断/1.5 贝叶斯框架：从似然到后验.md', encoding='utf-8').read()
js = open(r'd:/works/deepinv/第1章 逆问题与贝叶斯推断/第一章 配套实验/实验1.5-1/results_summary.json', encoding='utf-8').read()
js = json.loads(js)
sec = md[md.index('## 实验 1.5-1'):md.index('## 五、')]
rows = []
for ln in sec.splitlines():
    c = [x.strip() for x in ln.strip('|').split('|')]
    if len(c) == 6 and re.fullmatch(r'\d+', c[0]):
        rows.append(c)
print('解析到', len(rows), '行')
ok = True
for c, p in zip(rows, js['posterior']):
    n, ht, m, mo, v, w = c
    hh, tt = [x.strip() for x in ht.split('/')]
    chk = {'n': int(n) == p['n'], 'heads': int(hh) == p['heads'], 'tails': int(tt) == p['tails'],
           'mean': abs(float(m) - p['mean']) < 5e-5, 'var': abs(float(v) - p['var']) < 5e-7,
           'prior_w': abs(float(w.rstrip('%')) - 100 * p['prior_weight']) < 0.06,
           'mode': (p['mode'] is None and not re.fullmatch(r'[\d.]+', mo))
                   or (p['mode'] is not None and abs(float(mo) - p['mode']) < 5e-5)}
    ok &= all(chk.values())
    print('n=' + n, 'OK' if all(chk.values()) else chk)

mc = js['mc_replay']
cites = {'85.8 非单调': (float(sec.count('85.8\\%') > 0), 100 * mc['mean_path']['prob_non_monotone']),
         '75.9 冲高回落': (float(sec.count('75.9\\%') > 0), 100 * mc['mean_path']['prob_overshoot_then_fall']),
         '99.8 逐次上升': (float(sec.count('99.8\\%') > 0), 100 * mc['variance_monotonicity']['prob_stepwise_increase_somewhere']),
         '10.29 解析': (float(sec.count('10.29\\%') > 0), 100 * mc['by_n'][1]['prob_mean_equals_prior_exact'])}
for k, (present, val) in cites.items():
    good = present > 0 and abs(round(val, 1) - float(k.split()[0])) < 0.06
    ok &= good
    print(f'  md 引用 {k}%  vs json {val:.4f}% ->', 'OK' if good else 'FAIL')
print('反例数值 Beta(1,3)=0.0375 / Beta(2,3)=0.04:',
      abs(mc['variance_monotonicity']['stepwise_counterexample']['Beta(1,3)_var'] - 0.0375) < 1e-9
      and abs(mc['variance_monotonicity']['stepwise_counterexample']['Beta(2,3)_var'] - 0.04) < 1e-9)
print('\n图文与 json 全部一致:', bool(ok and len(rows) == len(js['posterior'])))
