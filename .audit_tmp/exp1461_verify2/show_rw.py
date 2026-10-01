# -*- coding: utf-8 -*-
import json, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
rs = json.load(open(r'd:\works\deepinv\.audit_tmp\exp1461_verify2\seedscan_results.json', encoding='utf-8'))
for r in rs:
    parts = []
    for N in (8, 32, 64):
        d = r[f'N{N}']
        parts.append(f"N{N}: rw {d['rw_mlm']:.4f} vs {d['rw_unif']:.4f} | hit {d['hit_mlm']:.3f} vs {d['hit_unif']:.3f} (diff {d['diff']:+.3f})")
    print(r['task'], 'ts=', r['train_seed'], 'ss=', r['samp_seed'], 'loss', r['loss_mlm'], '/', r['loss_unif'])
    for p in parts:
        print('   ', p)
