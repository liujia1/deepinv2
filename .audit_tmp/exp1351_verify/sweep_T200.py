# -*- coding: utf-8 -*-
"""T=200 对照：合理步长DPS(ζ=0.3) vs DOC，裁决讲义"多步采样DPS接近DOC"说法"""
import numpy as np, torch, json
from sweep_dps_doc import build_schedule, make_score, make_trials, run_dps, run_doc_original, GM_MEANS

T2 = 200
betas2, ab2 = build_schedule(T2)
score2 = make_score(ab2)
trials2 = make_trials(20, T2)
rng = np.random.RandomState(42)
x0_stars2 = []
for _ in range(20):
    comp = rng.choice(2, p=[0.5, 0.5])
    x0s = rng.normal(GM_MEANS[comp], 0.5)
    rng.randn()  # 跳过y的噪声随机数，保持与make_trials一致的流
    x0_stars2.append(x0s)

out = {}
for zeta in (0.3, 0.5):
    errs = []
    for k, tr in enumerate(trials2):
        gen = torch.Generator().manual_seed(5000 + k)
        x0 = run_dps(tr, betas2, ab2, score2, 'beta', zeta, gen, T2)
        errs.append(abs(x0 - x0_stars2[k]))
    errs = np.array(errs)
    name = f'DPS_beta ζ={zeta} (T=200)'
    out[name] = {'mean': round(float(errs.mean()), 4), 'std': round(float(errs.std()), 4)}
    print(name, out[name])

with open('sweep_T200_results.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('done')
