# -*- coding: utf-8 -*-
import sys, io, math, py_compile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import torch
PY = r'd:\works\deepinv\第12章 Score ≡ ELBO：殊途同归\第十二章 配套实验\实验12.6-1\12.6-1.py'
py_compile.compile(PY, doraise=True)
print('compile OK')
T, s = 200, 0.008
st = torch.arange(T + 1, dtype=torch.float64)
ac = torch.cos((st / T + s) / (1 + s) * math.pi / 2) ** 2
ac = ac / ac[0]
b = (1 - ac[1:] / ac[:-1]).clamp(max=0.999).float()
ab = torch.cumprod(1 - b, 0)
print(f'终端√ᾱ_T={ab[-1].sqrt():.6f}  首端ᾱ_0={ab[0]:.4f}  β范围[{b.min():.5f},{b.max():.5f}]')
# 新norm自洽性: score = eps/√(1-ᾱ_50)
import json, os
js = json.load(open(os.path.join(os.path.dirname(PY), 'results_summary.json'), encoding='utf-8'))
dv = js['dual_verification']
expect = dv['eps_pred_norm'] / math.sqrt(1 - ab[50].item())
print(f"eps={dv['eps_pred_norm']:.4f} score存档={dv['score_norm']:.4f} 换算={expect:.4f} 自洽={abs(dv['score_norm']-expect)<0.02}")
