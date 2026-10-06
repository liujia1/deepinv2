# -*- coding: utf-8 -*-
"""核验cosine重训后的checkpoint与JSON一致性"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import torch

BASE = r'd:\works\deepinv\第12章 Score ≡ ELBO：殊途同归\第十二章 配套实验\实验12.6-1'
js = json.load(open(os.path.join(BASE, 'results_summary.json'), encoding='utf-8'))

for name, key in [('dsm_model_checkpoint.pth', 'sampling_path_DSM'),
                  ('vlb_model_checkpoint.pth', 'variational_path_VLB')]:
    ck = torch.load(os.path.join(BASE, name), map_location='cpu', weights_only=False)
    print('='*58)
    print(name)
    print('  keys:', list(ck.keys()))
    print('  schedule:', ck.get('schedule'), '| is_final:', ck.get('is_final'), '| epoch:', ck.get('epoch'))
    print('  含优化器状态:', 'optimizer_state_dict' in ck)
    tl_ck = ck['train_losses']
    tl_js = js[key]['loss_curve']
    same = len(tl_ck) == len(tl_js) and all(abs(a - b) < 1e-12 for a, b in zip(tl_ck, tl_js))
    print(f'  train_losses长度={len(tl_ck)}, 与JSON曲线逐位一致: {same}')
    print(f'  ckpt loss={ck["loss"]:.10f} vs JSON final_loss={js[key]["final_loss"]:.10f} '
          f'-> {abs(ck["loss"]-js[key]["final_loss"])<1e-12}')

# 双验证数值有效性: score_norm 应等于 eps_norm/sqrt(1-abar_50)
T, s = 200, 0.008
import math
_steps = torch.arange(T + 1, dtype=torch.float64)
_ac = torch.cos((_steps / T + s) / (1 + s) * math.pi / 2) ** 2
_ac = _ac / _ac[0]
betas = (1 - _ac[1:] / _ac[:-1]).clamp(max=0.999).float()
ab = torch.cumprod(1 - betas, 0)
dv = js['dual_verification']
expect_score = dv['eps_pred_norm'] / math.sqrt(1 - ab[50])
print('='*58)
print(f'cosine ᾱ_50 = {ab[50]:.4f}')
print(f"score_norm 存档={dv['score_norm']:.4f} vs 由eps换算={expect_score:.4f} "
      f"(应严格相等: {abs(dv['score_norm']-expect_score)<0.02})")
