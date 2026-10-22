# -*- coding: utf-8 -*-
"""补 17.5-1 的 G=1 纯 MC 基线：复用主脚本的定义与 train_moi（文本切片 exec），
   只训练 G=1，评估后写出结果片段，供人工并入 results_summary.json。
   产出的 ckpt_moipool_G1.pt 会被主脚本后续运行直接加载（tag 一致）。"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')
import torch

BASE = r'h:\deepinv\deepinv2\第17章 自监督学习与等变架构\第十七章 配套实验\实验17.5-1'
src = open(os.path.join(BASE, '17.5-1.py'), encoding='utf-8').read()
seg_defs = src[: src.index("# Step 1: 单算子基线")]
seg_train = src[src.index("def moi_loss"): src.index("# 训练MOI G=1")]

ns = {'__name__': '__audit__', '__file__': os.path.join(BASE, '17.5-1.py')}
exec(compile(seg_defs, 'seg_defs', 'exec'), ns)
exec(compile(seg_train, 'seg_train', 'exec'), ns)

model = ns['SmallUNetMOI'](in_ch=2, out_ch=1, base=24).to(ns['device'])
t0 = time.time()
losses = ns['train_moi'](model, ns['train_loader'], ns['N_EPOCHS'], 1,
                         ns['device'], tag='moipool_G1')
print(f'\nG=1 训练完成，用时 {(time.time()-t0)/60:.1f} min')
psnr_v, obs_v, miss_v, ssim_v = ns['evaluate_combined'](
    model, ns['test_loader'], ns['test_mask'], ns['SIGMA'], ns['device'], tag='MOI G=1')
out = {'G': 1, 'psnr': float(psnr_v), 'obs': float(obs_v),
       'miss': float(miss_v), 'ssim': float(ssim_v),
       'gap': float(obs_v - miss_v), 'final_loss': float(losses[-1])}
print('\n' + json.dumps(out, ensure_ascii=False, indent=2))
with open(os.path.join(BASE, '_g1_result.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('已写出 _g1_result.json')
