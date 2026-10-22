# -*- coding: utf-8 -*-
"""修复后确认（p2_13）：
   1) 主脚本的 train_moi(tag='moipool_G1') 能识别 is_final ckpt、跳过训练；
   2) evaluate_combined 复算出的数值与 _g1_result.json 完全一致（同评估协议）。"""
import sys, os, json
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
losses = ns['train_moi'](model, ns['train_loader'], ns['N_EPOCHS'], 1,
                         ns['device'], tag='moipool_G1')
assert len(losses) == ns['N_EPOCHS'], f"应直接加载 20 轮记录，实得 {len(losses)}"

psnr_v, obs_v, miss_v, ssim_v = ns['evaluate_combined'](
    model, ns['test_loader'], ns['test_mask'], ns['SIGMA'], ns['device'], tag='MOI G=1(复核)')

ref = json.load(open(os.path.join(BASE, '_g1_result.json'), encoding='utf-8'))
print(f"\n复算: PSNR={psnr_v:.6f} obs={obs_v:.4f} miss={miss_v:.4f} ssim={ssim_v:.6f}")
print(f"参考: PSNR={ref['psnr']:.6f} obs={ref['obs']:.4f} miss={ref['miss']:.4f} ssim={ref['ssim']:.6f}")
assert abs(psnr_v - ref['psnr']) < 1e-6, 'PSNR 不可复现'
assert abs(ssim_v - ref['ssim']) < 1e-6, 'SSIM 不可复现'
print('✓ ckpt 加载分支与评估数值均与训练脚本一致')
