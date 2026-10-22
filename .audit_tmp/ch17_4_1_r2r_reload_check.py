# -*- coding: utf-8 -*-
"""修复后确认（p2_14）：主脚本 load_checkpoint(ckpt_R2R.pt) 识别 is_final、
   跳过训练，并用 evaluate_psnr/ssim(EVAL_SEED) 复现 α=0.5 重训结果。"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
import torch

BASE = r'h:\deepinv\deepinv2\第17章 自监督学习与等变架构\第十七章 配套实验\实验17.4-1'
src = open(os.path.join(BASE, '17.4-1.py'), encoding='utf-8').read()
seg_defs = src[: src.index("# Step 1: SURE原理验证")]

ns = {'__name__': '__audit__', '__file__': os.path.join(BASE, '17.4-1.py')}
exec(compile(seg_defs, 'seg_defs', 'exec'), ns)

model = ns['SmallUNet']().to(ns['device'])
opt = torch.optim.Adam(model.parameters(), lr=ns['LR'])
ckpt = os.path.join(BASE, 'ckpt_R2R.pt')
start, losses, is_final = ns['load_checkpoint'](ckpt, model, opt, ns['device'])
print(f"load_checkpoint → start_epoch={start}, is_final={is_final}, n_losses={len(losses)}")
assert is_final and start >= ns['N_EPOCHS'], '主脚本应跳过重训'

p = ns['evaluate_psnr'](model, ns['test_loader'], seed=ns['EVAL_SEED'])
s = ns['evaluate_ssim'](model, ns['test_loader'], seed=ns['EVAL_SEED'])
ref = json.load(open(os.path.join(BASE, '_r2r_alpha05_result.json'), encoding='utf-8'))
print(f"复算: PSNR={p:.6f} SSIM={s:.6f}")
print(f"参考: PSNR={ref['psnr']:.6f} SSIM={ref['ssim']:.6f}")
assert abs(p - ref['psnr']) < 1e-6 and abs(s - ref['ssim']) < 1e-6
print('✓ 主脚本路径可复现 α=0.5 的 R2R 数值')

# 与旧 α=0.1 结果的差异（修复收益）
old = 25.41660626879837
print(f"对比旧 α=0.1: {old:.2f} → {p:.2f} dB（+{p-old:.2f} dB）")
