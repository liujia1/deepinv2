# -*- coding: utf-8 -*-
import io, sys, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import torch
ARCH = r'D:\works\deepinv\第9章 VAE与重参数化\第九章 配套实验\实验9.3-1\vae_mnist_checkpoint.pth'
RERUN = r'D:\works\deepinv\.audit_tmp\exp931_verify\vae_mnist_checkpoint.pth'
a = torch.load(ARCH, map_location='cpu', weights_only=False)['history']
b = torch.load(RERUN, map_location='cpu', weights_only=False)['history']
print('%-4s %10s %10s %8s | %8s %8s %6s | %7s %7s %6s' %
      ('ep', 'arch_loss', 'rerun_loss', 'd%', 'arch_kld', 'rerun_kld', 'd%', 'arch_ad', 'rerun_ad', 'diff'))
for i in range(len(a['loss'])):
    print('%-4d %10.2f %10.2f %7.2f%% | %8.3f %8.3f %5.1f%% | %7d %7d %6s' % (
        i + 1, a['loss'][i], b['loss'][i], 100 * (b['loss'][i] / a['loss'][i] - 1),
        a['kld'][i], b['kld'][i], 100 * (b['kld'][i] / a['kld'][i] - 1),
        a['active_dims'][i], b['active_dims'][i],
        '' if a['active_dims'][i] == b['active_dims'][i] else '*'))
print()
rj = json.load(open(r'D:\works\deepinv\.audit_tmp\exp931_verify\results_summary.json', encoding='utf-8'))
for k in ('test_loss', 'test_bce', 'test_kld', 'test_active_dims', 'test_recon_psnr_db'):
    print('rerun', k, '=', rj.get(k))
print('rerun keys:', sorted(rj.keys()))
# 权重是否逐位相同（判断是否同一随机流）
import numpy as np
ca = torch.load(ARCH, map_location='cpu', weights_only=False)['encoder_state_dict']['fc_mu.weight']
cb = torch.load(RERUN, map_location='cpu', weights_only=False)['encoder_state_dict']['fc_mu.weight']
print('fc_mu.weight max|diff| =', (ca - cb).abs().max().item())
# smoke: 初始化探针
sm = open(r'D:\works\deepinv\.audit_tmp\exp931_smoke\out.log', encoding='utf-8', errors='replace').read().splitlines()
for l in sm:
    if '探针' in l or '阈值' in l or '每维KL' in l or 'PSNR' in l or '活跃维度' in l:
        print('SMOKE:', l)
