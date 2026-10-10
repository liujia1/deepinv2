# -*- coding: utf-8 -*-
"""核查 Colab 刚生成的 5.5-1 sampling_results.npz：
键完整性/形状/数值健全性 + 用共享闸门跑一遍（=三个下游将看到的判定）。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP551 = os.path.abspath(os.path.join(
    HERE, '..', '..',
    '第5章 朗之万动力学与得分函数', '第五章配套实验', '实验5.5-1'))
sys.path.insert(0, EXP551)
from pnp_ula_common import load_sampling_result, upstream_chain_gate, print_gate

data, path = load_sampling_result(EXP551)
print('文件:', path)
print('键:', sorted(data.files))

cfg = [int(v) for v in data['chain_config'].tolist()]
print('chain_config [maxit,burnin,half_mark,thin] =', cfg)

pm = data['post_mean']; pv = data['post_var']
mc = data['mc_samples']; xt = data['x_true']; yo = data['y_obs']
print(f'post_mean {pm.shape} [{pm.min():.4f},{pm.max():.4f}]')
print(f'post_var  {pv.shape} min={pv.min():.3e} mean={pv.mean():.3e} (有负值? {bool((pv < 0).any())})')
print(f'mc_samples{mc.shape} [{mc.min():.4f},{mc.max():.4f}]')
print(f'x_true {xt.shape} [{xt.min():.3f},{xt.max():.3f}], y_obs {yo.shape}')
h1 = data['post_var_h1']; h2 = data['post_var_h2']
print(f'post_var_h1 {h1.shape}, post_var_h2 {h2.shape}')

tp = data['trace_psnr']; tf = data['trace_fid']
print(f'trace_psnr {tp.shape}: 首值 {tp[0]:.2f}, 3/4处 {tp[int(.75*len(tp))]:.2f}, 末值 {tp[-1]:.2f}')
print(f'trace_fid  {tf.shape}')

# MMSE 指标复核（用与5.5-1同口径的 PSNR；measures 接受 torch 张量）
import torch
from sampling_tools.measures import PSNR, NRMSE, SSIM
_xt, _pm = torch.from_numpy(xt), torch.from_numpy(pm)
print(f'复核 PSNR(MMSE)={PSNR(_xt, _pm):.4f} dB, NRMSE={NRMSE(_xt, _pm):.4f}, SSIM={SSIM(_xt, _pm):.4f}')
print(f'抽稀样本均值 vs npz post_mean  RMS差异={np.sqrt(np.mean((mc.mean(0)-pm)**2)):.3e}')

print()
status, warnings, extras = upstream_chain_gate(data, path, purpose='下游分析')
print_gate(status, warnings, '下游结论有效')
print('\nextras:', {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in extras.items()})
