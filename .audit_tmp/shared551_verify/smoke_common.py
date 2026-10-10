# -*- coding: utf-8 -*-
"""共享模块 pnp_ula_common 短链烟雾测试（真模型、CPU、短链）。
验证：
  1) build_problem 同种子可复现（两次构建 y 完全一致）
  2) run_pnp_ula 主链模式(want_metric_curves=True)返回键/计数/诊断
  3) run_pnp_ula 扫描模式(want_metric_curves=False, trace_fn)与5.6-3用法一致
不跑长链：40步量级仅验证接线与数值口径。
"""
import os
import sys
import math

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP551 = os.path.join(
    HERE, '..', '..',
    '第5章 朗之万动力学与得分函数', '第五章配套实验', '实验5.5-1')
EXP551 = os.path.abspath(EXP551)
sys.path.insert(0, EXP551)

from pnp_ula_common import build_problem, run_pnp_ula, iat_geyer  # noqa: E402

print('[1] build_problem #1 ...')
prob1 = build_problem(device=torch.device('cpu'))
y1 = prob1.y.detach().cpu().numpy().copy()
x1 = prob1.x.detach().cpu().numpy()
print('    image_shape =', prob1.image_shape, 'sigma =', float(prob1.sigma))

print('[2] build_problem #2（同种子，y 必须逐元素相等）...')
prob2 = build_problem(device=torch.device('cpu'))
y2 = prob2.y.detach().cpu().numpy()
assert np.array_equal(y1, y2), '同种子两次构建的 y 不一致！'
print('    OK: y 可复现')

print('[3] 主链模式 maxit=40, burnin=8, n_samples=10 ...')
res = run_pnp_ula(prob1, maxit=40, burnin=8, n_samples=10, seed=2026,
                  want_fourier=True, want_metric_curves=True,
                  progress=False)
assert res['burnin'] == 8 and res['half_mark'] == 24, res['half_mark']
assert res['thinning_step'] == 4
assert len(res['trace_fid']) == 31, len(res['trace_fid'])
assert len(res['psnr_values']) == 31
assert len(res['nrmse_values']) == 31 and len(res['ssim_values']) == 31
assert res['mc_samples'].shape[1:] == x1.shape
assert res['mc_samples'].shape[0] >= 1
assert res['post_mean'].shape == x1.shape
assert res['post_var'].shape == x1.shape
assert res['post_mean_h1'].shape == x1.shape
assert res['post_var_h2'].shape == x1.shape
assert res['trace_extra'].shape == (0,)  # 未给 trace_fn
iat, ess = iat_geyer(res['trace_fid'])
assert ess > 0 and math.isfinite(iat)
assert math.isfinite(res['half_sigma_corr'])
assert res['n_post'] == 31
print(f"    OK: mc={res['mc_samples'].shape}, psnr_last={res['psnr_last']:.2f}, "
      f"gain={res['last_quarter_gain']:+.3f}, iat={iat:.2f}, ess={ess:.1f}, "
      f"half_corr={res['half_sigma_corr']:.3f}")

print('[4] 扫描模式 maxit=40, burnin=10, 无metric曲线, trace_every=3, trace_fn=像素均值 ...')
res2 = run_pnp_ula(prob2, maxit=40, burnin=10, n_samples=12, seed=7,
                   eps=prob2.eps0 * 2.0, delta_frac=0.5,
                   want_fourier=False, want_metric_curves=False,
                   trace_every=3, progress=False,
                   trace_fn=lambda X: float(X.mean()))
post_steps = 40 - 1 - 10  # i = burnin+1 .. maxit-1
assert len(res2['trace_fid']) == post_steps, len(res2['trace_fid'])
assert len(res2['trace_extra']) == post_steps, len(res2['trace_extra'])
assert len(res2['trace_psnr_points']) == post_steps // 3
assert res2['absfouriercoeff'] is None
assert math.isfinite(res2['last_quarter_gain'])
assert res2['eps'] == prob2.eps0 * 2.0
print(f"    OK: trace_extra={res2['trace_extra'].shape}, "
      f"psnr_points={len(res2['trace_psnr_points'])}, gain={res2['last_quarter_gain']:+.3f}, "
      f"delta={res2['delta']:.3e}")

print('\nSMOKE_COMMON_OK')
