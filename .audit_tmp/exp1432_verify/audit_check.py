# -*- coding: utf-8 -*-
"""14.3-2 审计辅助脚本：
1) checkpoint 内容审计（epoch 数、losses 尾部值、与 JSON 对照）
2) 验证 κ(扩散条件路径弯曲) 是否为 VP-cosine 调度的固有属性：
   即使按标准做法用 z~N(0,I)（而非 source 混合分布），路径是否仍显著弯曲
"""
import sys, io, os, json
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))

# ---- 1) checkpoint 审计 ----
for name in ['cfm_independent_final.pth', 'cfm_ot_final.pth']:
    ck = torch.load(os.path.join(BASE, name), map_location='cpu', weights_only=False)
    losses = ck.get('losses', [])
    print(f"[{name}] keys={list(ck.keys())}")
    print(f"  losses 条数={len(losses)}, 首={losses[0]:.6f}, 末={losses[-1]:.6f}")
    sd = ck['model_state_dict']
    n_params = sum(v.numel() for v in sd.values())
    print(f"  参数量={n_params}")

ck_ind = torch.load(os.path.join(BASE, 'cfm_independent_checkpoint.pth'), map_location='cpu', weights_only=False)
ck_ot = torch.load(os.path.join(BASE, 'cfm_ot_checkpoint.pth'), map_location='cpu', weights_only=False)
print(f"[中间ckpt] ind epoch={ck_ind.get('epoch')}, ot epoch={ck_ot.get('epoch')}")

# ---- 2) κ 固有属性检验：真实高斯噪声 vs source 混合分布 ----
def sample_source(n, rng):
    mode = rng.random(n) < 0.5
    x = np.where(mode, -2, 2) + 0.3 * rng.standard_normal(n)
    y = np.where(mode, 2, -2) + 0.3 * rng.standard_normal(n)
    return np.stack([x, y], axis=1)

def sample_target(n, rng):
    mode = rng.random(n) < 0.5
    x = np.where(mode, 2, -2) + 0.3 * rng.standard_normal(n)
    y = np.where(mode, 2, -2) + 0.3 * rng.standard_normal(n)
    return np.stack([x, y], axis=1)

def kappa_path(z, x0, t_vals):
    ab = np.cos(np.pi / 2 * (1 - t_vals)) ** 2
    ab = np.maximum(ab, 1e-10)
    path = np.sqrt(ab)[:, None] * x0[None, :] + np.sqrt(1 - ab)[:, None] * z[None, :]
    direct = np.linalg.norm(path[-1] - path[0])
    actual = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    return 1.0 - direct / (actual + 1e-12)

rng = np.random.default_rng(0)
n = 2000
t_vals = np.linspace(0, 1, 101)
tg = sample_target(n, rng)

# 情形A：本实验做法，z 来自 source 混合分布（随机配对）
zs = sample_source(n, rng)
kA = np.mean([kappa_path(zs[i], tg[i], t_vals) for i in range(500)])
# 情形B：标准扩散做法，z ~ N(0,I)
zg = rng.standard_normal((n, 2))
kB = np.mean([kappa_path(zg[i], tg[i], t_vals) for i in range(500)])
# 情形C：对照——线性插值路径 κ 应为 ~0
k_lin = np.mean([1.0 - np.linalg.norm(tg[i] - zs[i]) /
                 (np.linalg.norm(tg[i] - zs[i]) + 1e-12) if False else
                 (lambda p: 1.0 - np.linalg.norm(p[-1]-p[0]) / (np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))+1e-12))(
                     np.outer(1-t_vals, zs[i]) + np.outer(t_vals, tg[i]))
                 for i in range(500)])
print(f"κ(扩散条件路径, z=source混合[本实验]) = {kA:.4f}")
print(f"κ(扩散条件路径, z~N(0,I)[标准扩散] ) = {kB:.4f}")
print(f"κ(线性插值路径, 对照)               = {k_lin:.6f}")
