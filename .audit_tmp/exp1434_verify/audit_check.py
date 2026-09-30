# -*- coding: utf-8 -*-
"""核查14.3-4的两处"免责"说法:
  A. 5步时 σ=0.5 与 σ=0 的差(0.329 vs 0.374)是否真在"采样噪声边缘" —— 换多个评估种子重复测量
  B. "最优档依赖模型误差, 只能由实测给出" —— 用解析精确 v*, s* 跑同样的5步欧拉-丸山,
     若精确模型下 σ=0 单调最优, 则 σ=0.5 占优确实来自模型误差(固有属性), 反之则存疑。
"""
import sys, io, importlib.util
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("exp1434", r"d:\works\deepinv\.audit_tmp\exp1434_verify\14.3-4.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)   # 会重跑训练(确定性, 与原图/JSON一致)

print("\n" + "=" * 60)
print("核查A: 5步 σ=0 vs σ=0.5 的差距是否在采样噪声内(换5个评估种子)")
rows = []
for sd in [11, 22, 33, 44, 55]:
    ref = m.sample_reference(2000, seed=sd)
    d0 = m.sliced_w2(m.sample_sde_sigma(m.model_v, 0.0, 5, 2000, seed=sd), ref, seed=sd)
    d5 = m.sliced_w2(m.sample_sde_sigma(m.model_v, 0.5, 5, 2000, seed=sd), ref, seed=sd)
    rows.append((sd, d0, d5))
    print(f"  seed={sd}: SW2(σ=0)={d0:.4f}  SW2(σ=0.5)={d5:.4f}  差(0-0.5)={d0-d5:+.4f}")
diffs = [a - b for _, a, b in rows]
print(f"  差值均值 {np.mean(diffs):+.4f} ± 标准误 {np.std(diffs, ddof=1)/np.sqrt(len(diffs)):.4f}")

print("\n核查B: 精确解析模型(零模型误差)下的5步 σ 扫描")
def sample_exact(sigma, n_steps, n_samples, seed):
    """欧拉-丸山, v/s 直接用解析真值(无网络)"""
    g = np.random.default_rng(seed)
    x = g.standard_normal((n_samples, 2))
    h = 1.0 / n_steps
    for n in range(n_steps):
        t_n = n / n_steps
        v = m.mixture_velocity(x, t_n)
        s = m.mixture_score(x, t_n)
        x = x + h * (v + 0.5 * sigma ** 2 * s) + np.sqrt(h) * sigma * g.standard_normal(x.shape)
    return x

ref = m.sample_reference(2000, seed=42)
for ns in [5, 100]:
    line = []
    for sigma in [0.0, 0.5, 1.0, 2.0, 4.0]:
        xs = sample_exact(sigma, ns, 2000, seed=2028 + ns)
        line.append(m.sliced_w2(xs, ref))
    print(f"  精确模型 {ns}步: " + "  ".join(f"σ={s}:{d:.4f}" for s, d in zip([0, 0.5, 1, 2, 4], line)))
print("  (对照) 训练模型 5步: σ=0:0.374  σ=0.5:0.329  σ=1:0.344  σ=2:0.566  σ=4:274.95")
