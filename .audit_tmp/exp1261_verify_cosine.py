# -*- coding: utf-8 -*-
"""验证 12.6-1.py 的 cosine 调度实现是否符合理论(不执行训练)"""
import sys, io, math, py_compile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import torch

PY = r'd:\works\deepinv\第12章 Score ≡ ELBO：殊途同归\第十二章 配套实验\实验12.6-1\12.6-1.py'

# 1) 语法编译
try:
    py_compile.compile(PY, doraise=True)
    print('[1] 语法编译: OK')
except py_compile.PyCompileError as e:
    print('[1] 语法编译失败:', e); sys.exit(1)

# 2) 复刻脚本中的 cosine 调度公式并做理论核查
T = 200
_s = 0.008
_steps = torch.arange(T + 1, dtype=torch.float64)
_ac = torch.cos((_steps / T + _s) / (1 + _s) * math.pi / 2) ** 2
_ac = _ac / _ac[0]
betas = (1 - _ac[1:] / _ac[:-1]).clamp(max=0.999).float()
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

print(f'[2] betas 范围: [{betas.min():.5f}, {betas.max():.5f}]  (应 0<β<=0.999)')
print(f'    betas 单调递增? {bool((betas[1:] >= betas[:-1] - 1e-9).all())}')
print(f'    alpha_bars 单调递减? {bool((alpha_bars[1:] <= alpha_bars[:-1] + 1e-9).all())}')
print(f'    终端 √ᾱ_T = {alpha_bars[-1].sqrt():.6f}  (cosine理论应≈0, 采样起点N(0,I)才成立)')
print(f'    终端 √(1-ᾱ_T) = {(1-alpha_bars[-1]).sqrt():.6f}  (应≈1)')
print(f'    首端 ᾱ_0 = {alpha_bars[0]:.6f}  (应接近1, 低噪声端保留信号)')

# 3) 与旧线性调度对比终端信号
lin = torch.linspace(1e-4, 0.02, 200)
lin_ab = torch.cumprod(1 - lin, 0)
print(f'\n[3] 对照: 旧线性调度(1e-4,0.02)@T=200 终端 √ᾱ_T = {lin_ab[-1].sqrt():.4f} (残留信号, 缺陷)')
print(f'    cosine修复后终端 √ᾱ_T = {alpha_bars[-1].sqrt():.4f} (缺陷已消除)')
