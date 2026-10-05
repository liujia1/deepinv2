# -*- coding: utf-8 -*-
"""诊断: T=200 线性调度(1e-4,0.02)的终端SNR是否匹配DDPM采样起点 x_T~N(0,I)"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import torch

def schedule_stats(betas, label):
    alphas = 1 - betas
    ab = torch.cumprod(alphas, 0)
    T = len(betas)
    print(f"{label}: T={T}, beta[{betas[0]:.1e},{betas[-1]:.4f}]")
    print(f"  ᾱ_T = {ab[-1]:.4f}, √ᾱ_T = {ab[-1].sqrt():.4f}, √(1-ᾱ_T) = {(1-ab[-1]).sqrt():.4f}")
    print(f"  终端真实分布 x_T = {ab[-1].sqrt():.3f}·x₀ + {(1-ab[-1]).sqrt():.3f}·ε  vs 采样起点 N(0,I)")
    print()

# 当前实验设置
schedule_stats(torch.linspace(1e-4, 0.02, 200), "当前(讲义)")
# DDPM原文设置
schedule_stats(torch.linspace(1e-4, 0.02, 1000), "DDPM原文")
# 按 T 缩放 5 倍
schedule_stats(torch.linspace(5e-4, 0.1, 200), "T=200等比缩放")

# MNIST 数据在 [-1,1] 归一化后的大致 std
print("参考: 若√ᾱ_T≈0.36, 终端分布中信号分量std≈0.36×std(x₀), 与纯噪声起点差距显著")
