# -*- coding: utf-8 -*-
"""复核10.1-1: 检查checkpoint元数据与history"""
import torch, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

base = r"d:\works\deepinv\第10章 层级VAE与扩散的变分推导\第十章 配套实验\实验10.1-1"
for name in ['hvae_mnist_checkpoint.pth', 'svae_mnist_checkpoint.pth']:
    ck = torch.load(base + "\\" + name, map_location='cpu', weights_only=False)
    print("=" * 60)
    print(name)
    print("  keys:", sorted(ck.keys()))
    print("  epoch:", ck.get('epoch'), " is_final:", ck.get('is_final'))
    print("  d1,d2,beta:", ck.get('d1'), ck.get('d2'), ck.get('beta'))
    print("  d_z,beta:", ck.get('d_z'), ck.get('beta'))
    print("  final loss:", ck.get('loss'))
    h = ck.get('history', {})
    for k, v in h.items():
        print(f"  history[{k}] len={len(v)}")
        if len(v) <= 30:
            print("   ", [round(x, 3) for x in v])
    print("  参数shapes示例:", {k: tuple(v.shape) for k, v in list(ck.get('enc1_state_dict', ck.get('encoder_state_dict', {})).items())[:4]})
