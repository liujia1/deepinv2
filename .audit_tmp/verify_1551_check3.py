# -*- coding: utf-8 -*-
import torch, os, sys
sys.stdout.reconfigure(encoding='utf-8')
D = r'd:\works\deepinv\第15章 扩散模型的架构实践：UNet → DiT\第十五章 配套实验\实验15.5-1'
for n in ['adam_const', 'adamw_const']:
    ck = torch.load(os.path.join(D, f'{n}_checkpoint.pth'), map_location='cpu', weights_only=False)
    h = ck['loss_history']
    print(n)
    for i in range(9, 20):
        print(f"  E{i+1}: {h[i]:.6f}")
    mn = min(range(10, 20), key=lambda i: h[i])
    print(f"  E10后最低点: E{mn+1}={h[mn]:.6f}, E20={h[-1]:.6f}")
    # 全程单调性
    inc = [i+2 for i in range(len(h)-1) if h[i+1] > h[i]]
    print(f"  上升的轮次(相对上一轮): {inc}")
