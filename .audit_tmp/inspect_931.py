# -*- coding: utf-8 -*-
import io, sys, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import torch
P = r'D:\works\deepinv\第9章 VAE与重参数化\第九章 配套实验\实验9.3-1\vae_mnist_checkpoint.pth'
ck = torch.load(P, map_location='cpu', weights_only=False)
print('keys:', list(ck.keys()))
print('epoch:', ck.get('epoch'), 'is_final:', ck.get('is_final'), 'loss:', ck.get('loss'))
for k, v in ck['encoder_state_dict'].items():
    print(' enc', k, tuple(v.shape))
for k, v in ck['decoder_state_dict'].items():
    print(' dec', k, tuple(v.shape))
h = ck.get('history', {})
for k, v in h.items():
    print(k, 'len', len(v))
print(json.dumps({k: v for k, v in h.items()}, indent=1, default=str))
print('opt keys:', list(ck['optimizer_state_dict'].keys()) if 'optimizer_state_dict' in ck else None)
print('file mtime:', os.path.getmtime(P))
