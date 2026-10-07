# -*- coding: utf-8 -*-
import io, sys, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import torch

for name, path in [
    ('9.3-1 vae', r'D:\works\deepinv\第9章 VAE与重参数化\第九章 配套实验\实验9.3-1\vae_mnist_checkpoint.pth'),
    ('9.1-1 vae', r'D:\works\deepinv\第9章 VAE与重参数化\第九章 配套实验\实验9.1-1\vae_mnist_checkpoint.pth'),
    ('9.1-1 ae', r'D:\works\deepinv\第9章 VAE与重参数化\第九章 配套实验\实验9.1-1\ae_mnist_checkpoint.pth'),
    ('9.5-1 beta0', r'D:\works\deepinv\第9章 VAE与重参数化\第九章 配套实验\实验9.5-1\vae_beta0_checkpoint.pth'),
    ('9.5-1 beta1', r'D:\works\deepinv\第9章 VAE与重参数化\第九章 配套实验\实验9.5-1\vae_beta1_checkpoint.pth'),
    ('9.5-1 beta4', r'D:\works\deepinv\第9章 VAE与重参数化\第九章 配套实验\实验9.5-1\vae_beta4_checkpoint.pth'),
]:
    try:
        ck = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(name, 'ERR', e); continue
    print('=' * 70)
    print(name, 'keys=', list(ck.keys()))
    print('  epoch', ck.get('epoch'), 'is_final', ck.get('is_final'), 'loss', ck.get('loss'),
          'beta', ck.get('beta'), 'latent', ck.get('latent_dim'))
    h = ck.get('history') or ck.get('train_history') or {}
    if isinstance(h, dict):
        for k, v in h.items():
            if isinstance(v, (list, tuple)):
                s = ['%.3f' % x if isinstance(x, float) else str(x) for x in v]
                print('  %-14s len=%d  first3=%s last3=%s' % (k, len(v), s[:3], s[-3:]))
            else:
                print('  %-14s = %s' % (k, v))
    # latent dim of encoder
    for k, v in (ck.get('encoder_state_dict') or {}).items():
        if 'fc_mu' in k or 'mu' in k or 'fc2' in k:
            print('  ', k, tuple(v.shape))
