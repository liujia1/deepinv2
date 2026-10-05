# -*- coding: utf-8 -*-
"""检查实验12.6-1的checkpoint元数据与损失曲线"""
import torch, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = r'd:\works\deepinv\第12章 Score ≡ ELBO：殊途同归\第十二章 配套实验\实验12.6-1'

for name in ['dsm_model_checkpoint.pth', 'vlb_model_checkpoint.pth']:
    p = f'{BASE}\\{name}'
    ck = torch.load(p, map_location='cpu', weights_only=False)
    print('='*60)
    print(name)
    print('  keys:', list(ck.keys()))
    print('  epoch:', ck.get('epoch'))
    print('  loss:', ck.get('loss'))
    print('  is_final:', ck.get('is_final'))
    tl = ck.get('train_losses')
    if tl is not None:
        print(f'  train_losses 长度: {len(tl)}')
        print(f'  前5个: {[round(v,6) for v in tl[:5]]}')
        print(f'  后5个: {[round(v,6) for v in tl[-5:]]}')
        print(f'  最终: {tl[-1]:.6f}')
    print('  参数张量数:', len(ck['model_state_dict']))
    if 'optimizer_state_dict' in ck:
        osd = ck['optimizer_state_dict']
        n_state = sum(1 for st in osd.get('state', {}).values() if 'exp_avg' in st)
        print(f'  optimizer 含 exp_avg 的张量对数: {n_state}')
