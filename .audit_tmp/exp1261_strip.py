# -*- coding: utf-8 -*-
"""剥离成品checkpoint中的优化器状态（符合项目规范：成品不存优化器状态）"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import torch

BASE = r'd:\works\deepinv\第12章 Score ≡ ELBO：殊途同归\第十二章 配套实验\实验12.6-1'

for name in ['dsm_model_checkpoint.pth', 'vlb_model_checkpoint.pth']:
    p = os.path.join(BASE, name)
    before = os.path.getsize(p)
    ck = torch.load(p, map_location='cpu', weights_only=False)
    assert ck.get('is_final', False), f'{name} 不是 final checkpoint, 跳过'
    had_opt = 'optimizer_state_dict' in ck
    ck.pop('optimizer_state_dict', None)
    torch.save(ck, p)
    after = os.path.getsize(p)
    print(f'{name}: 含优化器={had_opt}  大小 {before} -> {after} 字节 (省 {before-after})')
    # 复核关键字段仍在
    ck2 = torch.load(p, map_location='cpu', weights_only=False)
    print(f'  keys={list(ck2.keys())}  epoch={ck2["epoch"]}  loss={ck2["loss"]:.6f}  len(train_losses)={len(ck2["train_losses"])}')
