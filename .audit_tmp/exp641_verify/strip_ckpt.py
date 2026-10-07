# -*- coding: utf-8 -*-
"""剥离成品 checkpoint 中的 optimizer_state_dict (项目规范: is_final=True 不存优化器状态)"""
import torch, os, hashlib

P = r'd:\works\deepinv\第6章 得分匹配：从去噪中学习得分\第六章 配套实现\实验6.4-1\ssm_checkpoint.pth'
ck = torch.load(P, map_location='cpu', weights_only=False)
print('before keys:', sorted(ck.keys()))
assert ck.get('is_final', False), '非成品检查点, 中止'
ck.pop('optimizer_state_dict', None)
torch.save(ck, P)
ck2 = torch.load(P, map_location='cpu', weights_only=False)
print('after keys :', sorted(ck2.keys()))
print('epoch:', ck2['epoch'], 'loss:', ck2['loss'], 'n_losses:', len(ck2['train_losses']),
      'model_equal:', torch.equal(ck['model_state_dict']['net.0.weight'], ck2['model_state_dict']['net.0.weight']))
print('size:', os.path.getsize(P))
