# -*- coding: utf-8 -*-
"""模拟掉线场景: 把 adam_const 的成品checkpoint篡改为"中断在第1轮"状态"""
import torch
p = 'extend_adam_const_checkpoint.pth'
ck = torch.load(p, map_location='cpu', weights_only=False)
print("篡改前: epoch =", ck['epoch'], ", is_final =", ck['is_final'], ", len(loss_history) =", len(ck['loss_history']))
ck['is_final'] = False
ck['loss_history'] = ck['loss_history'][:1]   # 只保留第1轮, 模拟当时中断
ck['val_history'] = ck['val_history'][:1]
# 中断态必须带optimizer/scheduler状态; cosine调度器状态对adam_const为None, 补占位以符合中断格式
ck['optimizer_state_dict'] = None
ck['scheduler_state_dict'] = None
torch.save(ck, p)
print("已篡改为中断态 (epoch=0, is_final=False)")
