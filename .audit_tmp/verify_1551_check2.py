# -*- coding: utf-8 -*-
import torch, os, json, sys
sys.stdout.reconfigure(encoding='utf-8')
D = r'd:\works\deepinv\第15章 扩散模型的架构实践：UNet → DiT\第十五章 配套实验\实验15.5-1'
names = ['adam_const', 'adamw_const', 'adam_cosine', 'adamw_cosine']
hv = {}
for n in names:
    ck = torch.load(os.path.join(D, f'extend_{n}_checkpoint.pth'), map_location='cpu', weights_only=False)
    hv[n] = ck['val_history']
diff = [hv['adamw_cosine'][i] - hv['adamw_const'][i] for i in range(100)]
print('AdamW cosine-const 差值 E14-E33:')
for i in range(13, 33):
    print(f"  E{i+1}: {diff[i]:+.6f}")
print('AdamW 全部负值轮:', [(i+1, round(diff[i], 6)) for i in range(100) if diff[i] < 0])
# 时间戳
import datetime
for f in names:
    pass
for f in sorted(os.listdir(D)):
    if f.endswith(('.png', '.json', '.pth', '.py')):
        t = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(D, f)))
        print(f"{t:%Y-%m-%d %H:%M:%S}  {f}")
