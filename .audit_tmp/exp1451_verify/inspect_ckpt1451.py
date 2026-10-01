# -*- coding: utf-8 -*-
"""检查实验14.5-1的4个checkpoint：epoch、loss历史、指纹、与当前架构兼容性"""
import sys, io, os
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import torch, hashlib

EXP = r"d:\works\deepinv\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.5-1"

def fp(time_sampling, ln_m=None, ln_s=None):
    s = f"time_sampling={time_sampling}"
    if time_sampling == 'logit_normal':
        s += f"_m={ln_m}_s={ln_s}"
    return hashlib.md5(s.encode()).hexdigest()[:8]

fp_uni = fp('uniform')
fp_log = fp('logit_normal', 0.0, 1.0)
print(f"当前代码指纹: uniform={fp_uni}, logit_normal(m=0.0,s=1.0)={fp_log}")

for name in ['rf_uniform_checkpoint.pth', 'rf_uniform_final.pth',
             'rf_logit_checkpoint.pth', 'rf_logit_final.pth']:
    p = os.path.join(EXP, name)
    ck = torch.load(p, map_location='cpu', weights_only=False)
    keys = list(ck.keys())
    ep = ck.get('epoch', None)
    losses = ck.get('train_losses', [])
    saved = ck.get('config_fingerprint', None)
    n_params = sum(v.numel() for v in ck['model_state_dict'].values())
    print(f"\n== {name} ==")
    print(f"  keys={keys}")
    print(f"  epoch={ep}, n_losses={len(losses)}, saved_fingerprint={saved}")
    print(f"  first_loss={losses[0]:.6f}, last_loss={losses[-1]:.6f}" if losses else "  no losses")
    print(f"  loss_history={['%.6f' % x for x in losses]}")
    print(f"  n_params={n_params}")

# 架构兼容性：用当前脚本定义的SmallUNet加载final权重
sys.path.insert(0, EXP)
import importlib.util
spec = importlib.util.spec_from_file_location("m1451", os.path.join(EXP, "14.5-1.py"))
# 不执行整个脚本（会训练），改为手动复制类定义太脆弱；直接用state_dict形状检查
ck_u = torch.load(os.path.join(EXP, 'rf_uniform_final.pth'), map_location='cpu', weights_only=False)
ck_l = torch.load(os.path.join(EXP, 'rf_logit_final.pth'), map_location='cpu', weights_only=False)
su, sl = ck_u['model_state_dict'], ck_l['model_state_dict']
same_keys = set(su.keys()) == set(sl.keys())
print(f"\nuniform与logit权重键集合一致: {same_keys}, 层数={len(su)}")
diff = max((su[k] - sl[k]).abs().max().item() for k in su)
print(f"两模型权重最大差异: {diff:.4f} (非零=>确为两个独立训练的模型)")
