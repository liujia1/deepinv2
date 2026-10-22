# -*- coding: utf-8 -*-
"""验证 17.5-1.py 修改后：
   (1) G=1 时 moi_loss 退化为纯 MC：loss_cons==0, loss==loss_mc
   (2) G=1 时每步只调用 1 次模型前向（跳过了恒为 0 的一致性项）
   (3) G=4 时每步 4 次前向（主重建 + 4 个 s 中 1 个自项跳过？否：batch 混合时 4 个 s 都可能有交叉样本）
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import torch

BASE = r'h:\deepinv\deepinv2\第17章 自监督学习与等变架构\第十七章 配套实验\实验17.5-1'
src = open(os.path.join(BASE, '17.5-1.py'), encoding='utf-8').read()

# 只执行“配置+模型+数据+评估”定义段（到 Step 1 之前），不触发训练
# 注意必须从文件开头切：SAVE_DIR 定义在 if/else 缩进块内，中间切入会 IndentationError
seg_defs = src[: src.index("# Step 1: 单算子基线")]
seg_train = src[src.index("def moi_loss"): src.index("# 训练MOI G=1")]

ns = {'__name__': '__audit__', '__file__': os.path.join(BASE, '17.5-1.py')}
exec(compile(seg_defs, 'seg_defs', 'exec'), ns)
exec(compile(seg_train, 'seg_train', 'exec'), ns)
print('定义段执行完毕; device =', ns['device'])
print('数据集: train', len(ns['mnist_train']), 'test', len(ns['mnist_test']))

model = ns['SmallUNetMOI'](in_ch=2, out_ch=1, base=24).to(ns['device'])
pool1 = ns['test_mask'].unsqueeze(0)          # G=1
pool4 = torch.cat([ns['test_mask'].unsqueeze(0),
                   torch.rand(3, ns['IMG_SIZE'], ns['IMG_SIZE'], device=ns['device']) > 0.5], 0)

x = torch.rand(8, 1, ns['IMG_SIZE'], ns['IMG_SIZE'], device=ns['device'])
g1 = torch.zeros(8, dtype=torch.long, device=ns['device'])
g4 = torch.randint(0, 4, (8,), device=ns['device'])

calls = {'n': 0}
_orig_fwd = model.forward
def counting_forward(*a, **k):
    calls['n'] += 1
    return _orig_fwd(*a, **k)
model.forward = counting_forward

for G, pool, g in [(1, pool1, g1), (4, pool4, g4)]:
    calls['n'] = 0
    loss, mc, cons = ns['moi_loss'](model, x, pool, ns['SIGMA'], g)
    print(f'G={G}: loss={loss.item():.6f} mc={mc:.6f} cons={cons:.6g} 前向次数={calls["n"]}')
    if G == 1:
        assert cons == 0.0, 'G=1 的 cons 必须为 0'
        assert abs(loss.item() - mc) < 1e-9, 'G=1 的 loss 必须等于 loss_mc'
        assert calls['n'] == 1, 'G=1 只应前向 1 次（主重建）'
print('\n[PASS] G=1 退化为纯 MC，且无冗余前向')
print('[INFO] G=4 前向次数应为 5（主重建 + 4 个 s 均有交叉样本）')
