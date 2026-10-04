# -*- coding: utf-8 -*-
"""实验15.5-1 核对: checkpoint vs results_summary.json vs results_summary_extend100.json vs 讲义数值"""
import torch, os, json, sys
sys.stdout.reconfigure(encoding='utf-8')

D = r'd:\works\deepinv\第15章 扩散模型的架构实践：UNet → DiT\第十五章 配套实验\实验15.5-1'
names = ['adam_const', 'adamw_const', 'adam_cosine', 'adamw_cosine']

print('=' * 30, '原20轮实验', '=' * 30)
hist = {}
for n in names:
    ck = torch.load(os.path.join(D, f'{n}_checkpoint.pth'), map_location='cpu', weights_only=False)
    h = ck['loss_history']
    hist[n] = h
    print(f"{n}: epoch={ck['epoch']} is_final={ck['is_final']} len={len(h)}")
    print(f"  E1={h[0]:.6f} E10={h[9]:.6f} E20={h[-1]:.6f}")

js = json.load(open(os.path.join(D, 'results_summary.json'), encoding='utf-8'))
ok = all(abs(hist[n][-1] - js['final_losses'][n]) < 1e-6 for n in names)
print('json final_losses 与 checkpoint 末轮:', '一致' if ok else '不一致!')
ranking = sorted(names, key=lambda n: hist[n][-1])
name2label = {
    'adam_const': 'Adam + Constant', 'adamw_const': 'AdamW + Constant',
    'adam_cosine': 'Adam + Warmup+Cosine', 'adamw_cosine': 'AdamW + Warmup+Cosine'}
print('重算排名:', [name2label[r] for r in ranking], '==json:',
      [name2label[r] for r in ranking] == js['ranking_best_to_worst'])

# 讲义百分比复算
a, aw, ac, awc = (hist[n][-1] for n in names)
print(f"  Adam比AdamW高(Constant): {(a-aw)/aw*100:.1f}%  (讲义21%)")
print(f"  Cosine比Constant高(AdamW): {(hist['adamw_cosine'][-1]-aw)/aw*100:.1f}%  (讲义8.9%)")
print(f"  Cosine比Constant低(Adam): {(a-hist['adam_cosine'][-1])/a*100:.1f}%  (讲义10.8%)")
print(f"  cosine下Adam优于AdamW: {(hist['adamw_cosine'][-1]-hist['adam_cosine'][-1])/hist['adamw_cosine'][-1]*100:.2f}%  (讲义约0.5%)")

print()
print('=' * 30, '延长100轮实验', '=' * 30)
hj = json.load(open(os.path.join(D, 'results_summary_extend100.json'), encoding='utf-8'))
ht, hv = {}, {}
for n in names:
    ck = torch.load(os.path.join(D, f'extend_{n}_checkpoint.pth'), map_location='cpu', weights_only=False)
    ht[n], hv[n] = ck['loss_history'], ck['val_history']
    print(f"{n}: epoch={ck['epoch']} is_final={ck['is_final']} len={len(ht[n])}/{len(hv[n])}")
    print(f"  E1 train={ht[n][0]:.6f}  E100 train={ht[n][-1]:.6f} val={hv[n][-1]:.6f}")
    print(f"  最后5轮val: {[round(x,6) for x in hv[n][-5:]]}")

ok = all(abs(ht[n][-1] - hj['final_train_losses'][n]) < 1e-6 and
         abs(hv[n][-1] - hj['final_val_losses'][n]) < 1e-6 for n in names)
print('extend json 与 checkpoint 末轮 train/val:', '一致' if ok else '不一致!')

# 与原20轮首轮对照 (讲义: 0.196194/0.195617/0.418601/0.417453 vs 0.1962/0.1956/0.4186/0.4175)
print('首轮一致性(extend vs 原):', {n: (round(ht[n][0], 6), round(hist[n][0], 6)) for n in names})

# 交叉点与负值统计
for opt, lab in [('adam', 'Adam'), ('adamw', 'AdamW')]:
    diff = [hv[f'{opt}_cosine'][i] - hv[f'{opt}_const'][i] for i in range(100)]
    negs = [i + 1 for i, x in enumerate(diff) if x < 0]
    print(f"{lab}: 首次转负epoch={negs[0] if negs else None} (json={hj['crossover_epoch_cosine_minus_constant'][lab]}) "
          f"负值轮数(<0)={len(negs)} (<=0轮数={sum(1 for x in diff if x <= 0)}) 最小差值={min(diff):+.6f}")
    if negs:
        # 打印负值区间
        runs, s = [], negs[0]
        for x in negs[1:] + [None]:
            if x is None or x != s + 1 or x != s + 1:
                pass
        # 简化: 打印前12个负值epoch
        print(f"   负值epoch示例: {negs[:12]} ... {negs[-3:]}")

# val排名复算
rank_v = sorted(names, key=lambda n: hv[n][-1])
print('val排名重算:', [name2label[r] for r in rank_v], '==json:',
      [name2label[r] for r in rank_v] == hj['val_ranking_best_to_worst'])

# 讲义: Adam比同调度AdamW高23%~29%
for sched in ['const', 'cosine']:
    imp = (hv[f'adam_{sched}'][-1] - hv[f'adamw_{sched}'][-1]) / hv[f'adamw_{sched}'][-1] * 100
    print(f"  Adam vs AdamW ({sched}) val口径: +{imp:.1f}%")

# verdict复算
cos_better = all(hv[f'{o}_cosine'][-1] < hv[f'{o}_const'][-1] for o in ['adam', 'adamw'])
print('verdict_cosine_overtakes_on_val 重算:', cos_better, '==json:',
      cos_better == hj['verdict_cosine_overtakes_on_val'])
