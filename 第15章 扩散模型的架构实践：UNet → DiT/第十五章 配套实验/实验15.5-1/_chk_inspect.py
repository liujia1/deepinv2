# -*- coding: utf-8 -*-
"""全量审计: 4个extend checkpoint vs results_summary_extend100.json vs 图"""
import torch, os, json
d = os.path.dirname(os.path.abspath(__file__))
names = ['adam_const', 'adamw_const', 'adam_cosine', 'adamw_cosine']
hist_t, hist_v = {}, {}
for n in names:
    ck = torch.load(os.path.join(d, f'extend_{n}_checkpoint.pth'), map_location='cpu', weights_only=False)
    th, vh = ck['loss_history'], ck['val_history']
    hist_t[n], hist_v[n] = th, vh
    print(f"=== {n} === epoch={ck['epoch']} is_final={ck['is_final']} len={len(th)}/{len(vh)}")
    print(f"  E1:  Train={th[0]:.6f} Val={vh[0]:.6f}")
    print(f"  E14: Train={th[13]:.6f} Val={vh[13]:.6f}")
    print(f"  E20: Train={th[19]:.6f} Val={vh[19]:.6f}")
    print(f"  E50: Train={th[49]:.6f} Val={vh[49]:.6f}")
    print(f"  E100:Train={th[-1]:.6f} Val={vh[-1]:.6f}")
    print(f"  最后5轮Val: {[round(x,6) for x in vh[-5:]]}")

js = json.load(open(os.path.join(d, 'results_summary_extend100.json'), encoding='utf-8'))
print("\n=== json一致性 ===")
ok = True
for n in names:
    ft, fv = js['final_train_losses'][n], js['final_val_losses'][n]
    dt, dv = abs(hist_t[n][-1]-ft), abs(hist_v[n][-1]-fv)
    flag = 'OK' if max(dt, dv) < 1e-6 else f'MISMATCH dt={dt:.2e} dv={dv:.2e}'
    ok &= max(dt, dv) < 1e-6
    print(f"  {n}: {flag}")
print("json与checkpoint:", "完全一致" if ok else "存在不一致!")

print("\n=== 重算cosine-constant差值与交叉点 ===")
for opt in ['adam', 'adamw']:
    diff = [hist_v[f'{opt}_cosine'][i] - hist_v[f'{opt}_const'][i] for i in range(100)]
    negs = [i+1 for i, x in enumerate(diff) if x < 0]
    first = negs[0] if negs else None
    print(f"  {opt}: 首次穿零epoch={first}, 负值轮数={len(negs)}, 负值区间示例={negs[:8]}...{negs[-3:] if negs else ''}")
    print(f"    diff@E10={diff[9]:+.6f} E14={diff[13]:+.6f} E20={diff[19]:+.6f} E30={diff[29]:+.6f} E50={diff[49]:+.6f} E100={diff[-1]:+.6f}")
    js_ce = js['crossover_epoch_cosine_minus_constant'][opt.capitalize()]
    print(f"    json记录交叉={js_ce} -> {'一致' if js_ce == first else '不一致!'}")

print("\n=== 排名复算 ===")
rank = sorted(names, key=lambda n: hist_v[n][-1])
print("  val排名:", rank)
print("  json排名:", [k.replace(' + ', '_').replace('Warmup+Cosine', '_cosine').lower().replace('adamw + constant','adamw_const') for k in []] or "对照:", )
label2name = {js['val_ranking_best_to_worst'][i]: names[i] if False else None for i in range(4)}
mapping = {'Adam + Constant':'adam_const','AdamW + Constant':'adamw_const','AdamW + Warmup+Cosine':'adamw_cosine','Adam + Warmup+Cosine':'adam_cosine'}
json_rank = [mapping[k] for k in js['val_ranking_best_to_worst']]
print("  json映射:", json_rank, "->", "一致" if json_rank == rank else "不一致!")
