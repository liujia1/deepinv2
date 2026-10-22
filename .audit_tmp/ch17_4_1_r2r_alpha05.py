# -*- coding: utf-8 -*-
"""按 p2_14 修复：主实验 R2R 改用 α=0.5（敏感性扫描最优），重训 30 epochs，
   保存 ckpt_R2R.pt（与主脚本格式一致，主脚本将来直接加载），并写出新评估值。"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')
import torch

BASE = r'h:\deepinv\deepinv2\第17章 自监督学习与等变架构\第十七章 配套实验\实验17.4-1'
src = open(os.path.join(BASE, '17.4-1.py'), encoding='utf-8').read()
seg_defs = src[: src.index("# Step 1: SURE原理验证")]
seg_r2r = src[src.index("def r2r_loss"): src.index("# 训练R2R模型")]

ns = {'__name__': '__audit__', '__file__': os.path.join(BASE, '17.4-1.py')}
exec(compile(seg_defs, 'seg_defs', 'exec'), ns)
exec(compile(seg_r2r, 'seg_r2r', 'exec'), ns)

device, LR, N_EPOCHS, SIGMA = ns['device'], ns['LR'], ns['N_EPOCHS'], ns['SIGMA']
train_loader, test_loader, EVAL_SEED = ns['train_loader'], ns['test_loader'], ns['EVAL_SEED']
r2r_loss, add_noise = ns['r2r_loss'], ns['add_noise']
ALPHA = 0.5
print(f'重训 R2R：alpha={ALPHA}, epochs={N_EPOCHS}, device={device}')

model = ns['SmallUNet']().to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
t0 = time.time()
losses = []
for epoch in range(N_EPOCHS):
    model.train()
    s, n = 0.0, 0
    for batch_x, _ in train_loader:
        batch_x = batch_x.to(device)
        y = add_noise(batch_x, SIGMA)
        opt.zero_grad()
        loss = r2r_loss(model, y, SIGMA, alpha=ALPHA)
        loss.backward()
        opt.step()
        s += loss.item(); n += 1
    losses.append(s / n)
    print(f'Epoch {epoch+1}/{N_EPOCHS} loss={losses[-1]:.6f} '
          f'({(time.time()-t0)/60:.1f} min)', flush=True)

ckpt = os.path.join(BASE, 'ckpt_R2R.pt')
ns['save_checkpoint'](ckpt, model, opt, N_EPOCHS - 1, losses, is_final=True,
                      extra={'loss': losses[-1]}, log_path=False)
print('已保存 ckpt_R2R.pt')

p = ns['evaluate_psnr'](model, test_loader, seed=EVAL_SEED)
s = ns['evaluate_ssim'](model, test_loader, seed=EVAL_SEED)
out = {'alpha': ALPHA, 'psnr': float(p), 'ssim': float(s),
       'final_loss': float(losses[-1]), 'epochs': N_EPOCHS,
       'minutes': (time.time() - t0) / 60}
print(json.dumps(out, ensure_ascii=False, indent=2))
with open(os.path.join(BASE, '_r2r_alpha05_result.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
