# -*- coding: utf-8 -*-
"""验证：同一checkpoint，不同采样种子 → 质量波动（证明"变差"是种子运气，非模型退化）"""
import sys, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
EXP = r"D:\works\deepinv\第11章 扩散模型：变分视角\第十一章 配套实验\实验11.4-1"
OUT = r"D:\works\deepinv\.audit_tmp\exp1141_seedvar"; os.makedirs(OUT, exist_ok=True)
device = torch.device('cpu')

src = open(os.path.join(EXP, '11.4-1.py'), encoding='utf-8').read()
g = {'torch': torch, 'nn': nn, 'F': F, 'np': np}
exec(compile(src[src.index('class SinusoidalTimeEmbedding'):src.index('# ============================================================\n# DDPM采样算法')], 'c', 'exec'), g)
SmallUNet = g['SmallUNet']
T=200; betas=torch.linspace(1e-4,0.02,T); alphas=1-betas
ab=torch.cumprod(alphas,0); abp=torch.cat([torch.ones(1),ab[:-1]])
sab=torch.sqrt(ab); s1m=torch.sqrt(1-ab)
pv=betas*(1-abp)/(1-ab); sra=1/torch.sqrt(alphas); bos=betas/s1m

ck=torch.load(os.path.join(EXP,'ddpm_epsilon_checkpoint.pth'),map_location=device,weights_only=False)
m=SmallUNet(pred_type='epsilon'); m.load_state_dict(ck['model_state_dict']); m.eval()

@torch.no_grad()
def sample(seed):
    torch.manual_seed(seed)
    x=torch.randn(8,1,28,28)
    for ti in reversed(range(T)):
        t=torch.full((8,),ti,dtype=torch.long)
        mean=sra[ti]*(x-bos[ti]*m(x,t))
        x=mean if ti==0 else mean+torch.sqrt(pv[ti])*torch.randn_like(x)
    return x

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
seeds=[2024,42,123,777]
fig,axes=plt.subplots(len(seeds),8,figsize=(16,2.3*len(seeds)))
for r,sd in enumerate(seeds):
    xs=sample(sd)
    for i in range(8):
        axes[r,i].imshow(np.clip((xs[i,0].numpy()+1)/2,0,1),cmap='gray'); axes[r,i].axis('off')
    axes[r,0].set_ylabel(f'seed={sd}',fontsize=10)
plt.suptitle('同一epsilon-checkpoint，仅换采样种子 → 质量明显波动（证明差异来自种子而非模型）')
plt.tight_layout(); plt.savefig(os.path.join(OUT,'seedvar.png'),dpi=110); plt.close()
print('saved', os.path.join(OUT,'seedvar.png'))
