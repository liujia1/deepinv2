# -*- coding: utf-8 -*-
"""F: 后验可信区间覆盖率校准检验（68%/95%），并与 oracle 方差对照"""
import sys, math, json
import numpy as np
import torch
from skimage.data import camera
from skimage.transform import resize

EXP_DIR = r"d:\works\deepinv\第4章 MCMC与ULA算法\第四章 配套实验\实验4.4-1"
sys.path.insert(0, EXP_DIR)
from sampling_tools import blur_operators, chambolle_prox_TV

torch.set_num_threads(8)
dev = torch.device('cpu')

torch.manual_seed(42)
np.random.seed(42)
u = resize(camera().astype(np.float64), (128, 128), anti_aliasing=True)
x_true = torch.tensor(u, dtype=torch.float32)
A, AT, AAT = blur_operators([9, 9], (128, 128), "uniform", dev)
Ax = A(x_true)
sigma = torch.linalg.norm(Ax - Ax.mean()) / math.sqrt(128*128*10**2)
y = Ax + sigma*torch.randn_like(x_true)
sigma2 = sigma.item()**2
beta, lam = 0.3, 1.0
gamma = 0.98/(float(AAT)/sigma2 + 1.0/lam)

n_iter, burn = 12000, 4000
X = y.clone()
M = None; S = None; k = 0
samps = []
for i in range(n_iter):
    P = chambolle_prox_TV(X, dev, {'lambda': lam*beta, 'MaxIter': 10})
    gp = (X - P)/lam
    X = X - gamma*(AT(A(X)-y)/sigma2 + gp) + math.sqrt(2*gamma)*torch.randn_like(X)
    if i >= burn:
        if M is None:
            M = X.clone(); S = torch.zeros_like(X); k = 1
        else:
            k += 1; Mn = M + (X-M)/k; S += (X-M)*(X-Mn); M = Mn
        if i % 20 == 0:
            samps.append(X.clone())

Samp = torch.stack(samps)                       # (Ns,128,128)
V = S/(k-1); Std = torch.sqrt(torch.clamp(V, min=0))
err = (Samp - x_true[None]).abs()
cov68 = float((err <= Std[None]).float().mean())
cov95 = float((err <= 1.96*Std[None]).float().mean())
# 边际标准化误差的峰度尾部（近似 N(0,1) 时覆盖率应标称）
z = ((Samp - x_true[None])/Std[None]).cpu().numpy().ravel()
import numpy as np
q68 = float(np.mean(np.abs(z) <= 1.0)); q95 = float(np.mean(np.abs(z) <= 1.96))
psnr_mmse = 10*math.log10(255.0**2/torch.mean((M-x_true)**2).item())

# 样本自相关：像素(64,64)
tr = Samp[:, 64, 64].cpu().numpy()
tr -= tr.mean()
ac = np.correlate(tr, tr, 'full')[len(tr)-1:]/np.sum(tr**2)
tau = 1 + 2*np.sum(ac[1:50][ac[1:50] > 0])

out = dict(n_saved_samples=Samp.shape[0], n_chain_samples=k,
           cov68=cov68, cov95=cov95, z68=q68, z95=q95,
           std_mean=float(Std.mean()), psnr_mmse=psnr_mmse,
           per_sample_pixel_tau=float(tau))
print(json.dumps(out, indent=2))
with open(r'd:\works\deepinv\_review_scratch\chains_F.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
