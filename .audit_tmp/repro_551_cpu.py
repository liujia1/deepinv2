# -*- coding: utf-8 -*-
"""CPU reproduction of experiment 5.5-1 PnP-ULA.
Only change vs original: load_model runs on CPU (no DataParallel/.cuda()).
RNG streams differ from CUDA so exact values won't match, but ballpark validates
that reported PSNR/SSIM are real, not fabricated.
"""
import os, sys, math, time
import numpy as np
import torch
import torch.nn as nn

EXP = r"d:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.5-1"
sys.path.insert(0, EXP)
from sampling_tools.blur_operators import blur_operators
from sampling_tools.measures import NRMSE, PSNR, SSIM
from sampling_tools.welford import welford
from sampling_tools.load_model import DnCNN
from PIL import Image

device = torch.device('cpu')
np.random.seed(42)
torch.manual_seed(42)

def load_model_cpu(path):
    net = DnCNN(channels=1, num_of_layers=17)
    sd = torch.load(path, map_location='cpu', weights_only=True)
    sd2 = { (k[7:] if k.startswith('module.') else k): v for k, v in sd.items() }
    net.load_state_dict(sd2)
    net.eval()
    return net

im = np.array(Image.open(os.path.join(EXP, "cman.png")))
print("image", im.shape)
x = torch.Tensor(im/255.)

kernel_len = [5, 5]; size = [im.shape[0], im.shape[1]]
A, AT, AAT_norm = blur_operators(kernel_len, size, "uniform", device)
y0 = A(x)
BSNRdb = 40
sigma = torch.linalg.matrix_norm(A(x)-torch.mean(A(x)), ord='fro')/math.sqrt(torch.numel(x)*10**(BSNRdb/10))
y = y0 + sigma * torch.randn_like(x)
print(f"sigma={sigma.item():.6f}  AAT_norm={AAT_norm.item():.6f}")

gradf = lambda xx: AT(A(xx)-y)/sigma**2
L_y = AAT_norm/(sigma**2)

model = load_model_cpu(os.path.join(EXP, "Pretrained_models", "RealSN_DnCNN_noise5.pth"))
denoise = lambda xx: (xx - model(xx[None][None].to(device))[0][0]).detach()

eps = (5/255)**2
alpha = 1; L_net = 1.0
max_lambd = 1.0/((2.0*alpha*L_net)/eps+4.0*L_y)
lambd = max_lambd*0.99
C_upper = torch.tensor(1.); C_lower = torch.tensor(0.)
projbox = lambda xx: torch.clamp(xx, min=C_lower, max=C_upper)

def Markov_kernel(X, delta):
    return projbox(X - delta*gradf(X) + alpha*delta/eps*(denoise(X)-X) + math.sqrt(2*delta)*torch.randn_like(X))

delta_max = (1.0)/(L_net/eps+L_y)
delta = delta_max*0.99
print(f"eps={eps:.8f} delta={delta:.3e} delta_max={delta_max:.3e}")

maxit = 200; burnin = np.int64(maxit*0.05); n_samples = np.int64(40)
X = y.clone(); MC_X=[]; count=0; thinning_step=np.int64(maxit/n_samples)
psnr_curve=[]
t0=time.time()
for i_x in range(maxit):
    X = Markov_kernel(X, delta)
    if i_x == burnin:
        post_meanvar = welford(X); absfouriercoeff = welford(torch.fft.fft2(X).abs()); count=0
    elif i_x > burnin:
        post_meanvar.update(X); absfouriercoeff.update(torch.fft.fft2(X).abs())
        psnr_curve.append(PSNR(x, post_meanvar.get_mean()))
        if count == thinning_step-1:
            MC_X.append(X.detach().cpu().numpy()); count=0
        else:
            count+=1
elapsed=time.time()-t0
post_mean = post_meanvar.get_mean()
print(f"\n[CPU] elapsed={elapsed:.1f}s  n_MC={len(MC_X)}")
print(f"initial : NRMSE={NRMSE(x,y):.4f} PSNR={PSNR(x,y):.2f} SSIM={SSIM(x,y):.4f}")
print(f"postmean: NRMSE={NRMSE(post_mean,x):.4f} PSNR={PSNR(post_mean,x):.2f} SSIM={SSIM(x,post_mean):.4f}")
