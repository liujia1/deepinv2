# -*- coding: utf-8 -*-
"""Tiny smoke test for 5.6-3 new code paths (real MAP + short ULA). Not the full sweep."""
import os, sys, math
import numpy as np, torch

device = torch.device('cpu')
np.random.seed(42); torch.manual_seed(42)

EXP551 = r"d:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.5-1"
sys.path.insert(0, EXP551)
from sampling_tools.blur_operators import blur_operators
from sampling_tools.load_model import DnCNN

npz = os.path.join(EXP551, "sampling_results.npz")
d = np.load(npz)
x_true, y_obs = d['x_true'], d['y_obs']
print("shapes", x_true.shape, y_obs.shape)

_x_t = torch.tensor(x_true, dtype=torch.float32, device=device)
_y_t = torch.tensor(y_obs, dtype=torch.float32, device=device)
A, AT, AAT_norm = blur_operators([5, 5], list(x_true.shape), "uniform", device)
BSNRdb = 40.0
sigma = torch.linalg.matrix_norm(A(_x_t) - torch.mean(A(_x_t)), ord='fro') / math.sqrt(torch.numel(_x_t) * 10 ** (BSNRdb / 10))
L_y = AAT_norm / (sigma ** 2)
gradf = lambda xx: AT(A(xx) - _y_t) / sigma ** 2
_C_hi = torch.tensor(1.0, device=device); _C_lo = torch.tensor(0.0, device=device)
projbox = lambda xx: torch.clamp(xx, min=_C_lo, max=_C_hi)

net = DnCNN(channels=1, num_of_layers=17)
sd = torch.load(os.path.join(EXP551, "Pretrained_models", "RealSN_DnCNN_noise5.pth"), map_location=device, weights_only=True)
sd2 = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
net.load_state_dict(sd2); net.eval()
denoise = lambda xx: (xx - net(xx[None][None].to(device))[0][0]).detach()

eps0 = (5/255)**2; L_net = 1.0; alpha = 1.0

def pnp_map(maxit=5, delta_frac=0.99):
    delta_max = 1.0 / (L_net / eps0 + L_y); delta = float(delta_max) * delta_frac
    X = _y_t.clone()
    for _ in range(maxit):
        X = projbox(X - delta * gradf(X) + alpha * delta / eps0 * (denoise(X) - X))
    return X.detach().cpu().numpy()

def run_ula(eps, delta_frac, maxit, n_samples, seed=0, burn_frac=0.05):
    torch.manual_seed(seed)
    delta_max = 1.0 / (L_net / eps + L_y); delta = float(delta_max) * delta_frac
    X = _y_t.clone(); burnin = int(maxit*burn_frac); thin = max(1, int((maxit-burnin)/n_samples))
    samples, trace = [], []; count = 0
    for i in range(maxit):
        X = projbox(X - delta*gradf(X) + alpha*delta/eps*(denoise(X)-X) + math.sqrt(2*delta)*torch.randn_like(X))
        if i > burnin:
            trace.append(float(X.mean()))
            if count == thin-1: samples.append(X.detach().cpu().numpy()); count = 0
            else: count += 1
    return np.array(samples), np.array(trace)

def psnr(x, y):
    mse = np.mean((x-y)**2); return 10*np.log10(max(x.max(),1.0)**2/mse)

x_map = pnp_map(maxit=8)
print("MAP ok, PSNR(8it)=%.2f" % psnr(x_true, x_map))
s, tr = run_ula(eps0, 0.99, maxit=40, n_samples=10, seed=1)
print("ULA ok, n_samples=%d, trace=%d" % (len(s), len(tr)))
print("var map mean=%.3e" % np.mean(np.var(s, axis=0, ddof=1)))
t = tr - tr.mean(); rho = float(np.corrcoef(t[:-1], t[1:])[0,1]); print("rho=%.3f" % rho)
print("ALL SMOKE CHECKS PASSED")
