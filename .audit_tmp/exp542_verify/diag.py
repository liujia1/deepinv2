# -*- coding: utf-8 -*-
"""
5.4-2 复核诊断（CPU, cman 中心128x128裁剪, 种子42, 5x5 uniform, BSNR40）
目标：
  A  : 完全复刻代码 TV-ULA 设置 (alpha=1, gamma=0.05, niter=2000, chambolle200)
  A2 : 同A仅延长到4000步 (chambolle100) -> 隔离"迭代不足"效应
  B  : 规范TV权重 alpha=20 (gamma=0.0025, prox lambda=0.05), 8000步 -> 隔离"先验权重"效应
  P  : 复刻 PnP-ULA, 5000步 -> 观察2000步后是否仍在爬升(官方1万步=33.22dB)
记录：后验均值PSNR随迭代轨迹、平均后验std、墙钟时间
"""
import os, sys, math, time, json
import torch
import numpy as np
from PIL import Image

torch.set_num_threads(os.cpu_count())
EXP = r'd:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.4-2'
sys.path.insert(0, EXP)
from sampling_tools import chambolle_prox_TV, blur_operators  # noqa
from sampling_tools.measures import PSNR as psnr_fn  # noqa

im = np.array(Image.open(os.path.join(EXP, 'cman.png')))
H, W = im.shape
h0, w0 = (H - 128) // 2, (W - 128) // 2
x = torch.Tensor(im[h0:h0 + 128, w0:w0 + 128] / 255.)
device = torch.device('cpu')

torch.manual_seed(42)
A, AT, AAT_norm = blur_operators([5, 5], [128, 128], 'uniform', device)
y0 = A(x)
sigma = torch.linalg.matrix_norm(A(x) - torch.mean(A(x)), ord='fro') / math.sqrt(torch.numel(x) * 10 ** 4)
y = y0 + sigma * torch.randn_like(x)
L_y = float(AAT_norm / sigma ** 2)
gradf = lambda X: AT(A(X) - y) / sigma ** 2
projbox = lambda X: torch.clamp(X, min=0, max=1)
print(f'sigma={float(sigma):.6f} L_y={L_y:.1f} PSNR(y)={float(psnr_fn(x,y)):.3f}', flush=True)

BURN = 200  # 与官方lab一致：5%烧入

def run_tv(tag, niter, alpha, gamma, prox_lam, inner, milestones):
    delta = 0.99 / (1.0 / gamma + L_y)
    X = y.clone()
    sm = torch.zeros_like(X); s2 = torch.zeros_like(X); cnt = 0
    hist = []
    t0 = time.time()
    for k in range(niter):
        g = gradf(X)
        pr = chambolle_prox_TV(X, device, {'lambda': prox_lam, 'maxiter': inner})
        drift_tv = (X - pr) / gamma
        X = projbox(X - delta * (g + drift_tv) + math.sqrt(2 * delta) * torch.randn_like(X))
        if k >= BURN:
            cnt += 1
            sm += X; s2 += X * X
        if (k + 1) in milestones and cnt > 0:
            mean = sm / cnt
            std = torch.sqrt(torch.clamp(s2 / cnt - mean ** 2, min=0))
            hist.append((k + 1, round(float(psnr_fn(x, mean)), 3), round(float(std.mean()), 5)))
            print(f'[{tag}] iter={k+1:6d} meanPSNR={hist[-1][1]:.3f} avgStd={hist[-1][2]:.5f} '
                  f'elapsed={time.time()-t0:.0f}s', flush=True)
    eff = niter * delta / gamma
    print(f'[{tag}] delta={delta:.3e} total_time={niter*delta:.4f} effective_prox_steps={eff:.2f}', flush=True)
    return {'tag': tag, 'niter': niter, 'alpha': alpha, 'gamma': gamma, 'prox_lam': prox_lam,
            'delta': delta, 'total_diffusion_time': niter * delta, 'eff_prox_steps': eff,
            'milestones': hist, 'elapsed_s': time.time() - t0}

def run_pnp(niter, milestones):
    sys.path.insert(0, os.path.join(EXP, 'sampling_tools'))
    from load_model import DnCNN
    net = DnCNN(channels=1, num_of_layers=17)
    sd = torch.load(os.path.join(EXP, 'Pretrained_models', 'RealSN_DnCNN_noise5.pth'),
                    map_location='cpu', weights_only=True)
    net.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
    net.eval()
    eps = (5 / 255) ** 2
    delta = 0.99 / (1.0 / eps + L_y)
    denoise = lambda z: (z - net(z[None][None])[0][0]).detach()
    X = y.clone()
    sm = torch.zeros_like(X); s2 = torch.zeros_like(X); cnt = 0
    hist = []
    t0 = time.time()
    with torch.no_grad():
        for k in range(niter):
            score = (denoise(X) - X) / eps
            X = projbox(X - delta * (gradf(X) - score) + math.sqrt(2 * delta) * torch.randn_like(X))
            if k >= BURN:
                cnt += 1
                sm += X; s2 += X * X
            if (k + 1) in milestones and cnt > 0:
                mean = sm / cnt
                std = torch.sqrt(torch.clamp(s2 / cnt - mean ** 2, min=0))
                hist.append((k + 1, round(float(psnr_fn(x, mean)), 3), round(float(std.mean()), 5)))
                print(f'[P] iter={k+1:6d} meanPSNR={hist[-1][1]:.3f} avgStd={hist[-1][2]:.5f} '
                      f'elapsed={time.time()-t0:.0f}s', flush=True)
    print(f'[P] delta={delta:.3e} total_time={niter*delta:.4f} eff_denoiser_steps={niter*delta/eps:.2f}', flush=True)
    return {'tag': 'P', 'niter': niter, 'eps': eps, 'delta': delta,
            'total_diffusion_time': niter * delta, 'eff_den_steps': niter * delta / eps,
            'milestones': hist, 'elapsed_s': time.time() - t0}

results = {}
results['A'] = run_tv('A coded(a=1,g=.05)', 2000, 1, 0.05, 0.05, 100,
                      [500, 1000, 2000])
results['A2'] = run_tv('A2 long(a=1)', 4000, 1, 0.05, 0.05, 100,
                       [500, 1000, 2000, 4000])
results['B'] = run_tv('B strong(a=20,g=.0025)', 8000, 20, 0.0025, 0.05, 100,
                      [500, 1000, 2000, 4000, 6000, 8000])
results['P'] = run_pnp(5000, [500, 1000, 2000, 3000, 5000])

with open(os.path.join(os.path.dirname(__file__), 'diag_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print('DONE', flush=True)
