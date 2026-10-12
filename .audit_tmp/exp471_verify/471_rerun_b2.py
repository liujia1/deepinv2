"""复核脚本 B2（精简版）：seed 敏感性 + 更长链对照"""
import math, os, sys, json, time, warnings
import numpy as np
import torch
torch.set_num_threads(4)
warnings.filterwarnings('ignore')

EXP = r"D:\works\deepinv\第4章 MCMC与ULA算法\第四章 配套实验\实验4.7-1"
OUT = r"D:\works\deepinv\.audit_tmp\exp471_verify"
sys.path.insert(0, EXP)
device = torch.device('cpu')
from sampling_tools import *   # noqa
from PIL import Image
import arviz

im = np.array(Image.open(os.path.join(EXP, "cman.png")))
x = torch.Tensor(im / 255.).to(device)
A, AT, AAT_norm = blur_operators([5, 5], list(im.shape[:2]), "uniform", device)
model = load_model(os.path.join(EXP, "Pretrained_models", "RealSN_DnCNN_noise5.pth"), device)
denoise = lambda z: (z - model(z[None][None])[0][0]).detach()
eps = (5 / 255) ** 2

def run(seed, maxit, n_samples=50):
    torch.manual_seed(seed); np.random.seed(seed)
    sigma = torch.linalg.matrix_norm(A(x) - torch.mean(A(x)), ord='fro') / math.sqrt(torch.numel(x) * 10 ** 4)
    y = A(x) + sigma * torch.randn_like(x)
    gradf = lambda z: AT(A(z) - y) / sigma ** 2
    L_y = AAT_norm / sigma ** 2
    delta = 0.99 / (1.0 / eps + L_y)
    burnin = int(maxit * 0.1)
    thinning = max(1, int((maxit - burnin) / n_samples))
    X = y.clone(); MC = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(maxit):
            X = X - delta * gradf(X) + delta / eps * (denoise(X) - X) + math.sqrt(2 * delta) * torch.randn_like(X)
            if i == burnin:
                pm = welford(X); count = 0
            elif i > burnin:
                pm.update(X)
                if count == thinning - 1:
                    MC.append(X.numpy()); count = 0
                else:
                    count += 1
    chain = np.array(MC).reshape(len(MC), -1)
    var_sp = np.var(chain, axis=0)
    order = np.argsort(var_sp)
    ess = [float(arviz.ess(chain[:, order[-1]])), float(arviz.ess(chain[:, order[len(order)//2]])), float(arviz.ess(chain[:, order[0]]))]
    out = dict(seed=seed, maxit=maxit, chain=len(MC),
               psnr=float(PSNR(x, pm.get_mean())), ssim=float(SSIM(x, pm.get_mean())),
               nrmse=float(NRMSE(x, pm.get_mean())),
               ess_slow=round(ess[0], 2), ess_med=round(ess[1], 2), ess_fast=round(ess[2], 2),
               ess_ratio_med=round(ess[1] / len(MC), 3),
               elapsed=round(time.time() - t0, 1))
    print(json.dumps(out), flush=True)
    return out

res = []
for seed in (43, 44):
    res.append(run(seed, 500))
for maxit in (2000, 5000):
    res.append(run(42, maxit))
with open(os.path.join(OUT, "rerun_b2.json"), 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=2)
print("done", flush=True)
