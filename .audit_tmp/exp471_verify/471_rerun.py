"""实验4.7-1 复核脚本 B：CPU 全流程重跑（PnP-ULA 分支强制开启）+ 更长链对照
不覆写实验目录任何产物，结果写入 .audit_tmp/exp471_verify
"""
import math, os, sys, json, time, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

kernel_len = [5, 5]; size = list(im.shape[:2])
A, AT, AAT_norm = blur_operators(kernel_len, size, "uniform", device)

model_path = os.path.join(EXP, "Pretrained_models", "RealSN_DnCNN_noise5.pth")
model = load_model(model_path, device)
denoise = lambda z: (z - model(z[None][None].to(device))[0][0]).detach()

eps = (5 / 255) ** 2
alpha = 1.0
L_net = 1.0

def run(seed, maxit, burnin_frac=0.1, n_samples=50, collect_psnr_every=1):
    torch.manual_seed(seed); np.random.seed(seed)
    y0 = A(x)
    sigma = torch.linalg.matrix_norm(A(x) - torch.mean(A(x)), ord='fro') / math.sqrt(torch.numel(x) * 10 ** (40 / 10))
    y = y0 + sigma * torch.randn_like(x)
    gradf = lambda z: AT(A(z) - y) / sigma ** 2
    L_y = AAT_norm / (sigma ** 2)
    delta = 0.99 / (L_net / eps + L_y)
    burnin = int(maxit * burnin_frac)
    thinning = max(1, int((maxit - burnin) / n_samples))

    X = y.clone()
    pm = welford(X.clone())  # placeholder, re-init at burnin
    MC, psnr_run, ssim_run = [], [], []
    count = 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(maxit):
            noise = math.sqrt(2 * delta) * torch.randn_like(X)
            X = X - delta * gradf(X) + alpha * delta / eps * (denoise(X) - X) + noise
            if i == burnin:
                pm = welford(X)
                count = 0
            elif i > burnin:
                pm.update(X)
                if (i - burnin) % collect_psnr_every == 0:
                    psnr_run.append(PSNR(x, pm.get_mean()))
                    ssim_run.append(SSIM(x, pm.get_mean()))
                if count == thinning - 1:
                    MC.append(X.detach().numpy()); count = 0
                else:
                    count += 1
    el = time.time() - t0
    chain = np.array(MC).reshape(len(MC), -1)
    var_sp = np.var(chain, axis=0)
    ess = {
        'slow': float(arviz.ess(chain[:, int(np.argmax(var_sp))])),
        'med':  float(arviz.ess(chain[:, int(np.argsort(var_sp)[len(var_sp)//2])])),
        'fast': float(arviz.ess(chain[:, int(np.argmin(var_sp))])),
    }
    return dict(seed=seed, maxit=maxit, chain_len=len(MC),
                sigma=float(sigma.item()), delta=float(delta), L_y=float(L_y),
                psnr=float(PSNR(x, pm.get_mean())), nrmse=float(NRMSE(x, pm.get_mean())),
                ssim=float(SSIM(x, pm.get_mean())), ess=ess, elapsed=el,
                psnr_run=psnr_run, ssim_run=ssim_run)

results = []
# 与讲义同设置（500 迭代），检验 seed 敏感性与 CPU/GPU 差异
for seed in [42, 43, 44]:
    r = run(seed, maxit=500)
    results.append({k: v for k, v in r.items() if k not in ('psnr_run', 'ssim_run')})
    print(f"seed={r['seed']} PSNR={r['psnr']:.2f} SSIM={r['ssim']:.4f} NRMSE={r['nrmse']:.4f} "
          f"ESS(s/m/f)={r['ess']['slow']:.1f}/{r['ess']['med']:.1f}/{r['ess']['fast']:.1f} "
          f"chain={r['chain_len']} t={r['elapsed']:.1f}s")

# 更长链：检验"低 ESS 是固有属性还是链长不足"
for maxit in [2000, 5000]:
    r = run(42, maxit=maxit)
    results.append({k: v for k, v in r.items() if k not in ('psnr_run', 'ssim_run')})
    print(f"maxit={r['maxit']} PSNR={r['psnr']:.2f} SSIM={r['ssim']:.4f} "
          f"ESS(s/m/f)={r['ess']['slow']:.1f}/{r['ess']['med']:.1f}/{r['ess']['fast']:.1f} "
          f"chain={r['chain_len']} t={r['elapsed']:.1f}s")

with open(os.path.join(OUT, "rerun_results.json"), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 画 PSNR 收敛曲线对比（500 vs 5000）
r500 = run(42, 500)
r5000 = run(42, 5000)
plt.figure(figsize=(8, 5))
plt.plot(r500['psnr_run'], label='maxit=500 (original)')
plt.plot(r5000['psnr_run'], label='maxit=5000')
plt.xlabel('iteration'); plt.ylabel('PSNR of running posterior mean (dB)')
plt.legend(); plt.grid(alpha=0.3); plt.title('4.7-1 CPU rerun: PSNR vs iterations')
plt.savefig(os.path.join(OUT, "psnr_vs_iter.png"), dpi=130, bbox_inches='tight')
print("saved psnr_vs_iter.png")
