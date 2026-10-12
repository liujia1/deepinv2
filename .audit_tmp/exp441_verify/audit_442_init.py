# -*- coding: utf-8 -*-
"""实验4.4-2 决定性复核：σ² 初值记忆 & 偏差方向（忠实复刻当前 4.4-2.py，调用真实 sampling_tools）
用法: python audit_442_init.py <low|true|high>
每个进程约5-6分钟(CPU)。输出 JSON 行到 audit_442_init_out.jsonl
"""
import sys, os, math, json
import numpy as np
import torch

sys.path.insert(0, r"d:/works/deepinv/第4章 MCMC与ULA算法/第四章 配套实验/实验4.4-2")
from skimage.data import camera
from sampling_tools import blur_operators, chambolle_prox_TV, TVnorm

mode = sys.argv[1] if len(sys.argv) > 1 else 'true'
dev = torch.device('cpu')
torch.manual_seed(42)  # 与原实验一致：power iter + y 的 RNG 流

x = torch.Tensor(camera().astype(np.float64))
nx, ny = x.shape
dimx = nx * ny
A, AT, AAT_norm = blur_operators([9, 9], (nx, ny), 'uniform', dev)
Ax = A(x)
sigma = torch.linalg.matrix_norm(Ax - torch.mean(Ax), ord='fro') / math.sqrt(dimx * 10 ** 3)
y = Ax + sigma * torch.randn_like(x)
s2_true = float(sigma ** 2)

proxg = lambda z, lam: chambolle_prox_TV(z, dev, {'lambda': lam, 'MaxIter': 10})
gradg = lambda z, lam, lp: (z - proxg(z, lam)) / lp
grad_f_x = lambda z, s2: AT(A(z) - y) / s2
df_ds2 = lambda z, s2: (torch.linalg.matrix_norm(y - A(z), ord='fro') ** 2) / (2 * s2 ** 2)

lambda_prox = 1.0
th_init = 0.01
s2_init = {'low': 1.0, 'true': s2_true, 'high': 20.0}[mode]
c_eta, c_s2, d_exp = 10, 10000, 0.8
dstep = lambda i: ((i + 1) ** (-d_exp)) / dimx
warmup, total, burn = 1000, 1000, 700

# warmup（固定 theta, sigma2）
gamma = 0.98 / (AAT_norm / s2_init + 1.0 / lambda_prox)
X = y.clone()
lwu = lambda_prox * th_init
for k in range(warmup):
    gf = grad_f_x(X, s2_init)
    gg = gradg(X, lwu, lambda_prox)
    X = X - gamma * gf - gamma * gg + math.sqrt(2 * gamma) * torch.randn_like(X)
    if (k + 1) % 500 == 0:
        print(f"[{mode}] warmup {k+1}/{warmup}", flush=True)

eta = math.log(th_init)
theta = th_init
s2 = s2_init
tl, sl = [theta], [s2]
score_th_res = []   # Fisher theta 分数 -TV + d/theta
for k in range(total):
    Lf = AAT_norm / s2
    gamma = 0.98 / (Lf + 1.0 / lambda_prox)
    gg = gradg(X, lambda_prox * theta, lambda_prox)
    gf = grad_f_x(X, s2)
    X = X - gamma * gf - gamma * gg + math.sqrt(2 * gamma) * torch.randn_like(X)

    gx = TVnorm(X).item()
    r2 = df_ds2(X, s2).item()
    gth = -gx + dimx / theta
    eta = max(math.log(0.001), min(math.log(1.0), eta + c_eta * dstep(k) * (gth * theta)))
    theta = math.exp(eta)
    gs2 = r2 - dimx / (2 * s2)
    s2 = max(0.1, min(100.0, s2 + c_s2 * dstep(k) * gs2))
    tl.append(theta); sl.append(s2)
    if k >= burn:
        score_th_res.append(gth)
    if (k + 1) % 200 == 0:
        print(f"[{mode}] sapg {k+1}/{total} theta={theta:.5f} s2={s2:.4f}", flush=True)

th_est = float(np.mean(tl[burn:]))
s2_est = float(np.mean(sl[burn:]))
rel = abs(s2_est - s2_true) / s2_true * 100

# MAP（与原脚本一致）
xm = y.clone()
lm = lambda_prox * th_est
gm = 0.98 / (AAT_norm / s2_est + 1.0 / lambda_prox)
for _ in range(200):
    xm = xm - gm * (grad_f_x(xm, s2_est) + gradg(xm, lm, lambda_prox))
snr = lambda e: 10 * math.log10(torch.sum(x ** 2).item() / torch.sum((x - e) ** 2).item())
fg = grad_f_x(xm, s2_est) + gradg(xm, lm, lambda_prox)

res = dict(mode=mode, s2_init=s2_init, s2_true=s2_true, theta_est=th_est,
           s2_est=s2_est, rel_err_pct=rel, snr_obs=snr(y), snr_map=snr(xm),
           map_grad_norm=float(torch.norm(fg)), map_min=float(xm.min()), map_max=float(xm.max()),
           theta_score_mean=float(np.mean(score_th_res)),
           theta_score_rel=float(np.mean(score_th_res) / (dimx / th_est)),
           tl_tail_mean=float(np.mean(tl[burn:])), sl_tail_std=float(np.std(sl[burn:])))
out = r"d:/works/deepinv/.audit_tmp/exp441_verify/audit_442_init_out.jsonl"
with open(out, 'a', encoding='utf-8') as f:
    f.write(json.dumps(res, ensure_ascii=False) + '\n')
np.savez(rf"d:/works/deepinv/.audit_tmp/exp441_verify/chains_{mode}.npz", tl=np.array(tl), sl=np.array(sl))
print("RESULT", json.dumps(res, ensure_ascii=False), flush=True)
