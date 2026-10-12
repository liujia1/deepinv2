# -*- coding: utf-8 -*-
"""实验4.4-2 MAP 侧核查（约10分钟, CPU）：
(A) 小图上验证 chambolle 修复版散度(-p[-2]) 比 旧版(-p[-1]=0) 得到更低 prox 目标能量
(B) 用估计参数做长 MAP(1000步)，记录梯度范数/SNR 随迭代曲线 -> 200步是否收敛
(C) theta 扫描 + oracle sigma2 -> 估计 theta 是否接近 SNR 最优点
"""
import sys, math, json
import numpy as np
import torch

sys.path.insert(0, r"d:/works/deepinv/第4章 MCMC与ULA算法/第四章 配套实验/实验4.4-2")
from skimage.data import camera
from sampling_tools import blur_operators, chambolle_prox_TV

dev = torch.device('cpu')

# ---------- (A) 64x64 小图 prox 能量对比 ----------
def grad_im(u):
    z = u[1:, :] - u[:-1, :]
    dux = torch.vstack([z, torch.zeros(1, u.shape[1])])
    z = u[:, 1:] - u[:, :-1]
    duy = torch.hstack([z, torch.zeros(u.shape[0], 1)])
    return dux, duy

def div_new(p1, p2):  # 修复版
    z = p2[:, 1:-1] - p2[:, :-2]
    v = torch.hstack([p2[:, :1], z, -p2[:, -2:-1]])
    z = p1[1:-1, :] - p1[:-2, :]
    u = torch.vstack([p1[:1, :], z, -p1[-2:-1, :]])
    return v + u

def div_old(p1, p2):  # 上游旧版（末行/末列恒0）
    z = p2[:, 1:-1] - p2[:, :-2]
    v = torch.hstack([p2[:, :1], z, torch.zeros((p2.shape[0], 1))])
    z = p1[1:-1, :] - p1[:-2, :]
    u = torch.vstack([p1[:1, :], z, torch.zeros((1, p1.shape[1]))])
    return v + u

def prox(g, lambd, div, iters=500):
    px = torch.zeros_like(g); py = torch.zeros_like(g); tau = 0.249
    for _ in range(iters):
        u = div(px, py) - g / lambd
        upx, upy = grad_im(u)
        tmp = torch.sqrt(upx * upx + upy * upy)
        px = (px + tau * upx) / (1 + tau * tmp)
        py = (py + tau * upy) / (1 + tau * tmp)
    return g - lambd * div(px, py)

def tv_correct(im):
    dh = torch.zeros_like(im); dh[:, :-1] = im[:, 1:] - im[:, :-1]
    dv = torch.zeros_like(im); dv[:-1, :] = im[1:, :] - im[:-1, :]
    return torch.sum(torch.sqrt(dh ** 2 + dv ** 2))

g64 = torch.Tensor(camera()[::8, ::8].astype(np.float64)) + 3.0 * torch.randn(64, 64)
lam = 4.0
xn = prox(g64, lam, div_new); xo = prox(g64, lam, div_old)
En = (0.5 * torch.sum((xn - g64) ** 2) + lam * tv_correct(xn)).item()
Eo = (0.5 * torch.sum((xo - g64) ** 2) + lam * tv_correct(xo)).item()
print(f"(A) prox能量: 修复版={En:.1f} 旧版={Eo:.1f} 旧版相对高={100*(Eo-En)/En:.3f}%", flush=True)

# ---------- (B)(C) 512 全尺寸 MAP ----------
torch.manual_seed(42)
x = torch.Tensor(camera().astype(np.float64))
nx, ny = x.shape; dimx = nx * ny
A, AT, AAT = blur_operators([9, 9], (nx, ny), 'uniform', dev)
Ax = A(x)
sigma = torch.linalg.matrix_norm(Ax - torch.mean(Ax), ord='fro') / math.sqrt(dimx * 10 ** 3)
y = Ax + sigma * torch.randn_like(x)
s2_true = float(sigma ** 2)
s2_est = 4.826514
theta_est = 0.023081
lp = 1.0
proxg = lambda z, t: chambolle_prox_TV(z, dev, {'lambda': t, 'MaxIter': 10})
gradg = lambda z, t: (z - proxg(z, t)) / lp
gf = lambda z, s2: AT(A(z) - y) / s2
snr = lambda e: 10 * math.log10(torch.sum(x ** 2).item() / torch.sum((x - e) ** 2).item())

def run_map(theta, s2, niter=1000, track=False):
    xm = y.clone()
    gm = 0.98 / (AAT / s2 + 1.0 / lp)
    hist = []
    for i in range(niter):
        g_all = gf(xm, s2) + gradg(xm, lp * theta)
        xm = xm - gm * g_all
        if track and (i % 20 == 0 or i in (199, 399, 999)):
            hist.append((i, float(torch.norm(g_all)), snr(xm)))
    g_all = gf(xm, s2) + gradg(xm, lp * theta)
    return snr(xm), float(torch.norm(g_all)), hist

# (B) 估计参数的收敛曲线（与JSON口径一致：200步 vs 1000步）
snr200, gn200, _ = run_map(theta_est, s2_est, 200)
snr1k, gn1k, hist = run_map(theta_est, s2_est, 1000, track=True)
print(f"(B) theta_est,s2_est: 200步 SNR={snr200:.2f} |gn|={gn200:.3f}; 1000步 SNR={snr1k:.2f} |gn|={gn1k:.3f}", flush=True)
print("    曲线(i,|gn|,SNR):", [(i, round(g, 3), round(s, 2)) for i, g, s in hist[:8]], "...",
      [(i, round(g, 3), round(s, 2)) for i, g, s in hist[-3:]], flush=True)

# (C) theta 扫描：s2_est vs oracle s2_true
print("(C) theta 扫描:", flush=True)
sweep = []
for th in [0.005, 0.01, 0.023081, 0.04, 0.08, 0.16]:
    s_a, _, _ = run_map(th, s2_est, 600)
    s_t, _, _ = run_map(th, s2_true, 600)
    sweep.append((th, s_a, s_t))
    print(f"    theta={th:.4f}  SNR(s2_est)={s_a:.2f}  SNR(s2_true)={s_t:.2f}", flush=True)

out = dict(A=dict(E_new=En, E_old=Eo), B=dict(snr200=snr200, gn200=gn200, snr1000=snr1k, gn1000=gn1k,
                                              hist=hist), C=sweep,
           snr_obs=snr(y), s2_true=s2_true)
with open(r"d:/works/deepinv/.audit_tmp/exp441_verify/audit_442_map_out.json", 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("DONE", flush=True)
