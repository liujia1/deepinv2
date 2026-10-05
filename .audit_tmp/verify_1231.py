# -*- coding: utf-8 -*-
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, torch

T = 1000
for dt in [torch.float32, torch.float64]:
    betas = torch.linspace(1e-4, 0.02, T, dtype=dt)
    alphas = 1.0 - betas
    ab = torch.cumprod(alphas, dim=0)
    abp = torch.cat([torch.ones(1, dtype=dt), ab[:-1]])
    pv = betas * (1 - abp) / (1 - ab)
    wt = betas**2 / (2 * pv * alphas * (1 - ab))
    lam = betas[1:] * (1 - ab[1:]) / ((1 - abp[1:]) * alphas[1:])
    dsm = lam / (1 - ab[1:])
    ratio = (dsm.numpy() / wt[1:].numpy()).astype(np.float64)
    print(f"--- {dt} ---")
    print(f"ratio mean={ratio.mean():.17g} std={ratio.std():.3e} maxdev={np.max(np.abs(ratio-2)):.3e}")
    w = wt[1:].numpy().astype(np.float64)
    amin = int(np.argmin(w)); amax = int(np.argmax(w))
    print(f"vlb min={w.min():.6f}@t={amin+1}, max={w.max():.6f}@t={amax+1}, decades={np.log10(w.max()/w.min()):.3f}")
    print("wt t=1,999:", w[0], w[998], " w@t=999/ min:", w[998]/w.min())
    if dt == torch.float64:
        for t in [1, 10, 50, 100, 250, 500, 750, 999]:
            print(f"t={t}: abar={ab[t].item():.6f} sigma={ (1-ab[t]).sqrt().item():.6f} snr={(ab[t]/(1-ab[t])).item():.4f}")
        print("Karras:")
        for t in [1, 50, 100, 250, 500, 750, 999]:
            s = ab[t].sqrt().item(); sve = (1-ab[t]).sqrt().item(); svp = ((1-ab[t])/ab[t]).sqrt().item()
            print(f"t={t}: s={s:.6f} sig_ve={sve:.6f} sig_vp={svp:.6g}")
        print("VLB sampled (direct index t):")
        for t in [1, 10, 50, 100, 250, 500, 750, 999]:
            print(f"t_{t}: {wt[t].item():.10g}")
