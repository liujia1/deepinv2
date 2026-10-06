# -*- coding: utf-8 -*-
import numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
mu2, sigma2 = 1.0, 0.25
for L in [5, 10, 50, 100]:
    dt = 1.0 / L
    alpha_step = 1 - dt
    total_kl = 0
    for t in range(1, L + 1):
        ab_t = alpha_step ** t
        var_t = ab_t * sigma2 + (1 - ab_t)
        kl_t = 0.5 * (ab_t * mu2 + var_t - 1 - np.log(var_t))
        total_kl += kl_t * dt
    print(f'L={L}  新公式KL={total_kl:.6f}')
