# 用 mpmath 高精度直接求 DFT 特征值 H[f,f]，裁决 float64 实测衰减在 f=20/25 是否为浮点噪声
from mpmath import mp, mpf, exp, pi, nsum, fsum
mp.dps = 60
import numpy as np

n = 64; sigma = 3.0
# 构造与代码一致的高核（float64 网格值，但用 mp.mpf 精确求和）
ax = list(range(0, n//2)) + list(range(-n//2, 0))
h64 = np.exp(-(np.meshgrid(ax, ax)[0]**2 + np.meshgrid(ax, ax)[1]**2)/(2*sigma**2))
h64 = h64/h64.sum()

def H_exact(f):
    # H[f,f] = sum_{u,v} h[u,v] * exp(-2πi f (u+v)/n)  （np.fft 约定）
    # 用复数高精度求和
    s = mp.mpc(0)
    for u in range(n):
        for v in range(n):
            th = -2*pi*mpf(f*(ax[u]+ax[v])/n)
            s += mp.e**(mp.j*th)*mpf(h64[u,v])
    return abs(s)

for f in [15, 20, 25]:
    exact = H_exact(f)
    cont = exp(-2*pi**2*mpf(sigma)**2*(f**2+f**2)/mpf(n)**2)
    print(f"f={f}: 高精度|H[f,f]|={mp.nstr(exact,6)}  连续近似={mp.nstr(cont,6)}")
meas = {15: 3.339e-09, 20: 8.487e-16, 25: 1.562e-17}
for f, m in meas.items():
    print(f"f={f}: float64实测/高精度 = {mp.nstr(mp.mpf(m)/H_exact(f),6)}")
