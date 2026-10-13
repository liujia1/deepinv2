# -*- coding: utf-8 -*-
"""独立复核 3.5-2：复现 PSNR/差异比 + CP 解 ROF 正确性 + λ_tikh 最优性 + 边缘集中稳健性"""
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio

np.random.seed(42)
n = 128
x_true = resize(shepp_logan_phantom(), (n, n), order=1, preserve_range=True)
sigma = 0.15
y = x_true + sigma * np.random.randn(n, n)
lam_tv, lam_tikh = 0.15, 0.5

dxp = lambda u: np.roll(u, -1, axis=1) - u
dyp = lambda u: np.roll(u, -1, axis=0) - u
dxm_ad = lambda p: p - np.roll(p, 1, axis=1)
dym_ad = lambda p: p - np.roll(p, 1, axis=0)

def cp_tv(y, lam, maxiter=500):
    n1, n2 = y.shape
    x = np.zeros((n1, n2)); x_bar = np.zeros((n1, n2))
    px = np.zeros((n1, n2)); py = np.zeros((n1, n2))
    L2 = 8.0; s = 1/np.sqrt(L2); t = 1/np.sqrt(L2)
    for k in range(maxiter):
        px += s*dxp(x_bar); py += s*dyp(x_bar)
        np_ = np.sqrt(px**2+py**2); fac = np.maximum(1.0, np_/lam)
        px/=fac; py/=fac
        xo = x.copy(); div = dxm_ad(px)+dym_ad(py)
        x = (xo + t*div + t*y)/(1+t); x = np.maximum(0, x)
        x_bar = 2*x - xo
    return x

def h1(y, lam):
    nn = y.shape[0]; d = 2-2*np.cos(2*np.pi*np.arange(nn)/nn)
    DX, DY = np.meshgrid(d, d); DtD = DX+DY
    return np.real(np.fft.ifft2(np.fft.fft2(y)/(1+lam*DtD)))

x_tv = cp_tv(y, lam_tv, 500); x_tikh = h1(y, lam_tikh)

def rof(x, y, lam):  # TV 目标 (各向异 L1，与代码 dxp/dyp 一致用 |·| 求和)
    return 0.5*np.sum((x-y)**2) + lam*(np.sum(np.abs(dxp(x)))+np.sum(np.abs(dyp(x))))
def rof_iso(x, y, lam):  # 各向同性 ||Dx||_{2,1}
    g = np.stack([dxp(x), dyp(x)], -1); return 0.5*np.sum((x-y)**2)+lam*np.sum(np.linalg.norm(g,axis=2))

pn = peak_signal_noise_ratio(x_true, y)
pt = peak_signal_noise_ratio(x_true, np.clip(x_tikh,0,1))
pv = peak_signal_noise_ratio(x_true, np.clip(x_tv,None,1))
print("=== JSON 复现 ===")
print(f"psnr_noisy={pn:.2f}(J16.45) psnr_tikh={pt:.2f}(J22.91) psnr_tv={pv:.2f}(J26.57)")

# 差异边缘/平坦 (复刻代码口径)
gx,gy=np.gradient(x_true); eg=np.sqrt(gx**2+gy**2)
thr=np.quantile(eg[eg>1e-6],0.90); em=eg>max(thr,0.02); fm=eg<0.005
dm=np.abs(x_tv-x_tikh); de=dm[em].mean(); df=dm[fm].mean()
print(f"diff_edge={de:.4f}(J0.0592) diff_flat={df:.4f}(J0.0474) ratio={de/df:.2f}(J1.25)")

# CP 是否真解 ROF: 目标值应显著低于 y 与 x_true
print("\n=== CP 解 ROF 正确性 (各向同性目标) ===")
for tag,xx in [("y",y),("x_true",x_true),("x_tikh",x_tikh),("x_tv",x_tv)]:
    print(f"  ROF({tag})={rof_iso(xx,y,lam_tv):.4f}")
# 迭代收敛性: 500 vs 5000 步
x_tv2=cp_tv(y,lam_tv,5000)
print(f"  ||x_tv(500)-x_tv(5000)||/||x_tv||={np.linalg.norm(x_tv-x_tv2)/np.linalg.norm(x_tv2):.4f}  ROF(5000)={rof_iso(x_tv2,y,lam_tv):.4f}")

# λ_tikh=0.5 是否 H1 的 PSNR 最优
print("\n=== H1 λ 扫描 (验证 0.5 为 PSNR 最优) ===")
best=(0,-1)
for lam in np.arange(0.05,1.51,0.05):
    p=peak_signal_noise_ratio(x_true,np.clip(h1(y,lam),0,1))
    if p>best[1]: best=(lam,p)
    if abs(lam-0.5)<1e-9 or abs(lam-best[0])<1e-9: pass
print(f"  扫描最优 λ*={best[0]:.2f} PSNR={best[1]:.2f}; λ=0.5 PSNR={peak_signal_noise_ratio(x_true,np.clip(h1(y,0.5),0,1)):.2f}")
for lam in [0.4,0.45,0.5,0.55,0.6]:
    print(f"    λ={lam}: PSNR={peak_signal_noise_ratio(x_true,np.clip(h1(y,lam),0,1)):.3f}")

# 边缘集中稳健性: 多口径
print("\n=== '差异集中于边缘' 稳健性 (多口径) ===")
tot=dm.sum(); etot=dm[em].sum(); area_e=em.mean()
print(f"  边缘带面积占比={area_e*100:.1f}%  边缘带差异能量占比={etot/tot*100:.1f}%")
for q in [0.8,0.9,0.95]:
    t2=np.quantile(eg[eg>1e-6],q); e2=eg>t2; f2=eg<0.005
    print(f"  edge=quantile{q}: edge_mean={dm[e2].mean():.4f} flat_mean={dm[f2].mean():.4f} ratio={dm[e2].mean()/dm[f2].mean():.2f}")
# top-10% 差异像素落在边缘带内的比例
k=npint=int(0.1*dm.size); idx=np.argsort(dm.ravel())[::-1][:k]
in_edge=np.isin(idx, np.flatnonzero(em.ravel())).mean()
print(f"  差异最大10%像素中落在边缘带的比例={in_edge*100:.1f}% (随机基线={area_e*100:.1f}%)")
