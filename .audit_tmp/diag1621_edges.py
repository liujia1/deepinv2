# -*- coding: utf-8 -*-
"""诊断 16.2-1 文档中三个无代码依据的定量断言：
   1) "边缘处误差放大约 3 倍"
   2) "边缘幅度平均损失约 30%"
   3) "水平/垂直边缘保留度分别为 0.68/0.70"
严格复用 16.2-1.py 的 IC-free 生成方式（256 生成 -> resize 128 -> x n/n_data）。
"""
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize
from skimage.metrics import peak_signal_noise_ratio as psnr

np.random.seed(42)
n, n_data = 128, 256

phantom_fine = resize(shepp_logan_phantom(), (n_data, n_data), order=0,
                      preserve_range=True, anti_aliasing=False)
phantom_fine = phantom_fine / phantom_fine.max()
phantom = resize(phantom_fine, (n, n), order=1, preserve_range=True, anti_aliasing=True)
phantom = phantom / phantom.max()


def gen(theta):
    sf = radon(phantom_fine, theta=theta, circle=True)
    sr = resize(sf, (n, len(theta)), order=1, preserve_range=True, anti_aliasing=True)
    return sr * (n / n_data)


theta_sparse = np.linspace(0, 180, 30, endpoint=False)
theta_limited = np.linspace(0, 120, 120, endpoint=False)
theta_full = np.linspace(0, 180, 180, endpoint=False)

r_sparse = iradon(gen(theta_sparse), theta=theta_sparse, circle=True, filter_name='ramp')
r_limited = iradon(gen(theta_limited), theta=theta_limited, circle=True, filter_name='ramp')
r_full = iradon(gen(theta_full), theta=theta_full, circle=True, filter_name='ramp')

print("PSNR  sparse=%.2f  limited=%.2f  full=%.2f"
      % (psnr(phantom, r_sparse, data_range=1.0),
         psnr(phantom, r_limited, data_range=1.0),
         psnr(phantom, r_full, data_range=1.0)))

# ---------- 边缘掩码：由真值梯度定义 ----------
def grad(a):
    gy, gx = np.gradient(a)
    return gy, gx


gy, gx = grad(phantom)
gmag = np.hypot(gy, gx)
edge = gmag > 0.15 * gmag.max()          # 强边缘（体模内部灰阶跳变 ~0.2-0.5）
inner = gmag < 0.01 * gmag.max()         # 平坦内部
print("\n边缘像素数 %d（占 %.1f%%），平坦内部像素数 %d"
      % (edge.sum(), 100 * edge.mean(), inner.sum()))

err = np.abs(r_limited - phantom)
print("\n[1] 边缘 vs 内部平均绝对误差：")
print("    边缘 mean|err| = %.4f" % err[edge].mean())
print("    内部 mean|err| = %.4f" % err[inner].mean())
print("    比值 = %.2f 倍" % (err[edge].mean() / err[inner].mean()))
# 全图版（含 FOV 外）
print("    （全图 mean|err| = %.4f，边缘/全图 = %.2f）"
      % (err.mean(), err[edge].mean() / err.mean()))

# ---------- 边缘幅度保留度 ----------
def band_ratio(recon):
    _, gxr = np.gradient(recon)
    gyr, _ = np.gradient(recon)
    mr = np.hypot(gyr, gxr)
    return (np.median(mr[edge]) / np.median(gmag[edge]))


print("\n[2] 边缘梯度幅度保留度（中位数比，1.0=完全保留）：")
for nm, r in (("sparse", r_sparse), ("limited", r_limited), ("full", r_full)):
    print("    %-8s %.3f  -> 损失 %.1f%%" % (nm, band_ratio(r), 100 * (1 - band_ratio(r))))

# ---------- 方向性：水平边缘（|gy|>|gx|） vs 垂直边缘 ----------
hor = edge & (np.abs(gy) > np.abs(gx))
ver = edge & (np.abs(gx) >= np.abs(gy))
print("\n[3] 方向性保留度：")
print("    水平边缘像素 %d，垂直边缘像素 %d" % (hor.sum(), ver.sum()))
for nm, r, key in (("rr", r_limited, 'both'), ("sparse", r_sparse, 'both')):
    gyr, gxr = np.gradient(r)
    # 水平边缘的梯度主要在 y 方向 -> 看 gy 的保留
    ret_h = np.median(np.abs(gyr[hor])) / np.median(np.abs(gy[hor]))
    ret_v = np.median(np.abs(gxr[ver])) / np.median(np.abs(gx[ver]))
    print("    %-8s 水平边保留 %.3f，垂直边保留 %.3f" % (nm, ret_h, ret_v))

# 方向性误差：沿缺失方向（60-180 缺）条纹
print("\n[4] 有限角误差的方向分布（傅里叶域）：")
E = np.fft.fftshift(np.abs(np.fft.fft2(r_limited - phantom)))
H, W = E.shape
cy, cx = H // 2, W // 2
yy, xx = np.mgrid[0:H, 0:W]
rr = np.hypot(yy - cy, xx - cx)
ang = np.degrees(np.arctan2(-(yy - cy), xx - cx)) % 180   # 0-180
mask = rr > 15
for a0 in range(0, 180, 30):
    m = mask & (ang >= a0) & (ang < a0 + 30)
    print("    角度 %3d-%3d 度：平均能量 %.2e" % (a0, a0 + 30, E[m].mean()))
