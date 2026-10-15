# -*- coding: utf-8 -*-
# 诊断2：x_rand 是不是"构造"出来的（含 A-零空间成分），而非独立随机噪声
import numpy as np, os
from PIL import Image

D = r"d:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.4-2"
A = np.load(os.path.join(D, 'transform_A.npy'))
maomi = np.array(Image.open(os.path.join(D, 'maomi_grayscale.png')).convert('L')) / 255.0
mohu  = np.array(Image.open(os.path.join(D, 'maomi_mohu.png')).convert('L')) / 255.0
rand  = np.array(Image.open(os.path.join(D, 'random_gray.png')).convert('L')) / 255.0

x_true = maomi.ravel(); y = mohu.ravel(); x_rand = rand.ravel()
d = x_rand - x_true
print("||d|| =", np.linalg.norm(d), " ||A d|| =", np.linalg.norm(A @ d))
print("d 的 mean/std:", d.mean(), d.std())

U, s, Vt = np.linalg.svd(A)
c = Vt @ d  # 在右奇异向量基下的系数
sig = s.copy()
# 能量占比按奇异值分档
for thr, name in [(1e-2, 'sigma<1%*smax'), (0.05, 'sigma<5%'), (0.1, 'sigma<10%')]:
    mask = sig < thr * sig[0]
    print(f"{name}: 方向数={mask.sum()}, d能量占比={np.sum(c[mask]**2)/np.sum(c**2):.4f}")
# 大奇异值方向的能量（会被A放大）
mask_big = sig >= 0.5*sig[0]
print("sigma>=50%: 方向数={}, 能量占比={:.6f}".format(mask_big.sum(), np.sum(c[mask_big]**2)/np.sum(c**2)))

# 检查 c 与 1/sigma 的关系：是否像"最小范数解修正"
# 若 x_rand = x_true + V diag(min(1/sigma, t)) U^T r 之类
# 简化：看 |c_j| * sigma_j 的分布是否近似均匀（=A d 在白噪声系数下均匀）
Ad = A @ d
print("||Ad|| =", np.linalg.norm(Ad))
print("c 的前10大 |c| 对应的 sigma 排名(降序索引):",
      np.argsort(-np.abs(c))[:10], "对应sigma:", s[np.argsort(-np.abs(c))[:10]])

# d 的自相关：是否白噪声？
dd = d.reshape(32,32) - d.reshape(32,32).mean()
num = np.sum(dd[:, :-1]*dd[:, 1:]); den = np.sum(dd*dd)
print("d 水平相邻相关系数:", num/den)
num2 = np.sum(dd[:-1, :]*dd[1:, :])
print("d 垂直相邻相关系数:", num2/den)

# 对照组：真随机噪声的 d_random = x_rand_new - x_true 的 ||A d|| 期望
rng = np.random.default_rng(1)
errs = []
for _ in range(200):
    xr = rng.uniform(0,1,1024)
    errs.append(np.linalg.norm(A @ (xr - x_true)))
print("若 d 为独立随机: ||A d|| 中位数 =", np.median(errs))

# 若把随机噪声只投到小奇异值空间：x_true + P_small(rand - mean)
thresh = 0.02
mask = s < thresh*s[0]
Vsmall = Vt[mask]  # rows
r = rng.uniform(0,1,1024); r = r - r.mean()
proj = r - Vt.T @ (Vt @ r)  # 投到mask补空间? 直接构造:
alpha = np.zeros(1024); alpha[mask] = (Vt @ r)[mask] / 1.0
x_constructed = x_true + Vt.T @ alpha
print("构造示例 ||A x_c - y|| =", np.linalg.norm(A@x_constructed - y), " ||x_c - x_true|| =", np.linalg.norm(x_constructed - x_true))

# 关键定量: x_rand 若真是独立 uniform 噪声, 其 ||Ax_rand - y|| 应该在 ~3.9; 实际0.039
# 反推: x_rand 与 y 的拟合精度相当于解空间内的点。用最小二乘检查 x_rand 是否≈ argmin
xls, *_ = np.linalg.lstsq(A, y, rcond=None)
print("lstsq 解误差 ||A xls - y|| =", np.linalg.norm(A@xls - y))
print("||x_rand - xls|| =", np.linalg.norm(x_rand - xls), " ||x_true - xls|| =", np.linalg.norm(x_true - xls))
print("x_rand 像素直方图是否均匀: KS-ish 分段计数")
cnt, _ = np.histogram(rand, bins=8, range=(0,1))
print(cnt)

# 每个奇异值下 c_j 是否≈ 与 1/sigma_j 成比例受限
ratio = np.abs(c) * s
print("|c_j|*sigma_j: 均值 {:.4f}, std {:.4f}, 最大 {:.4f}".format(ratio.mean(), ratio.std(), ratio.max()))
