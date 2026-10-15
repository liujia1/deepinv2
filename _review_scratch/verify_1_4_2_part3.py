# -*- coding: utf-8 -*-
"""part3: A 是通用模糊算子，还是只为这只猫定制的算子？"""
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

EXP = r"d:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.4-2"
def load_png(n): return np.array(Image.open(EXP + "\\" + n).convert("L")).astype(np.float64) / 255.0

A = np.load(EXP + r"\transform_A.npy").astype(np.float64)
xt = load_png("maomi_grayscale.png").ravel()
y = load_png("maomi_mohu.png").ravel()
n = 1024

ii, jj = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
def dct_mode(p, q):
    f = np.cos(np.pi * p * (ii + 0.5) / 32) * np.cos(np.pi * q * (jj + 0.5) / 32)
    return f.ravel()

print("===== 最低若干 DCT 模式的增益/保持 =====")
print(" mode   gain    corr_same")
for p, q in [(0,0),(1,0),(0,1),(1,1),(2,0),(0,2),(2,2),(3,1),(3,3),(5,5),(8,8)]:
    f = dct_mode(p, q)
    of = A @ f
    gain = np.linalg.norm(of) / np.linalg.norm(f)
    c = abs(of @ f) / (np.linalg.norm(of) * np.linalg.norm(f))
    print(f" ({p},{q})  {gain:.4f}  {c:.4f}")

print("\n===== 平移/翻转等变性（真模糊算子必须满足）=====")
def roll2(v, di, dj):
    return np.roll(np.roll(v.reshape(32,32), di, 0), dj, 1).ravel()
tests = {
    "右移3": roll2(xt, 0, 3),
    "下移4": roll2(xt, 4, 0),
    "水平翻转": xt.reshape(32,32)[:, ::-1].ravel(),
    "垂直翻转": xt.reshape(32,32)[::-1, :].ravel(),
}
for name, xs in tests.items():
    out = A @ xs
    ref_blur = gaussian_filter(xs.reshape(32,32), 1.0).ravel()
    c_blur = np.corrcoef(out, ref_blur)[0,1]
    c_const = abs(out - out.mean())
    print(f"{name}: 输出std={out.std():.4f}, corr(输出, 该图σ1模糊)={c_blur:.4f}, 输出零均值范数={np.linalg.norm(c_const):.3f}")

print("\n===== 其他平滑图像：高斯斑 / 低频正弦 / 平滑随机图 =====")
rng = np.random.default_rng(0)
blob = np.exp(-((ii-10)**2 + (jj-16)**2) / (2*5.0**2))
sine = 0.5 + 0.3*np.cos(2*np.pi*3*ii/32)
smooth_rand = gaussian_filter(rng.uniform(size=(32,32)), 3.0)
for name, img in [("高斯斑", blob), ("3周期正弦", sine), ("平滑噪声", smooth_rand)]:
    v = img.ravel()
    out = A @ v
    ref = gaussian_filter(img, 1.0).ravel()
    print(f"{name}: 输入std={v.std():.3f} 输出std={out.std():.4f} corr(输出,σ1模糊)={np.corrcoef(out, ref)[0,1]:.4f} "
          f"corr(输出,平坦)={np.corrcoef(out, np.full(n, v.mean()))[0,1]:.4f}")

print("\n===== A 是否含“猫专用”秩通道 =====")
# 去掉常数秩1通道后，A 在“中心化猫”方向上的作用
m = np.ones(n) / np.sqrt(n)
xtc = xt - (xt @ m) * m
yc = y - (y @ m) * m
Ac = A - np.outer(A @ m, m)  # 去掉常数列通道的近似
# 直接检验：A 把 xtc 映射到 yc 的倍数？
v = A @ xtc
print("||A xtc||=%.4f ||yc||=%.4f corr=%.6f 标量比=%.6f" %
      (np.linalg.norm(v), np.linalg.norm(yc), np.corrcoef(v, yc)[0,1], (v @ yc)/(yc @ yc)))
# 与 xtc 正交的零均值向量经过 A 的放大
w = rng.standard_normal(n); w -= w @ m * m; w -= (w @ xtc)/(xtc@xtc) * xtc; w /= np.linalg.norm(w)
print("随机正交零均值方向经 A 的增益 ||A w||=%.4f（对比猫方向 ||A xtc||/||xtc||=%.4f）"
      % (np.linalg.norm(A @ w), np.linalg.norm(A @ xtc)/np.linalg.norm(xtc)))
# 对一系列幅度 t: A (mean + t*xtc) 是否≈ mean-blur + t*yc 线性成立
for t in [-1.0, 0.5, 2.0]:
    xs = xt + t * xtc
    out = A @ xs
    pred = A @ xt + t * (A @ xtc)
    print(f"  线性自洽 t={t}: ||差||=%.2e" % np.linalg.norm(out - pred))

print("\n===== 零空间方向长什么样（v_1023 等）=====")
U, s, Vt = np.linalg.svd(A)
for j in [1023, 1022, 1021, 1000]:
    v = Vt[j].reshape(32,32)
    # 空间频率粗估：与 DCT 模式最大相关
    best = (0, None)
    for p in range(32):
        for q in range(32):
            c = abs((v.ravel()) @ dct_mode(p,q)) / np.linalg.norm(v) / np.linalg.norm(dct_mode(p,q))
            if c > best[0]: best = (c, (p,q))
    print(f"v_{j}: σ={s[j]:.2e} 最接近 DCT 模式 {best[1]} corr={best[0]:.3f}")

print("\n===== 概率对数尾部 =====")
from scipy.stats import beta as B
Fd = B(12, 500)
for f0 in [0.5, 0.9, 0.99, 0.9999, 0.99999]:
    print(f"  log10 P(F >= {f0}) = {Fd.logsf(f0)/np.log(10):.1f}")
# 要达到真值拟合水平需要 F 多高（粗略）：残余需 <=0.045
med = 4.264677953
need = 1 - (0.045/med)**2
print(f"粗略要求 F >= 1-(0.045/{med:.2f})^2 = {need:.6f}")
print(f"  log10 P(F >= {need:.6f}) = {Fd.logsf(need)/np.log(10):.1f}")
