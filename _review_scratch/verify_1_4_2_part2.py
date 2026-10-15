# -*- coding: utf-8 -*-
"""实验1.4-2 复核 part2：A 的本质、y 生成、random_gray 构造取证"""
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = r"d:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.4-2"
OUT = r"d:\works\deepinv\_review_scratch"

def load_png(name):
    return np.array(Image.open(EXP + "\\" + name).convert("L")).astype(np.float64) / 255.0

A = np.load(EXP + r"\transform_A.npy").astype(np.float64)
xt = load_png("maomi_grayscale.png").ravel()
y = load_png("maomi_mohu.png").ravel()
xr = load_png("random_gray.png").ravel()
raw_y = np.array(Image.open(EXP + r"\maomi_mohu.png").convert("L"))

U, s, Vt = np.linalg.svd(A)

print("===== 奇异谱分布 =====")
pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
print("分位:", pcts)
print("σ   :", np.round(np.percentile(s, pcts), 5))
for thr in [0.5, 0.3, 0.1, 0.05, 0.0114, 1e-3, 1e-4]:
    print(f"  σ < {thr}: {(s < thr).sum()} 个")
print("Σσ² =", (s ** 2).sum())
# MP 检验：g=0.01 iid 高斯的体谱
g = A.std()
mp_lo, mp_hi = g * (np.sqrt(1024) - 1), g * (np.sqrt(1024) + 1)
print(f"iid N(μ,{g:.4f}²) 的 MP 体谱区间: [{mp_lo:.4f}, {mp_hi:.4f}]")
print("体谱(去掉最大1和最小24)范围:", s[1:-24].min(), s[1:-24].max())

print("\n===== 元素分布 =====")
from scipy import stats as st
print("skew=%.4f kurtosis=%.4f（高斯:0,0）" % (st.skew(A.ravel()), st.kurtosis(A.ravel())))
print("行内 std 的变异:", end=" ")
rs = A.std(axis=1)
print("mean=%.5f std=%.5f" % (rs.mean(), rs.std()))

print("\n===== 频率响应（对 DCT 正弦基施加 A）=====")
# 构造 2D 余弦基 f_{pq}，比较 ||A f|| 与 ||f||，并看输出是否仍是该频率
amp = np.zeros((32, 32))
leak = np.zeros((32, 32))
for p in range(32):
    for q in range(32):
        ii, jj = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
        f = np.cos(np.pi * p * (ii + 0.5) / 32) * np.cos(np.pi * q * (jj + 0.5) / 32)
        f = f.ravel()
        of = (A @ f).reshape(32, 32)
        amp[p, q] = np.linalg.norm(A @ f) / np.linalg.norm(f)
        # 输出在同频率基上的相关（判断是否只是衰减还是完全打散）
        f2 = f.reshape(32, 32)
        leak[p, q] = abs((of * f2).sum()) / (np.linalg.norm(of) * np.linalg.norm(f2) + 1e-30)
print("低频(0,0) 增益=%.3f  同频保持=%.3f" % (amp[0, 0], leak[0, 0]))
print("中频(4,4) 增益=%.3f  同频保持=%.3f" % (amp[4, 4], leak[4, 4]))
print("高频(16,16) 增益=%.3f 同频保持=%.3f" % (amp[16, 16], leak[16, 16]))
print("同频保持中位数(所有频率):", np.median(leak), " 增益中位数:", np.median(amp))

print("\n===== A 对猫图作用的定量检查 =====")
Axt = A @ xt
Axr = A @ xr
def imgstats(v, name):
    X = v.reshape(32, 32)
    print(f"{name}: mean={X.mean():.4f} std={X.std():.4f} min={X.min():.4f} max={X.max():.4f}")
imgstats(xt, "xt    ")
imgstats(y, "y     ")
imgstats(Axt, "A xt  ")
imgstats(Axr, "A xr  ")
print("corr(Axt, y) =", np.corrcoef(Axt, y)[0, 1])
print("corr(Axt, xt)=", np.corrcoef(Axt, xt)[0, 1])
# 与真正的高斯模糊对比
from scipy.ndimage import gaussian_filter
X = xt.reshape(32, 32)
best = (1e9, None)
for sig in np.linspace(0.5, 4, 36):
    B = gaussian_filter(X, sig)
    e = np.linalg.norm(B.ravel() - Axt) / np.linalg.norm(Axt)
    if e < best[0]:
        best = (e, sig)
print("与 gaussian_filter(xt,σ) 最佳相对误差 = %.4f at σ=%.2f" % best)
best2 = (1e9, None)
for sig in np.linspace(0.5, 4, 36):
    B = gaussian_filter(X, sig)
    e = np.linalg.norm(B.ravel() - y) / np.linalg.norm(y)
    if e < best2[0]:
        best2 = (e, sig)
print("y 与 gaussian_filter(xt,σ) 最佳相对误差 = %.4f at σ=%.2f" % best2)

# A 是否与对称高斯模糊矩阵相似：构造 gaussian_filter 矩阵 Bmat
def gaussian_matrix(sig):
    Bm = np.zeros((1024, 1024))
    for j in range(1024):
        e = np.zeros(1024); e[j] = 1.0
        Bm[:, j] = gaussian_filter(e.reshape(32, 32), sig).ravel()
    return Bm
for sig in [1.0, 1.5, 2.0]:
    Bm = gaussian_matrix(sig)
    rel = np.linalg.norm(A - Bm) / np.linalg.norm(A)
    print(f"||A - GaussBlur(σ={sig})||/||A|| = {rel:.4f}  (||B||_F={np.linalg.norm(Bm):.2f})")

print("\n===== 保存可视化 =====")
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
ax[0, 0].imshow(Axt.reshape(32, 32), cmap="gray"); ax[0, 0].set_title("A @ x_true")
ax[0, 1].imshow(y.reshape(32, 32), cmap="gray"); ax[0, 1].set_title("y (maomi_mohu)")
ax[0, 2].imshow((y - Axt).reshape(32, 32), cmap="gray"); ax[0, 2].set_title("y - A xt (放大可见)")
ax[0, 3].imshow(Axr.reshape(32, 32), cmap="gray"); ax[0, 3].set_title("A @ x_random")
for k, idx in enumerate([0, 256, 512, 1000]):
    ax[1, k].imshow(A[idx].reshape(32, 32), cmap="RdBu_r"); ax[1, k].set_title(f"A 的第 {idx} 行")
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig(OUT + r"\diag_A_action.png", dpi=110); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
bm = A.reshape(32, 32, 32, 32).mean(axis=(1, 3))
ax[0].imshow(bm, cmap="RdBu_r"); ax[0].set_title("32x32 块均值 [i,k]")
ax[1].imshow(amp, cmap="viridis"); ax[1].set_title("频率增益 ||A f||/||f||")
ax[2].imshow(leak, cmap="viridis", vmin=0, vmax=1); ax[2].set_title("同频保持 |corr|")
plt.tight_layout(); plt.savefig(OUT + r"\diag_A_spectrum.png", dpi=110); plt.close()

print("\n===== y 生成方式 =====")
r = y - Axt
z = r / r.std()
print("||r||=%.5f 逐像素std=%.6f mean=%.2e max|.|=%.5f" % (np.linalg.norm(r), r.std(), r.mean(), np.abs(r).max()))
for k in (1, 2, 3):
    print(f"  |z|<{k}σ: {(np.abs(z) < k).mean():.3f}")
qonly = np.round(Axt * 255) / 255 - Axt
print("若 y=量化(Axt): 纯量化残差范数=%.5f" % np.linalg.norm(qonly))
print("y 唯一灰度级数:", len(np.unique(raw_y)), " 范围:", raw_y.min(), raw_y.max())

print("\n===== random_gray 构造取证 =====")
e = y - Axt
Ad = A @ (xr - xt)
alpha = float(Ad @ e / (e @ e))
print("三角等号: ||Ad||=%.8f  |et-er|=%.8f" % (np.linalg.norm(Ad), abs(np.linalg.norm(e) - np.linalg.norm(y - Axr))))
print("Ad=αe 共线: α=%.6f 相对偏离=%.3e" % (alpha, np.linalg.norm(Ad - alpha * e) / np.linalg.norm(Ad)))
xls = np.linalg.lstsq(A, y, rcond=None)[0]
blend = (1 - alpha) * xt + alpha * xls
print("混合前越界比例=%.3f 范围[%.3f,%.3f]" % (((blend < 0) | (blend > 1)).mean(), blend.min(), blend.max()))
recon = np.round(np.clip(blend, 0, 1) * 255) / 255
print("clip+量化后逐像素完全一致比例: %.5f" % (recon == xr).mean())
print("最大像素差异:", np.abs(recon - xr).max())
raw_xr = np.array(Image.open(EXP + r"\random_gray.png").convert("L"))
print("random_gray 饱和(0/255)比例=%.3f" % (((raw_xr == 0) | (raw_xr == 255)).mean()))
h = xls - xt
beta = float((xr - xt) @ h / (h @ h))
print("d=β(xls-xt): β=%.6f 余弦=%.8f 残差范数=%.5f" %
      (beta, np.linalg.norm((xr - xt) @ h) / (np.linalg.norm(xr - xt) * np.linalg.norm(h)),
       np.linalg.norm(xr - xt - beta * h)))
# 纯零空间投影检验（24维）
mask = s < 0.01 * s[0]
Vs = Vt[mask]
d = xr - xt
pl = d - Vs.T @ (Vs @ d)
print("d 落在24维近零空间正交补的范数=%.5f（对比 d 范数 %.4f，量化噪声 %.4f）" %
      (np.linalg.norm(pl), np.linalg.norm(d), np.sqrt(1024 / 12) / 255))
# 用 24 维能否精确表示（量化前）
print("24维张成解释 d 的能量比例:", 1 - (pl @ pl) / (d @ d))

print("\n===== 误差放大理论核对 =====")
ne = U.T @ e
xerr = Vt @ (xls - xt)
print("谱恒等式 xerr_j == ne_j/σ_j 最大偏差:",
      np.abs(xerr - ne / s).max() / np.abs(xerr).max())
top = np.argsort(np.abs(xerr))[-6:][::-1]
print("误差能量最大的6个方向 j: σ_j, |u_j^T e|, |xerr_j|")
for j in top:
    print(f"  j={j:4d} σ={s[j]:.3e} 噪声投影={abs(ne[j]):.3e} 误差分量={abs(xerr[j]):.3f}")
kappa = s[0] / s[-1]
amp_real = (np.linalg.norm(xls - xt) / np.linalg.norm(xt)) / (np.linalg.norm(e) / np.linalg.norm(y))
print(f"实际放大倍数={amp_real:.1f}, κ上界={kappa:.1f}, 相对噪声={np.linalg.norm(e)/np.linalg.norm(y):.2e}")

print("\n===== 随机方向落入近零空间的概率 =====")
from scipy.stats import beta as B
Fd = B(24 / 2, (1024 - 24) / 2)
for f0 in [0.5, 0.9, 0.99, 0.9999]:
    print(f"  P(F >= {f0}) = {1 - Fd.cdf(f0):.3e}")
print("done")
