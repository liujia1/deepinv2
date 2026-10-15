# -*- coding: utf-8 -*-
"""part4: y 来源像素级取证 + 块结构/可视化数值核对"""
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

EXP = r"d:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.4-2"
def load_raw(n): return np.array(Image.open(EXP + "\\" + n).convert("L"))
A = np.load(EXP + r"\transform_A.npy").astype(np.float64)
raw_xt = load_raw("maomi_grayscale.png"); raw_y = load_raw("maomi_mohu.png")
xt = raw_xt.astype(np.float64) / 255
y = raw_y.astype(np.float64) / 255
Axt = A @ xt.ravel()

# 1) y 是否就是 round(A xt)？
qA = np.round(Axt * 255).clip(0, 255).reshape(32, 32)
print("y vs round(255*A@xt): 完全相同像素比例 =", (qA == raw_y).mean())
diff = qA.astype(int) - raw_y.astype(int)
vals, cnt = np.unique(diff, return_counts=True)
print("  差值分布(灰度级:个数):", dict(zip(vals.tolist(), cnt.tolist())))

# 2) y 是否独立用 gaussian_filter 生成？
best = None
for sig in np.arange(0.6, 2.01, 0.05):
    for mode in ["nearest", "reflect", "constant"]:
        B = gaussian_filter(xt, sig, mode=mode)
        qB = np.round(B * 255).clip(0, 255)
        acc = (qB == raw_y).mean()
        if best is None or acc > best[0]:
            best = (acc, sig, mode)
print("y vs round(255*gaussian_filter): 最佳完全相同比例 = %.4f at σ=%.2f mode=%s" % best)

# 3) PIL 风格高斯模糊（半径离散核）尝试
try:
    from PIL import ImageFilter
    im = Image.fromarray(raw_xt)
    for rad in [1, 2, 3, 4]:
        bp = np.array(im.filter(ImageFilter.GaussianBlur(radius=rad)))
        print(f"  PIL GaussianBlur({rad}): 完全相同比例 = {(bp == raw_y).mean():.4f}")
except Exception as ex:
    print("PIL test skipped:", ex)

# 4) 差异像素位置：边界还是边缘
di = np.argwhere(diff != 0)
print("不一致像素数:", len(di), " 其中位于图像外两圈的比例:",
      ((di[:, 0] < 2) | (di[:, 0] > 29) | (di[:, 1] < 2) | (di[:, 1] > 29)).mean())

# 5) 块均值统计（解释原图 A 可视化）
A4 = A.reshape(32, 32, 32, 32)
bm = A4.mean(axis=(1, 3))
print("块均值: mean=%.3e std=%.3e min=%.3e max=%.3e" % (bm.mean(), bm.std(), bm.min(), bm.max()))
Az = zoom(A, 32/1024, order=1)
print("zoom 输出: std=%.3e min=%.3e max=%.3e" % (Az.std(), Az.min(), Az.max()))
# A 元素的块内/块间方差分解
grand = A.mean()
between = (((A4.mean(axis=(2,3)) - grand)**2).sum())
within = ((A4 - A4.mean(axis=(2,3), keepdims=True))**2).sum()
tot = ((A - grand)**2).sum()
print("方差分解: 块间=%.1f%% 块内=%.1f%%" % (100*between/tot, 100*within/tot))

# 6) 行和（常值输入的响应）形状
rs = A.sum(axis=1).reshape(32,32)
print("行和图 vs y 的 corr =", np.corrcoef(rs.ravel(), y.ravel())[0,1])
print("行和图 vs round(Axt) corr =", np.corrcoef(rs.ravel(), qA.ravel())[0,1])

# 7) 代码 line144 口径核对：新噪声的 ||A(u-xt)|| 与 E_mc=||Au-y|| 是否真的差不多
rng = np.random.default_rng(2026)
Xmc = rng.uniform(0, 1, size=(1024, 500))
Emc = np.linalg.norm(A @ Xmc - y[:, None], axis=0)
Ealt = np.linalg.norm(A @ (Xmc - xt[:, None]), axis=0)
print("MC ||Au-y|| 中位=%.4f  vs  ||A(u-xt)|| 中位=%.4f  (比值 %.3f)"
      % (np.median(Emc), np.median(Ealt), np.median(Emc)/np.median(Ealt)))
