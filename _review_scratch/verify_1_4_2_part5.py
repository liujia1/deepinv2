# -*- coding: utf-8 -*-
"""part5: A 的猫专用通道定量 + MC口径核对（修复版）"""
import numpy as np
from PIL import Image

EXP = r"d:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.4-2"
def load_png(n): return np.array(Image.open(EXP + "\\" + n).convert("L")).astype(np.float64) / 255.0
A = np.load(EXP + r"\transform_A.npy").astype(np.float64)
xt = load_png("maomi_grayscale.png").ravel()
y = load_png("maomi_mohu.png").ravel()
n = 1024
r = A @ np.ones(n)
A0 = A - np.outer(r, np.ones(n)) / n
print("||A0 @ xt|| =", np.linalg.norm(A0 @ xt), "（若≈0 说明猫只经均值通道响应）")
print("||A0 @ 1|| =", np.linalg.norm(A0 @ np.ones(n)))
m = np.ones(n)/np.sqrt(n)
w = xt - (xt @ m)*m
print("||xtc|| =", np.linalg.norm(w))
# A0 谱
s0 = np.linalg.svd(A0, compute_uv=False)
print("A0 奇异值: max=%.4f 中位=%.4f min=%.2e; <0.1个数=%d; <0.05个数=%d; <0.0114个数=%d"
      % (s0[0], np.median(s0), s0[-1], (s0<0.1).sum(), (s0<0.05).sum(), (s0<0.0114).sum()))

# MC 口径
rng = np.random.default_rng(2026)
Xmc = rng.uniform(0,1,size=(n,500))
Emc = np.linalg.norm(A @ Xmc - y[:,None], axis=0)
Ealt = np.linalg.norm(A @ (Xmc - xt[:,None]), axis=0)
print("MC ||Au-y|| 中位=%.4f  ||A(u-xt)|| 中位=%.4f 比值=%.4f"
      % (np.median(Emc), np.median(Ealt), np.median(Emc)/np.median(Ealt)))

# y 与 mean(xt)*r 的关系（r 通道单独产生的 y 近似）
yr = (y.sum()/n) * r / r.mean()
print("r 通道重建 y 相对误差=%.4f, corr=%.5f" % (np.linalg.norm(yr-y)/np.linalg.norm(y),
      np.corrcoef(yr, y)[0,1]))
