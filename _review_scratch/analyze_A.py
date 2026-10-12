# -*- coding: utf-8 -*-
"""A 阶段产物快速分析：后验 std 的空间结构、RMSE 曲线平台、均值与真值残差结构"""
import numpy as np
from skimage.data import camera
from skimage.transform import resize

OUT = r"d:\works\deepinv\_review_scratch"
M = np.load(OUT + "/A_mean.npy")
S = np.load(OUT + "/A_std.npy")
curve = np.load(OUT + "/A_rmse.npy")
x = resize(camera().astype(np.float64), (128, 128), anti_aliasing=True)

print(f"RMSE 曲线: burn后首值={curve[0]:.3f}, 1000={curve[1000]:.3f}, "
      f"3000={curve[3000]:.3f}, 末值={curve[-1]:.3f}")
print(f"后1000点 RMSE 漂移: {curve[1000]-curve[-1]:+.3f} (相对首降 {curve[0]-curve[-1]:.3f})")

# 用梯度模长把像素分成 边缘带 / 平坦区
gx = np.abs(np.diff(x, axis=0, prepend=x[:1, :]))
gy = np.abs(np.diff(x, axis=1, prepend=x[:, :1]))
g = gx + gy
edge = g > np.percentile(g, 90)
flat = g < np.percentile(g, 30)
print(f"\n后验 std: 全体均值={S.mean():.3f} | 边缘带(top10%)={S[edge].mean():.3f} | "
      f"平坦区(bot30%)={S[flat].mean():.3f} | 边缘/平坦={S[edge].mean()/S[flat].mean():.1f}x")

err = M - x
print(f"\n均值偏差: RMSE={np.sqrt((err**2).mean()):.3f} | 偏差均值={err.mean():+.3f} | "
      f"边缘RMSE={np.sqrt((err[edge]**2).mean()):.2f} | 平坦RMSE={np.sqrt((err[flat]**2).mean()):.2f}")
print(f"均值范围 [{M.min():.1f},{M.max():.1f}]；负像素占比={(M<0).mean()*100:.2f}%；>255占比={(M>255).mean()*100:.3f}%")

# MMSE 在平坦区是 残噪声vs过平滑 的权衡：看平坦区均值的局部起伏（应远小于原噪声sigma=6.55）
def local_roughness(a):
    d = np.diff(a, axis=0)
    return np.sqrt((d**2).mean())
print(f"\n平坦区相邻像素差RMS: 真值={local_roughness(x[flat.reshape(x.shape)] if False else x):.3f}, "
      f"MMSE={local_roughness(M):.3f}, 观测噪声参考sigma=6.55")
