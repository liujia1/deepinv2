# -*- coding: utf-8 -*-
"""diag2: 追查脉冲噪声不可复现来源 + SSIM分解修复"""
import io, sys
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

n = 256
x = resize(shepp_logan_phantom(), (n, n), order=0, preserve_range=True, anti_aliasing=False)
x = x / x.max()

def seq(tag, extra_unique=False):
    np.random.seed(42)
    if extra_unique:
        _ = np.unique(np.round(x, 6))
    yg = np.clip(x + 0.1 * np.random.randn(n, n), 0, 1)
    yp = np.clip(np.random.poisson(x / 0.08).astype(np.float64) * 0.08, 0, 1)
    ys = random_noise(x, mode='s&p', amount=0.01)
    print(f"{tag}: sp PSNR={peak_signal_noise_ratio(x, ys, data_range=1.0):.3f}")
    return ys

a = seq("A 纯复现")
b = seq("B 含np.unique")
print("A==B:", np.array_equal(a, b))

# random_noise 同状态重复调用是否确定
np.random.seed(42); np.random.randn(n, n); np.random.poisson(x / 0.08)
r1 = random_noise(x, mode='s&p', amount=0.01)
np.random.seed(42); np.random.randn(n, n); np.random.poisson(x / 0.08)
r2 = random_noise(x, mode='s&p', amount=0.01)
print("同状态两次random_noise相同:", np.array_equal(r1, r2))

# 直接看skimage源码里s&p用的RNG函数
import inspect
from skimage.util import random_noise as rn
src = inspect.getsource(rn)
seg = src[src.find("'s&p'")-200: src.find("'s&p'")+600]
print("---- skimage s&p 源码片段 ----")
print(seg)

# ---- SSIM 分解（修复 full=True 返回元组）----
np.random.seed(42)
yg = np.clip(x + 0.1 * np.random.randn(n, n), 0, 1)
yp = np.clip(np.random.poisson(x / 0.08).astype(np.float64) * 0.08, 0, 1)
mcs_g, map_g = structural_similarity(x, yg, data_range=1.0, win_size=7, full=True)
mcs_p, map_p = structural_similarity(x, yp, data_range=1.0, win_size=7, full=True)
bg = (x == 0)
inner = np.zeros_like(bg); inner[20:-20, 20:-20] = True
flat = bg & inner
print(f"\n背景像素占比 {bg.mean():.2%}")
print(f"高斯  SSIM图: 背景平坦区={map_g[flat].mean():.4f} 非背景区={map_g[~flat].mean():.4f} 全图均值={map_g.mean():.4f} (skimme={mcs_g:.4f})")
print(f"泊松  SSIM图: 背景平坦区={map_p[flat].mean():.4f} 非背景区={map_p[~flat].mean():.4f} 全图均值={map_p.mean():.4f} (skimage={mcs_p:.4f})")
C3 = 0.03**2
print(f"平坦区结构项塌缩理论值 C3/(2σ²+C3), σ=0.1: {C3/(2*0.01+C3):.4f}")
