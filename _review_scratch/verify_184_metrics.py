# -*- coding: utf-8 -*-
"""独立复核 18.4-1 缓存：
1) 用本地 celeba + 种子42 重建全部观测 y，并与缓存中的基线图(A^T y / 插值)逐像素对比；
2) 对缓存中每张重建图独立重算 PSNR/SSIM，与 JSON 声称值对比。
环境：torch 2.11+cpu / deepinv 0.4.2
"""
import os, sys, pickle
import numpy as np
import torch
import deepinv as dinv
from deepinv.physics import Downsampling, BlurFFT, Inpainting, GaussianNoise
from deepinv.utils import load_image
import torch.nn.functional as F

print("deepinv:", dinv.__version__)
EXP_DIR = r"d:\works\deepinv\第18章 综合项目：用扩散模型求解自定义逆问题\第十八章 配套实验\实验18.4-1"
torch.manual_seed(42); np.random.seed(42)

x_true = load_image(os.path.join(EXP_DIR, "celeba_example.jpg"), img_size=(256, 256), resize_mode='resize')
if x_true.ndim == 3:
    x_true = x_true.unsqueeze(0)
print("x_true:", tuple(x_true.shape), float(x_true.min()), float(x_true.max()))

def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return 10 * np.log10(1.0 / mse)

def ssim(a, b):
    from skimage.metrics import structural_similarity as s
    return s(a.squeeze().permute(1, 2, 0).numpy().clip(0, 1),
             b.squeeze().permute(1, 2, 0).numpy().clip(0, 1), channel_axis=2, data_range=1.0)

# 严格按 18.4-1.py Step1 的顺序重建场景（含 RNG 调用顺序）
from deepinv.physics.functional import gaussian_blur as gbf
light_kernel = gbf(sigma=(1.5, 1.5)); heavy_kernel = gbf(sigma=(3.0, 3.0))
light_blur = BlurFFT(img_size=(3, 256, 256), filter=light_kernel); light_blur.set_noise_model(GaussianNoise(sigma=0.01))
heavy_blur = BlurFFT(img_size=(3, 256, 256), filter=heavy_kernel); heavy_blur.set_noise_model(GaussianNoise(sigma=0.05))
y_light = light_blur(x_true); y_heavy = heavy_blur(x_true)
sr2 = Downsampling(factor=2, filter='bicubic', img_size=(3, 256, 256)); sr2.set_noise_model(GaussianNoise(sigma=0.01))
sr4 = Downsampling(factor=4, filter='bicubic', img_size=(3, 256, 256)); sr4.set_noise_model(GaussianNoise(sigma=0.01))
y_sr2 = sr2(x_true); y_sr4 = sr4(x_true)
inp50 = Inpainting(img_size=(3, 256, 256), mask=0.5); inp50.set_noise_model(GaussianNoise(sigma=0.01))
inp80 = Inpainting(img_size=(3, 256, 256), mask=0.2); inp80.set_noise_model(GaussianNoise(sigma=0.01))
y_inp50 = inp50(x_true); y_inp80 = inp80(x_true)
print("mask50 keep=", float(inp50.mask.mean()), " mask80 keep=", float(inp80.mask.mean()))

phys_map = {'轻度模糊': (light_blur, y_light, 'adj'), '重度模糊': (heavy_blur, y_heavy, 'adj'),
            '2倍超分': (sr2, y_sr2, 'interp'), '4倍超分': (sr4, y_sr4, 'interp'),
            '50%修复': (inp50, y_inp50, 'adj'), '80%修复': (inp80, y_inp80, 'adj')}

with open(os.path.join(EXP_DIR, "experiment_cache.pkl"), "rb") as f:
    cache = pickle.load(f)

# ---- (1) 基线逐像素比对 ----
print("\n=== 观测/基线复现一致性（重建的 A^T y vs 缓存基线图）===")
results = {}
for step in ("step2_results", "step3_results", "step4_results"):
    for sname, methods in cache.get(step, {}).items():
        results.setdefault(sname, {}).update(methods)

for sname, (phys, y, kind) in phys_map.items():
    if kind == 'interp':
        base = F.interpolate(y, size=(256, 256), mode='bilinear', align_corners=False)
        cname = '插值基线'
    else:
        base = phys.A_adjoint(y)
        cname = '伴随重建'
    cached = results[sname].get(cname)
    if cached is None:
        print(sname, "缓存缺基线", cname); continue
    d = (base - cached['img']).abs()
    obs_psnr = psnr(x_true, y) if y.shape[-1] == 256 else float('nan')
    print(f"{sname}: max|Δ|={float(d.max()):.3e} mean|Δ|={float(d.mean()):.3e}  "
          f"观测y PSNR(x,y)={obs_psnr:.2f}")

# ---- (2) 全部缓存重建图指标独立复算 ----
print("\n=== 缓存重建图 PSNR/SSIM 独立复算 vs JSON声称 ===")
import json
with open(os.path.join(EXP_DIR, "results_summary.json"), encoding='utf-8') as f:
    js = json.load(f)
for sname, methods in results.items():
    for mname, m in methods.items():
        img = m['img']
        p = psnr(x_true, img); s = ssim(x_true, img)
        j = js[sname][mname]
        flag = "OK" if abs(p - j['psnr']) < 0.05 and abs(s - j['ssim']) < 0.002 else "!! MISMATCH"
        extra = ""
        if 'psnr_mmse' in m:
            extra = f" | MMSE图重算={p:.2f} 声称psnr_mmse={m['psnr_mmse']:.2f}"
        print(f"[{sname}] {mname:14s} 重算PSNR={p:6.2f} json={j['psnr']:6.2f}  "
              f"重算SSIM={s:.4f} json={j['ssim']:.4f} {flag}{extra}")
