# -*- coding: utf-8 -*-
"""检查 18.4-1 缓存：版本、指标、重建图、重算PSNR核对"""
import os, sys, pickle
import numpy as np
import torch

EXP_DIR = r"d:\works\deepinv\第18章 综合项目：用扩散模型求解自定义逆问题\第十八章 配套实验\实验18.4-1"
pkl_path = os.path.join(EXP_DIR, "experiment_cache.pkl")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("pkl top-level keys:", list(data.keys()))
print("cache_version:", data.get("cache_version"))

for step in ("step2_results", "step3_results", "step4_results"):
    block = data.get(step, {})
    print(f"\n===== {step} ({len(block)} scenarios) =====")
    for sname, methods in block.items():
        for mname, m in methods.items():
            img = m.get("img")
            extra = ""
            for k in ("psnr_std", "psnr_mmse", "n_samples", "time_total"):
                if k in m:
                    extra += f" {k}={m[k]}"
            shape = tuple(img.shape) if isinstance(img, torch.Tensor) else None
            rng = (float(img.min()), float(img.max())) if isinstance(img, torch.Tensor) else None
            print(f"  [{sname}] {mname}: psnr={m['psnr']:.3f} ssim={m.get('ssim'):.4f} "
                  f"time={m['time']:.3f} img={shape} range={rng}{extra}")
