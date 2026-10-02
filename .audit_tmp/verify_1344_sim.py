# -*- coding: utf-8 -*-
import numpy as np
from skimage.metrics import structural_similarity as compute_ssim

def _sobel_edges(a):
    try:
        from skimage.filters import sobel
        e = sobel(a.astype(float))
    except Exception:
        dx = np.abs(np.diff(a.astype(float), axis=1))
        dy = np.abs(np.diff(a.astype(float), axis=0))
        e = np.zeros_like(a, dtype=float)
        e[:, 1:] += dx; e[:, :-1] += dx
        e[1:, :] += dy; e[:-1, :] += dy
    return (e > 0.05).astype(bool)

def _edge_iou(a, b):
    ea, eb = _sobel_edges(a), _sobel_edges(b)
    inter = np.logical_and(ea, eb).sum()
    union = np.logical_or(ea, eb).sum()
    return float(inter) / float(union) if union > 0 else 0.0

# 完全复刻 13.4-4.py 步骤2 的模拟数据
np.random.seed(42)
depth_map_sim = np.zeros((100, 100))
depth_map_sim[20:80, 20:80] = 0.5
depth_map_sim[30:70, 30:70] = 0.7
depth_map_sim[40:60, 40:60] = 0.9

np.random.seed(43)
ctrlnet_result_sim = np.zeros((100, 100))
ctrlnet_result_sim[20:80, 20:80] = 0.6 + np.random.randn(60, 60) * 0.05
ctrlnet_result_sim[30:70, 30:70] = 0.75 + np.random.randn(40, 40) * 0.05
ctrlnet_result_sim[40:60, 40:60] = 0.9 + np.random.randn(20, 20) * 0.05

np.random.seed(44)
xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
cfg_base = (xx + yy) / 2
cfg_result_sim = cfg_base + np.random.randn(100, 100) * 0.08
cfg_result_sim = np.clip(cfg_result_sim, 0, 1)

np.random.seed(45)
none_result = np.random.rand(100, 100)

ssim_ctrlnet = compute_ssim(depth_map_sim, ctrlnet_result_sim, data_range=1.0)
ssim_cfg = compute_ssim(depth_map_sim, cfg_result_sim, data_range=1.0)
ssim_none = compute_ssim(depth_map_sim, none_result, data_range=1.0)
iou_ctrlnet = _edge_iou(depth_map_sim, ctrlnet_result_sim)
iou_cfg = _edge_iou(depth_map_sim, cfg_result_sim)

print("SSIM_ControlNet=", round(float(ssim_ctrlnet), 4))
print("SSIM_CFG=", round(float(ssim_cfg), 4))
print("SSIM_none=", round(float(ssim_none), 4))
print("IoU_ControlNet=", round(float(iou_ctrlnet), 4))
print("IoU_CFG=", round(float(iou_cfg), 4))
