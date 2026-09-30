# -*- coding: utf-8 -*-
"""14.3-3 checkpoint 体检：结构可加载性 / 训练轮数 / loss 记录"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

FOLDER = r"d:\works\deepinv\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.3-3"

for name in ['rf_final.pth', 'rf_checkpoint.pth', 'ddpm_final.pth', 'ddpm_checkpoint.pth']:
    p = os.path.join(FOLDER, name)
    ck = torch.load(p, map_location='cpu', weights_only=False)
    keys = list(ck.keys())
    tl = ck.get('train_losses', None)
    print(f"== {name} ==")
    print(f"  keys: {keys}")
    print(f"  epoch: {ck.get('epoch', 'N/A')}")
    if tl is not None:
        print(f"  train_losses 长度(epochs): {len(tl)}")
        print(f"  首3个: {[round(float(v),6) for v in tl[:3]]}")
        print(f"  末3个: {[round(float(v),6) for v in tl[-3:]]}")
    sd = ck['model_state_dict']
    n_par = sum(v.numel() for v in sd.values())
    print(f"  参数量: {n_par}, 层数: {len(sd)}")
    # 关键层形状抽查
    for k in list(sd.keys())[:3]:
        print(f"    {k}: {tuple(sd[k].shape)}")

# 验证当前代码的 SmallUNet 能否加载 final ckpt
import importlib.util
spec = importlib.util.spec_from_file_location("m1433mod", os.path.join(FOLDER, "14.3-3.py"))
# 不执行整个脚本，仅手动比对 state_dict 结构：用 torch.load 提取
ck_rf = torch.load(os.path.join(FOLDER, 'rf_final.pth'), map_location='cpu', weights_only=False)
ck_ddpm = torch.load(os.path.join(FOLDER, 'ddpm_final.pth'), map_location='cpu', weights_only=False)
same_arch = set(ck_rf['model_state_dict'].keys()) == set(ck_ddpm['model_state_dict'].keys())
print(f"\nrf 与 ddpm state_dict 键集合一致(同架构): {same_arch}")
