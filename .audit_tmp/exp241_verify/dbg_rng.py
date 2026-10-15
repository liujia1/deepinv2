# -*- coding: utf-8 -*-
"""定位 2.4-1.py 中 y3(图3噪声) 的 RNG 消耗点：逐步打印 RNG 状态指纹与 PSNR"""
import sys, os
import numpy as np
import torch
sys.stdout.reconfigure(encoding='utf-8')
EXP_DIR = r"d:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\第二章 配套实验\实验2.4-1"
sys.path.insert(0, EXP_DIR)
os.chdir(EXP_DIR)
import importlib.util
spec = importlib.util.spec_from_file_location("exp241_dbg", os.path.join(EXP_DIR, "2.4-1.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def rng_fp(tag):
    g = torch.random.get_rng_state()
    print(f"  [RNG] {tag}: sha1={__import__('hashlib').sha1(g.numpy().tobytes()).hexdigest()[:12]}")

def psnr(a, b):
    return mod.compute_psnr(a, b)

x_gt_test = mod.create_rich_test_image(64).to(mod.device)
rng_fp("import+create_test 之后(顶层seed42)")

torch.manual_seed(42); np.random.seed(42)
train_set = mod.SyntheticRichDataset(num_samples=128, size=64, seed_offset=0)
rng_fp("train_set 构造后")
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
rng_fp("DataLoader 构造后")

model_after = mod.build_model()
rng_fp("build_model(model_after) 后")
ckpt = torch.load(os.path.join(EXP_DIR, 'checkpoints', 'foe_final.pth'), map_location=mod.device)
model_after.load_state_dict(ckpt['model_state_dict'])
rng_fp("torch.load+load_state_dict 后")

torch.manual_seed(42); np.random.seed(42)
model_before = mod.build_model()
rng_fp("model_before 重建(重seed42)后")

y2 = mod.degrade(x_gt_test, 0.12)
print(f"  y2 PSNR={psnr(y2, x_gt_test):.4f}  (归档 JSON=18.4953)")
rng_fp("y2 抽样后")
y3 = mod.degrade(x_gt_test, 0.12)
print(f"  y3 PSNR={psnr(y3, x_gt_test):.4f}  (本次重跑图3实测=20.04 / 归档旧图=20.01)")

# 对照：如果 model_before 不重新 seed（即跳过 manual_seed(42)），y3 会是多少
print("  → 若 y2 匹配而 y3≈20.04，则 20.04 就是当前代码的确定值；旧图 20.01 来自旧代码 RNG 时序")
