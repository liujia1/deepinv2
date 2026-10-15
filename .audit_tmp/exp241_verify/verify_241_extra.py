# -*- coding: utf-8 -*-
"""补充核查: (a) 滤波器两两相关性 (b) 联合惩罚能量图集中度: 学得的 vs 随机滤波器 vs TV"""
import sys, os
import numpy as np
import torch
import torch.nn.functional as F
sys.stdout.reconfigure(encoding='utf-8')
EXP_DIR = r"d:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\第二章 配套实验\实验2.4-1"
sys.path.insert(0, EXP_DIR)
import importlib.util
spec = importlib.util.spec_from_file_location("exp241", os.path.join(EXP_DIR, "2.4-1.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
torch.set_num_threads(1)
device = torch.device('cpu')

ck = torch.load(os.path.join(EXP_DIR, 'checkpoints', 'foe_final.pth'), map_location=device, weights_only=False)
m = mod.build_model(); m.load_state_dict(ck['model_state_dict']); m.eval()
x_gt = mod.create_rich_test_image(64)

def concentration(penalty_map):
    """结构处(真值梯度非零像素)能量占比 / 均匀占比"""
    gx = x_gt[:, :, :, 1:] - x_gt[:, :, :, :-1]
    gy = x_gt[:, :, 1:, :] - x_gt[:, :, :-1, :]
    struct = ((gx[:, :, :63, :] + gy[:, :, :, :63]).abs() > 1e-6).float()  # 均取前63行/列 → (1,1,63,63)
    struct_full = torch.zeros(1, 1, 64, 64)
    struct_full[:, :, 1:, 1:] = struct
    m_ = penalty_map.reshape(1, 1, 64, 64)
    frac_struct = float(m_[struct_full > 0].sum() / m_.sum())
    frac_area = float((struct_full > 0).float().mean())
    return frac_struct, frac_area

with torch.no_grad():
    fl = m.get_normalized_filters(); al = m.alphas.view(1, -1, 1, 1)
    resp = F.conv2d(x_gt, fl, padding=3)
    pen_learned = (al * torch.log(1 + resp ** 2)).sum(dim=1)  # 1,64,64
    print("[a] 滤波器两两 |cos| (归一化后, 中心区域):")
    Ff = fl.numpy()[:, 0].reshape(6, 49)
    Ff = Ff / np.linalg.norm(Ff, axis=1, keepdims=True)
    C = np.abs(Ff @ Ff.T)
    print(np.array2string(C, precision=2, suppress_small=True))
    off = C[~np.eye(6, dtype=bool)]
    print(f"  非对角 |cos| 均值={off.mean():.3f}, 最大={off.max():.3f}")

    print("[b] 联合惩罚能量在结构(线/框像素)处的集中度:")
    fs, fa = concentration(pen_learned)
    print(f"  学得的FoE: 结构处能量占比={fs:.3f}, 结构面积占比={fa:.3f} → 集中比={fs/fa:.2f}")
    # 随机滤波器对照（与 model_before 相同的 seed42 初始化）
    torch.manual_seed(42); np.random.seed(42)
    mr = mod.FoEModule(6, 7, 1.0, 0.01)
    flr = mr.get_normalized_filters(); alr = mr.alphas.view(1, -1, 1, 1)
    rr = F.conv2d(x_gt, flr, padding=3)
    pen_rand = (alr * torch.log(1 + rr ** 2)).sum(dim=1)
    fs_r, _ = concentration(pen_rand)
    print(f"  随机FoE:  结构处能量占比={fs_r:.3f} → 集中比={fs_r/fa:.2f}")
    # TV 对照
    gx = (x_gt[:, :, :, 1:] - x_gt[:, :, :, :-1]).abs()
    gy = (x_gt[:, :, 1:, :] - x_gt[:, :, :-1, :]).abs()
    tv = F.pad(gx[:, :, :63, :] + gy[:, :, :, :63], (1, 0, 1, 0))
    fs_tv, _ = concentration(tv)
    print(f"  TV(|∇x|): 结构处能量占比={fs_tv:.3f} → 集中比={fs_tv/fa:.2f}")
    # 均值模糊残差(高频)对照：结构处天然高，说明该现象是否平凡
    blur = F.avg_pool2d(F.pad(x_gt, (3, 3, 3, 3), mode='reflect'), 7, stride=1)
    hp = (x_gt - blur).abs()
    fs_hp, _ = concentration(hp)
    print(f"  高通|x-avg7(x)|: 结构处能量占比={fs_hp:.3f} → 集中比={fs_hp/fa:.2f}  (若≈学得的FoE则'结构处能量高'是平凡现象)")
