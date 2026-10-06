# -*- coding: utf-8 -*-
"""复核脚本3：
C) 存档图1b/图2/图3 背景灰度测量 → 判定其生成约定（[0,1]直接显示 or [-1,1]显示）
D) 模拟当前代码路径（旧checkpoint + [-1,1]起点采样）会产出什么
E) 模拟旧代码路径复现存档图2（旧checkpoint + [0,1]起点采样, 直接显示）
"""
import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

EXP = r"D:\works\deepinv\第11章 扩散模型：变分视角\第十一章 配套实验\实验11.4-1"
OUT = r"D:\works\deepinv\.audit_tmp\exp1141_verify"
os.makedirs(OUT, exist_ok=True)
device = torch.device('cpu')

# ---- C) 测量存档PNG子图块的中位灰度 ----
def tile_stats(path, grid):
    arr = np.array(Image.open(path).convert('L'), dtype=float) / 255.0
    H, W = arr.shape
    rows = np.array_split(np.arange(H), grid[0])
    cols = np.array_split(np.arange(W), grid[1])
    out = []
    for rr in rows:
        for cc in cols:
            sub = arr[rr[0]:rr[-1]+1, cc[0]:cc[-1]+1]
            # 去掉边缘10%以防标题/边框干扰
            h, w = sub.shape
            sub = sub[int(h*0.12):int(h*0.95), int(w*0.05):int(w*0.95)]
            out.append((np.median(sub), sub.min(), sub.max()))
    return out

print("== 存档图背景灰度判定 ==")
st1b = tile_stats(os.path.join(EXP, '步骤1b_前向加噪.png'), (2, 3))
for i, (m, lo, hi) in enumerate(st1b):
    print(f"1b tile#{i}: median={m:.3f} min={lo:.3f} max={hi:.3f}")
st2 = tile_stats(os.path.join(EXP, '步骤2_DDPM采样对比.png'), (2, 8))
meds2 = [t[0] for t in st2]
print("图2 各tile median:", ['%.3f' % m for m in meds2])
st3 = tile_stats(os.path.join(EXP, '步骤3_逐步去噪过程.png'), (1, 7))
print("图3 各tile median:", ['%.3f' % t[0] for t in st3])
print("判定依据: 旧约定([0,1]数据直接显示)背景≈0.0; 新约定([-1,1]数据(x+1)/2显示)背景≈0.5")

# ---- 模型与采样 ----
src = open(os.path.join(EXP, '11.4-1.py'), encoding='utf-8').read()
cls_start = src.index('class SinusoidalTimeEmbedding')
cls_end = src.index('# ============================================================\n# DDPM采样算法')
g = {'torch': torch, 'nn': nn, 'F': F, 'np': np}
exec(compile(src[cls_start:cls_end], 'cls', 'exec'), g)
SmallUNet = g['SmallUNet']

T = 200
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
posterior_var = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)
sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)
beta_over_sqrt_1m_ab = betas / sqrt_one_minus_alpha_bars

def load_model(path, pred_type):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = SmallUNet(pred_type=pred_type)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    return m

model_eps = load_model(os.path.join(EXP, 'ddpm_epsilon_checkpoint.pth'), 'epsilon')
model_x0 = load_model(os.path.join(EXP, 'ddpm_x0_checkpoint.pth'), 'x0')

@torch.no_grad()
def ddpm_sample(model, shape, pred_type, x_T):
    x = x_T.clone()
    for t_idx in reversed(range(T)):
        t = torch.full((shape[0],), t_idx, dtype=torch.long)
        pred = model(x, t)
        if pred_type == 'epsilon':
            mean = sqrt_recip_alphas[t_idx] * (x - beta_over_sqrt_1m_ab[t_idx] * pred)
        else:
            ab = alpha_bars[t_idx]; abp = alpha_bars_prev[t_idx]; a = alphas[t_idx]
            mean = (torch.sqrt(abp) * betas[t_idx] / (1 - ab)) * pred + \
                   (torch.sqrt(a) * (1 - abp) / (1 - ab)) * x
        x = mean if t_idx == 0 else mean + torch.sqrt(posterior_var[t_idx]) * torch.randn_like(x)
    return x

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# E1) 旧约定路径复现: 数据[0,1], x_T~N(0,I), 直接imshow
#     旧checkpoint就是在[0,1]数据下训练的, 旧代码采样起点也是N(0,I)
#     注: 旧代码ddpm_sample与现在逻辑一致(该段未变), 起点x_T~N(0,I)
shapes = (8, 1, 28, 28)
torch.manual_seed(123)
x_T = torch.randn(shapes)
s_eps_old = ddpm_sample(model_eps, shapes, 'epsilon', x_T)
s_x0_old = ddpm_sample(model_x0, shapes, 'x0', x_T)
print("\n== 旧约定采样(数据[0,1]训练, N(0,I)起点) 最终x范围 ==")
print("eps: min=%.3f max=%.3f 背景中位=%.3f" % (s_eps_old.min(), s_eps_old.max(), s_eps_old.median()))
print("x0 : min=%.3f max=%.3f 背景中位=%.3f" % (s_x0_old.min(), s_x0_old.max(), s_x0_old.median()))

fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
for i in range(8):
    axes[0, i].imshow(s_eps_old[i, 0].numpy(), cmap='gray'); axes[0, i].axis('off')
    axes[1, i].imshow(s_x0_old[i, 0].numpy(), cmap='gray'); axes[1, i].axis('off')
plt.suptitle('E1: checkpoint([0,1]训练)+N(0,I)起点, 直接显示 (接近旧代码路径)')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'E1_old_path.png'), dpi=110); plt.close()

# D) 当前代码路径: checkpoint是[0,1]训练的, 但当前代码若重跑会跳过训练(is_final=True),
#    采样起点N(0,I)不变(采样起点与transform无关!), 显示(x+1)/2 → 背景≈0.5灰底
fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
for i in range(8):
    axes[0, i].imshow(np.clip((s_eps_old[i, 0].numpy() + 1) / 2, 0, 1), cmap='gray'); axes[0, i].axis('off')
    axes[1, i].imshow(np.clip((s_x0_old[i, 0].numpy() + 1) / 2, 0, 1), cmap='gray'); axes[1, i].axis('off')
plt.suptitle('E2: 同样样本按当前代码的(x+1)/2显示 → 若重跑当前代码, 图将变成灰底')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'E2_new_display.png'), dpi=110); plt.close()

# F) 图1a曲线与checkpoint history一致性: 从PNG中提取曲线端点
arr = np.array(Image.open(os.path.join(EXP, '步骤1a_训练曲线.png')))
print("\n图1a尺寸:", arr.shape)
# 蓝色像素 (matplotlib default blue ~ (31,119,180)), 红色 ~ (255,0,0)
rgb = arr[..., :3].astype(int)
blue = (np.abs(rgb - [31, 119, 180]).sum(-1) < 90)
red = (np.abs(rgb - [214, 39, 40]).sum(-1) < 90)
for name, mask in [('blue(eps)', blue), ('red(x0)', red)]:
    ys, xs = np.where(mask)
    print(f"{name}: x像素范围[{xs.min()},{xs.max()}], y顶部={ys.min()}, y底部={ys.max()}, 像素数={len(xs)}")
print("预期: eps曲线起点高(0.196)终点0.0475; x0起点0.100终点0.0334 → blue顶部y应远小于red顶部y")
print("完成: ", OUT)
