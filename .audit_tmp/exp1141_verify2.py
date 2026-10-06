# -*- coding: utf-8 -*-
"""实验11.4-1 复核脚本2：
A) 图1b精确复现检验：旧约定([0,1]数据, 直接imshow) vs 新约定([-1,1]数据, (x+1)/2)
B) 采样交叉检验：旧checkpoint分别配合 [0,1] / [-1,1] 两种采样约定
"""
import sys, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
torch.manual_seed(42)
np.random.seed(42)

EXP = r"D:\works\deepinv\第11章 扩散模型：变分视角\第十一章 配套实验\实验11.4-1"
OUT = r"D:\works\deepinv\.audit_tmp\exp1141_verify"
os.makedirs(OUT, exist_ok=True)
device = torch.device('cpu')

# ---- 调度（与实验一致）----
T = 200
beta_min, beta_max = 1e-4, 0.02
betas = torch.linspace(beta_min, beta_max, T).to(device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
alpha_bars_prev = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
posterior_var = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)
sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)
beta_over_sqrt_1m_ab = betas / sqrt_one_minus_alpha_bars

print("ᾱ: t=20 %.4f t=50 %.4f t=100 %.4f t=150 %.4f t=199 %.4f" %
      (alpha_bars[20], alpha_bars[50], alpha_bars[100], alpha_bars[150], alpha_bars[199]))
print("信号幅度√ᾱ: t=150 %.4f t=199 %.4f" % (np.sqrt(alpha_bars[150]), np.sqrt(alpha_bars[199])))

# ---- 模型（从原脚本head段提取类定义）----
src = open(os.path.join(EXP, '11.4-1.py'), encoding='utf-8').read()
cut = src.index('# ============================================================\n# DDPM采样算法')
ns = {}
exec(compile(src[:src.index('print("\\n" + "="*60)')], 'head', 'exec'), ns)  # 到设备配置前
# head段里还有 SAVE_DIR 逻辑，重新定义需要的：
import torch.nn as nn
# 直接exec类定义部分
cls_start = src.index('class SinusoidalTimeEmbedding')
cls_end = src.index('# ============================================================\n# DDPM采样算法')
g = {'torch': torch, 'nn': nn, 'F': F, 'np': np}
exec(compile(src[cls_start:cls_end], 'cls', 'exec'), g)
SmallUNet = g['SmallUNet']

def load_model(path, pred_type):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = SmallUNet(pred_type=pred_type)
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m

model_eps = load_model(os.path.join(EXP, 'ddpm_epsilon_checkpoint.pth'), 'epsilon')
model_x0 = load_model(os.path.join(EXP, 'ddpm_x0_checkpoint.pth'), 'x0')

# ---- A) 图1b 复现检验 ----
# 旧约定：ToTensor only，直接 imshow x_t
ds_old = datasets.MNIST(os.path.join(EXP, 'data'), train=False, download=False,
                        transform=transforms.ToTensor())
# 新约定：[-1,1]
ds_new = datasets.MNIST(os.path.join(EXP, 'data'), train=False, download=False,
                        transform=transforms.Compose([transforms.ToTensor(),
                                                      transforms.Lambda(lambda x: x * 2 - 1)]))
img_old = ds_old[0][0]  # [1,28,28]
img_new = ds_new[0][0]
print("测试图0: 旧约定 min/max = %.3f/%.3f, 背景均值=%.3f" % (img_old.min(), img_old.max(), img_old.mean()))
print("测试图0: 新约定 min/max = %.3f/%.3f, 背景均值=%.3f" % (img_new.min(), img_new.max(), img_new.mean()))

def q_sample(x_0, t_idx, noise):
    return sqrt_alpha_bars[t_idx] * x_0 + sqrt_one_minus_alpha_bars[t_idx] * noise

t_show = [0, 20, 50, 100, 150, 199]
tiles_old, tiles_new = [], []
for t_idx in t_show:
    torch.manual_seed(42)
    eps = torch.randn_like(img_old)
    xt_old = q_sample(img_old, t_idx, eps)          # 旧: 数据[0,1], 直接显示
    xt_new_vis = (q_sample(img_new, t_idx, eps) + 1) / 2  # 新: 数据[-1,1], (x+1)/2显示
    tiles_old.append(np.clip(xt_old[0].numpy(), 0, 1))
    tiles_new.append(np.clip(xt_new_vis[0].numpy(), 0, 1))

# 读取存档图1b, 自动分块：用投影法找6个子图
arch = np.array(Image.open(os.path.join(EXP, '步骤1b_前向加噪.png')).convert('L'), dtype=float) / 255.0
print("存档1b尺寸:", arch.shape)

def find_tiles(arch_img):
    # 找非白行列区域（子图块）
    nonwhite = arch_img < 0.95
    rows = nonwhite.any(axis=1); cols = nonwhite.any(axis=0)
    # 行分段
    def segments(mask):
        segs = []; s = None
        for i, v in enumerate(mask):
            if v and s is None: s = i
            if not v and s is not None:
                if i - s > 50: segs.append((s, i))
                s = None
        if s is not None: segs.append((s, len(mask)))
        return segs
    return segments(rows), segments(cols)

rsegs, csegs = find_tiles(arch)
print("行分段:", rsegs, "列分段:", csegs)

if len(rsegs) >= 2 and len(csegs) >= 3:
    # 对每个tile, 在存档区域内做最小二乘匹配（允许平移, 搜索±20px）
    k = 0
    for (r0, r1) in rsegs[:2]:
        for (c0, c1) in csegs[:3]:
            if k >= 6: break
            region = arch[r0:r1, c0:c1]
            # 存档tile是28px图像放大显示, 下采样region到28x28再比较
            reg_t = torch.from_numpy(region).float()[None, None]
            def mse(tile):
                down = F.interpolate(reg_t, size=(28, 28), mode='area')[0, 0].numpy()
                return np.mean((down - tile) ** 2)
            # 也尝试tile反色无关——直接比较
            m_old = mse(tiles_old[k]); m_new = mse(tiles_new[k])
            print(f"tile#{k} 存档vs旧约定 MSE={m_old:.6f}  vs新约定 MSE={m_new:.6f}")
            k += 1

# ---- B) 采样交叉检验 ----
@torch.no_grad()
def ddpm_sample(model, shape, pred_type='epsilon', data_shift=0.0):
    model.eval()
    x = torch.randn(shape, device=device)
    for t_idx in reversed(range(T)):
        t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        pred = model(x, t)
        if pred_type == 'epsilon':
            model_mean = sqrt_recip_alphas[t_idx] * (x - beta_over_sqrt_1m_ab[t_idx] * pred)
        else:
            ab = alpha_bars[t_idx]; ab_prev = alpha_bars_prev[t_idx]; a = alphas[t_idx]
            model_mean = (torch.sqrt(ab_prev) * betas[t_idx] / (1 - ab)) * pred + \
                         (torch.sqrt(a) * (1 - ab_prev) / (1 - ab)) * x
        if t_idx == 0:
            x = model_mean
        else:
            x = model_mean + torch.sqrt(posterior_var[t_idx]) * torch.randn_like(x)
    return x

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

n = 8
shapes = (n, 1, 28, 28)
torch.manual_seed(42); s_eps = ddpm_sample(model_eps, shapes, 'epsilon')
torch.manual_seed(42); s_x0 = ddpm_sample(model_x0, shapes, 'x0')

fig, axes = plt.subplots(2, n, figsize=(16, 4.4))
for i in range(n):
    axes[0, i].imshow(s_eps[i, 0].numpy(), cmap='gray')  # 旧约定：直接显示
    axes[0, i].axis('off')
    axes[1, i].imshow(s_x0[i, 0].numpy(), cmap='gray')
    axes[1, i].axis('off')
axes[0, 0].set_title('eps raw-display(旧约定复现)', fontsize=9)
axes[1, 0].set_title('x0 raw-display', fontsize=9)
plt.suptitle('B1: 旧checkpoint + 原始显示(数据[0,1]约定)')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'B1_old_convention.png'), dpi=110); plt.close()

fig, axes = plt.subplots(2, n, figsize=(16, 4.4))
for i in range(n):
    axes[0, i].imshow(np.clip((s_eps[i, 0].numpy() + 1) / 2, 0, 1), cmap='gray')  # 新约定显示
    axes[0, i].axis('off')
    axes[1, i].imshow(np.clip((s_x0[i, 0].numpy() + 1) / 2, 0, 1), cmap='gray')
    axes[1, i].axis('off')
axes[0, 0].set_title('eps (x+1)/2(新约定显示)', fontsize=9)
axes[1, 0].set_title('x0 (x+1)/2', fontsize=9)
plt.suptitle('B2: 旧checkpoint 但按新[-1,1]约定显示 (当前代码重跑将得到的图)')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'B2_new_display.png'), dpi=110); plt.close()

print("eps样本 原始范围: min=%.3f max=%.3f mean=%.3f" % (s_eps.min(), s_eps.max(), s_eps.mean()))
print("x0 样本 原始范围: min=%.3f max=%.3f mean=%.3f" % (s_x0.min(), s_x0.max(), s_x0.mean()))
print("完成: ", OUT)
