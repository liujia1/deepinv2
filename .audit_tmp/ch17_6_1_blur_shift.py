# -*- coding: utf-8 -*-
"""核实 p2_12：17.6-1 Step4 中"高斯模糊+平移 相对误差17%"的性质
   (a) 忠实复现 check_equivariance_shift（20次随机平移, MAX_SHIFT=8, test_x=MNIST前4张32x32）
   (b) 对比 zero-padding conv 与 circular-padding conv
   (c) 检查误差的空间分布：是否全部集中在边界 _kernel_size//2=3 像素内
   (d) 复现 inpainting 平移相对误差的量级（344.7%）作为环境对照
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

torch.manual_seed(0)
np.random.seed(0)

SAVE_DIR = r'h:\deepinv\deepinv2\第17章 自监督学习与等变架构\第十七章 配套实验\实验17.6-1'
IMG_SIZE, MAX_SHIFT, KEEP_RATIO = 32, 8, 0.5
device = torch.device('cpu')

transform = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])
mnist_test = datasets.MNIST(root=os.path.join(SAVE_DIR, 'mnist_data'),
                            train=False, download=False, transform=transform)
loader = torch.utils.data.DataLoader(mnist_test, batch_size=6, shuffle=False)
test_imgs, _ = next(iter(loader))
test_x = test_imgs[:4].to(device)
print(f'test_x shape = {tuple(test_x.shape)}')

# ---- 复现实验中的核与算子 ----
_kernel_size, _sigma_k = 7, 1.5
_k = torch.zeros(1, 1, _kernel_size, _kernel_size)
for i in range(_kernel_size):
    for j in range(_kernel_size):
        _k[0, 0, i, j] = np.exp(-((i-_kernel_size//2)**2 + (j-_kernel_size//2)**2) / (2*_sigma_k**2))
_k = _k / _k.sum()

def blur_zero(x):
    return F.conv2d(x, _k, padding=_kernel_size//2)

def blur_circ(x):
    return F.conv2d(F.pad(x, (_kernel_size//2,)*4, mode='circular'), _k, padding=0)

def check_equivariance_shift(A_fn, x, n_tests=20):
    """与 17.6-1.py L664-684 完全一致"""
    errors, Ax_norms = [], []
    for _ in range(n_tests):
        dy = torch.randint(-MAX_SHIFT, MAX_SHIFT+1, (1,)).item()
        dx = torch.randint(-MAX_SHIFT, MAX_SHIFT+1, (1,)).item()
        Tg = lambda z: torch.roll(z, shifts=(dy, dx), dims=(2, 3))
        ATg_x = A_fn(Tg(x)); Ax = A_fn(x); TgAx = Tg(Ax)
        errors.append(((ATg_x - TgAx) ** 2).mean().sqrt().item())
        Ax_norms.append(Ax.abs().mean().item())
    abs_err = np.mean(errors)
    rel_err = abs_err / (np.mean(Ax_norms) + 1e-8)
    return abs_err, rel_err

# 固定随机种子复现实验中的 20 次抽样序列
torch.manual_seed(1234)
a0, r0 = check_equivariance_shift(blur_zero, test_x)
torch.manual_seed(1234)
a1, r1 = check_equivariance_shift(blur_circ, test_x)
print(f'\n[1] zero-padding conv : abs={a0:.4f}  rel={r0*100:.1f}%   (JSON 记录 17.0%)')
print(f'[2] circular  conv    : abs={a1:.3e}  rel={r1*100:.3e}%  (应为数值零)')

# ---- (c) 误差空间分布：固定一个平移，逐像素平均 |A(Tx)-T(Ax)| ----
dy, dx = 5, -6
Tg = lambda z: torch.roll(z, shifts=(dy, dx), dims=(2, 3))
err_map = (blur_zero(Tg(test_x)) - Tg(blur_zero(test_x))).abs().mean(dim=(0, 1))
# T(Ax) 的边界（zero padding）经 roll 后会被搬到对侧，并向内扩展 |dy|/|dx|；
# 再加上卷积核半径 3，距边界 m = max(|dy|,|dx|)+3 以内的像素都可能受影响
m = max(abs(dy), abs(dx)) + _kernel_size//2
inner = err_map[m:-m, m:-m]
print(f'\n[3] 误差空间分布 (平移 dy={dy}, dx={dx})，受影响带宽度 m={m}')
print(f'    边界带(m像素) max|err| = {max(err_map[:m].max(), err_map[-m:].max(), err_map[:, :m].max(), err_map[:, -m:].max()):.3e}')
print(f'    内部区域({IMG_SIZE-2*m}x{IMG_SIZE-2*m}) max|err| = {inner.max():.3e}   (<1e-7 说明其余区域严格等变)')
print(f'    边界带总误差占比 = {(err_map.sum()-inner.sum())/err_map.sum()*100:.2f}%')

# ---- (d) inpainting 对照 ----
g = torch.Generator().manual_seed(2025)
mask = (torch.rand(IMG_SIZE, IMG_SIZE, generator=g) < KEEP_RATIO)
def inpaint_A(x):
    return x * mask.to(x.device).unsqueeze(0).unsqueeze(0).expand_as(x)
torch.manual_seed(1234)
a2, r2 = check_equivariance_shift(inpaint_A, test_x)
print(f'\n[4] Inpainting(固定随机掩码) 平移: abs={a2:.4f}  rel={r2*100:.1f}%  (实验记录 344.7%, 量级一致即可)')

# ---- (e) circular conv 对旋转 90° 是否也严格等变 ----
def check_rot(A_fn, x):
    AT = A_fn(torch.rot90(x, 1, [2, 3])); Ax = A_fn(x)
    err = ((AT - torch.rot90(Ax, 1, [2, 3])) ** 2).mean().sqrt().item()
    return err, err / (Ax.abs().mean().item() + 1e-8)
e0, f0 = check_rot(blur_zero, test_x)
e1, f1 = check_rot(blur_circ, test_x)
print(f'\n[5] 旋转90°: zero-padding rel={f0*100:.2e}%, circular rel={f1*100:.2e}%')
print('    核 exp(-(i^2+j^2)/2s^2) 在 7x7 上关于中心 90° 对称 => 两种边界都严格旋转等变')
