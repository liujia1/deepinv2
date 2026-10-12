"""实验4.7-1 复核脚本 A：确定性参数复算（不覆写任何实验产物）"""
import math, os, sys, json
import numpy as np
import torch

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

EXP = r"D:\works\deepinv\第4章 MCMC与ULA算法\第四章 配套实验\实验4.7-1"
sys.path.insert(0, EXP)
device = torch.device('cpu')

from sampling_tools import *   # noqa
from PIL import Image

im = np.array(Image.open(os.path.join(EXP, "cman.png")))
x = torch.Tensor(im / 255.).to(device)
print("im.shape", im.shape, "dtype", im.dtype)

kernel_len = [5, 5]; size = [im.shape[0], im.shape[1]]
A, AT, AAT_norm = blur_operators(kernel_len, size, "uniform", device)
print(f"AAT_norm = {AAT_norm:.6f}")

y0 = A(x)
BSNRdb = 40
sigma = torch.linalg.matrix_norm(A(x) - torch.mean(A(x)), ord='fro') / math.sqrt(torch.numel(x) * 10 ** (BSNRdb / 10))
print(f"sigma = {sigma.item():.8f}   (json/doc: 0.002484)")
y = y0 + sigma * torch.randn_like(x)

L_y = AAT_norm / (sigma ** 2)
eps = (5 / 255) ** 2
delta_max = 1.0 / (1.0 / eps + L_y)
delta = delta_max * 0.99
print(f"L_y = {L_y:.4f}  (doc: 160910.78)")
print(f"eps = {eps:.8f}  (doc: 0.000384)")
print(f"delta_max = {delta_max:.10e}")
print(f"delta = {delta:.10e}  (doc: 0.000006)")
print(f"1/eps = {1/eps:.2f}  vs L_y = {L_y:.2f}  -> L_net=1 假设的影响: {(1/eps)/L_y:.2f}x")

# ---- 追踪主循环的 count / 样本数 / welford 计数 ----
maxit, burnin, n_samples = 500, int(500 * 0.1), 50
thinning_step = max(1, int((maxit - burnin) / n_samples))
print(f"thinning_step = {thinning_step}")
n_MC = 0; n_welford_updates = 0; k_welford = 1; count = None
for i_x in range(maxit):
    if i_x == burnin:
        count = 0
    elif i_x > burnin:
        n_welford_updates += 1
        k_welford += 1
        if count == thinning_step - 1:
            n_MC += 1; count = 0
        else:
            count += 1
print(f"MC_X 长度 = {n_MC} (json chain_length=49), welford 样本数 k = {k_welford}")

# ---- 初始指标 ----
print(f"initial NRMSE {NRMSE(x, y):.4f}  PSNR {PSNR(x, y):.2f}  SSIM {SSIM(x, y):.4f}")
