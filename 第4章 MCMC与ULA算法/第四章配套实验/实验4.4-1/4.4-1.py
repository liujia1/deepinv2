"""
实验4.4-1 2D图像MYULA后验采样（去卷积）
对应章节：4.4 MYULA——近端ULA
知识点：不可微后验的挑战；Moreau-Yoshida包络光滑化；近端算子替代不可微梯度；
        MYULA迭代公式；Lipschitz常数 L = L_f + 1/lambda_prox；后验均值/方差/MMSE

修改说明：
  从原参考实验4.3.py迁移，将Huber-TV改为Moreau-TV（Chambolle近端算子），
  与4.4节MYULA理论完全对齐。使用PyTorch + sampling_tools模块。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import os
import sys
import warnings
import logging

import torch
from skimage.data import camera
from skimage.transform import resize
from PIL import Image

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

# ====== 中文字体配置（兼容本地和Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验4.4-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)
else:
    _chinese_path = '.chinese'
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")

SCRIPT_DIR = SAVE_DIR
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from sampling_tools import *
# ========================================================

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ══════════════════════════════════════════════════════════
# 1. 图像和正向模型
# ══════════════════════════════════════════════════════════
u_np = camera().astype(np.float64) / 255.0
u_np = resize(u_np, (128, 128), anti_aliasing=True)
x_true = torch.Tensor(u_np).to(device)
nx, ny = x_true.shape
dimx = nx * ny

kernel_len = [9, 9]
type_blur = "uniform"
A, AT, AAT_norm = blur_operators(kernel_len, (nx, ny), type_blur, device)

Ax = A(x_true)
BSNR = 30
sigma = torch.linalg.matrix_norm(Ax - torch.mean(Ax), ord='fro') / math.sqrt(dimx * 10 ** (BSNR / 10))
y = Ax + sigma * torch.randn_like(x_true)
sigma2 = (sigma ** 2).item()

print(f"Image size: {nx}x{ny}, BSNR={BSNR}, sigma={sigma.item():.6f}, sigma^2={sigma2:.6f}")

# ══════════════════════════════════════════════════════════
# 2. 似然梯度
# ══════════════════════════════════════════════════════════
# f(x) = ||y - Ax||^2 / (2*sigma^2)
# grad_f(x) = A^T(Ax - y) / sigma^2

def grad_f(x):
    return AT(A(x) - y) / sigma2

# ══════════════════════════════════════════════════════════
# 3. Moreau-Yoshida近端TV（MYULA核心）
# ══════════════════════════════════════════════════════════
# g(x) = beta * TV(x), 不可微
# Moreau-Yoshida包络: g_lambda(x) = min_z { g(z) + ||x-z||^2 / (2*lambda) }
# 近端算子: prox_{lambda*g}(x) = argmin_z { g(z) + ||x-z||^2 / (2*lambda) }
# MYULA梯度: nabla g_lambda(x) = (x - prox_{lambda*g}(x)) / lambda
#
# 对 g(x) = beta * TV(x):
#   prox_{lambda*g}(x) = prox_{lambda*beta*TV}(x)
#   由近端算子的正齐次性: prox_{lambda*beta*TV}(x) = prox_{(lambda*beta)*TV}(x)
#   Chambolle算法求解: prox_{tau*TV}(x), 其中 tau = lambda*beta

lambda_prox = 1.0
beta = 0.01

def prox_g(x, lam_beta):
    return chambolle_prox_TV(x, device, {'lambda': lam_beta, 'MaxIter': 25})

def grad_g_myula(x, lam_beta):
    return (x - prox_g(x, lam_beta)) / lambda_prox

# ══════════════════════════════════════════════════════════
# 4. MYULA参数
# ══════════════════════════════════════════════════════════
# Lipschitz常数: L = L_f + L_{g_lambda}
#   L_f = ||A^T A|| / sigma^2 = AAT_norm / sigma^2
#   L_{g_lambda} = beta / lambda_prox
L_f = AAT_norm / sigma2
L_g = beta / lambda_prox
L_total = L_f + L_g
gamma = 0.98 / L_total

lam_beta = lambda_prox * beta

n_iter = 5000
n_burn_in = 1000

print(f"\nMYULA参数:")
print(f"  beta={beta}, lambda_prox={lambda_prox}")
print(f"  L_f={L_f:.2f}, L_g={L_g:.4f}, L_total={L_total:.2f}")
print(f"  gamma=0.98/L={gamma:.6f}")
print(f"  n_iter={n_iter}, burn_in={n_burn_in}")

# ══════════════════════════════════════════════════════════
# 5. MYULA主循环
# ══════════════════════════════════════════════════════════
# MYULA迭代:
#   X_{m+1} = X_m - gamma * (nabla f(X_m) + beta * nabla g_lambda(X_m))
#             + sqrt(2*gamma) * Z_{m+1}
# 其中 nabla g_lambda(x) = (x - prox_{lambda*g}(x)) / lambda

print(f"\nRunning MYULA...")

X = y.clone().detach()
welford_stats = welford(X)
mmse_errors = []

for i in range(n_iter):
    grad_likelihood = grad_f(X)
    grad_prior = beta * grad_g_myula(X, lam_beta)

    Z = torch.randn_like(X)
    X = X - gamma * (grad_likelihood + grad_prior) + math.sqrt(2 * gamma) * Z

    if i >= n_burn_in:
        welford_stats.update(X)
        X_mean = welford_stats.get_mean()
        rmse = torch.sqrt(torch.mean((X_mean - x_true) ** 2)).item()
        mmse_errors.append(rmse)

    if (i + 1) % 1000 == 0:
        print(f"  iter {i+1}/{n_iter}")

X_mean = welford_stats.get_mean()
X_var = welford_stats.get_var()
X_std = torch.sqrt(torch.clamp(X_var, min=0))
X_last = X.clone()

# ══════════════════════════════════════════════════════════
# 6. PSNR计算
# ══════════════════════════════════════════════════════════
def psnr(x_ref, x_est, max_val=1.0):
    mse = torch.mean((x_ref - x_est) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * math.log10(max_val ** 2 / mse)

psnr_obs = psnr(x_true, y)
psnr_mean = psnr(x_true, X_mean)
psnr_last = psnr(x_true, X_last)

# ══════════════════════════════════════════════════════════
# 7. 可视化
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0][0].imshow(x_true.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
axes[0][0].set_title(r'Ground truth $x$')
axes[0][0].axis('off')

axes[0][1].imshow(y.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
axes[0][1].set_title(rf'Observation $y$ (PSNR={psnr_obs:.1f} dB)')
axes[0][1].axis('off')

axes[0][2].imshow(X_std.cpu().numpy(), cmap='gray')
axes[0][2].set_title(r'Posterior std $\sqrt{\mathrm{Var}(x|y)}$')
axes[0][2].axis('off')

axes[1][0].imshow(X_mean.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
axes[1][0].set_title(rf'Posterior mean / MMSE (PSNR={psnr_mean:.1f} dB)')
axes[1][0].axis('off')

axes[1][1].imshow(X_last.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
axes[1][1].set_title(rf'Single sample (PSNR={psnr_last:.1f} dB)')
axes[1][1].axis('off')

axes[1][2].plot(mmse_errors)
axes[1][2].set_xlabel(r'Iteration (after burn-in)')
axes[1][2].set_ylabel(r'RMSE')
axes[1][2].set_title(r'MMSE error convergence')
axes[1][2].grid(True, alpha=0.3)

fig.suptitle(r'Experiment 4.4-1: MYULA posterior sampling (deblurring)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'myula_deblurring_results.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════
# 8. 输出结论
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("实验4.4-1 2D图像MYULA后验采样（去卷积）")
print("=" * 60)
print(f"\n[步骤1] 正向模型: {nx}x{ny}图像, BSNR={BSNR}dB")
print(f"  sigma^2 = {sigma2:.6f}")
print(f"\n[步骤2] MYULA参数:")
print(f"  L_f = ||A^TA||/sigma^2 = {L_f:.2f}")
print(f"  L_g = beta/lambda_prox = {L_g:.4f}")
print(f"  L_total = {L_total:.2f}")
print(f"  gamma = 0.98/L = {gamma:.6f}")
print(f"\n[步骤3] 结果:")
print(f"  PSNR(observation) = {psnr_obs:.2f} dB")
print(f"  PSNR(MMSE)        = {psnr_mean:.2f} dB")
print(f"  PSNR(single)      = {psnr_last:.2f} dB")

print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print(r"1. 不可微先验 g(x)=beta*TV(x) 的梯度不存在，ULA不可直接使用")
print(r"2. Moreau-Yoshida包络光滑化: g_lambda(x) = min_z{g(z)+||x-z||^2/(2*lambda)}")
print(r"3. MYULA梯度: nabla g_lambda(x) = (x - prox_{lambda*g}(x)) / lambda")
print(r"4. 近端算子 prox_{lambda*g} 由Chambolle算法迭代求解")
print(r"5. Lipschitz常数: L = L_f + beta/lambda_prox")
print(r"6. lambda权衡: 小lambda->近似精确但L大(收敛慢); 大lambda->近似粗糙但L小(收敛快)")
print(r"7. 后验均值(MMSE)比单个样本更稳定，PSNR更高")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")
