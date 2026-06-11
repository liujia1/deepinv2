"""
实验4.4-1 2D图像MYULA后验采样（去卷积）
对应章节：4.4 MYULA——近端ULA
知识点：不可微后验的挑战；Moreau-Yoshida包络光滑化；近端算子替代不可微梯度；
        MYULA迭代公式；Lipschitz常数 L = L_f + beta/lambda_prox；后验均值/方差/MMSE

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
# 图像值域 [0,255]，与 4.4-2 保持一致，避免归一化导致 sigma^2 过小
u_np = camera().astype(np.float64)
u_np = resize(u_np, (128, 128), anti_aliasing=True)
x_true = torch.Tensor(u_np).to(device)
nx, ny = x_true.shape
dimx = nx * ny

kernel_len = [9, 9]
type_blur = "uniform"
A, AT, AAT_norm = blur_operators(kernel_len, (nx, ny), type_blur, device)

Ax = A(x_true)
BSNR = 20  # 教学实验用BSNR=20，避免sigma过小导致L_f过大、步长过小
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
beta = 0.3  # 平衡正则化强度：beta过大导致过度平滑，过小则去模糊不足

# chambolle_prox_TV 求解 prox_{tau*TV}(x)，其中 tau = lambda_prox * beta
lam_beta = lambda_prox * beta  # tau

def prox_g(x, tau):
    """
    近端算子 prox_{tau*TV}(x)，tau = lambda_prox * beta

    MaxIter=10: MCMC采样中前后两步X变化很小，近端算子不需要每次完全收敛
    性能优化建议：
      1. 可进一步降低MaxIter到5（快速近似）
      2. 若chambolle_prox_TV支持warm-start（传入上一步对偶变量），可大幅加速
      3. GPU并行化可提升数倍速度
    """
    return chambolle_prox_TV(x, device, {'lambda': tau, 'MaxIter': 10})

def grad_g_myula(x, tau, lam):
    """
    MYULA梯度: nabla g_lambda(x) = (x - prox_{lambda*g}(x)) / lambda

    参数:
        x: 当前状态
        tau: 近端算子参数 (lambda_prox * beta)
        lam: Moreau-Yoshida参数 (lambda_prox)
    """
    return (x - prox_g(x, tau)) / lam

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

n_iter = 10000
n_burn_in = 4000

# 断言保护：burn_in 必须小于 n_iter
assert n_burn_in < n_iter, f"burn_in({n_burn_in}) 必须小于 n_iter({n_iter})"

print(f"\nMYULA参数:")
print(f"  beta={beta}, lambda_prox={lambda_prox}")
print(f"  L_f={L_f:.2f}, L_g={L_g:.4f}, L_total={L_total:.2f}")
print(f"  gamma=0.98/L={gamma:.6f}")
print(f"  n_iter={n_iter}, burn_in={n_burn_in}")

# ══════════════════════════════════════════════════════════
# 5. MYULA主循环
# ══════════════════════════════════════════════════════════
# MYULA迭代:
#   grad_prior = beta * nabla g_lambda(X)
#   X_{m+1} = X_m - gamma * (nabla f(X_m) + grad_prior) + sqrt(2*gamma) * Z_{m+1}
# 其中 nabla g_lambda(x) = (x - prox_{lambda*g}(x)) / lambda

print(f"\nRunning MYULA...")

X = y.clone().detach()
welford_stats = None  # burn-in 结束后再初始化，避免初始值 y 计入统计
mmse_errors = []

for i in range(n_iter):
    grad_likelihood = grad_f(X)
    grad_prior = beta * grad_g_myula(X, lam_beta, lambda_prox)

    Z = torch.randn_like(X)
    X = X - gamma * (grad_likelihood + grad_prior) + math.sqrt(2 * gamma) * Z

    if i >= n_burn_in:
        # burn-in 结束后，用第一个样本初始化 welford
        # 假设 welford(X) 构造时已将 X 计入统计（n=1）
        # 若 sampling_tools 的实现不同，需改为先 welford(shape) 再 update(X)
        if welford_stats is None:
            welford_stats = welford(X)
        else:
            welford_stats.update(X)
        X_mean = welford_stats.get_mean()
        rmse = torch.sqrt(torch.mean((X_mean - x_true) ** 2)).item()
        mmse_errors.append(rmse)

    if (i + 1) % 1000 == 0:
        print(f"  iter {i+1}/{n_iter}")

X_mean = welford_stats.get_mean()
X_var = welford_stats.get_var()
# Welford数值累积可能产生极小负值，clamp保护sqrt
X_std = torch.sqrt(torch.clamp(X_var, min=0))
X_last = X.clone()

# ══════════════════════════════════════════════════════════
# 6. PSNR计算
# ══════════════════════════════════════════════════════════
def psnr(x_ref, x_est, max_val=255.0):
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

axes[0][0].imshow(x_true.cpu().numpy(), cmap='gray', vmin=0, vmax=255)
axes[0][0].set_title(r'Ground truth $x$')
axes[0][0].axis('off')

axes[0][1].imshow(y.cpu().numpy(), cmap='gray', vmin=0, vmax=255)
axes[0][1].set_title(rf'Observation $y$ (PSNR={psnr_obs:.1f} dB)')
axes[0][1].axis('off')

axes[0][2].imshow(X_std.cpu().numpy(), cmap='gray')
axes[0][2].set_title(r'Posterior std $\sqrt{\mathrm{Var}(x|y)}$')
axes[0][2].axis('off')

axes[1][0].imshow(X_mean.cpu().numpy(), cmap='gray', vmin=0, vmax=255)
axes[1][0].set_title(rf'Posterior mean / MMSE (PSNR={psnr_mean:.1f} dB)')
axes[1][0].axis('off')

axes[1][1].imshow(X_last.cpu().numpy(), cmap='gray', vmin=0, vmax=255)
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
# 图像值域统计（MYULA 无约束，样本可能超出 [0,1]）
print(f"  X_mean range: [{X_mean.min().item():.3f}, {X_mean.max().item():.3f}]")
print(f"  X_last range: [{X_last.min().item():.3f}, {X_last.max().item():.3f}]")

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
