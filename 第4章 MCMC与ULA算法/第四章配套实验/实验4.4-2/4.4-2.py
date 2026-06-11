"""
实验4.4-2 MYULA经验贝叶斯（SAPG参数自适应估计）
对应章节：4.4 MYULA（近端ULA）、3.6（经验贝叶斯、边际似然）
知识点：不可微后验的挑战；Moreau-Yoshida包络光滑化；近端算子替代不可微梯度；
        MYULA迭代公式；Lipschitz常数 L = L_f + theta/lambda_prox；
        Fisher恒等式；SAPG随机近似投影梯度；经验贝叶斯自适应参数估计

修改说明：
  从原参考实验4.7.py迁移，适配配套实验风格（matplotlib.use('Agg')、
  中文字体模块、Colab支持、LaTeX格式符号）。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
from PIL import Image
import math
import os
import sys
import warnings
import logging

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
    SAVE_DIR = os.path.join(_gdrive, '实验4.4-2')
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ══════════════════════════════════════════════════════════
# 1. 图像和正向模型
# ══════════════════════════════════════════════════════════
if os.path.exists(os.path.join(SCRIPT_DIR, "cman.png")):
    x_np = np.array(Image.open(os.path.join(SCRIPT_DIR, "cman.png"))).astype(np.float64)
else:
    from skimage.data import camera
    x_np = camera().astype(np.float64)

x_true = torch.Tensor(x_np).to(device)
nx, ny = x_true.shape
dimx = nx * ny

kernel_len = [9, 9]
type_blur = "uniform"
A, AT, AAT_norm = blur_operators(kernel_len, (nx, ny), type_blur, device)

Ax = A(x_true)
BSNR = 30
sigma = torch.linalg.matrix_norm(Ax - torch.mean(Ax), ord='fro') / math.sqrt(dimx * 10 ** (BSNR / 10))
y = Ax + sigma * torch.randn_like(x_true)
sigma2_init = (sigma ** 2).item()

print(f"Image: {nx}x{ny}, BSNR={BSNR}, true sigma^2={sigma2_init:.4f}")

# ══════════════════════════════════════════════════════════
# 2. 似然
# ══════════════════════════════════════════════════════════
# f(x) = ||y - Ax||^2 / (2*sigma^2)
# df_dsigma2 = ∂f/∂σ² = ||y - Ax||^2 / (2*sigma^4)
# grad_f_x = ∂f/∂x = A^T(Ax - y) / sigma^2

f_y = lambda x, s2: (torch.linalg.matrix_norm(y - A(x), ord='fro')**2) / (2 * s2)
df_dsigma2 = lambda x, s2: (torch.linalg.matrix_norm(y - A(x), ord='fro')**2) / (2 * s2**2)
grad_f_x = lambda x, s2: AT(A(x) - y) / s2

# ══════════════════════════════════════════════════════════
# 3. Moreau-Yoshida近端TV
# ══════════════════════════════════════════════════════════
# g(x) = TV(x), 不可微
# MY梯度: nabla g_lambda(x) = (x - prox_{lambda*g}(x)) / lambda
# 对 theta*g(x) = theta*TV(x):
#   prox_{lambda*(theta*g)}(x) = prox_{(lambda*theta)*TV}(x)
#   Chambolle参数: tau = lambda*theta

lambda_prox = 1.0

# 近端算子参数
# MaxIter=10: MCMC采样中前后两步X变化很小，近端算子不需要每次完全收敛
# 性能优化建议：
#   1. 可进一步降低MaxIter到5（快速近似）
#   2. 若chambolle_prox_TV支持warm-start（传入上一步对偶变量），可大幅加速
#   3. GPU并行化可提升数倍速度
proxg = lambda x, lam: chambolle_prox_TV(x, device, {'lambda': lam, 'MaxIter': 10})
gradg = lambda x, lam, lprox: (x - proxg(x, lam)) / lprox

# ══════════════════════════════════════════════════════════
# 4. SAPG参数
# ══════════════════════════════════════════════════════════
th_init = 0.01

min_th = 0.001
max_th = 1
min_sigma2 = 0.1
max_sigma2 = 100

d_exp = 0.8
delta_step = lambda i: ((i + 1) ** (-d_exp)) / dimx

# 超参数学习率缩放因子
# c_eta=10: 对数域参数 eta=log(theta) 梯度数量级较小，需要较小步长
# c_sigma2=10000: 线性域参数 sigma^2 梯度波动剧烈（残差项 ||y-Ax||^2/(2*sigma^4)），
#                 需要较大步长补偿；实际步长 = c * delta_step(k)
# 注意：更换图像尺寸或噪声水平时，可能需要重新调参
c_eta = 10
c_sigma2 = 10000

warmupSteps = 1000
total_iter = 1000
n_burn_in = 700   # 70% burn-in，与 4.4-1 风格统一

# Lipschitz常数
L_f_init = AAT_norm / sigma2_init
L_g_init = th_init / lambda_prox
L_total_init = L_f_init + L_g_init
gamma = 0.98 / L_total_init
print(f"MCMC step size gamma = {gamma:.6f}")

# ══════════════════════════════════════════════════════════
# 5. MYULA暖启动
# ══════════════════════════════════════════════════════════
print('\nRunning Warm up...\n')

X_wu = y.to(device).detach().clone()
fix_sigma2 = sigma2_init
fix_theta = th_init

for k in tqdm(range(warmupSteps)):
    grad_f = grad_f_x(X_wu, fix_sigma2)
    lam_wu = lambda_prox * fix_theta   # prox参数: lambda*theta
    grad_g = fix_theta * gradg(X_wu, lam_wu, lambda_prox)
    Z = torch.randn_like(X_wu)
    X_wu = X_wu - gamma * grad_f - gamma * grad_g + math.sqrt(2 * gamma) * Z

# 对数尺度参数
eta_init = math.log(th_init)
min_eta = math.log(min_th)
max_eta = math.log(max_th)

# ══════════════════════════════════════════════════════════
# 6. SAPG主循环
# ══════════════════════════════════════════════════════════
print('\nRunning SAPG algorithm...\n')

X = X_wu.clone()
eta = eta_init
theta = th_init
sigma2 = sigma2_init

theta_list = [theta]
sigma2_list = [sigma2]

alpha = 1  # TV is 1-homogeneous

for k in tqdm(range(total_iter)):
    # SAPG使用单步样本近似期望（随机近似），无需多样本平均
    L_f = AAT_norm / sigma2
    L_g = theta / lambda_prox
    gamma = 0.98 / (L_f + L_g)

    lam = lambda_prox * theta   # prox参数: lambda*theta
    grad_g = theta * gradg(X, lam, lambda_prox)
    grad_f = grad_f_x(X, sigma2)

    Z = torch.randn_like(X)
    X = X - gamma * grad_f - gamma * grad_g + math.sqrt(2 * gamma) * Z

    g_x = TVnorm(X).item()
    gf_sigma2 = df_dsigma2(X, sigma2).item()

    grad_theta_logp = -g_x + dimx / (alpha * theta)
    grad_eta_logp = grad_theta_logp * theta
    eta = eta + c_eta * delta_step(k) * grad_eta_logp
    eta = max(min_eta, min(max_eta, eta))
    theta = math.exp(eta)

    grad_sigma2_logp = gf_sigma2 - dimx / (2 * sigma2)
    sigma2 = sigma2 + c_sigma2 * delta_step(k) * grad_sigma2_logp
    sigma2 = max(min_sigma2, min(max_sigma2, sigma2))

    theta_list.append(theta)
    sigma2_list.append(sigma2)

theta_burnin = theta_list[n_burn_in:]
sigma2_burnin = sigma2_list[n_burn_in:]
theta_est = float(np.mean(theta_burnin))
sigma2_est = float(np.mean(sigma2_burnin))
true_sigma2 = (sigma ** 2).item()

print(f"\nEstimated theta  = {theta_est:.6f}")
print(f"Estimated sigma2 = {sigma2_est:.6f}")
print(f"True sigma2      = {true_sigma2:.6f}")

# ══════════════════════════════════════════════════════════
# 7. 收敛图
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

iters = np.arange(len(theta_list))
axes[0].plot(iters, theta_list, label=r'$\theta_n$', color='steelblue')
axes[0].axvline(n_burn_in, color='red', linestyle='--', label=r'Burn-in end')
axes[0].axhline(theta_est, color='orange', linestyle='--', label=rf'Mean after burn-in: {theta_est:.4f}')
axes[0].set_xlabel(r'Iteration')
axes[0].set_ylabel(r'$\theta$')
axes[0].set_title(r'$\theta$ (TV regularization) convergence')
axes[0].legend()

axes[1].plot(iters, sigma2_list, label=r'$\sigma^2_n$', color='darkorange')
axes[1].axvline(n_burn_in, color='red', linestyle='--', label=r'Burn-in end')
axes[1].axhline(sigma2_est, color='blue', linestyle='--', label=rf'Estimate: {sigma2_est:.5f}')
axes[1].axhline(true_sigma2, color='green', linestyle=':', label=rf'True value: {true_sigma2:.5f}')
axes[1].set_xlabel(r'Iteration')
axes[1].set_ylabel(r'$\sigma^2$')
axes[1].set_title(r'$\sigma^2$ (noise variance) convergence')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'sapg_convergence.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════
# 8. MAP重建
# ══════════════════════════════════════════════════════════
print('\nRunning MAP reconstruction with estimated parameters...')

x_map = y.clone()
n_map_iter = 200

lam_map = lambda_prox * theta_est
L_f_map = AAT_norm / sigma2_est
L_g_map = theta_est / lambda_prox
gamma_map = 0.98 / (L_f_map + L_g_map)

for _ in tqdm(range(n_map_iter)):
    grad_f_map = grad_f_x(x_map, sigma2_est)
    grad_g_map = theta_est * gradg(x_map, lam_map, lambda_prox)
    x_map = x_map - gamma_map * (grad_f_map + grad_g_map)

# MAP收敛验证：打印最终梯度范数
final_grad = grad_f_x(x_map, sigma2_est) + theta_est * gradg(x_map, lam_map, lambda_prox)
print(f"  MAP final gradient norm: {torch.norm(final_grad).item():.6f}")
# 图像值域统计（梯度下降无约束，MAP结果可能超出 [0,255]）
print(f"  x_map range: [{x_map.min().item():.1f}, {x_map.max().item():.1f}]")

# ══════════════════════════════════════════════════════════
# 9. SNR指标与可视化
# ══════════════════════════════════════════════════════════
def snr(x_ref, x_est):
    noise = x_ref - x_est
    return 10 * torch.log10(torch.sum(x_ref**2) / torch.sum(noise**2)).item()

snr_obs = snr(x_true, y)
snr_map = snr(x_true, x_map)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, img, title in zip(axes,
                           [x_true.cpu().numpy(), y.cpu().numpy(), x_map.cpu().numpy()],
                           [r'Ground truth $x$',
                            rf'Observation $y$ (SNR={snr_obs:.1f} dB)',
                            rf'MAP reconstruction (SNR={snr_map:.1f} dB, $\theta$={theta_est:.4f})']):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis('off')

fig.suptitle(r'Experiment 4.4-2: MYULA Empirical Bayes (SAPG parameter estimation)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'sapg_results.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════
# 10. 输出结论
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("实验4.4-2 MYULA经验贝叶斯（SAPG参数自适应估计）")
print("=" * 60)
print(f"\n[步骤1] 正向模型: {nx}x{ny}图像, BSNR={BSNR}dB")
print(f"  true sigma^2 = {true_sigma2:.6f}")
print(f"\n[步骤2] SAPG估计:")
print(f"  estimated theta  = {theta_est:.6f}")
print(f"  estimated sigma2 = {sigma2_est:.6f}")
print(f"\n[步骤3] MAP重建:")
print(f"  SNR(observation) = {snr_obs:.2f} dB")
print(f"  SNR(MAP)         = {snr_map:.2f} dB")
print(f"  SNR improvement  = {snr_map - snr_obs:.2f} dB")
print(f"  x_map range: [{x_map.min().item():.1f}, {x_map.max().item():.1f}]")

print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print(r"1. 经验贝叶斯: 从观测数据y中自动估计超参数(theta, sigma^2)")
print(r"2. Fisher恒等式: nabla_theta log p(y|theta) = -E[TV(X)] + d/(alpha*theta)")
print(r"3. SAPG: 用MYULA样本的平均TV替代期望, 随机近似边际似然梯度")
print(r"4. 对数尺度参数化: eta=log(theta), 自动保证theta>0")
print(r"5. 投影到容许集: 防止参数越界")
print(r"6. burn-in截断: 只使用平稳分布区域的样本计算均值")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")
