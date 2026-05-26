"""
实验4.5-1 半二次最小化 vs GLM（Gibbs采样）：MAP vs MMSE
对应章节：4.5 Gibbs采样——利用条件结构的采样
知识点：Gibbs交替采样；半二次最小化→GLM；高斯尺度混合表示；
        GIG分布采样；Gibbs=带噪声的交替最小化；MAP vs MMSE

修改说明：
  从原参考实验4.5.py迁移，适配配套实验风格（matplotlib.use('Agg')、
  中文字体模块、Colab支持、LaTeX格式符号）。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from scipy.stats import geninvgauss
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
    SAVE_DIR = os.path.join(_gdrive, '实验4.5-1')
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
# ========================================================

np.random.seed(42)
torch.manual_seed(42)

# ══════════════════════════════════════════════════════════
# 1. 辅助函数
# ══════════════════════════════════════════════════════════

def CG(x, Ax, sigma, lamb, z, b, maxit=100, verbose=0):
    r = b - Ax(sigma, lamb, z, x)
    p = r
    rsold = torch.sum(r**2)
    for it in range(maxit):
        Ap = Ax(sigma, lamb, z, p)
        alpha = rsold / torch.sum(p * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(r**2)
        rsold = rsnew.clone()
        p = r + rsnew / rsold * p
        R = torch.mean(r**2)
        if R < 1e-9:
            break
    if verbose > 0:
        print("CG: it = ", it, ", mse = ", R)
    return x

def Du(u):
    M, N = u.shape
    Du_out = torch.zeros((2, M, N))
    Du_out[0, :, :-1] += u[:, 1:] - u[:, :-1]
    Du_out[1, :-1, :] += u[1:, :] - u[:-1, :]
    return Du_out

def DTp(p):
    C, M, N = p.shape
    DTp_out = torch.zeros((M, N))
    DTp_out[:, 1:] += p[0, :, :-1]
    DTp_out[:, :-1] -= p[0, :, :-1]
    DTp_out[1:, :] += p[1, :-1, :]
    DTp_out[:-1, :] -= p[1, :-1, :]
    return DTp_out

# ══════════════════════════════════════════════════════════
# 2. 合成图像与加噪
# ══════════════════════════════════════════════════════════
N = 100
x_grid = np.linspace(-1, 1, N)
y_grid = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x_grid, y_grid)

u1 = 0 * X + 0.5
u2 = X + Y
u2 -= u2.min()
u2 /= u2.max()

u3 = np.clip(1 - ((1 - X)**2 + (1 - Y)**2), a_min=0, a_max=None)
u3 -= u3.min()
u3 /= u3.max()

m1 = 0 * X
m1[N//8:-N//8, N//8:-N//8] = 1
u = u1.copy()
u[m1 == 1] = u2[m1 == 1] * 1.25

m2 = 0 * X
m2[(X - 1)**2 + (Y - 1)**2 <= 1] = 1
u[m2 == 1] = u3[m2 == 1]

sigma = 0.1
f = u + np.random.randn(N, N) * sigma
f_ = torch.from_numpy(f)

# ══════════════════════════════════════════════════════════
# 3. 步骤1：MAP——半二次最小化
# ══════════════════════════════════════════════════════════
# |t| = min_{z>0} { t^2/(2z) + z/2 }
# 交替更新：z = clip(|Du|, eps)，然后用CG求解u

print("=" * 60)
print("实验4.5-1 半二次最小化 vs GLM（Gibbs采样）")
print("=" * 60)

maxiter_hq = 50
lamb_hq = 10.0

Au_hq = lambda sigma, lamb, z, u: lamb_hq * DTp(Du(u) / z) + u / sigma**2
rhs_hq = f_ / sigma**2

u_ = torch.from_numpy(f)

print(f"\n[步骤1] 半二次最小化 MAP (lambda={lamb_hq}, {maxiter_hq} iters)")
for it in range(maxiter_hq):
    diff_u = torch.abs(Du(u_))
    z_ = np.clip(diff_u.numpy(), a_min=1e-6, a_max=None)
    z_ = torch.from_numpy(z_)
    u_ = CG(u_, Au_hq, sigma, lamb_hq, z_, rhs_hq, maxit=1000, verbose=0)
    TV = lamb_hq * diff_u.sum()
    Dat = torch.sum((u_ - f_)**2 / (2 * sigma**2))
    if (it + 1) % 10 == 0:
        print(f"  iter {it+1}: TV + Dat = {(TV + Dat).item():.6f}")

u_hq = u_
print("  半二次最小化完成")

# ══════════════════════════════════════════════════════════
# 4. 步骤2：MMSE——GLM Gibbs采样
# ══════════════════════════════════════════════════════════
# Gibbs交替采样：
#   z_j ~ GIG(lambda^2, |Du_j|^2, 1/2)  (辅助变量)
#   x ~ N(mu(z), Sigma(z))              (条件高斯，CG+扰动)
#
# 对比半二次最小化：
#   确定版：z = clip(|Du|, eps)  vs  Gibbs版：z ~ GIG(0.5, lambda*|Du|)
#   确定版：CG精确右端          vs  Gibbs版：CG求解带高斯扰动的右端

maxiter_glm = 50
lamb_glm = 20.0

u_ = torch.from_numpy(f)

Au_glm = lambda sigma, lamb, z, u: DTp(Du(u) / z) + u / sigma**2

u_sum_ = 0
u_sqr_ = 0

print(f"\n[步骤2] GLM Gibbs采样 MMSE (lambda={lamb_glm}, {maxiter_glm} iters)")
for it in range(maxiter_glm):
    diff_u = torch.abs(Du(u_))
    tmp = diff_u.numpy().ravel()
    tmp[tmp == 0] = 1e-30

    z_ = torch.from_numpy(geninvgauss.rvs(0.5, lamb_glm * tmp) * tmp / lamb_glm).reshape(2, N, N)
    eta = torch.normal(0, 1, size=(3, N, N))

    rhs = DTp(eta[:2] / torch.sqrt(z_)) + eta[2] / sigma + f_ / sigma**2
    u_ = CG(u_, Au_glm, sigma, lamb_glm, z_, rhs, maxit=1000, verbose=0)

    u_sum_ += u_
    u_sqr_ += u_**2

    TV = lamb_glm * diff_u.sum()
    Dat = torch.sum((u_ - f_)**2 / (2 * sigma**2))
    if (it + 1) % 10 == 0:
        print(f"  iter {it+1}: TV + Dat = {(TV + Dat).item():.6f}")

u_avg_ = u_sum_ / maxiter_glm
u_var_ = u_sqr_ / maxiter_glm - u_avg_**2
print("  GLM Gibbs采样完成")

# ══════════════════════════════════════════════════════════
# 5. 可视化
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0][0].imshow(u, cmap='gray', vmin=0, vmax=1)
axes[0][0].set_title(r'Ground truth $u$')
axes[0][0].axis('off')

axes[0][1].imshow(f, cmap='gray', vmin=0, vmax=1)
axes[0][1].set_title(rf'Noisy image ($\sigma={sigma}$)')
axes[0][1].axis('off')

axes[0][2].axis('off')

axes[1][0].imshow(u_hq.reshape(N, N).numpy(), cmap='gray', vmin=0, vmax=1)
axes[1][0].set_title(r'MAP (Half-quadratic minimization)')
axes[1][0].axis('off')

axes[1][1].imshow(u_avg_.reshape(N, N).numpy(), cmap='gray', vmin=0, vmax=1)
axes[1][1].set_title(r'MMSE (GLM Gibbs posterior mean)')
axes[1][1].axis('off')

axes[1][2].imshow(torch.sqrt(torch.clamp(u_var_, min=0)).numpy(), cmap='hot')
axes[1][2].set_title(r'Posterior std (uncertainty)')
axes[1][2].axis('off')

fig.suptitle(r'Experiment 4.5-1: Half-quadratic minimization vs GLM (Gibbs sampling)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'hq_vs_glm_results.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════
# 6. 输出结论
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print(r"1. 半二次最小化: |t| = min_{z>0} {t^2/(2z) + z/2}")
print(r"   确定性: z = clip(|Du|, eps), 然后CG求解u -> MAP")
print(r"2. GLM Gibbs: z ~ GIG(0.5, lambda*|Du|), 然后CG+扰动采样u -> MMSE")
print(r"3. Gibbs = 半二次最小化 + 噪声")
print(r"   确定版: z = clip(|Du|, eps)  vs  Gibbs版: z ~ GIG(0.5, lambda*|Du|)")
print(r"   确定版: CG精确右端          vs  Gibbs版: CG求解带高斯扰动的右端")
print(r"4. MAP收敛到众数(分段常数), MMSE收敛到分布(后验均值+不确定性)")
print(r"5. 统一转化律: 优化算法 + 噪声 = 采样算法")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")
