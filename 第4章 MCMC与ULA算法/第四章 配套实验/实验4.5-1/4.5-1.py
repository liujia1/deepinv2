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
# 0. GPU检测与缓存配置
# ══════════════════════════════════════════════════════════
_has_gpu = torch.cuda.is_available()
if _has_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Device: {device}")

CACHE_FILE = os.path.join(SAVE_DIR, 'experiment_4.5-1_cache.pth')
USE_CACHE = os.path.exists(CACHE_FILE)

if USE_CACHE:
    print(f"\n发现缓存文件: {CACHE_FILE}")
    print("加载缓存结果（跳过计算）...")
    cache = torch.load(CACHE_FILE, map_location=device, weights_only=False)
    u = cache['u']
    f = cache['f']
    f_ = cache['f_'].to(device)
    u_hq = cache['u_hq'].to(device)
    u_avg_ = cache['u_avg_'].to(device)
    u_var_ = cache['u_var_'].to(device)
    sigma = cache['sigma']
    N = cache['N']
    print("缓存加载完成")
else:
    print("\n未找到缓存文件，开始计算...")

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
        beta = rsnew / rsold
        rsold = rsnew.clone()
        p = r + beta * p
        R = torch.mean(r**2)
        if R < 1e-9:
            break
    if verbose > 0:
        print("CG: it = ", it, ", mse = ", R)
    return x

def Du(u):
    M, N = u.shape
    Du_out = torch.zeros((2, M, N), dtype=u.dtype, device=u.device)
    Du_out[0, :, :-1] += u[:, 1:] - u[:, :-1]
    Du_out[1, :-1, :] += u[1:, :] - u[:-1, :]
    return Du_out

def DTp(p):
    C, M, N = p.shape
    DTp_out = torch.zeros((M, N), dtype=p.dtype, device=p.device)
    DTp_out[:, 1:] += p[0, :, :-1]
    DTp_out[:, :-1] -= p[0, :, :-1]
    DTp_out[1:, :] += p[1, :-1, :]
    DTp_out[:-1, :] -= p[1, :-1, :]
    return DTp_out

# ★ 自检：验证Du/DTp伴随算子关系与边界处理(避免discrete boundary条件改错)
# 伴随关系: 对任意u, p, 应有 <Du(u), p> = <u, DTp(p)>  (内积守恒)
# 这是GLM中交替采样的数学基础(也是18.4-1标准化TV散度修复的依据)
# 检验规则: |<Du(u), p> - <u, DTp(p)>| / <u,u> < 1e-10 (浮点精度内)
print(f"[算子自检] Du/DTp伴随关系验证(实际运行, 非手算):")
np.random.seed(42)
_torch_device = torch.device('cpu')  # 算子自检不依赖GPU, 始终CPU
_M, _N = 20, 30
_u_test = torch.randn(_M, _N, device=_torch_device)
_p_test = torch.randn(2, _M, _N, device=_torch_device)
_Du_u = Du(_u_test)
_DTp_p = DTp(_p_test)
# 内积: <Du(u), p> = sum(Du_u * p), <u, DTp(p)> = sum(u * DTp_p)
_ip_left = (_Du_u * _p_test).sum().item()
_ip_right = (_u_test * _DTp_p).sum().item()
_diff = abs(_ip_left - _ip_right)
_norm = (_u_test * _u_test).sum().item()
print(f"  <Du(u), p>={_ip_left:.4f}, <u, DTp(p)>={_ip_right:.4f}, 差={_diff:.2e}, 相对={_diff/max(_norm,1e-10):.2e}")
print(f"  伴随关系成立: {'OK' if _diff / max(_norm, 1e-10) < 1e-10 else 'FAIL'}")
# 额外验证: DTp(Du(u)) = -离散Laplace算子·u (内部点), 边界处因fill=0有数值差异
_Lap = DTp(Du(_u_test))
_interior_err = (_Lap[1:-1, 1:-1] - ((_u_test[:-2, 1:-1] + _u_test[2:, 1:-1] + _u_test[1:-1, :-2] + _u_test[1:-1, 2:]) - 4*_u_test[1:-1, 1:-1])).abs().max().item()
print(f"  DTp(Du(u))与5点Laplace差(内部点): {_interior_err:.2e} (应≈0, 边界fill方式决定边缘值)")

# ══════════════════════════════════════════════════════════
# 2. 合成图像与加噪
# ══════════════════════════════════════════════════════════
if not USE_CACHE:
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
    f_ = torch.from_numpy(f).float().to(device)  # 统一为float32，移到GPU

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

print(f"\n[步骤1] 半二次最小化 MAP (lambda={lamb_hq}, {maxiter_hq} iters)")
if not USE_CACHE:
    u_ = torch.from_numpy(f).float().to(device)  # 统一为float32，移到GPU
    for it in range(maxiter_hq):
        diff_u = torch.abs(Du(u_))
        z_ = np.clip(diff_u.cpu().numpy(), a_min=1e-6, a_max=None)  # 移回CPU进行clip
        z_ = torch.from_numpy(z_).float().to(device)  # 统一为float32，移到GPU
        u_ = CG(u_, Au_hq, sigma, lamb_hq, z_, rhs_hq, maxit=1000, verbose=0)
        # 使用更新后的u_计算TV和Dat，确保一致性
        diff_u = torch.abs(Du(u_))
        TV = lamb_hq * diff_u.sum()
        Dat = torch.sum((u_ - f_)**2 / (2 * sigma**2))
        if (it + 1) % 10 == 0:
            print(f"  iter {it+1}: TV + Dat = {(TV + Dat).item():.6f}")

    u_hq = u_
    print("  半二次最小化完成")
else:
    print("  从缓存加载 MAP 结果")

# ══════════════════════════════════════════════════════════
# 4. 步骤2：MMSE——GLM Gibbs采样
# ══════════════════════════════════════════════════════════
# Gibbs交替采样：
#   z_j ~ GIG(lambda^2, |Du_j|^2, 1/2)  (辅助变量)
#   x ~ N(mu(z), Sigma(z))              (条件高斯，CG+扰动)
#
# 注意：迭代次数影响MMSE估计的精度。对于100x100图像，
# 50次迭代（含20次burn-in）仅使用30个有效样本，后验均值估计方差较大。
# 教学实验中可适当增加至200次以获得更稳定的结果。

maxiter_glm = 50
lamb_glm = 10.0  # 与lamb_hq保持一致，确保MAP与MMSE对比的公平性

# Au_glm: GLM条件高斯的系统矩阵
# 注意：lamb参数未在函数体中使用，因为λ已通过GIG采样隐含在z_中
# 保留lamb参数是为了与Au_hq保持一致的函数签名
Au_glm = lambda sigma, lamb, z, u: DTp(Du(u) / z) + u / sigma**2

u_sum_ = 0
u_sqr_ = 0
n_burn_in = 20  # burn-in期，丢弃前20次迭代的样本

print(f"\n[步骤2] GLM Gibbs采样 MMSE (lambda={lamb_glm}, {maxiter_glm} iters, burn-in={n_burn_in})")
if not USE_CACHE:
    u_ = torch.from_numpy(f).float().to(device)  # 统一为float32，移到GPU
    for it in range(maxiter_glm):
        diff_u = torch.abs(Du(u_))
        tmp = diff_u.cpu().numpy().ravel()  # 移回CPU进行GIG采样
        tmp[tmp == 0] = 1e-30

        # z_j ~ GIG(1/2, λ², |Du_j|²)
        # 令 c = |Du|/λ，X ~ geninvgauss(0.5, λ|Du|)，则 z = cX = X·|Du|/λ
        z_ = torch.from_numpy(geninvgauss.rvs(0.5, lamb_glm * tmp) * tmp / lamb_glm).float().reshape(2, N, N).to(device)
        eta = torch.normal(0, 1, size=(3, N, N)).to(device)

        rhs = DTp(eta[:2] / torch.sqrt(z_)) + eta[2] / sigma + f_ / sigma**2
        u_ = CG(u_, Au_glm, sigma, lamb_glm, z_, rhs, maxit=1000, verbose=0)

        # 仅在burn-in期后累加样本
        if it >= n_burn_in:
            u_sum_ += u_
            u_sqr_ += u_**2

        # 使用更新后的u_计算TV和Dat，确保一致性
        diff_u = torch.abs(Du(u_))
        TV = lamb_glm * diff_u.sum()
        Dat = torch.sum((u_ - f_)**2 / (2 * sigma**2))
        if (it + 1) % 10 == 0:
            print(f"  iter {it+1}: TV + Dat = {(TV + Dat).item():.6f}")

    n_samples = maxiter_glm - n_burn_in
    u_avg_ = u_sum_ / n_samples
    u_var_ = u_sqr_ / n_samples - u_avg_**2
    print(f"  GLM Gibbs采样完成 (有效样本数: {n_samples}, 总迭代={maxiter_glm}, burn-in={n_burn_in})")
else:
    print("  从缓存加载 MMSE 结果")

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

axes[1][0].imshow(u_hq.cpu().numpy(), cmap='gray', vmin=0, vmax=1)  # 移回CPU可视化
axes[1][0].set_title(r'MAP (Half-quadratic minimization)')
axes[1][0].axis('off')

axes[1][1].imshow(u_avg_.cpu().numpy(), cmap='gray', vmin=0, vmax=1)  # 移回CPU可视化
axes[1][1].set_title(r'MMSE (GLM Gibbs posterior mean)')
axes[1][1].axis('off')

axes[1][2].imshow(torch.sqrt(torch.clamp(u_var_, min=0)).cpu().numpy(), cmap='hot')  # 移回CPU可视化
axes[1][2].set_title(r'Posterior std (uncertainty)')
axes[1][2].axis('off')

fig.suptitle(r'Experiment 4.5-1: Half-quadratic minimization vs GLM (Gibbs sampling)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'hq_vs_glm_results.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════
# 6. 保存缓存（用于下次快速加载）
# ══════════════════════════════════════════════════════════
if not USE_CACHE:
    cache = {
        'u': u,
        'f': f,
        'f_': f_.cpu(),  # 保存时移回CPU，确保跨设备兼容
        'u_hq': u_hq.cpu(),
        'u_avg_': u_avg_.cpu(),
        'u_var_': u_var_.cpu(),
        'sigma': sigma,
        'N': N,
    }
    torch.save(cache, CACHE_FILE)
    print(f"\n缓存已保存至: {CACHE_FILE}")

# ══════════════════════════════════════════════════════════
# 7. 输出结论
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

# ══════════════════════════════════════════════════════════
# 保存数值结果到JSON文件
# ══════════════════════════════════════════════════════════
import json

def _to_native(obj):
    """递归将numpy/torch类型转换为Python原生类型，便于JSON序列化"""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, 'item') and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except (ValueError, RuntimeError, TypeError):
            return obj
    return obj

results_summary = {
    'experiment': '4.5-1',
    'title': '半二次最小化 vs GLM（Gibbs采样）',
    'setup': {
        'image_size': [int(N), int(N)],
        'sigma': float(sigma),
        'lamb_hq': float(lamb_hq),
        'lamb_glm': float(lamb_glm),
        'maxiter_hq': maxiter_hq,
        'maxiter_glm': maxiter_glm,
        'n_burn_in': n_burn_in,
    }
}

with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(_to_native(results_summary), f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
