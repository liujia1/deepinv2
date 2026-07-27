"""
经验贝叶斯参数估计：边际似然最大化
对应章节：3.6 收敛性分析与正则化参数选择（经验贝叶斯部分）
知识点：边际似然 p(y|alpha)、经验贝叶斯策略、高斯共轭闭式解、Fisher恒等式
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os
import sys

# ====== 中文字体配置（兼容本地和Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '经验贝叶斯参数估计')
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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")

np.random.seed(42)

# ══════════════════════════════════════════════════════════
# 1. 构造小规模问题
# ══════════════════════════════════════════════════════════
n = 50
m = 40
A = np.random.randn(m, n) / np.sqrt(m)

x_true = np.zeros(n)
x_true[5] = 1.0
x_true[15] = 0.8
x_true[30] = -0.5

sigma2 = 0.01
y = A @ x_true + np.sqrt(sigma2) * np.random.randn(m)

# ══════════════════════════════════════════════════════════
# 2. 边际似然（高斯共轭闭式解）
# ══════════════════════════════════════════════════════════
def marginal_likelihood(y, A, alpha, sigma2):
    """
    高斯共轭下的边际似然 log p(y|alpha)
    边际协方差: C = sigma^2 * I + alpha * A * A^T

    高斯共轭情形下 p(y|alpha) 有解析闭式解，无需 EM / MCMC / Monte Carlo gradient。

    Fisher恒等式（标准形式）:
        nabla_alpha log p(y|alpha) = E_{p(x|y,alpha)}[nabla_alpha log p(y,x|alpha)]
    当似然 p(y|x) 不依赖 alpha 时（即 sigma^2 固定、alpha 仅出现在先验中），
    nabla_alpha log p(y|x) = 0，上式右端退化为：
        E_{p(x|y,alpha)}[nabla_alpha log p(x|alpha)]
    Fisher恒等式在一般模型中提供梯度的随机近似途径，
    但本实验有闭式边际似然，可直接对 alpha 优化，无需借助Fisher恒等式。
    """
    m = A.shape[0]
    C = sigma2 * np.eye(m) + alpha * A @ A.T
    sign, logdet = np.linalg.slogdet(C)
    C_inv_y = np.linalg.solve(C, y)
    log_ml = -0.5 * m * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * y @ C_inv_y
    return log_ml

# ══════════════════════════════════════════════════════════
# 3. Tikhonov解（用于评估重建质量）
# ══════════════════════════════════════════════════════════
def tikhonov_solve(y, A, alpha, sigma2):
    return np.linalg.solve(
        A.T @ A / sigma2 + np.eye(A.shape[1]) / alpha,
        A.T @ y / sigma2
    )

# ══════════════════════════════════════════════════════════
# 4. 对比不同参数选择方法
# ══════════════════════════════════════════════════════════
alphas = np.logspace(-4, 2, 100)

# (a) 边际似然（经验贝叶斯）
log_ml = np.array([marginal_likelihood(y, A, alpha, sigma2) for alpha in alphas])
alpha_eb = alphas[np.argmax(log_ml)]

# (b) PSNR（需知真解，仅作参考）
peak = np.max(np.abs(x_true))
psnrs = []
for alpha in alphas:
    x_hat = tikhonov_solve(y, A, alpha, sigma2)
    mse = np.mean((x_hat - x_true) ** 2)
    psnrs.append(10 * np.log10(peak ** 2 / mse))
alpha_opt = alphas[np.argmax(psnrs)]

print("=" * 60)
print("经验贝叶斯参数估计")
print("=" * 60)
print(f"经验贝叶斯 alpha_hat = {alpha_eb:.4f}")
print(f"真实最优 alpha       = {alpha_opt:.4f}")
print(f"对数边际似然最大值: log p(y|alpha) = {np.max(log_ml):.2f}")

# ══════════════════════════════════════════════════════════
# 5. 可视化
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].semilogx(alphas, log_ml, 'b-', linewidth=2)
axes[0].axvline(alpha_eb, color='r', linestyle='--',
                label=rf'EB: $\hat{{\alpha}}={alpha_eb:.3f}$')
axes[0].axvline(alpha_opt, color='g', linestyle=':',
                label=rf'最优: $\alpha={alpha_opt:.3f}$')
axes[0].set_xlabel(r'正则化参数 $\alpha$')
axes[0].set_ylabel(r'$\log p(y|\alpha)$')
axes[0].set_title(r'边际似然 $\log p(y|\alpha)$')
axes[0].legend()
axes[0].grid(True)

axes[1].semilogx(alphas, psnrs, 'b-', linewidth=2)
axes[1].axvline(alpha_eb, color='r', linestyle='--',
                label=rf'EB: $\hat{{\alpha}}={alpha_eb:.3f}$')
axes[1].axvline(alpha_opt, color='g', linestyle=':',
                label=rf'最优: $\alpha={alpha_opt:.3f}$')
axes[1].set_xlabel(r'正则化参数 $\alpha$')
axes[1].set_ylabel(r'PSNR (dB)')
axes[1].set_title(r'重建质量 vs $\alpha$')
axes[1].legend()
axes[1].grid(True)

plt.suptitle(r'经验贝叶斯参数估计', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_经验贝叶斯参数估计.png'), dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════
# 6. 结果输出与教学说明
# ══════════════════════════════════════════════════════════
print(f"\n[核心发现]")
print(f"  1. 经验贝叶斯选择的 alpha_hat = {alpha_eb:.4f}")
print(f"     （使边际似然 p(y|alpha) 最大）")
print(f"  2. 真实最优（已知真解）alpha = {alpha_opt:.4f}")
print(f"  3. 经验贝叶斯在无需真解的情况下，自动选择了接近最优的参数")
print(f"  4. 边际似然 p(y|alpha) = integral p(y|x) p(x|alpha) dx")
print(f"     衡量的是 \u201c在参数 alpha 下，观测到数据 y 的可能性\u201d")

print(f"\n[与章节联系]")
print(f"  本实验实现了 3.6 节经验贝叶斯的核心思想：")
print(f"  正则化参数 alpha 的选择从 \u201c手动调参\u201d 升级为 \u201c数据驱动\u201d")
print(f"  高频学派方法（L曲线、交叉验证）缺乏概率论根基，")
print(f"  而边际似然最大化为参数选择提供了严格的贝叶斯解释")

print(f"\n{'=' * 60}")
print(f"实验完成")
print(f"输出图片: {os.path.join(SAVE_DIR, '步骤1_经验贝叶斯参数估计.png')}")
print(f"{'=' * 60}")

# ===== 保存数值结果 =====
import json
# 计算经验贝叶斯与最优alpha下的重建PSNR
x_hat_eb = tikhonov_solve(y, A, alpha_eb, sigma2)
x_hat_opt = tikhonov_solve(y, A, alpha_opt, sigma2)
mse_eb = np.mean((x_hat_eb - x_true) ** 2)
mse_opt = np.mean((x_hat_opt - x_true) ** 2)
psnr_eb = float(round(10 * np.log10(peak ** 2 / mse_eb), 2))
psnr_opt = float(round(10 * np.log10(peak ** 2 / mse_opt), 2))

results_summary = {
    'n': n,
    'm': m,
    'sigma2': float(round(sigma2, 4)),
    'alpha_eb': float(round(alpha_eb, 4)),
    'alpha_opt': float(round(alpha_opt, 4)),
    'log_ml_max': float(round(np.max(log_ml), 2)),
    'psnr_eb': psnr_eb,
    'psnr_opt': psnr_opt,
}

def _to_native(obj):
    import numpy as np
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return _to_native(obj.tolist())
    try:
        import torch
        if isinstance(obj, torch.Tensor): return _to_native(obj.detach().cpu().tolist())
    except: pass
    return obj

results_summary = {k: _to_native(v) for k, v in results_summary.items()}
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")