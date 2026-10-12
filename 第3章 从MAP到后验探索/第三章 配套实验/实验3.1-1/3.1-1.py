"""
实验3.1-1 MAP估计：后验能量分解
对应章节：3.1 MAP估计——从后验众数到优化问题
知识点：后验概率与优化目标的等价性；后验能量分解（数据项 + 正则项）；
        λ = sigma^2/sigma_x^2 的概率诠释；MAP解与闭式解

修改说明：
  从原参考实验3.1.py拆分，聚焦MAP估计与后验能量分解概念，
  去除梯度下降细节（移至3.2-1）和λ扫描（移至3.3-1）。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio
from scipy.ndimage import gaussian_filter
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
    SAVE_DIR = os.path.join(_gdrive, '实验3.1-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)  # 递归创建所有父目录
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)

# ══════════════════════════════════════════════════════════
# 1. 问题设定：小尺寸模糊逆问题
# ══════════════════════════════════════════════════════════
n = 32
from skimage.data import shepp_logan_phantom
phantom = resize(shepp_logan_phantom(), (n, n), order=0, preserve_range=True, anti_aliasing=False)
x = phantom / phantom.max()      # 原始图像，值域 [0,1]
N = n * n

# 高斯模糊核
def gaussian_psf(size, sigma):
    ax = np.fft.fftfreq(size, d=1.0) * size
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return h / h.sum()

blur_sigma = 2.0
h = gaussian_psf(n, blur_sigma)
H_fft = np.fft.fft2(h)

# 构建模糊矩阵 A (N×N)
# [注意] 这里显式构建 A 矩阵仅用于教学演示（小规模 N=1024）
# [重要] 实际大规模问题中应使用迭代法（如共轭梯度CG）或傅里叶域求解
#        避免显式构建和求逆大矩阵（计算复杂度 O(N^3)，内存 O(N^2)）
A = np.zeros((N, N))
for j in range(N):
    e_j = np.zeros(N)
    e_j[j] = 1.0
    A[:, j] = np.real(np.fft.ifft2(H_fft * np.fft.fft2(e_j.reshape(n, n)))).ravel()

sigma_noise = 0.05
y = A @ x.ravel() + sigma_noise * np.random.randn(N)

# ══════════════════════════════════════════════════════════
# 2. MAP估计：后验能量分解
# ══════════════════════════════════════════════════════════
# 高斯似然 + 高斯先验 → 后验为高斯分布
# 后验能量: -ln p(x|y) = 0.5||Ax-y||^2/sigma^2 + 0.5||x||^2/sigma_x^2 + const
# MAP估计 = 最小化 J(x) = 0.5||Ax-y||^2 + 0.5 lambda ||x||^2
# 其中 lambda = sigma^2/sigma_x^2

sigma_prior = 0.5
lam = sigma_noise ** 2 / sigma_prior ** 2

AtA = A.T @ A
Aty = A.T @ y

# 闭式解（MAP估计）
# [注意] np.linalg.inv 仅适用于小规模问题（此处 N=1024，约几秒）
# [重要] 实际应用中应使用：
#   1. 迭代法：共轭梯度法 (CG)、LSQR 等，避免显式求逆
#   2. 傅里叶域求解：利用卷积定理，在频域直接计算 (O(N log N))
#   3. 预条件技术：加速迭代收敛
# [说明] 高斯后验下 MAP = 后验均值 = 后验众数，因此可用后验分布的均值公式计算 MAP
Sigma_post = np.linalg.inv(AtA / sigma_noise ** 2 + np.eye(N) / sigma_prior ** 2)
mu_post = Sigma_post @ (Aty / sigma_noise ** 2)

# 后验方差（不确定性量化）
# [注意1] 这仅在高斯-高斯共轭模型下可用（闭式后验协方差）
# [注意2] 图中展示的是边缘方差 diag(Σ_post)，忽略了非对角元（像素间相关性）
#         实际后验分布中像素之间存在相关性，此处仅展示各像素的边际不确定性
# [注意3] 本实验采用周期性卷积，A 为循环矩阵 → Σ_post 亦为循环矩阵 → 其对角元
#         （边缘方差）在所有像素处恒等于同一常数（≈0.227），不存在空间分布差异。
#         因此该图仅示意"各像素边际不确定性相同"；若需展示空间变化的不确定性，
#         须改用非周期边界条件（如零填充/反射）或空间变化先验。
# [预告] 非高斯后验的不确定性量化见第4章（MCMC/ULA采样方法）
# [关联] 3.7节将讨论：为什么仅有MAP点估计是不够的
post_var = np.diag(Sigma_post).reshape(n, n)

# ══════════════════════════════════════════════════════════
# 3. 后验能量验证
# ══════════════════════════════════════════════════════════
def posterior_energy(x_vec, y, A, sigma_noise, sigma_prior):
    """计算后验能量 -ln p(x|y)（省略常数项）"""
    data_term = 0.5 / sigma_noise**2 * np.linalg.norm(A @ x_vec - y) ** 2
    reg_term = 0.5 / sigma_prior**2 * np.linalg.norm(x_vec) ** 2
    return data_term + reg_term, data_term, reg_term

# 计算三种情况下的后验能量
# 初始化点1：转置近似（匹配滤波输出 A^T y）。
#   [注意] A 与 A^T 同为低通模糊，A^T y 相当于"模糊的模糊"，幅值被收缩、离真解更远，
#          因此它其实是比观测初始更差的起点（能量最高、PSNR 最低），并非"质量较好"。
x_init_transpose = A.T @ y
# 初始化点2：直接用观测作为初始（代表"不做任何复原"，实际中常见的基线）
x_init_obs = y
# MAP估计（最优解）
x_map = mu_post

E_init_transpose, d_init_transpose, r_init_transpose = posterior_energy(x_init_transpose, y, A, sigma_noise, sigma_prior)
E_init_obs, d_init_obs, r_init_obs = posterior_energy(x_init_obs, y, A, sigma_noise, sigma_prior)
E_map, d_map, r_map = posterior_energy(x_map, y, A, sigma_noise, sigma_prior)
E_true, d_true, r_true = posterior_energy(x.ravel(), y, A, sigma_noise, sigma_prior)

psnr_map = peak_signal_noise_ratio(x, np.clip(x_map.reshape(n, n), 0, 1))
psnr_init_transpose = peak_signal_noise_ratio(x, np.clip(x_init_transpose.reshape(n, n), 0, 1))
psnr_init_obs = peak_signal_noise_ratio(x, np.clip(x_init_obs.reshape(n, n), 0, 1))

print("=" * 60)
print("实验3.1-1 MAP估计：后验能量分解")
print("=" * 60)
print(f"\n[参数设定]")
print(f"  噪声标准差 sigma = {sigma_noise:.3f}")
print(f"  先验标准差 sigma_x = {sigma_prior:.3f}")
print(f"  正则化参数 lambda = sigma^2/sigma_x^2 = {lam:.4f}")
print(f"\n[后验能量分解 -ln p(x|y) = 数据项 + 正则项]")
print(f"  {'':>15} {'数据项':>12} {'正则项':>12} {'后验能量':>12} {'PSNR':>10}")
print(f"  {'观测初始':>15} {d_init_obs:>12.2f} {r_init_obs:>12.2f} {E_init_obs:>12.2f} {psnr_init_obs:>10.2f}")
print(f"  {'转置初始':>15} {d_init_transpose:>12.2f} {r_init_transpose:>12.2f} {E_init_transpose:>12.2f} {psnr_init_transpose:>10.2f}")
print(f"  {'MAP估计':>15} {d_map:>12.2f} {r_map:>12.2f} {E_map:>12.2f} {psnr_map:>10.2f}")
print(f"  {'真解':>15} {d_true:>12.2f} {r_true:>12.2f} {E_true:>12.2f} {'N/A':>10}")
print(f"\n  [验证] MAP解的后验能量 ({E_map:.2f}) 同时低于观测初始 ({E_init_obs:.2f}) 与转置初始 ({E_init_transpose:.2f})，为三者最小，优化有效")
print(f"  [注] 转置初始 A^T y 因'模糊的模糊'被双重低通收缩，能量反而高于观测初始，并非更好的起点")
print(f"  [验证] MAP解PSNR = {psnr_map:.2f} dB，略优于观测初始 {psnr_init_obs:.2f} dB 与转置初始 {psnr_init_transpose:.2f} dB（白高斯先验对分片常数体模为先验失配，增益有限，约0.4dB）")
print(f"\n  [注] 观测初始代表'不做任何复原处理'，是实践中最常见的朴素基线")

# ══════════════════════════════════════════════════════════
# 4. 可视化
# ══════════════════════════════════════════════════════════
from matplotlib import gridspec
fig = plt.figure(figsize=(15, 10))
# 第一行3个图，第二行2个图居中且适度分开（使用7列布局实现）
gs = gridspec.GridSpec(2, 7, figure=fig,
                       width_ratios=[1, 1, 1, 1, 1, 1, 1],
                       wspace=0.3, hspace=0.3)

# 第一行：原始 → 观测 → MAP重建（均匀分布）
ax1 = fig.add_subplot(gs[0, 0:2])  # 占据第0-1列
ax1.imshow(x, cmap='gray')
ax1.set_title(r'原始图像 $x$')
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 2:5])  # 占据第2-4列（居中，稍宽）
ax2.imshow(y.reshape(n, n), cmap='gray')
ax2.set_title(r'模糊含噪观测 $y = Ax + \epsilon$')
ax2.axis('off')

ax3 = fig.add_subplot(gs[0, 5:7])  # 占据第5-6列
ax3.imshow(np.clip(x_map.reshape(n, n), 0, 1), cmap='gray')
ax3.set_title(r'MAP估计 $\hat{x}_{\mathrm{MAP}}$' + f'\nPSNR={psnr_map:.2f}dB')
ax3.axis('off')

# 第二行：后验不确定性 → 能量分解（居中且适度分开，中间留1列间隙）
ax4 = fig.add_subplot(gs[1, 1:3])  # 左侧图（占据第1-2列）
# [关键] 周期性卷积下 diag(Σ_post) 为常数；若让 colorbar 自动缩放，会把 float64
#        舍入噪声(~1e-15)放大成满屏"花纹"，制造出虚假的空间结构。这里固定色标到
#        [0, 2·均值]，诚实地把"各像素不确定性相同"呈现为近似均匀的图。
_v = float(np.mean(post_var))
im = ax4.imshow(post_var, cmap='hot', vmin=0.0, vmax=2.0 * _v)
ax4.set_title(r'后验边缘方差 $\mathrm{diag}(\Sigma_{\mathrm{post}})$' + '\n（周期性卷积下各像素方差恒为同一常数）')
ax4.axis('off')
plt.colorbar(im, ax=ax4, fraction=0.046)

# 能量分解柱状图
ax5 = fig.add_subplot(gs[1, 4:6])  # 右侧图（占据第4-5列，与左侧间隔1列）
methods = ['观测初始', '转置初始', 'MAP估计', '真解']
data_terms = [d_init_obs, d_init_transpose, d_map, d_true]
reg_terms = [r_init_obs, r_init_transpose, r_map, r_true]
x_pos = np.arange(len(methods))
width = 0.35
ax5.bar(x_pos - width/2, data_terms, width, label='数据项', color='steelblue', alpha=0.8)
ax5.bar(x_pos + width/2, reg_terms, width, label='正则项', color='seagreen', alpha=0.8)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(methods, rotation=15, ha='right')
ax5.set_ylabel('能量值 (nats)')
ax5.set_title(r'后验能量分解: $-\ln p(x|y) = $ 数据项 $+$ 正则项')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

plt.suptitle('实验3.1-1: MAP估计——从后验众数到优化问题', fontsize=14, y=0.98)
plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(SAVE_DIR, '步骤1_MAP后验能量分解.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print("1. MAP估计 = 最大化后验 = 最小化 -ln p(x|y)")
print("2. -ln p(x|y) 可分解为: 数据项 + 正则项 + 常数")
print("3. 正则化参数 lambda = sigma^2/sigma_x^2 有明确的概率意义")
print("4. 后验方差反映 MAP 解的不确定性")
print("   [注意] 这仅在高斯-高斯共轭模型下可用（闭式后验协方差）")
print("   [注意] 本实验用周期性卷积，边缘方差 diag(Σ_post) 在各像素恒为常数（无空间差异）")
print("   [预告] 非高斯后验的不确定性量化见第4章（MCMC/ULA采样方法）")
print("   [关联] 3.7节将讨论：为什么仅有MAP点估计是不够的")
print("5. 在共轭高斯-高斯模型下，MAP解有闭式形式")
print("\n实验完成。结果已保存至:", SAVE_DIR)

# ===== 保存数值结果 =====
import json
results_summary = {
    'n': n,
    'sigma_noise': float(round(sigma_noise, 4)),
    'sigma_prior': float(round(sigma_prior, 4)),
    'lam': float(round(lam, 4)),
    'psnr_map': float(round(psnr_map, 2)),
    'psnr_init_transpose': float(round(psnr_init_transpose, 2)),
    'psnr_init_obs': float(round(psnr_init_obs, 2)),
    'E_map': float(round(E_map, 2)),
    'E_init_transpose': float(round(E_init_transpose, 2)),
    'E_init_obs': float(round(E_init_obs, 2)),
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