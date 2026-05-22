"""
实验3.2-1 梯度下降优化：步长选择与收敛分析
对应章节：3.2 优化基础——凸性、光滑性与梯度下降
知识点：Lipschitz常数；梯度下降步长 tau < 2/L；收敛曲线；
        不同步长对收敛速率的影响；目标函数单调下降

修改说明：
  从原参考实验3.1.py拆分，聚焦梯度下降优化原理，
  去除MAP后验推导（移至3.1-1）和λ扫描（移至3.3-1）。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio
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
    SAVE_DIR = os.path.join(_gdrive, '实验3.2-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)  # 递归创建所有父目录
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

# ══════════════════════════════════════════════════════════
# 1. 问题设定
# ══════════════════════════════════════════════════════════
n = 32
N = n * n
from skimage.data import shepp_logan_phantom
phantom = resize(shepp_logan_phantom(), (n, n), order=0, preserve_range=True, anti_aliasing=False)
x = phantom / phantom.max()

def gaussian_psf(size, sigma):
    ax = np.concatenate((np.arange(0, size // 2), np.arange(-size // 2, 0)))
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return h / h.sum()

h = gaussian_psf(n, sigma=2.0)
H_fft = np.fft.fft2(h)

A = np.zeros((N, N))
for j in range(N):
    e_j = np.zeros(N)
    e_j[j] = 1.0
    A[:, j] = np.real(np.fft.ifft2(H_fft * np.fft.fft2(e_j.reshape(n, n)))).ravel()

sigma_noise = 0.05
y = A @ x.ravel() + sigma_noise * np.random.randn(N)

# ══════════════════════════════════════════════════════════
# 2. 目标函数与梯度
# ══════════════════════════════════════════════════════════
lam = 0.01

def objective(x_vec, A, y, lam):
    """Tikhonov目标函数 J(x) = 0.5||Ax-y||^2 + 0.5*lam*||x||^2"""
    return 0.5 * np.linalg.norm(A @ x_vec - y) ** 2 + 0.5 * lam * np.linalg.norm(x_vec) ** 2

def gradient(x_vec, A, y, lam):
    """梯度 nabla J(x) = A^T(Ax - y) + lam*x"""
    return A.T @ (A @ x_vec - y) + lam * x_vec

# ══════════════════════════════════════════════════════════
# 3. Lipschitz常数估计（幂迭代法）
# ══════════════════════════════════════════════════════════
# 目标函数的Hessian: H = A^T A + lam*I
# Lipschitz常数 L = lambda_max(A^T A) + lam = ||A||^2 + lam
# 根据梯度下降收敛理论，步长需满足 tau < 2/L

def estimate_lipschitz(A, lam, n_iter=50):
    """用幂迭代估计 ||A||^2，进而得到Lipschitz常数 L = ||A||^2 + lam"""
    N = A.shape[0]
    v = np.random.randn(N)
    v = v / np.linalg.norm(v)
    for _ in range(n_iter):
        v = A.T @ (A @ v)
        v = v / np.linalg.norm(v)
    A_norm_sq = np.dot(v, A.T @ (A @ v))
    return A_norm_sq + lam

L = estimate_lipschitz(A, lam, n_iter=50)
tau_opt = 1.0 / L          # 最优步长 (梯度下降理论中的标准选择)
tau_max = 2.0 / L           # 收敛的理论上限

print("=" * 60)
print("实验3.2-1 梯度下降优化：步长选择与收敛分析")
print("=" * 60)
print(f"\n[Lipschitz常数估计]")
print(f"  估计的 L = {L:.4f}")
print(f"  理论最优步长 tau_opt = 1/L = {tau_opt:.6f}")
print(f"  收敛上限 tau_max = 2/L = {tau_max:.6f}")

# ══════════════════════════════════════════════════════════
# 4. 梯度下降：不同步长对比
# ══════════════════════════════════════════════════════════
def gradient_descent(x0, A, y, lam, tau, n_iter):
    """梯度下降算法"""
    x = x0.copy()
    obj_hist = np.zeros(n_iter)
    for k in range(n_iter):
        x = x - tau * gradient(x, A, y, lam)
        obj_hist[k] = objective(x, A, y, lam)
    return x, obj_hist

x_init = np.zeros(N)

# 三种步长方案
taus = [
    ('tau_opt = 1/L', tau_opt),
    ('tau = 0.5/L (偏小)', 0.5 / L),
    ('tau = 1.8/L (偏大)', 1.8 / L),
]

max_iter = 1000
results = {}
for label, tau in taus:
    x_opt, obj_hist = gradient_descent(x_init, A, y, lam, tau, max_iter)
    psnr_val = peak_signal_noise_ratio(x, np.clip(x_opt.reshape(n, n), 0, 1))
    results[label] = {
        'x': x_opt, 'obj_hist': obj_hist, 'psnr': psnr_val, 'tau': tau
    }
    print(f"\n  {label}: tau={tau:.6f}")
    print(f"    最终目标函数: {obj_hist[-1]:.4f}")
    print(f"    重建PSNR: {psnr_val:.2f} dB")

# ══════════════════════════════════════════════════════════
# 5. 可视化
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 第一行：原始 → 观测 → 最优重建
axes[0, 0].imshow(x, cmap='gray')
axes[0, 0].set_title(r'原始图像 $x$')
axes[0, 0].axis('off')

axes[0, 1].imshow(y.reshape(n, n), cmap='gray')
axes[0, 1].set_title(r'模糊含噪观测 $y$')
axes[0, 1].axis('off')

best_key = 'tau_opt = 1/L'
axes[0, 2].imshow(np.clip(results[best_key]['x'].reshape(n, n), 0, 1), cmap='gray')
axes[0, 2].set_title(r'梯度下降最优重建' + f'\n{best_key}\nPSNR={results[best_key]["psnr"]:.2f}dB')
axes[0, 2].axis('off')

# 第二行：收敛曲线对比 → 步长偏小 → 步长偏大
colors = ['#2c7bb6', '#fdae61', '#d7191c']
for idx, (label, _) in enumerate(taus):
    obj_hist = results[label]['obj_hist']
    axes[1, 0].semilogy(obj_hist - obj_hist[-1] + 1e-16,
                        color=colors[idx], linewidth=1.5,
                        label=f'{label}')
axes[1, 0].axhline(y=1e-16, color='gray', linestyle='--', linewidth=0.5)
axes[1, 0].set_xlabel('迭代次数 $k$')
axes[1, 0].set_ylabel(r'$J(x_k) - J(x^*)$')
axes[1, 0].set_title(r'收敛曲线对比（对数坐标）')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# 步长偏小：收敛慢
small_label = 'tau = 0.5/L (偏小)'
axes[1, 1].semilogy(results[small_label]['obj_hist'] - results[best_key]['obj_hist'][-1] + 1e-16,
                    color=colors[1], linewidth=1.5)
axes[1, 1].set_xlabel('迭代次数 $k$')
axes[1, 1].set_ylabel(r'$J(x_k) - J(x^*)$')
axes[1, 1].set_title(f'步长偏小: tau={results[small_label]["tau"]:.6f}\n收敛慢，需更多迭代')
axes[1, 1].grid(True, alpha=0.3)

# 步长偏大：可能振荡
large_label = 'tau = 1.8/L (偏大)'
axes[1, 2].plot(results[large_label]['obj_hist'], color=colors[2], linewidth=1.5)
axes[1, 2].set_xlabel('迭代次数 $k$')
axes[1, 2].set_ylabel(r'$J(x_k)$')
axes[1, 2].set_title(f'步长偏大: tau={results[large_label]["tau"]:.6f}\n接近稳定性边界')
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle(r'实验3.2-1: 梯度下降优化——步长选择与收敛分析', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_梯度下降步长对比.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "-" * 50)
print("【收敛速率分析】")
print("-" * 50)
for label, _ in taus:
    obj_hist = results[label]['obj_hist']
    # 计算达到目标函数低于阈值所需的迭代次数
    threshold = obj_hist[-1] + 0.1  # 距最优值0.1以内
    n_to_thresh = np.where(obj_hist < threshold)[0]
    if len(n_to_thresh) > 0:
        print(f"  {label}: {n_to_thresh[0] + 1} 次迭代达到目标 {threshold:.2f}")
    else:
        print(f"  {label}: 未在 {max_iter} 次迭代内达到目标")

print(f"\n  [结论]")
print(f"    1. tau_opt = 1/L 提供最优收敛速率")
print(f"    2. tau < 1/L: 收敛慢，需要更多迭代")
print(f"    3. tau -> 2/L: 接近发散边界，可能振荡")
print(f"    4. tau >= 2/L: 发散（不收敛）")

print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print("1. 梯度下降收敛条件: tau < 2/L，其中 L 是 Lipschitz 常数")
print("2. 最优步长 tau_opt = 1/L 在光滑凸优化中提供线性收敛")
print("3. Lipschitz 常数 L = lambda_max( Hessian ) = ||A||^2 + lam")
print("4. 步长过小: 收敛缓慢，需大量迭代")
print("5. 步长接近上限: 可能振荡，需谨慎选择")
print("6. 幂迭代法是估计 ||A|| 的实用方法")
print("\n实验完成。结果已保存至:", SAVE_DIR)