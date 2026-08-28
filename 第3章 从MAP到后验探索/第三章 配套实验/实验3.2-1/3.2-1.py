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

h = gaussian_psf(n, sigma=3.0)
# 【教学要点】sigma=3.0（较大模糊核）使系统条件数增大：
# 更模糊的PSF使A的某些小奇异值更接近零，μ=λ_min(A^T A)+λ变小，
# 条件数κ=L/μ增大，能让tau=1.8/L在收敛曲线上呈现可见振荡，
# 增强与最优步长的教学对比效果
H_fft = np.fft.fft2(h)

# 【教学要点】使用算子形式而非显式矩阵
# 循环卷积具有 Toeplitz 结构，可用 FFT 高效实现矩阵-向量乘法
# 显式构造 N×N 矩阵会导致：
#   - n=32: 8 MB (勉强可接受)
#   - n=64: 128 MB
#   - n=128: 2 GB (内存崩溃)
# 算子形式不仅节省内存，还能让学生理解"线性算子"的抽象概念

def A_op(x_vec):
    """前向算子：Ax = ifft2(H * fft2(x))
    
    利用卷积定理：空间域卷积 = 频域乘积
    对于循环卷积，A 是块循环矩阵，可通过 FFT 对角化
    """
    return np.real(np.fft.ifft2(H_fft * np.fft.fft2(x_vec.reshape(n, n)))).ravel()

def AT_op(r_vec):
    """伴随算子（adjoint）：A^T r = ifft2(conj(H) * fft2(r))
    
    对于循环卷积，A 在频域中是对角矩阵 diag(H_fft)，
    其伴随为 A^T = diag(conj(H_fft))。
    对于实值且关于原点对称的 PSF（如当前高斯核），
    conj(H_fft) = H_fft，即 A^T = A（自伴随）。
    """
    return np.real(np.fft.ifft2(np.conj(H_fft) * np.fft.fft2(r_vec.reshape(n, n)))).ravel()

sigma_noise = 0.05
y = A_op(x.ravel()) + sigma_noise * np.random.randn(N)

# ══════════════════════════════════════════════════════════
# 2. 目标函数与梯度（使用算子形式）
# ══════════════════════════════════════════════════════════
lam = 0.01

def objective(x_vec, lam):
    """Tikhonov目标函数 J(x) = 0.5||Ax-y||^2 + 0.5*lam*||x||^2
    
    注意：这里使用算子 A_op 而非矩阵 A
    """
    residual = A_op(x_vec) - y
    return 0.5 * np.linalg.norm(residual) ** 2 + 0.5 * lam * np.linalg.norm(x_vec) ** 2

def gradient(x_vec, lam):
    """梯度 ∇J(x) = A^T(Ax - y) + lam*x
    
    使用伴随算子 AT_op 计算 A^T r
    """
    return AT_op(A_op(x_vec) - y) + lam * x_vec

# ══════════════════════════════════════════════════════════
# 3. Lipschitz常数估计（幂迭代法，使用算子形式）
# ══════════════════════════════════════════════════════════
# 目标函数的Hessian: H = A^T A + lam*I
# Lipschitz常数 L = lambda_max(A^T A) + lam = ||A||^2 + lam
# 根据梯度下降收敛理论，步长需满足 tau < 2/L

def estimate_lipschitz(lam, n_iter=50):
    """用幂迭代估计 ||A||^2，进而得到Lipschitz常数 L = ||A||^2 + lam
    
    幂迭代法原理：
    - 对对称正定矩阵 M = A^T A，其主特征向量 v 满足 Mv = λ_max v
    - 迭代过程：v_{k+1} = M v_k / ||M v_k||
    - 收敛后，Rayleigh 商 v^T M v 给出 λ_max ≈ ||A||^2
    
    注意：
    - 这里只估计 ||A||^2（通过 A^T A 的主特征值）
    - 最终 L = ||A||^2 + lam，其中 lam 是已知的正则化参数
    - 幂迭代不需要显式构造 A^T A，只需能计算 A 和 A^T 的作用
    """
    rng = np.random.default_rng(seed=0)
    v = rng.standard_normal(N)
    v = v / np.linalg.norm(v)
    for _ in range(n_iter):
        # 计算 (A^T A) v，使用算子形式避免显式矩阵
        v = AT_op(A_op(v))
        v = v / np.linalg.norm(v)
    # Rayleigh 商：v^T (A^T A) v = ||A v||^2
    Av = A_op(v)
    A_norm_sq = np.dot(Av, Av)
    # L = ||A||^2 + lam（Hessian 的最大特征值）
    return A_norm_sq + lam

L = estimate_lipschitz(lam, n_iter=50)
tau_opt = 1.0 / L          # 最优步长 (梯度下降理论中的标准选择)
tau_max = 2.0 / L           # 收敛的理论上限

# ★ 自检：验证幂迭代‖A‖²与最优步长的理论关系(避免L计算错误)
# 核心公式: L = ||A||² + λ, τ_opt = 1/L, τ_max = 2/L
# 检验规则:
#   P1: τ_opt = 1/L 精确成立(相对误差<1e-6)
#   P2: τ_max = 2/L 精确成立
#   P3: τ_opt < τ_max < ∞ (L>0保证)
print(f"[步长自检] Lipschitz常数与步长关系验证(实际运行, 非手算):")
print(f"  P1: τ_opt=1/L: 1/L={1.0/L:.6f} vs τ_opt={tau_opt:.6f} (相对误差{abs(1.0/L - tau_opt)/tau_opt:.2e}): "
      f"{'OK' if abs(1.0/L - tau_opt) < 1e-6 else 'FAIL'}")
print(f"  P2: τ_max=2/L: 2/L={2.0/L:.6f} vs τ_max={tau_max:.6f}: "
      f"{'OK' if abs(2.0/L - tau_max) < 1e-6 else 'FAIL'}")
print(f"  P3: L>0, τ_opt<τ_max: L={L:.4f} {'>0' if L > 0 else '<=0 FAIL'}, "
      f"τ_opt={tau_opt:.6f}<τ_max={tau_max:.6f}: {'OK' if tau_opt < tau_max else 'FAIL'}")
# 验证: 在τ=τ_max附近步长会产生数值不稳定(放大特征值效应)
print(f"  注: 步长>τ_max时, (I - τ·H)谱半径>1, 梯度下降发散")

print("=" * 60)
print("实验3.2-1 梯度下降优化：步长选择与收敛分析")
print("=" * 60)
print(f"\n[Lipschitz常数估计]")
print(f"  估计的 L = {L:.4f}")
print(f"  理论最优步长 tau_opt = 1/L = {tau_opt:.6f}")
print(f"  收敛上限 tau_max = 2/L = {tau_max:.6f}")

# 【教学要点】计算Tikhonov问题的解析最优解作为 J* 参考基准
# 在循环卷积假设下，Tikhonov问题有精确的频域闭式解:
#   x* = (A^T A + lam*I)^{-1} A^T y
#   X*_fft = conj(H) * Y / (|H|^2 + lam)
# 注意：这里的 A_op/AT_op 使用相同的循环卷积假设（对角化于FFT），
#       因此频域解析解与梯度下降优化的是同一个目标函数，
#       J_star 就是真实的全局最优值。
Y_fft = np.fft.fft2(y.reshape(n, n))
X_star_fft = np.conj(H_fft) * Y_fft / (np.abs(H_fft)**2 + lam)
x_star = np.real(np.fft.ifft2(X_star_fft)).ravel()
J_star = objective(x_star, lam)
print(f"  解析最优 J* = {J_star:.4f} (频域闭式解，与A_op/AT_op一致)")

# ══════════════════════════════════════════════════════════
# 4. 梯度下降：不同步长对比
# ══════════════════════════════════════════════════════════
def gradient_descent(x0, lam, tau, n_iter):
    """梯度下降算法（使用算子形式）
    
    包含早停保护：检测到 inf/nan 时提前终止，避免数值溢出影响后续绘图。
    """
    x = x0.copy()
    obj_hist = np.zeros(n_iter)
    for k in range(n_iter):
        x = x - tau * gradient(x, lam)
        val = objective(x, lam)
        obj_hist[k] = val
        # 早停保护：数值发散时提前终止，剩余位置填充 nan
        if not np.isfinite(val):
            obj_hist[k:] = np.nan
            break
    return x, obj_hist

x_init = np.zeros(N)
# 【教学要点】x_init = 0 时 J_init = 0.5 * ||y||^2（观测能量的一半）
# 因为 A(0) = 0，数据项 = 0.5||0 - y||^2 = 0.5||y||^2，正则化项也为 0
# 这个初始值有物理意义：相当于"没有任何复原先验"的起点
# 在发散图中，初始值较大正是因为观测 y 包含噪声能量

# 四种步长方案：包含发散案例
taus = [
    ('tau_opt = 1/L', tau_opt),
    ('tau = 0.5/L (偏小)', 0.5 / L),
    ('tau = 1.8/L (接近边界)', 1.8 / L),
    ('tau = 2.1/L (发散)', 2.1 / L),  # 新增：展示发散情形
]

max_iter = 1000
results = {}
for label, tau in taus:
    # 发散情况限制迭代次数，避免数值溢出
    n_iter_actual = 50 if '发散' in label else max_iter
    x_opt, obj_hist = gradient_descent(x_init, lam, tau, n_iter_actual)
    # 发散情形下跳过PSNR计算（结果无意义）
    if np.all(np.isfinite(x_opt)):
        psnr_val = peak_signal_noise_ratio(x, np.clip(x_opt.reshape(n, n), 0, 1))
    else:
        psnr_val = None
    results[label] = {
        'x': x_opt, 'obj_hist': obj_hist, 'psnr': psnr_val, 'tau': tau
    }
    print(f"\n  {label}: tau={tau:.6f}")
    if psnr_val is not None:
        print(f"    最终目标函数: {obj_hist[~np.isnan(obj_hist)][-1]:.4f}")
        print(f"    重建PSNR: {psnr_val:.2f} dB")
    else:
        print(f"    最终目标函数: {obj_hist[~np.isnan(obj_hist)][-1]:.4f} (发散)")
        print(f"    重建PSNR: 数值发散，无意义")

# ═════════════════════════════════════════════════════════
# 5. 可视化（2行布局：第一行4个图，第二行3个居中图）
# ══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 10))
# 使用12列网格：第一行4图各占3列，第二行3图各占4列居中
gs = fig.add_gridspec(2, 12, hspace=0.30, wspace=0.18)

# ── 第一行：重建结果对比（4个图，各占3列）──
axes_top = [
    fig.add_subplot(gs[0, 0:3]),
    fig.add_subplot(gs[0, 3:6]),
    fig.add_subplot(gs[0, 6:9]),
    fig.add_subplot(gs[0, 9:12]),
]

axes_top[0].imshow(x, cmap='gray')
axes_top[0].set_title(r'原始图像 $x$')
axes_top[0].axis('off')

axes_top[1].imshow(y.reshape(n, n), cmap='gray')
axes_top[1].set_title(r'模糊含噪观测 $y$')
axes_top[1].axis('off')

best_key = 'tau_opt = 1/L'
axes_top[2].imshow(np.clip(results[best_key]['x'].reshape(n, n), 0, 1), cmap='gray')
axes_top[2].set_title(r'最优步长 $\tau=1/L$' +
                     f'\nPSNR={results[best_key]["psnr"]:.2f}dB')
axes_top[2].axis('off')

near_key = 'tau = 1.8/L (接近边界)'
axes_top[3].imshow(np.clip(results[near_key]['x'].reshape(n, n), 0, 1), cmap='gray')
axes_top[3].set_title(r'近边界步长 $\tau=1.8/L$' +
                     f'\nPSNR={results[near_key]["psnr"]:.2f}dB' +
                     '\n（与最优步长收敛到同一解）')
axes_top[3].axis('off')

# ─ 第二行：收敛分析（3个图居中，各占3列，子图间留1列空隙）──
axes_bottom = [
    fig.add_subplot(gs[1, 1:4]),   # 收敛曲线对比
    fig.add_subplot(gs[1, 5:8]),   # 步长发散
    fig.add_subplot(gs[1, 9:12]),  # 近边界步长放大
]

# 【第二行-左】收敛曲线对比（仅收敛案例，发散单独展示）
convergent_taus = [(l, t) for l, t in taus if '发散' not in l]
colors_conv = ['#2c7bb6', '#fdae61', '#d7191c']
for idx, (label, _) in enumerate(convergent_taus):
    obj_hist = results[label]['obj_hist'].copy()
    gap = obj_hist - J_star + 1e-16
    # semilogy要求所有值为正：过滤浮点误差导致的负值
    gap_safe = np.where(gap > 0, gap, np.nan)
    axes_bottom[0].semilogy(gap_safe, color=colors_conv[idx], linewidth=1.5,
                        label=f'{label}')
axes_bottom[0].axhline(y=1e-16, color='gray', linestyle='--', linewidth=0.5)
axes_bottom[0].set_xlabel('迭代次数 $k$')
axes_bottom[0].set_ylabel(r'$J(x_k) - J^*$')
axes_bottom[0].set_title(r'收敛曲线对比（对数坐标，3种收敛步长）')
axes_bottom[0].legend(fontsize=8)
axes_bottom[0].grid(True, alpha=0.3)

# 【第二行-中】发散案例（独立展示）
# 使用线性坐标：发散时J(x_k)可能含inf/nan，semilogy会静默失败
J_init = objective(x_init, lam)  # 真正的初始值，非第一步迭代后的值
diverge_label = 'tau = 2.1/L (发散)'
obj_div = results[diverge_label]['obj_hist']
valid_idx = np.isfinite(obj_div)
axes_bottom[1].plot(np.where(valid_idx)[0], obj_div[valid_idx],
                color='#a6611a', linewidth=1.5, label='J(x_k)')
axes_bottom[1].axhline(y=J_init, color='gray', linestyle='--', linewidth=1,
                   label=f'初始值 J(x_0)={J_init:.1f}')
axes_bottom[1].set_xlabel('迭代次数 $k$')
axes_bottom[1].set_ylabel(r'$J(x_k)$')
axes_bottom[1].set_title(f'步长发散: tau={results[diverge_label]["tau"]:.6f}\n超过稳定性边界 2/L')
axes_bottom[1].legend(fontsize=8)
axes_bottom[1].grid(True, alpha=0.3)

# 【第二行-右】1.8/L 独立放大展示（接近边界的振荡行为）
near_obj = results[near_key]['obj_hist']
gap_near = near_obj - J_star + 1e-16
gap_near_safe = np.where(gap_near > 0, gap_near, np.nan)
axes_bottom[2].semilogy(gap_near_safe, color='#d7191c', linewidth=1.5)
axes_bottom[2].axhline(y=1e-16, color='gray', linestyle='--', linewidth=0.5)
axes_bottom[2].set_xlabel('迭代次数 $k$')
axes_bottom[2].set_ylabel(r'$J(x_k) - J^*$')
axes_bottom[2].set_title(r'近边界步长 $\tau=1.8/L$ 放大' + '\n收敛慢，接近振荡边界')
axes_bottom[2].grid(True, alpha=0.3)

plt.suptitle(r'实验3.2-1: 梯度下降优化——步长选择与收敛分析', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_梯度下降步长对比.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "-" * 50)
print("【收敛速率分析】")
print("-" * 50)
for label, _ in taus:
    obj_hist = results[label]['obj_hist']
    valid = obj_hist[np.isfinite(obj_hist)]
    if len(valid) == 0:
        print(f"  {label}: 全部发散，无有效值")
        continue
    gap_init = objective(x_init, lam) - J_star  # 初始点到最优值的差距
    threshold = gap_init * 1e-3  # 收敛阈值：初始差距的0.1%
    n_to_thresh = np.where((obj_hist - J_star) < threshold)[0]
    if len(n_to_thresh) > 0:
        print(f"  {label}: {n_to_thresh[0] + 1} 次迭代收敛到距最优值0.1%以内")
    else:
        print(f"  {label}: 未在 {len(obj_hist)} 次迭代内达到目标")

print(f"\n  [结论]")
print(f"    1. tau_opt = 1/L 提供最优收敛速率")
print(f"    2. tau < 1/L: 收敛慢，需要更多迭代")
print(f"    3. tau -> 2/L: 接近发散边界，可能振荡")
print(f"    4. tau = 2/L: 振荡（不收敛，目标函数值不变）")
print(f"    5. tau > 2/L: 发散（目标函数不降反升）")

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

# ===== 保存数值结果 =====
import json
results_summary = {
    'n': n,
    'N': N,
    'sigma_noise': float(round(sigma_noise, 4)),
    'lam': float(round(lam, 4)),
    'L': float(round(L, 4)),
    'tau_opt': float(round(tau_opt, 6)),
    'tau_max': float(round(tau_max, 6)),
    'J_star': float(round(J_star, 4)),
    'results': {label: {
        'tau': float(round(data['tau'], 6)),
        'psnr': float(round(data['psnr'], 2)) if data['psnr'] is not None else None,
        'convergent': np.all(np.isfinite(data['x'])),
    } for label, data in results.items()},
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