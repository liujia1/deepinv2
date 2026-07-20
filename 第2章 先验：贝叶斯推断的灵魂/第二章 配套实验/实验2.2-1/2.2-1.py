"""
实验2.2-1 高斯先验的推广：从Tikhonov到Sobolev平滑
对应章节：2.2 经典先验族
知识点：高斯先验→Tikhonov；梯度高斯先验→Sobolev平滑；闭式解；不同协方差结构的影响

素材来源：
  - IP22 (statistical_perspective.md): 高斯随机场采样 + MAP估计
  - 2.2章节: 高斯先验的推广形式

修改说明（基于代码评审v8）：
  1. 添加Sobolev频域闭式解，与迭代解对比验证
  2. 统一PSNR/SSIM格式化函数
  3. data_range改为理论值域1.0
  4. 收敛判断改为稳健相对阈值
  5. Tikhonov参数说明补充归一化讨论
  6. 补充verbose未收敛输出
  7. 修复axes条件分支逻辑
  8. 魔法数字统一管理
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import sys

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.2-1', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.2-1')
    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
    print("请确保 .chinese 文件夹已上传到 Google Drive 的正确位置")

np.random.seed(42)

# ─── 实验参数配置 ───
N_PIXELS    = 64
NOISE_LEVEL = 0.1
SOB_LAMBDAS = [0.05, 0.1, 0.2]
SOB_ITERS   = 200
SOB_TOL     = 1e-6
DATA_RANGE  = 1.0  # camera() 经 resize 后值域为 [0,1]，理论值域

def fmt_metrics(psnr, ssim):
    """统一格式化 PSNR 和 SSIM 输出"""
    return f"PSNR={psnr:.2f}dB, SSIM={ssim:.3f}"

n = N_PIXELS
x_true = resize(data.camera(), (n, n))

y = x_true + NOISE_LEVEL * np.random.randn(n, n)

def tikhonov_denoise(y, lam):
    """
    简单高斯先验: x ~ N(0, σ_x² I)
    正则项: ||x||_2²
    
    闭式解推导:
      目标函数: J(x) = 0.5||y-x||² + 0.5λ||x||²
      梯度: ∇J = (x-y) + λx = 0
      解: x = y/(1+λ)
    
    参数说明:
      - λ = σ²/σ_x²，其中σ为噪声标准差，σ_x为先验标准差
      - λ越大，先验影响越强，解越接近0
      - 注意：σ_x 应与图像值域匹配。camera()经resize后值域约[0,1]，
        若主观设定σ_x=1.0，相当于假设像素值可扩散到[-3,3]范围（3σ原则），
        这与实际数据特性不符，教学中需讨论此问题。
    """
    return y / (1 + lam)

def sobolev_denoise(y, lam, n_iter=SOB_ITERS, tol=SOB_TOL, verbose=False):
    """
    梯度高斯先验: ∇x ~ N(0, σ_∇² I)
    正则项: ||∇x||_2² = ∫|∇x|² dx (Sobolev半范数H¹)
    
    目标函数: J(x) = 0.5||y-x||² + 0.5λ||∇x||²
    
    梯度下降迭代:
      x^{k+1} = x^k - dt * ∇J(x^k)
              = x^k + dt * [(y-x^k) + λ·Δx^k]
      其中Δ为拉普拉斯算子
    
    数值稳定性分析:
      - 二维四邻域拉普拉斯算子的谱半径 ρ(Δ) = 8
      - 梯度下降稳定性条件: dt × (1 + λ·ρ(Δ)) < 1
      - 即: dt < 1/(1 + 8λ)
      - 采用自适应步长: dt = 0.9 / (1 + 8λ)
    
    边界条件说明:
      - 使用np.roll实现周期性边界条件（图像左右/上下相连）
      - 对于非周期性图像（如cameraman），边缘可能出现振铃伪影
      - 这是频域方法的固有特性，教学中可作为讨论点
    
    收敛诊断:
      - 记录目标函数值，当相对变化小于阈值时提前终止
      - 使用稳健相对阈值：|ΔJ| / (|J| + 1e-12) < tol
      - 分母加1e-12防止除零，适用于任意量级的目标函数
      - k > 0 确保第一次迭代不会误触发
    
    性能说明:
      - 每次迭代计算两次差分：一次用于梯度下降更新，一次用于目标函数监控
      - 如不需要收敛诊断，可去掉loss计算以减少开销
    """
    x = y.copy()
    dt = 0.9 / (1 + 8 * lam)
    
    prev_loss = np.inf
    converged_iter = n_iter
    
    for k in range(n_iter):
        # 用当前x的拉普拉斯更新x
        lap = (np.roll(x, 1, axis=0) + np.roll(x, -1, axis=0) +
               np.roll(x, 1, axis=1) + np.roll(x, -1, axis=1) - 4 * x)
        x = x + dt * ((y - x) + lam * lap)
        
        # 计算更新后x的目标函数（用于收敛诊断）
        # 等价形式：||∇x||² = Σ(前向差分)²，与拉普拉斯二次型 -x^T Δx 在周期边界下等价
        grad_x = np.roll(x, -1, axis=1) - x
        grad_y = np.roll(x, -1, axis=0) - x
        loss = 0.5 * np.sum((x - y)**2) + 0.5 * lam * np.sum(grad_x**2 + grad_y**2)
        
        if k > 0 and abs(prev_loss - loss) / (abs(prev_loss) + 1e-12) < tol:
            converged_iter = k + 1
            if verbose:
                print(f"    收敛于第 {k+1} 次迭代 (λ={lam:.3f})")
            break
        prev_loss = loss
    else:
        # 循环正常结束（未break），说明未提前收敛
        if verbose:
            print(f"    未收敛，已达最大迭代次数 {n_iter} (λ={lam:.3f})")
    
    return x, converged_iter

def sobolev_denoise_exact(y, lam):
    """
    Sobolev去噪的频域闭式解（与四邻域离散拉普拉斯一致）
    
    理论推导:
      目标函数: J(x) = 0.5||y-x||² + 0.5λ||∇x||²
      
      在离散频域中，四邻域拉普拉斯算子的特征值为：
        λ_Lap(ω) = 2cos(2πω_x) + 2cos(2πω_y) - 4
      其中 ω_x, ω_y 为离散归一化频率（范围 [-0.5, 0.5)）
      
      由Parseval定理，空域范数可转化为频域求和：
        ||∇x||² = Σ|∇x̂(ω)|² = Σ(-λ_Lap(ω))·|X̂(ω)|²
      其中 -λ_Lap(ω) ≥ 0，保证正则项非负
      
      频域目标函数:
        Ĵ(X̂) = 0.5|Ŷ - X̂|² - 0.5λ·λ_Lap(ω)·|X̂|²
      
      对每个频率分量求Wirtinger导数并令其为零:
        ∂Ĵ/∂X̂* = -(Ŷ - X̂) - λ·λ_Lap(ω)·X̂ = 0
        X̂(1 - λ·λ_Lap(ω)) = Ŷ
        X̂ = Ŷ / (1 - λ·λ_Lap(ω))
    
    实现说明:
      - 使用np.fft.fft2进行二维DFT
      - 频率网格通过np.fft.fftfreq构造（离散归一化频率，范围[-0.5, 0.5)）
      - 使用离散拉普拉斯特征值，确保与迭代法的四邻域拉普拉斯完全一致
      - 结果通过np.fft.ifft2返回空域
      
    教学价值:
      - 与迭代解对比，验证梯度下降收敛正确性（差异应达机器精度）
      - 展示频域方法的计算效率（O(n² log n) vs 迭代O(k·n²)）
      - 加深对离散Sobolev先验频域特性的理解
    """
    ny, nx = y.shape
    # 构造离散归一化频率网格（范围 [-0.5, 0.5)）
    freq_y = np.fft.fftfreq(ny)
    freq_x = np.fft.fftfreq(nx)
    omega_y, omega_x = np.meshgrid(freq_y, freq_x, indexing='ij')
    
    # 四邻域离散拉普拉斯的特征值
    lap_eigenval = 2*np.cos(2*np.pi*omega_x) + 2*np.cos(2*np.pi*omega_y) - 4
    
    # DFT
    y_hat = np.fft.fft2(y)
    # 频域滤波: X̂ = Ŷ / (1 - λ·λ_Lap)
    # 注意：lap_eigenval ≤ 0，故分母 ≥ 1，数值稳定
    x_hat = y_hat / (1 - lam * lap_eigenval)
    # IDFT返回实部
    x_exact = np.fft.ifft2(x_hat).real
    
    return x_exact

sigma_x_data = x_true.std()
sigma_x_prior = 1.0

lam_tikh_data = NOISE_LEVEL**2 / sigma_x_data**2
lam_tikh_prior = NOISE_LEVEL**2 / sigma_x_prior**2

x_tikh_data = tikhonov_denoise(y, lam_tikh_data)
x_tikh_prior = tikhonov_denoise(y, lam_tikh_prior)

x_sob_results = []
x_sob_exact_results = []
converged_iters = []

for lam_sob in SOB_LAMBDAS:
    # 迭代解
    x_sob, iters = sobolev_denoise(y, lam_sob, verbose=True)
    x_sob_results.append(x_sob)
    converged_iters.append(iters)
    # 闭式解（用于对比验证）
    x_sob_exact = sobolev_denoise_exact(y, lam_sob)
    x_sob_exact_results.append(x_sob_exact)

# 统一计算所有PSNR/SSIM（避免重复计算）
psnr_noisy = peak_signal_noise_ratio(x_true, y, data_range=DATA_RANGE)
psnr_tikh_data = peak_signal_noise_ratio(x_true, x_tikh_data, data_range=DATA_RANGE)
psnr_tikh_prior = peak_signal_noise_ratio(x_true, x_tikh_prior, data_range=DATA_RANGE)
psnr_sob = [peak_signal_noise_ratio(x_true, xs, data_range=DATA_RANGE) for xs in x_sob_results]
psnr_sob_exact = [peak_signal_noise_ratio(x_true, xs, data_range=DATA_RANGE) for xs in x_sob_exact_results]

ssim_noisy = structural_similarity(x_true, y, data_range=DATA_RANGE)
ssim_tikh_data = structural_similarity(x_true, x_tikh_data, data_range=DATA_RANGE)
ssim_tikh_prior = structural_similarity(x_true, x_tikh_prior, data_range=DATA_RANGE)
ssim_sob = [structural_similarity(x_true, xs, data_range=DATA_RANGE) for xs in x_sob_results]
ssim_sob_exact = [structural_similarity(x_true, xs, data_range=DATA_RANGE) for xs in x_sob_exact_results]

print("=" * 70)
print("实验2.2-1 高斯先验的推广：从Tikhonov到Sobolev平滑")
print("=" * 70)

print("\n" + "=" * 70)
print("【核心知识点】MAP与MMSE的等价性")
print("=" * 70)
print("""
在高斯-高斯模型下（高斯似然 + 高斯先验），后验分布为高斯分布。
由于高斯分布对称，均值 = 众数，因此：
  - MAP估计（后验众数）= MMSE估计（后验均值）
  - Tikhonov闭式解 x = y/(1+λ) 既是MAP也是MMSE
这是共轭先验的重要性质，详见第2章附录2A。
""")

print("=" * 70)
print("【参数设置】")
print("=" * 70)
print(f"  图像尺寸: {n}×{n}")
print(f"  噪声标准差 σ = {NOISE_LEVEL:.4f}")
print(f"  图像实际标准差 σ_x(数据) = {sigma_x_data:.4f}")
print(f"  先验假设标准差 σ_x(先验) = {sigma_x_prior:.4f} (主观设定)")
print(f"\n  Tikhonov参数:")
print(f"    - 从数据估计: λ = σ²/σ_x(数据)² = {lam_tikh_data:.4f}")
print(f"    - 主观先验:   λ = σ²/σ_x(先验)² = {lam_tikh_prior:.4f}")
print(f"\n  注意：σ_x(先验)=1.0 假设像素值域约[-3,3]（3σ原则），")
print(f"        但 camera() 经 resize 后值域实际约 [0,1]，")
print(f"        该假设与数据特性不匹配，导致 λ 偏小、正则化不足。")
print(f"\n  Sobolev参数 λ ∈ {SOB_LAMBDAS}")
print(f"  自适应步长: dt = 0.9/(1+8λ)")

print("\n" + "=" * 70)
print("【评估结果】")
print("=" * 70)
print(f"\n1. 含噪图像:")
print(f"   {fmt_metrics(psnr_noisy, ssim_noisy)}")

print(f"\n2. Tikhonov去噪 (简单高斯先验):")
print(f"   正则项: ||x||_2²")
print(f"   假设: 图像值小")
print(f"   [从数据估计σ_x] λ={lam_tikh_data:.4f}: {fmt_metrics(psnr_tikh_data, ssim_tikh_data)}")
print(f"   [主观先验σ_x=1] λ={lam_tikh_prior:.4f}: {fmt_metrics(psnr_tikh_prior, ssim_tikh_prior)}")
print(f"\n   → 教学解读：")
print(f"     两种参数选择方式展示了先验假设对结果的影响：")
print(f"     - 从数据估计σ_x({sigma_x_data:.3f})：λ={lam_tikh_data:.4f}，正则化较强")
print(f"     - 主观先验σ_x=1.0：λ={lam_tikh_prior:.4f}，正则化较弱（因假设值域过大）")
print(f"     对比PSNR({psnr_tikh_data:.2f} vs {psnr_tikh_prior:.2f}dB)说明：")
print(f"     先验假设需与数据特性匹配，不匹配的先验会引入偏差或不足。")

print(f"\n3. Sobolev去噪 (梯度高斯先验):")
print(f"   正则项: ||∇x||_2²")
print(f"   假设: 相邻像素相似")
for i, lam_sob in enumerate(SOB_LAMBDAS):
    print(f"   λ={lam_sob:.2f}: {fmt_metrics(psnr_sob[i], ssim_sob[i])}, 收敛于{converged_iters[i]}次迭代")

print(f"\n【迭代解 vs 闭式解验证】")
print(f"   迭代法使用空域梯度下降，闭式解使用频域DFT滤波")
print(f"   两者理论上等价，数值差异应接近机器精度（~1e-15）")
for i, lam_sob in enumerate(SOB_LAMBDAS):
    diff = np.max(np.abs(x_sob_results[i] - x_sob_exact_results[i]))
    print(f"   λ={lam_sob:.2f}: 最大差异 = {diff:.2e}")
    print(f"            迭代解 {fmt_metrics(psnr_sob[i], ssim_sob[i])}")
    print(f"            闭式解 {fmt_metrics(psnr_sob_exact[i], ssim_sob_exact[i])}")

print(f"\n【边界条件说明】")
print(f"  使用周期性边界条件（np.roll实现）")
print(f"  对于非周期性图像，边缘可能出现振铃伪影")
print(f"  这是频域方法的固有特性，教学中可讨论")

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

axes[0, 0].imshow(x_true, cmap='gray')
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(y, cmap='gray')
axes[0, 1].set_title(f'含噪图像\n{fmt_metrics(psnr_noisy, ssim_noisy)}')
axes[0, 1].axis('off')

axes[0, 2].imshow(x_tikh_data, cmap='gray')
axes[0, 2].set_title(f'Tikhonov (数据估计$\\sigma_x$)\n$\\lambda$={lam_tikh_data:.4f}\n{fmt_metrics(psnr_tikh_data, ssim_tikh_data)}')
axes[0, 2].axis('off')

axes[1, 0].imshow(x_tikh_prior, cmap='gray')
axes[1, 0].set_title(f'Tikhonov (主观先验$\\sigma_x=1$)\n$\\lambda$={lam_tikh_prior:.4f}\n{fmt_metrics(psnr_tikh_prior, ssim_tikh_prior)}')
axes[1, 0].axis('off')

sob_idx_01 = 1  # $\\lambda$=0.1 在 SOB_LAMBDAS 中的索引
lam_display = SOB_LAMBDAS[sob_idx_01]
axes[1, 1].imshow(x_sob_results[sob_idx_01], cmap='gray')
axes[1, 1].set_title(f'Sobolev ($\\lambda$={lam_display})\n{fmt_metrics(psnr_sob[sob_idx_01], ssim_sob[sob_idx_01])}')
axes[1, 1].axis('off')

best_sob_idx = np.argmax(psnr_sob)
axes[1, 2].imshow(x_sob_results[best_sob_idx], cmap='gray')
axes[1, 2].set_title(f'Sobolev (最优$\\lambda$={SOB_LAMBDAS[best_sob_idx]})\n{fmt_metrics(psnr_sob[best_sob_idx], ssim_sob[best_sob_idx])}\n(按PSNR自动选择)')
axes[1, 2].set_axis_off()

plt.suptitle('高斯先验的推广：Tikhonov vs Sobolev', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_去噪结果对比.png'), dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

center = n // 2
for i, (lam_sob, xs) in enumerate(zip(SOB_LAMBDAS, x_sob_results)):
    axes[0].plot(xs[center, :], linewidth=1, label=f'$\\lambda$={lam_sob:.2f}')
axes[0].plot(x_true[center, :], 'k--', linewidth=1.5, label='真实')
axes[0].plot(y[center, :], 'r-', linewidth=0.5, alpha=0.5, label='含噪')
axes[0].set_title('Sobolev不同$\\lambda$的剖面对比\n展示超参数敏感性')
axes[0].legend(fontsize=8)
axes[0].set_xlabel('像素索引')
axes[0].set_ylabel('灰度值')
axes[0].grid(True, alpha=0.3)

all_methods = ['Tikhonov\n(数据估计)', 'Tikhonov\n(主观先验)'] + [f'Sobolev\n$\\lambda$={lam}' for lam in SOB_LAMBDAS]
all_psnrs = [psnr_tikh_data, psnr_tikh_prior] + psnr_sob
all_ssim_scaled = [ssim_tikh_data*100, ssim_tikh_prior*100] + [s*100 for s in ssim_sob]

x_pos = np.arange(len(all_methods))
width = 0.35

ax1 = axes[1]
ax1.bar(x_pos - width/2, all_psnrs, width, color='steelblue', alpha=0.8, label='PSNR (dB)')
ax1.axhline(y=psnr_noisy, color='black', linestyle='--', label=f'含噪PSNR: {psnr_noisy:.2f}dB')

psnr_min, psnr_max = min(all_psnrs), max(all_psnrs)
y_bottom = min(psnr_min, psnr_noisy) - 1
ax1.set_ylim([y_bottom, psnr_max + 1])
ax1.set_ylabel('PSNR (dB)', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

ax2 = ax1.twinx()
ax2.bar(x_pos + width/2, all_ssim_scaled, width, color='seagreen', alpha=0.6, label='SSIM$\\times$100')

ssim_min, ssim_max = min(all_ssim_scaled), max(all_ssim_scaled)
ssim_noisy_scaled = ssim_noisy * 100
ax2.set_ylim([min(ssim_min, ssim_noisy_scaled) - 2, ssim_max + 2])
ax2.axhline(y=ssim_noisy_scaled, color='gray', linestyle=':', label=f'含噪SSIM$\\times$100: {ssim_noisy_scaled:.1f}')
ax2.set_ylabel('SSIM$\\times$100', color='seagreen')
ax2.tick_params(axis='y', labelcolor='seagreen')

ax1.set_xticks(x_pos)
ax1.set_xticklabels(all_methods, fontsize=8)
ax1.set_title('不同方法的性能对比\n(左轴: PSNR, 右轴: SSIM$\\times$100)')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_剖面对比与性能.png'), dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

t = np.linspace(-2, 2, 400)
axes[0].plot(t, t**2, 'b-', linewidth=2, label='L2: $t^2$ (高斯先验正则项)')
axes[0].plot(t, np.abs(t), 'r-', linewidth=2, label='L1: |t| (Laplace/TV正则项)')
axes[0].set_title('正则项形态对比（示意）\nL2均匀惩罚 vs L1促稀疏\n注：L1方法未在本实验实现，详见实验2.2-3')
axes[0].legend()
axes[0].set_xlabel('t')
axes[0].set_ylabel('惩罚值')
axes[0].set_ylim(-0.2, 4)
axes[0].grid(True, alpha=0.3)

# 局部方差计算（边界安全检查）
assert n >= 10, "N_PIXELS 过小，局部方差区域无效"
local_var_true = np.var(x_true[center-5:center+5, center-5:center+5])
local_var_tikh = np.var(x_tikh_data[center-5:center+5, center-5:center+5])
local_var_sob = [np.var(xs[center-5:center+5, center-5:center+5]) for xs in x_sob_results]

var_methods = ['真实', 'Tikhonov\n(数据估计)'] + [f'Sobolev\n$\\lambda$={lam}' for lam in SOB_LAMBDAS]
var_values = [local_var_true, local_var_tikh] + local_var_sob
var_colors = plt.cm.tab10(np.linspace(0, 0.5, len(var_methods)))

axes[1].bar(var_methods, var_values, color=var_colors[:len(var_methods)], alpha=0.7)
axes[1].set_ylabel('局部方差')
axes[1].set_title('局部方差保持度\n(中心$10\\times10$区域)')

# 动态计算文字偏移量，避免超出边框
var_max = max(var_values)
var_min = min(var_values)
var_range = var_max - var_min
offset = var_range * 0.02 if var_range > 0 else var_max * 0.02

for i, v in enumerate(var_values):
    # 根据数值大小动态调整文字位置
    text_y = v + offset
    # 如果文字可能超出标题区域，将文字移到柱内
    if text_y > var_max * 0.95:
        text_y = v - offset
        va = 'top'
    else:
        va = 'bottom'
    axes[1].text(i, text_y, f'{v:.4f}', ha='center', va=va, fontsize=8)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(SAVE_DIR, '步骤3_正则项形态与方差保持.png'), dpi=150, bbox_inches='tight')
plt.close()

# 新增：步骤4 - 迭代解与闭式解对比
n_rows = 2  # 上排迭代解，下排闭式解
n_cols = len(SOB_LAMBDAS)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False)

for i, lam_sob in enumerate(SOB_LAMBDAS):
    # 上排：迭代解
    axes[0, i].imshow(x_sob_results[i], cmap='gray')
    title1 = '迭代解 ($\\lambda$={:.2f})\n{}\n迭代{}次'.format(lam_sob, fmt_metrics(psnr_sob[i], ssim_sob[i]), converged_iters[i])
    axes[0, i].set_title(title1)
    axes[0, i].axis('off')
    
    # 下排：闭式解
    axes[1, i].imshow(x_sob_exact_results[i], cmap='gray')
    title2 = '闭式解 ($\\lambda$={:.2f})\n{}\n频域O($n^2\\log n$), n为边长'.format(lam_sob, fmt_metrics(psnr_sob_exact[i], ssim_sob_exact[i]))
    axes[1, i].set_title(title2)
    axes[1, i].axis('off')

plt.suptitle('Sobolev去噪：迭代解 vs 频域闭式解', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤4_迭代解与闭式解对比.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 70)
print("【高斯先验的特点总结】")
print("=" * 70)
print("\n1. 简单高斯先验 (Tikhonov):")
print("   - 假设: 图像值小")
print("   - 协方差: σ_x² I (对角矩阵)")
print("   - 优点: 有闭式解，计算高效")
print("   - 缺点: 过度平滑，丢失边缘")
print("   - 参数选择: σ_x需与数据值域匹配，可从数据估计或主观设定")

print("\n2. 梯度高斯先验 (Sobolev平滑):")
print("   - 假设: 相邻像素相似")
print("   - 协方差: 涉及梯度算子")
print("   - 优点: 编码空间结构，比简单高斯更合理")
print("   - 缺点: 仍会过度平滑边缘")
print("   - 数值稳定性: dt < 1/(1+8λ)")
print("   - 超参数敏感性: λ影响平滑程度，需调参")
print("   - 频域闭式解: X̂ = Ŷ / (1 - λ·λ_Lap)，计算效率更高")
print("     其中 λ_Lap = 2cos(2πω_x) + 2cos(2πω_y) - 4 ≤ 0 为离散拉普拉斯特征值")

print("\n3. 与Laplace/TV先验的对比:")
print("   - 高斯先验 (L2): 均匀惩罚所有值，无精确零值")
print("   - Laplace/TV先验 (L1): 促稀疏，产生精确零值")
print("   - L1更适合稀疏表示，L2更适合平滑先验")

print("\n【实验完成】")
print(f"结果已保存至: {SAVE_DIR}")

# ===== 保存数值结果 =====
import json
results_summary = {
    'image_size': n,
    'noise_level': float(NOISE_LEVEL),
    'psnr_noisy_dB': float(round(psnr_noisy, 2)),
    'ssim_noisy': float(round(ssim_noisy, 4)),
    'tikhonov_data': {
        'lambda': float(round(lam_tikh_data, 6)),
        'psnr_dB': float(round(psnr_tikh_data, 2)),
        'ssim': float(round(ssim_tikh_data, 4)),
    },
    'tikhonov_prior': {
        'lambda': float(round(lam_tikh_prior, 6)),
        'psnr_dB': float(round(psnr_tikh_prior, 2)),
        'ssim': float(round(ssim_tikh_prior, 4)),
    },
    'sobolev': {f'lambda_{lam:.2f}': {
        'psnr_dB': float(round(psnr_sob[i], 2)),
        'ssim': float(round(ssim_sob[i], 4)),
        'converged_iters': int(converged_iters[i]),
    } for i, lam in enumerate(SOB_LAMBDAS)},
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
