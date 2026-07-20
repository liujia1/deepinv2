"""
实验2.b-1 TGV：TV的改进
对应章节：附录2B TGV：TV的改进
知识点：阶梯效应根源；TGV定义；TGV vs TV对比；分段仿射 vs 分段常数

素材来源：
  - 2.5.py: TV去噪与阶梯效应展示
  - 附录2B: TGV理论
"""

import numpy as np
import os
import sys
import warnings

# ====== 静默模式配置 ======
SILENT_MODE = True  # True: 不弹窗、不显示警告；False: 正常交互模式

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
from skimage import data
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.b-1', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.b-1')
    os.makedirs(SAVE_DIR, exist_ok=True)
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    cn_font = setup_chinese_font(save_dir=_chinese_path)
    if cn_font:
        plt.rcParams['font.sans-serif'] = [cn_font] + plt.rcParams.get('font.sans-serif', [])
        plt.rcParams['font.family'] = 'sans-serif'
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
    print("请确保 .chinese 文件夹已上传到 Google Drive 的正确位置")

np.random.seed(42)

def gradient_forward(x):
    """
    前向差分梯度算子（零边界条件）
    
    (Dx)_h[i,j] = x[i,j+1] - x[i,j]  (水平方向)
    (Dx)_v[i,j] = x[i+1,j] - x[i,j]  (垂直方向)
    
    边界处理: 最后一列/行的梯度为0（零边界）
    """
    h, w = x.shape
    dx_h = np.zeros_like(x)
    dx_v = np.zeros_like(x)
    dx_h[:, :-1] = x[:, 1:] - x[:, :-1]
    dx_v[:-1, :] = x[1:, :] - x[:-1, :]
    return dx_h, dx_v

def divergence_backward(p_h, p_v):
    """
    后向差分散度算子（零边界条件）
    
    这是 gradient_forward 的伴随算子（负梯度）:
    div(p) = -D^* p，满足 <Dx, p> = <x, -div(p)>
    
    实现方式:
    div[i,j] = p_h[i,j-1] - p_h[i,j] + p_v[i-1,j] - p_v[i,j]
    
    边界处理: 超出边界的p值视为0（零边界）
    """
    h, w = p_h.shape
    div = np.zeros((h, w))
    div[:, 1:] += p_h[:, :-1]   # 加 p_h[i,j-1]
    div -= p_h                  # 减 p_h[i,j]（含边界，边界处p_h已为0）
    div[1:, :] += p_v[:-1, :]   # 加 p_v[i-1,j]
    div -= p_v                  # 减 p_v[i,j]（含边界，边界处p_v已为0）
    return div

def symmetrized_gradient_forward(w_h, w_v):
    """对称梯度算子 E(w) = [∂h w_h, ∂v w_v; (∂h w_v + ∂v w_h)/2]"""
    h, w = w_h.shape
    Ew_1 = np.zeros_like(w_h)
    Ew_2 = np.zeros_like(w_v)
    Ew_3 = np.zeros_like(w_h)
    
    Ew_1[:, :-1] = w_h[:, 1:] - w_h[:, :-1]
    Ew_2[:-1, :] = w_v[1:, :] - w_v[:-1, :]
    
    d_wv_h = np.zeros_like(w_v)
    d_wh_v = np.zeros_like(w_h)
    d_wv_h[:, :-1] = w_v[:, 1:] - w_v[:, :-1]
    d_wh_v[:-1, :] = w_h[1:, :] - w_h[:-1, :]
    Ew_3 = 0.5 * (d_wv_h + d_wh_v)
    
    return Ew_1, Ew_2, Ew_3

def symmetrized_divergence_backward(q_1, q_2, q_3):
    """
    对称梯度算子的伴随（负散度）
    
    注意: q_3 的交叉项贡献方向如下（这是对称梯度算子的数学性质）:
    - dq3_dv（q_3的垂直方向差分）加到 div_w_h（w_h的散度）
    - dq3_dh（q_3的水平方向差分）加到 div_w_v（w_v的散度）
    
    这种交叉方向是刻意的数学设计，因为对称梯度 E(w) 的 (1,2) 分量
    同时包含 ∂h w_v 和 ∂v w_h，其伴随算子自然产生交叉贡献。
    """
    h, w = q_1.shape
    div_w_h = np.zeros_like(q_1)
    div_w_v = np.zeros_like(q_2)
    
    div_w_h[:, 1:] += q_1[:, :-1]
    div_w_h -= q_1
    
    div_w_v[1:, :] += q_2[:-1, :]
    div_w_v -= q_2
    
    dq3_dh = np.zeros_like(q_3)
    dq3_dv = np.zeros_like(q_3)
    dq3_dh[:, 1:] += 0.5 * q_3[:, :-1]
    dq3_dh -= 0.5 * q_3
    dq3_dv[1:, :] += 0.5 * q_3[:-1, :]
    dq3_dv -= 0.5 * q_3
    
    div_w_h += dq3_dv
    div_w_v += dq3_dh
    
    return div_w_h, div_w_v

def tgv_denoise_primal_dual(y, alpha0, alpha1, n_iter=500, tau=0.25, sigma=0.25):
    """
    TGV去噪的原始-对偶方法实现
    
    TGV模型: min_{x,w} 0.5||x-y||^2 + alpha1*||Dx-w||_1 + alpha0*||Ew||_1
    
    其中:
    - x: 图像
    - w: 辅助向量场（梯度的仿射分量逼近）
    - D: 梯度算子
    - E: 对称梯度算子
    
    参数:
        y: 含噪图像
        alpha0: 二阶正则化参数
        alpha1: 一阶正则化参数
        n_iter: 迭代次数
        tau, sigma: 步长参数，需满足 tau*sigma*||K||^2 <= 1
    
    返回:
        x: 去噪后图像
        w_h, w_v: 辅助向量场
    """
    h, w = y.shape
    x = y.copy()
    w_h = np.zeros((h, w))
    w_v = np.zeros((h, w))
    p_h = np.zeros((h, w))
    p_v = np.zeros((h, w))
    q_1 = np.zeros((h, w))
    q_2 = np.zeros((h, w))
    q_3 = np.zeros((h, w))
    
    theta = 1.0
    x_bar = x.copy()
    w_h_bar = w_h.copy()
    w_v_bar = w_v.copy()
    
    for k in range(n_iter):
        Dx_h, Dx_v = gradient_forward(x_bar)
        
        p_h = p_h + sigma * (Dx_h - w_h_bar)
        p_v = p_v + sigma * (Dx_v - w_v_bar)
        
        norm_p = np.sqrt(p_h**2 + p_v**2)
        p_h = p_h / np.maximum(1, norm_p / alpha1)
        p_v = p_v / np.maximum(1, norm_p / alpha1)
        
        Ew_1, Ew_2, Ew_3 = symmetrized_gradient_forward(w_h_bar, w_v_bar)
        
        q_1 = q_1 + sigma * Ew_1
        q_2 = q_2 + sigma * Ew_2
        q_3 = q_3 + sigma * Ew_3
        
        norm_q = np.sqrt(q_1**2 + q_2**2 + 2*q_3**2)
        q_1 = q_1 / np.maximum(1, norm_q / alpha0)
        q_2 = q_2 / np.maximum(1, norm_q / alpha0)
        q_3 = q_3 / np.maximum(1, norm_q / alpha0)
        
        div_p = divergence_backward(p_h, p_v)
        div_q_h, div_q_v = symmetrized_divergence_backward(q_1, q_2, q_3)
        
        x_new = (x + tau * div_p + tau * y) / (1 + tau)
        w_h_new = w_h + tau * (div_q_h - p_h)
        w_v_new = w_v + tau * (div_q_v - p_v)
        
        x_bar = x_new + theta * (x_new - x)
        w_h_bar = w_h_new + theta * (w_h_new - w_h)
        w_v_bar = w_v_new + theta * (w_v_new - w_v)
        
        x = x_new
        w_h = w_h_new
        w_v = w_v_new
    
    return x, w_h, w_v

def find_smooth_gradient_region(image, row, min_length=20):
    """
    自动检测图像中的平滑渐变区域
    
    检测标准:
    1. 一阶梯度在中等范围内（排除过于平坦和过于陡峭的区域）
    2. 二阶差分较小（梯度变化平缓）
    
    这样可以找到真正的渐变区域，而非均匀背景或边缘
    """
    profile = image[row, :]
    grad = np.abs(np.diff(profile))
    grad2 = np.abs(np.diff(grad))
    
    grad_low = np.percentile(grad, 20)
    grad_high = np.percentile(grad, 70)
    
    medium_grad_mask = (grad >= grad_low) & (grad <= grad_high)
    
    small_curvature_mask = grad2 < np.percentile(grad2, 50)
    
    smooth_mask = medium_grad_mask[:-1] & small_curvature_mask
    
    regions = []
    start = None
    for i, is_smooth in enumerate(smooth_mask):
        if is_smooth and start is None:
            start = i
        elif not is_smooth and start is not None:
            if i - start >= min_length:
                regions.append((start, i))
            start = None
    if start is not None and len(smooth_mask) - start >= min_length:
        regions.append((start, len(smooth_mask)))
    
    if regions:
        longest = max(regions, key=lambda r: r[1] - r[0])
        return slice(longest[0], longest[1])
    else:
        return slice(len(profile)//4, len(profile)//2)

n = 128
camera = resize(data.camera(), (n, n))

sigma_noise = 0.1
camera_noisy = random_noise(camera, mode='gaussian', var=sigma_noise**2)

alpha_tv = 0.2
camera_tv = denoise_tv_chambolle(camera_noisy, weight=alpha_tv)

alpha0 = 0.3
alpha1 = 0.15
print("===== TGV：TV的改进 =====")
print(f"\n[步长选择说明]")
print(f"  Chambolle-Pock收敛条件: τ·σ·||K||² ≤ 1")
print(f"  其中 ||K|| 是联合算子 K=[D;E] 的谱范数")
print(f"  对于离散梯度算子，||K|| ≈ √8 ≈ 2.83")
print(f"  因此实际约束: τ·σ ≤ 1/||K||² ≈ 1/8 = 0.125")
print(f"  本实验取 τ=σ=0.25，τ·σ=0.0625 < 0.125 ✓")
print(f"  迭代次数: 500 (TGV通常需要500-1000次收敛)")

camera_tgv, w_h, w_v = tgv_denoise_primal_dual(camera_noisy, alpha0, alpha1, n_iter=500)

psnr_noisy = peak_signal_noise_ratio(camera, camera_noisy)
psnr_tv = peak_signal_noise_ratio(camera, camera_tv)
psnr_tgv = peak_signal_noise_ratio(camera, camera_tgv)

ssim_noisy = structural_similarity(camera, camera_noisy, data_range=1.0)
ssim_tv = structural_similarity(camera, camera_tv, data_range=1.0)
ssim_tgv = structural_similarity(camera, camera_tgv, data_range=1.0)

print(f"\n噪声水平 σ = {sigma_noise}")
print(f"\n去噪结果对比:")
print(f"  含噪图像: PSNR = {psnr_noisy:.2f} dB, SSIM = {ssim_noisy:.4f}")
print(f"  TV去噪:   PSNR = {psnr_tv:.2f} dB, SSIM = {ssim_tv:.4f}")
print(f"  TGV去噪:  PSNR = {psnr_tgv:.2f} dB, SSIM = {ssim_tgv:.4f}")
print(f"\nTGV参数:")
print(f"  α₀ (二阶正则化) = {alpha0}")
print(f"  α₁ (一阶正则化) = {alpha1}")
print(f"  α₀/α₁ = {alpha0/alpha1:.2f} (建议约2)")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(camera_noisy, cmap='gray')
axes[0].set_title(f'含噪图像\nPSNR={psnr_noisy:.2f}dB')
axes[0].axis('off')

axes[1].imshow(camera, cmap='gray')
axes[1].set_title('原始图像')
axes[1].axis('off')

axes[2].imshow(camera_tv, cmap='gray')
axes[2].set_title(f'TV去噪\n边缘锐利，渐变区阶梯化\nPSNR={psnr_tv:.2f}dB')
axes[2].axis('off')

axes[3].imshow(camera_tgv, cmap='gray')
axes[3].set_title(f'TGV去噪\n渐变区更自然\nPSNR={psnr_tgv:.2f}dB')
axes[3].axis('off')

plt.suptitle('TV vs TGV：阶梯效应的改善', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_TV_vs_TGV对比.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
    plt.show()
plt.close(fig)

row = n // 2
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(camera[row, :], 'k-', linewidth=2, label='真实信号')
axes[0].plot(camera_tv[row, :], 'r-', linewidth=1.5, label='TV（阶梯效应）')
axes[0].plot(camera_tgv[row, :], 'g-', linewidth=1.5, label='TGV（更平滑）')
axes[0].set_title('中心行剖面对比')
axes[0].legend()
axes[0].set_xlabel('像素索引')
axes[0].set_ylabel('灰度值')
axes[0].grid(True, alpha=0.3)

grad_region = find_smooth_gradient_region(camera, row)
axes[1].plot(camera[row, grad_region], 'k-', linewidth=2, label='真实（渐变区域）')
axes[1].plot(camera_tv[row, grad_region], 'r-', linewidth=1.5, label='TV（阶梯效应）')
axes[1].plot(camera_tgv[row, grad_region], 'g-', linewidth=1.5, label='TGV（渐变自然）')
axes[1].set_title(f'渐变区域放大（自动检测: 索引{grad_region.start}-{grad_region.stop}）')
axes[1].legend()
axes[1].set_xlabel('像素索引')
axes[1].set_ylabel('灰度值')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_阶梯效应改善分析.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
    plt.show()
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

w_mag = np.sqrt(w_h**2 + w_v**2)
im = axes[0].imshow(w_mag, cmap='hot')
axes[0].set_title('辅助向量场 |w|（梯度仿射分量）')
axes[0].axis('off')
plt.colorbar(im, ax=axes[0], fraction=0.046)

h, w = camera_tgv.shape
skip = 8
Y, X = np.mgrid[0:h:skip, 0:w:skip]
scale = np.percentile(np.sqrt(w_h**2 + w_v**2), 90) * skip * 10 + 1e-6
axes[1].quiver(X, Y, w_h[::skip, ::skip], -w_v[::skip, ::skip], 
               scale=scale, alpha=0.7, color='blue')
axes[1].imshow(camera_tgv, cmap='gray', alpha=0.3)
axes[1].set_title('向量场w叠加在去噪图像上')
axes[1].axis('off')

axes[2].hist(w_mag.flatten(), bins=50, density=True, alpha=0.7, color='blue')
axes[2].set_xlabel('|w|')
axes[2].set_ylabel('概率密度')
axes[2].set_title('辅助向量场幅值分布')
axes[2].grid(True, alpha=0.3)

plt.suptitle('TGV核心机制：辅助向量场w可视化', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_辅助向量场可视化.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
    plt.show()
plt.close(fig)

alpha_ratios = [0.5, 1.0, 2.0, 4.0, 8.0]
alpha1_fixed = 0.15
tgv_results = []
psnr_results = []
ssim_results = []

assert len(alpha_ratios) <= 5, "alpha_ratios最多5个元素，否则步骤4子图布局会溢出"

print("\n===== 参数敏感性实验 =====")
print(f"固定 α₁ = {alpha1_fixed}，变化 α₀/α₁ 比值:")
print(f"  (共{len(alpha_ratios)}组实验，每组500次迭代，预计耗时3-10分钟)")

for i, ratio in enumerate(alpha_ratios):
    print(f"  [{i+1}/{len(alpha_ratios)}] 正在计算 α₀/α₁ = {ratio:.1f} ... ", end='', flush=True)
    a0 = ratio * alpha1_fixed
    x_tgv, _, _ = tgv_denoise_primal_dual(camera_noisy, a0, alpha1_fixed, n_iter=500)
    tgv_results.append(x_tgv)
    psnr_r = peak_signal_noise_ratio(camera, x_tgv)
    ssim_r = structural_similarity(camera, x_tgv, data_range=1.0)
    psnr_results.append(psnr_r)
    ssim_results.append(ssim_r)
    print(f"完成 (PSNR={psnr_r:.2f}dB, SSIM={ssim_r:.4f})")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, (ratio, img) in enumerate(zip(alpha_ratios[:5], tgv_results[:5])):
    ax = axes[i // 3, i % 3]
    ax.imshow(img, cmap='gray')
    ax.set_title(f'α₀/α₁ = {ratio:.1f}\nPSNR={psnr_results[i]:.2f}dB, SSIM={ssim_results[i]:.3f}')
    ax.axis('off')

axes[1, 2].plot(alpha_ratios, psnr_results, 'bo-', linewidth=2, markersize=8, label='PSNR')
axes[1, 2].set_xlabel('α₀/α₁')
axes[1, 2].set_ylabel('PSNR (dB)', color='blue')
axes[1, 2].tick_params(axis='y', labelcolor='blue')
axes[1, 2].grid(True, alpha=0.3)
ax2 = axes[1, 2].twinx()
ax2.plot(alpha_ratios, ssim_results, 'rs-', linewidth=2, markersize=8, label='SSIM')
ax2.set_ylabel('SSIM', color='red')
ax2.tick_params(axis='y', labelcolor='red')
axes[1, 2].set_title('参数敏感性分析')

plt.suptitle('TGV参数敏感性：α₀/α₁对去噪效果的影响', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤4_参数敏感性分析.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
    plt.show()
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

methods = ['含噪', 'TV', 'TGV']
psnrs = [psnr_noisy, psnr_tv, psnr_tgv]
ssims = [ssim_noisy, ssim_tv, ssim_tgv]
colors = ['gray', 'red', 'green']

x_pos = np.arange(len(methods))
bars1 = axes[0].bar(x_pos, psnrs, color=colors, alpha=0.7)
axes[0].set_ylabel('PSNR (dB)')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(methods)
axes[0].set_title('PSNR对比')
axes[0].set_ylim([0, max(psnrs) + 5])
for i, p in enumerate(psnrs):
    axes[0].text(i, p + 0.3, f'{p:.2f}', ha='center', fontsize=10)

bars2 = axes[1].bar(x_pos, ssims, color=colors, alpha=0.7)
axes[1].set_ylabel('SSIM')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(methods)
axes[1].set_title('SSIM对比')
axes[1].set_ylim([0, 1.05])
for i, s in enumerate(ssims):
    axes[1].text(i, s + 0.02, f'{s:.3f}', ha='center', fontsize=10)

table_data = [
    ['惩罚项', '||∇x||₁', 'α₁||∇x-w||₁ + α₀||Ew||₁'],
    ['偏好解', '分段常数', '分段仿射'],
    ['渐变区域', '阶梯效应', '自然渐变'],
    ['边缘保持', '能', '能'],
    ['参数数量', '1个 (α)', '2个 (α₀, α₁)'],
    ['计算复杂度', '低', '较高'],
]
axes[2].axis('off')
table = axes[2].table(cellText=table_data,
                      colLabels=['特性', 'TV', 'TGV'],
                      loc='center',
                      cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)
axes[2].set_title('TV vs TGV 特性对比', fontsize=12, pad=20)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤5_性能与特性对比.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
    plt.show()
plt.close(fig)

print("\n===== 附录2B 核心结论 =====")
print("\n1. 阶梯效应根源:")
print("   TV只使用一阶导数，偏好分段常数函数")
print("   渐变区域被近似为一系列阶梯")
print("\n2. TGV改进思路:")
print("   引入二阶导数信息，允许分段仿射函数")
print("   TGV = min_w [α₁||∇x-w||₁ + α₀||Ew||₁]")
print("   辅助向量场w逼近梯度的仿射分量")
print("\n3. TV与TGV的关系:")
print("   当α₀→∞时，TGV退化为TV")
print("   TV是TGV的特例")
print("\n4. 参数选择:")
print("   α₀/α₁ ≈ 2 通常效果较好")
print("   α₀控制二阶平滑，α₁控制一阶平滑")
print("\n5. 应用场景:")
print("   PET/MRI重建、图像修复等渐变区域重要的场景")
print("\n6. 局限性:")
print("   TGV仍是显式先验，参数需手工调节")
print("   更根本的改进需要走向隐式先验（2.4节）")

# ===== 保存数值结果 =====
import json
results_summary = {
    'image_size': n,
    'noise_sigma': float(sigma_noise),
    'psnr_noisy_dB': float(round(psnr_noisy, 2)),
    'psnr_tv_dB': float(round(psnr_tv, 2)),
    'psnr_tgv_dB': float(round(psnr_tgv, 2)),
    'ssim_noisy': float(round(ssim_noisy, 4)),
    'ssim_tv': float(round(ssim_tv, 4)),
    'ssim_tgv': float(round(ssim_tgv, 4)),
    'alpha0': float(alpha0),
    'alpha1': float(alpha1),
    'alpha0_alpha1_ratio': float(round(alpha0/alpha1, 2)),
    'sensitivity_alpha_ratio': {f'ratio_{r:.1f}': {'psnr_dB': float(round(p, 2)), 'ssim': float(round(s, 4))} for r, p, s in zip(alpha_ratios, psnr_results, ssim_results)},
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
