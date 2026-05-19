import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import sys
import os

# Jupyter 中渲染 LaTeX 公式
try:
    from IPython.display import display, Markdown
    _in_jupyter = True
except ImportError:
    _in_jupyter = False

# ====== 中文字体配置（兼容本地和 Google Colab）======
_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验1.3-3', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验1.3-3')
else:
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    pass
# ========================================================

np.random.seed(42)

print("="*60)
print("实验1.3-3：逆问题中的'幽灵'——几乎不可见的扰动")
print("="*60)
print("本实验演示：两个视觉截然不同的图像，其观测几乎相同")
print("对应 1.3 节 '逆问题中的幽灵：几乎不可见的扰动'")

# ---- 1. 准备图像和模糊算子 ----
n = 64
phantom = resize(shepp_logan_phantom(), (n, n), order=3, preserve_range=True, anti_aliasing=True)
x = phantom / phantom.max()

def gaussian_psf(size, sigma):
    ax = np.concatenate((np.arange(0, size // 2), np.arange(-size // 2, 0)))
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return h / h.sum()

def blur(x, h):
    return np.real(np.fft.ifft2(np.fft.fft2(h) * np.fft.fft2(x)))

sigma = 3.0
h = gaussian_psf(n, sigma)

# ---- 2. 构造"幽灵"图像 ----
# 高频振荡图案落在模糊算子的近似零空间中
# 物理解释：高斯模糊在傅里叶域是低通滤波器，高频分量被强烈衰减
# - 高频分量对应模糊算子的小奇异值方向（与实验 1.7 呼应）
# - 频率越高 → 傅里叶域衰减越强 → 越接近零空间

xx, yy = np.meshgrid(np.arange(n), np.arange(n))

# 扫描不同频率，展示衰减规律
# 去掉高频（28, 30），避免数值精度问题导致放大因子失真
# 频率25时衰减已接近机器精度，足以展示指数增长趋势
frequencies = [5, 10, 15, 20, 25]
attenuations = []

for freq in frequencies:
    d_test = np.sin(2 * np.pi * freq * xx / n) * np.sin(2 * np.pi * freq * yy / n)
    # 使用与验证相同的 blur() 算子（FFT 圆周卷积），确保一致性
    Ad_test = blur(d_test, h)
    attenuation = np.linalg.norm(Ad_test) / np.linalg.norm(d_test)
    attenuations.append(attenuation)

# 使用频率 15 作为幽灵分量（中等衰减，视觉效果明显）
freq = 15
alpha = 0.3  # 幽灵分量的幅度系数，意图明确：控制扰动强度

# 二维正弦波：避免纯水平条纹的"刻意"感，更接近真实场景
# 使用两个正交方向的正弦波叠加，形成棋盘状高频纹理
d = alpha * np.sin(2 * np.pi * freq * xx / n) * np.sin(2 * np.pi * freq * yy / n)

# 构造"幽灵"图像
x_prime = x + d

# ---- 3. 验证"几乎幽灵"性质 ----
Ax = blur(x, h)
Ax_prime = blur(x_prime, h)
Ad = blur(d, h)

rel_meas_err = np.linalg.norm(Ad) / np.linalg.norm(Ax)
rel_img_err = np.linalg.norm(d) / np.linalg.norm(x)

# ---- 4. 计算各频率的误差放大因子 ----
# 放大因子 = 图像相对差异 / 测量相对差异
# 对于频率 f 的扰动，放大因子 ≈ 1 / attenuation(f)
amplification_factors = [1.0 / a for a in attenuations]

# 建立频率到索引的映射，避免线性搜索（O(n) → O(1)）
freq_to_idx = {f: i for i, f in enumerate(frequencies)}

print(f"\n实验设置:")
print(f"- 图像尺寸: {n}x{n}")
print(f"- 高斯模糊核: σ=3.0")
print(f"- 幽灵分量频率: {freq}")
print(f"- 扫描频率范围: {frequencies[0]} ~ {frequencies[-1]}")
print("-"*60)

# ---- 5. 可视化 ----
# 计算傅里叶幅度谱，展示幽灵分量在频域的位置
X_fft = np.abs(np.fft.fftshift(np.fft.fft2(x)))
X_prime_fft = np.abs(np.fft.fftshift(np.fft.fft2(x_prime)))
D_fft = np.abs(np.fft.fftshift(np.fft.fft2(d)))

fig, axes = plt.subplots(3, 4, figsize=(18, 13))

axes[0, 0].imshow(x, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('图像 x')
axes[0, 0].axis('off')

axes[0, 1].imshow(x_prime, cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title(f"图像 x'（x + 幽灵分量）\n‖x-x'‖/‖x‖ = {rel_img_err:.1%}")
axes[0, 1].axis('off')

axes[0, 2].imshow(d, cmap='RdBu_r', vmin=-alpha, vmax=alpha)
axes[0, 2].set_title(f'差异 d = x\' - x\n（高频分量，α={alpha}）')
axes[0, 2].axis('off')

axes[0, 3].plot(frequencies, attenuations, 'o-', linewidth=2, markersize=8)
axes[0, 3].axvline(freq, color='red', linestyle='--', alpha=0.8,
                   label=f'选用频率={freq}')
axes[0, 3].set_xlabel('空间频率')
axes[0, 3].set_ylabel('衰减因子 ‖Ad‖/‖d‖')
axes[0, 3].set_title('频率-衰减规律\n频率越高→越接近零空间')
axes[0, 3].legend(fontsize=9)
axes[0, 3].grid(True)

ax_vmin, ax_vmax = Ax.min(), Ax.max()
axes[1, 0].imshow(Ax, cmap='gray', vmin=ax_vmin, vmax=ax_vmax)
axes[1, 0].set_title('Ax（x 的观测）')
axes[1, 0].axis('off')

axes[1, 1].imshow(Ax_prime, cmap='gray', vmin=ax_vmin, vmax=ax_vmax)
axes[1, 1].set_title("Ax'（x' 的观测）")
axes[1, 1].axis('off')

Ax_diff = Ax_prime - Ax
# 对差异进行归一化，确保即使值很小也能清晰显示
vmax_diff = np.abs(Ax_diff).max()
if vmax_diff > 0:
    Ax_diff_norm = Ax_diff / vmax_diff  # 归一化到 [-1, 1]
    axes[1, 2].imshow(Ax_diff_norm, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 2].set_title(f"Ax' - Ax（观测差异，已归一化）\n‖Ax'-Ax‖/‖Ax‖ = {rel_meas_err:.3%}\n（原始 max={vmax_diff:.2e}）")
else:
    axes[1, 2].text(0.5, 0.5, '差异为零',
                    ha='center', va='center', transform=axes[1, 2].transAxes,
                    fontsize=14, color='gray')
    axes[1, 2].set_title(f"Ax' - Ax（观测差异）\n‖Ax'-Ax‖/‖Ax‖ = 0%")
axes[1, 2].axis('off')

# 方案 A：放大因子对比图（对数坐标，使不同量级的放大因子都可见）
# 使用索引作为 x 位置，确保柱子均匀分布且完整显示
x_positions = np.arange(len(frequencies))
axes[1, 3].bar(x_positions, amplification_factors, color='steelblue', edgecolor='navy', width=0.6)
axes[1, 3].axvline(freq_to_idx[freq], color='red', linestyle='--', alpha=0.8, label=f'选用频率={freq}')
axes[1, 3].set_xlabel('空间频率')
axes[1, 3].set_ylabel('误差放大因子（对数坐标）')
axes[1, 3].set_title('各频率的误差放大因子\n（图像差异/测量差异）')
axes[1, 3].set_xticks(x_positions)
axes[1, 3].set_xticklabels([str(f) for f in frequencies])  # 用频率值作为标签
axes[1, 3].set_xlim(-0.5, len(frequencies) - 0.5)  # 确保所有柱子完整显示
axes[1, 3].legend(fontsize=9)
axes[1, 3].grid(True, axis='y', alpha=0.3)
axes[1, 3].set_yscale('log')  # 对数坐标使小值可见
# 标注选用频率对应的放大因子
amp_val = amplification_factors[freq_to_idx[freq]]
axes[1, 3].text(freq_to_idx[freq], amp_val * 1.5,
                f'{amp_val:.0f}×',
                ha='center', fontsize=10, color='red', fontweight='bold')

# ---- 第三行：傅里叶幅度谱对比 ----
# 展示幽灵分量在频域的位置，与衰减因子曲线形成呼应
axes[2, 0].imshow(np.log1p(X_fft), cmap='viridis')
axes[2, 0].set_title('log(1+|FFT(x)|)\n原始图像频谱')
axes[2, 0].axis('off')

axes[2, 1].imshow(np.log1p(X_prime_fft), cmap='viridis')
axes[2, 1].set_title("log(1+|FFT(x')|)\n幽灵图像频谱")
axes[2, 1].axis('off')

# 差异频谱：突出显示幽灵分量在频域的峰值位置
axes[2, 2].imshow(np.log1p(D_fft), cmap='hot')
axes[2, 2].set_title(f'log(1+|FFT(d)|)\n幽灵分量频谱（频率={freq}）')
axes[2, 2].axis('off')

# 频谱差异的定量对比：展示高频区域的能量分布
# 计算径向平均频谱，与衰减因子曲线呼应

center = n // 2
grid_Y, grid_X = np.ogrid[:n, :n]
r = np.sqrt((grid_X - center)**2 + (grid_Y - center)**2).astype(int)

# 径向平均频谱：只计算到 n//2，避免角落处超出范围的径向距离
r_max = n // 2
mask = r <= r_max
radial_X = np.bincount(r[mask].ravel(), X_fft[mask].ravel()) / np.bincount(r[mask].ravel())
radial_D = np.bincount(r[mask].ravel(), D_fft[mask].ravel()) / np.bincount(r[mask].ravel())

axes[2, 3].semilogy(radial_X, 'b-', linewidth=2, label='|FFT(x)|', alpha=0.7)
axes[2, 3].semilogy(radial_D, 'r-', linewidth=2, label='|FFT(d)|', alpha=0.7)
# 棋盘纹 sin(2πf·x)·sin(2πf·y) 的傅里叶峰在径向频率 f√2 处
freq_radial = freq * np.sqrt(2)
axes[2, 3].axvline(freq_radial, color='red', linestyle='--', alpha=0.8, label=f'幽灵径向频率≈{freq_radial:.1f}')
axes[2, 3].set_xlabel('空间频率（径向）')
axes[2, 3].set_ylabel('平均幅度（对数）')
axes[2, 3].set_title('径向平均频谱\n与衰减因子曲线呼应')
axes[2, 3].legend(fontsize=9)
axes[2, 3].grid(True, alpha=0.3)
axes[2, 3].set_xlim(0, r_max)

plt.suptitle('"几乎幽灵"现象\n两个视觉截然不同的图像，其观测几乎相同', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_3_3_几乎幽灵.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("几乎幽灵验证")
print("="*60)
print(f"图像相对差异: ‖x - x'‖/‖x‖ = {rel_img_err:.1%}")
print(f"测量相对差异: ‖Ax - Ax'‖/‖Ax‖ = {rel_meas_err:.3%}")
print(f"放大因子: 图像差异/测量差异 = {rel_img_err / rel_meas_err:.0f} 倍")
print("\n" + "="*60)
print("频率 vs 衰减 vs 放大因子")
print("="*60)
print(f"{'频率':>6s}  {'衰减因子':>10s}  {'放大因子':>10s}")
print("-" * 32)
for f, a, amp in zip(frequencies, attenuations, amplification_factors):
    if amp == np.inf:
        amp_str = "∞"
    else:
        amp_str = f"{amp:.0f}×"
    print(f"{f:6d}  {a:10.4f}  {amp_str:>10s}")

print("\n" + "="*60)
print("实验结论：")
print("="*60)
print("1. 高频扰动（幽灵分量）在图像空间中差异巨大")
print("2. 但经过正向算子（高斯模糊）后，观测差异极小")
print("3. 频率越高，衰减越强，放大因子越大")
print("4. 这正是'不适定性'的本质：问题本身存在'幽灵'方向")
print("5. 即使零噪声，不引入额外信息也无法唯一确定解")
print("="*60)

if _in_jupyter:
    display(Markdown(r"""
**理论回顾**：
- **几乎幽灵**（almost-ghost）：两个图像 $x$ 和 $x'$，测量几乎相同但视觉截然不同
- **根本原因**：差异 $d = x' - x$ 落在 $A$ 的近似零空间中，$\|Ad\| \ll \|d\|$
- **核心结论**：不适定性不是算法的缺陷，而是问题本身的内在性质

**物理直觉**：
- 高斯模糊在傅里叶域是低通滤波器，高频分量被强烈衰减
- 高频幽灵分量对应模糊算子的小奇异值方向
- 算子 $A$ 抹去了幽灵方向上的信息，逆问题无法区分这些方向
"""))
else:
    print("\n理论回顾：")
    print("- 几乎幽灵（almost-ghost）：两个图像 x 和 x'，测量几乎相同但视觉截然不同")
    print("- 根本原因：差异 d = x' - x 落在 A 的近似零空间中，‖Ad‖ ≪ ‖d‖")
    print("- 核心结论：不适定性不是算法的缺陷，而是问题本身的内在性质")
    print("\n物理直觉：")
    print("- 高斯模糊在傅里叶域是低通滤波器，高频分量被强烈衰减")
    print("- 高频幽灵分量对应模糊算子的小奇异值方向")
    print("- 算子 A 抹去了幽灵方向上的信息，逆问题无法区分这些方向")
