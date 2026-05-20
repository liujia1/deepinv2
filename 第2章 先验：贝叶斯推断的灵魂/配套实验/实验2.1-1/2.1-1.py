"""
实验2.1-1 先验与正则化的数值验证
对应章节：2.1 先验的数学角色：正则化的概率诠释
知识点：贝叶斯定理；先验→正则化对应关系；-ln p(x|y) = 数据项 + 正则项；λ = σ²/σ_x²

素材来源：
  - M1 CompImLab25.ipynb: Tikhonov闭式解
  - IP22 statistical_perspective.md: 高斯先验MAP推导
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import os
import sys

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.1-1', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.1-1')
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _chinese_path)
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=_chinese_path)

np.random.seed(42)

n = 128
x = resize(data.camera(), (n, n))

noise_lev = 1e-1
y = x + noise_lev * np.random.randn(n, n)

sigma = noise_lev
sigma_x = 1.0
lambda_Tikh = sigma**2 / sigma_x**2
x_Tikh = y / (1 + lambda_Tikh)

data_term = 0.5 / sigma**2 * np.sum((y - x)**2)
reg_term_gauss = 0.5 / sigma_x**2 * np.sum(x**2)
posterior_energy = data_term + reg_term_gauss

grad_at_map = -1/sigma**2 * (y - x_Tikh) + 1/sigma_x**2 * x_Tikh

print("===== 先验 = 正则化的数值验证 =====")
print(f"噪声水平 σ = {sigma:.4f}")
print(f"先验标准差 σ_x = {sigma_x:.4f}")
print(f"\n核心等式验证:")
print(f"  -ln p(x|y) = 数据项 + 正则项 + 常数")
print(f"  数据项 = 1/(2σ²) ||y-x||² = {data_term:.4f}")
print(f"  正则项 = 1/(2σ_x²) ||x||² = {reg_term_gauss:.4f}")
print(f"  后验能量 = {posterior_energy:.4f}")
print(f"\n正则化参数的贝叶斯诠释:")
print(f"  λ = σ²/σ_x² = {lambda_Tikh:.4f}")
print(f"  MAP 处梯度范数（应为0）: {np.linalg.norm(grad_at_map):.2e}")
print(f"  闭式解 x_Tikh = y/(1+λ) 已验证")

Orig_MSE = mean_squared_error(x, y)
Tikh_MSE = mean_squared_error(x, x_Tikh)
Orig_PSNR = peak_signal_noise_ratio(x, y)
Tikh_PSNR = peak_signal_noise_ratio(x, x_Tikh)

print(f"\n含噪  MSE: {Orig_MSE:.4f}, PSNR: {Orig_PSNR:.2f} dB")
print(f"Tikh  MSE: {Tikh_MSE:.4f}, PSNR: {Tikh_PSNR:.2f} dB")

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].imshow(x, cmap='gray')
axs[0].set_title('原始图像 x')

axs[1].imshow(y, cmap='gray')
axs[1].set_title(f'含噪观测 y\nσ={noise_lev:.2f}, PSNR={Orig_PSNR:.2f}dB')

axs[2].imshow(x_Tikh, cmap='gray')
axs[2].set_title(f'Tikhonov重建 (高斯先验)\nλ=σ²/σ_x²={lambda_Tikh:.4f}\nPSNR={Tikh_PSNR:.2f}dB')

for ax in axs:
    ax.axis('off')

plt.suptitle('先验 = 正则化：高斯先验 → Tikhonov正则化', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_高斯先验与Tikhonov验证.png'), dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

t = np.linspace(-2, 2, 400)
axes[0].plot(t, t**2, 'b-', linewidth=2, label='L2: t² (高斯先验正则项)')
axes[0].set_title('高斯先验对应的正则项形态')
axes[0].legend()
axes[0].set_xlabel('t')
axes[0].set_ylabel('惩罚值')
axes[0].set_ylim(-0.2, 4)
axes[0].grid(True, alpha=0.3)

lambdas = [0.001, 0.01, 0.1, 1.0]
center_row = n // 2
for lam_val in lambdas:
    x_hat = y / (1 + lam_val)
    axes[1].plot(x_hat[center_row, :], linewidth=1.5, label=f'λ={lam_val}')
axes[1].plot(x[center_row, :], 'k--', linewidth=1, label='真实')
axes[1].set_title('不同λ下中心行剖面\nλ小→欠正则化，λ大→过正则化')
axes[1].legend()
axes[1].set_xlabel('像素索引')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_正则项形态与λ影响.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n===== 结论 =====")
print("1. -ln p(x|y) = 数据项 + 正则项，验证了先验与正则项的对应关系")
print("2. λ = σ²/σ_x² 给出了正则化参数的概率诠释")
print("3. 不同λ值影响重建质量：λ小噪声残留，λ大过度平滑")
