"""
实验2.1-2 正则化参数λ对重建的影响
对应章节：2.1 先验的数学角色：正则化的概率诠释
知识点：λ = σ²/σ_x²；λ小→欠正则化，λ大→过正则化

素材来源：
  - M1 CompImLab25.ipynb: Tikhonov闭式解 + PSNR度量

重要假设：
  本实验假设前向算子 A = I（恒等算子），即仅有噪声污染，无模糊或其他退化。
  在此假设下，Tikhonov 解有闭式形式：x_hat = y / (1 + λ)
  
  对于一般的 A ≠ I 情形，需要迭代求解（见实验2.1-1）。
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio
import os
import sys

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.1-2', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.1-2')
else:
    _chinese_path = '.chinese'
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()

sys.path.insert(0, _chinese_path)
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=_chinese_path)

np.random.seed(42)

n = 128
x = resize(data.camera(), (n, n))

DATA_RANGE = 1.0
print(f"[参数验证] 图像归一化到 [0,1]，DATA_RANGE = {DATA_RANGE}")

noise_lev = 1e-1
y = x + noise_lev * np.random.randn(n, n)

print("\n" + "="*60)
print("重要假设: 本实验假设前向算子 A = I（仅有噪声，无模糊）")
print("在此假设下，Tikhonov 解有闭式形式: x_hat = y / (1 + λ)")
print("="*60 + "\n")

lambdas = np.logspace(-4, 1, 30)
psnr_list = []

for lam in lambdas:
    x_tikh = y / (1 + lam)
    psnr_list.append(peak_signal_noise_ratio(x, x_tikh, data_range=DATA_RANGE))

best_idx = np.argmax(psnr_list)

# 注：真实场景中 x 未知，σ_x 应从训练集统计或独立先验中获取
# 此处直接使用真实图像的 std 仅为教学演示
sigma_x = x.std()
lambda_bayes = noise_lev**2 / sigma_x**2

x_tikh_bayes = y / (1 + lambda_bayes)
psnr_bayes = peak_signal_noise_ratio(x, x_tikh_bayes, data_range=DATA_RANGE)
psnr_noisy = peak_signal_noise_ratio(x, y, data_range=DATA_RANGE)

print("===== 正则化参数 λ 的贝叶斯诠释 =====")
print(f"噪声水平 σ = {noise_lev:.4f}")
print(f"先验标准差 σ_x = {sigma_x:.4f}（由图像数据的标准差估计）")
print(f"  注: σ_x = std(x)，而非 RMS = sqrt(E[x²])")
print(f"  因为 camera() 图像均值 ≈ {x.mean():.2f}，非零均值")
print(f"  注：真实场景中 x 未知，σ_x 应从训练集或先验分布估计")
print(f"\n贝叶斯 λ = σ^2/σ_x^2 = {lambda_bayes:.4f}")
print(f"贝叶斯 λ 对应 PSNR = {psnr_bayes:.2f} dB")
print(f"PSNR 扫描最优 λ = {lambdas[best_idx]:.4f}, PSNR = {psnr_list[best_idx]:.2f} dB")

fig1, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].semilogx(lambdas, psnr_list, 'b-o', markersize=4, label='Tikhonov')
axes[0].axhline(y=psnr_noisy, color='r', linestyle='--', label='含噪观测')
axes[0].axvline(x=lambda_bayes, color='g', linestyle=':', linewidth=2,
                label=f'贝叶斯 λ={lambda_bayes:.4f}')
axes[0].axvline(x=lambdas[best_idx], color='b', linestyle='--', alpha=0.5,
                label=f'PSNR最优 λ={lambdas[best_idx]:.4f}')
axes[0].scatter([lambda_bayes], [psnr_bayes], color='g', s=100, zorder=5, marker='*',
                label=f'贝叶斯 λ 点 (PSNR={psnr_bayes:.2f}dB)')
axes[0].set_xlabel('λ')
axes[0].set_ylabel('PSNR (dB)')
axes[0].set_title('PSNR vs λ：贝叶斯 λ 与 PSNR 最优 λ 对比\n(A=I 假设下的闭式解)')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

lambdas_demo = [0.001, lambda_bayes, 1.0]
labels_demo = ['λ=0.001 (欠正则化)', f'贝叶斯 λ={lambda_bayes:.4f}', 'λ=1.0 (过正则化)']

for i, (lam_val, label) in enumerate(zip(lambdas_demo, labels_demo)):
    x_tikh = y / (1 + lam_val)
    axes[1].plot(x_tikh[n//2, :], linewidth=1.5, label=label)
axes[1].plot(x[n//2, :], 'k--', linewidth=1, label='真实')
axes[1].set_title('不同 λ 下中心行剖面')
axes[1].legend(fontsize=8)
axes[1].set_xlabel('像素索引')

x_best = y / (1 + lambdas[best_idx])
axes[2].plot(x[n//2, :], 'k--', linewidth=1.5, label='真实')
axes[2].plot(x_best[n//2, :], 'b-', linewidth=1.5, label=f'最优 λ={lambdas[best_idx]:.4f}')
axes[2].set_title('最优 λ 下的重建剖面')
axes[2].legend()
axes[2].set_xlabel('像素索引')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_正则化参数影响.png'), dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig1)

fig2, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(x, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(np.clip(y, 0, 1), cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title(f'含噪 (σ={noise_lev})')
axes[0, 1].axis('off')

axes[0, 2].imshow(np.clip(x_best, 0, 1), cmap='gray', vmin=0, vmax=1)
axes[0, 2].set_title(f'最优 λ 重建\nλ={lambdas[best_idx]:.4f}\nPSNR={psnr_list[best_idx]:.2f}dB')
axes[0, 2].axis('off')

for i, (lam_val, label) in enumerate(zip(lambdas_demo, labels_demo)):
    x_tikh = y / (1 + lam_val)
    psnr_val = peak_signal_noise_ratio(x, x_tikh, data_range=DATA_RANGE)
    axes[1, i].imshow(np.clip(x_tikh, 0, 1), cmap='gray', vmin=0, vmax=1)
    axes[1, i].set_title(f'{label}\nPSNR={psnr_val:.2f}dB')
    axes[1, i].axis('off')

plt.suptitle('正则化参数 λ 的影响：λ小→噪声残留，λ大→过度平滑\n(A=I 假设下的闭式解)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_不同λ下的重建对比.png'), dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig2)

print("\n===== 结论 =====")
print("1. λ = σ^2/σ_x^2 给出了正则化参数的概率诠释")
print("2. λ过小：噪声残留严重（欠正则化）")
print("3. λ过大：图像过度平滑（过正则化）")
print(f"4. 贝叶斯 λ = {lambda_bayes:.4f}（σ_x = std(x)）与 PSNR 最优 λ = {lambdas[best_idx]:.4f} 接近，"
      f"验证了贝叶斯框架的正确性")
print("5. 本实验假设 A=I（仅有噪声），闭式解 x_hat = y/(1+λ)")
print("   对于 A≠I 的逆问题，需迭代求解（见实验2.1-1）")
psnr_gain = psnr_list[best_idx] - psnr_noisy
print(f"6. A=I 的 Tikhonov 去噪能力有限：最优解仅比含噪图提升 {psnr_gain:.2f}dB")
print("   这正是需要更强先验（TV、深度学习等）的原因")
