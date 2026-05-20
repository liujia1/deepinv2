"""
实验2.1-1 先验与正则化的数值验证（含模糊逆问题）
对应章节：2.1 先验的数学角色：正则化的概率诠释
知识点：贝叶斯定理；先验→正则化对应关系；-ln p(x|y) = 数据项 + 正则项；λ = σ²/σ_x²；
         非平凡前向算子 A ≠ I 下的 Tikhonov 重建

素材来源：
  - M1 CompImLab25.ipynb: Tikhonov闭式解
  - IP22 statistical_perspective.md: 高斯先验MAP推导
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from scipy.ndimage import gaussian_filter
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

x_std = np.std(x)
print(f"[参数验证] 图像标准差 std(x) = {x_std:.4f}")

blur_sigma = 2.0

def A(u):
    """Forward: Gaussian blur (mode='reflect' 避免边界伪影)"""
    return gaussian_filter(u, sigma=blur_sigma, mode='reflect')

def AT(u):
    """Adjoint: 高斯模糊在 reflect 边界下是自伴算子 A^T = A"""
    return gaussian_filter(u, sigma=blur_sigma, mode='reflect')

print("\n[自伴性验证] 检验 <Ax, z> ≈ <x, Az> (随机向量点积)")
print("  注: 自伴算子满足 <Ax,z> = <x,Az>，用多次随机测试验证")
n_tests = 5
rel_errors = []
for i in range(n_tests):
    z_test = np.random.randn(n, n)
    Ax_z = np.sum(A(x) * z_test)
    x_Az = np.sum(x * A(z_test))
    rel_err = abs(Ax_z - x_Az) / abs(Ax_z)
    rel_errors.append(rel_err)
    print(f"  测试 {i+1}: <Ax,z>={Ax_z:.6f}, <x,Az>={x_Az:.6f}, 相对误差={rel_err:.2e}")
print(f"  最大相对误差: {max(rel_errors):.2e} (≈0 验证自伴性)")

noise_lev = 1e-1
y = A(x) + noise_lev * np.random.randn(n, n)

sigma = noise_lev
sigma_x = x_std
lambda_Tikh = sigma**2 / sigma_x**2

print(f"\n[参数设定]")
print(f"  噪声水平 σ = {sigma:.4f}")
print(f"  先验标准差 σ_x = {sigma_x:.4f} (匹配图像实际标准差)")
print(f"  正则化参数 λ = σ²/σ_x² = {lambda_Tikh:.4f}")

def estimate_operator_norm(A, AT, shape, n_iter=20):
    """使用幂迭代估计算子范数 ||A||
    
    幂迭代求 A^T A 的最大特征值:
    v_{k+1} = A^T A v_k / ||A^T A v_k||
    收敛后: ||A||² ≈ ||A^T A v|| / ||v||
    
    注: 目标函数 J(x) = 0.5||y-Ax||² + 0.5λ||x||² 的
    Lipschitz 常数为 L = ||A||² + λ
    """
    v = np.random.randn(*shape)
    v = v / np.linalg.norm(v)
    
    for k in range(n_iter):
        v_new = AT(A(v))
        v_new_norm = np.linalg.norm(v_new)
        if v_new_norm < 1e-12:
            print(f"  [警告] 幂迭代在第 {k} 步遇到零向量")
            return 0.0
        v = v_new / v_new_norm
    
    v_norm = np.linalg.norm(v)
    assert abs(v_norm - 1.0) < 1e-10, f"数值检查失败: ||v|| = {v_norm} (应为1.0)"
    
    A_norm_sq = np.linalg.norm(AT(A(v)))
    
    return np.sqrt(A_norm_sq)

A_norm_est = estimate_operator_norm(A, AT, (n, n))
L_est = A_norm_est**2 + lambda_Tikh
lr_max = 2.0 / L_est
lr = 0.5

print(f"\n[学习率理论依据]")
print(f"  估计的 ||A|| ≈ {A_norm_est:.4f}")
print(f"  Lipschitz 常数 L = ||A||² + λ ≈ {L_est:.4f}")
print(f"  理论上界 lr < 2/L ≈ {lr_max:.4f}")
print(f"  实际使用 lr = {lr} ({'安全' if lr < lr_max else '风险!'})")

def solve_tikhonov_gd(y, A, AT, lam, n_iter=500, lr=0.5, record_interval=50):
    """梯度下降最小化 J(x) = 0.5||y - A(x)||² + 0.5*λ||x||²

    梯度: ∇J(x) = -A^T(y - A(x)) + λx
    
    返回:
        x_hat: 估计结果
        obj_hist: 目标函数值历史
        iter_hist: 对应的迭代次数历史
    """
    x_hat = AT(y).copy()
    obj_hist = []
    iter_hist = []
    for i in range(n_iter):
        Ax = A(x_hat)
        if i % record_interval == 0:
            obj = 0.5 * np.sum((y - Ax)**2) + 0.5 * lam * np.sum(x_hat**2)
            obj_hist.append(obj)
            iter_hist.append(i)
        grad = -AT(y - Ax) + lam * x_hat
        x_hat = x_hat - lr * grad
    
    obj_final = 0.5 * np.sum((y - A(x_hat))**2) + 0.5 * lam * np.sum(x_hat**2)
    obj_hist.append(obj_final)
    iter_hist.append(n_iter)
    
    return x_hat, obj_hist, iter_hist

x_Tikh, obj_hist, iter_hist = solve_tikhonov_gd(y, A, AT, lambda_Tikh, n_iter=500, lr=lr)

print("\n===== 先验 = 正则化的数值验证（含模糊逆问题）=====")

data_term_at_map = 0.5 / sigma**2 * np.sum((y - A(x_Tikh))**2)
reg_term_at_map = 0.5 / sigma_x**2 * np.sum(x_Tikh**2)
posterior_energy_at_map = data_term_at_map + reg_term_at_map

data_term_at_truth = 0.5 / sigma**2 * np.sum((y - A(x))**2)
reg_term_at_truth = 0.5 / sigma_x**2 * np.sum(x**2)
posterior_energy_at_truth = data_term_at_truth + reg_term_at_truth

print(f"\n[后验能量计算]")
print(f"  注: 实际推断时只有 y，无法使用真实 x")
print(f"  此处展示两种计算方式供对比:")
print(f"\n  方式1 (MAP解处，实际可用):")
print(f"    数据项 = 1/(2σ²)||y - A(x_Tikh)||² = {data_term_at_map:.4f}")
print(f"    正则项 = 1/(2σ_x²)||x_Tikh||² = {reg_term_at_map:.4f}")
print(f"    后验能量 = {posterior_energy_at_map:.4f}")
print(f"\n  方式2 (真实解处，仅供理论对比):")
print(f"    数据项 = 1/(2σ²)||y - A(x)||² = {data_term_at_truth:.4f}")
print(f"    正则项 = 1/(2σ_x²)||x||² = {reg_term_at_truth:.4f}")
print(f"    后验能量 = {posterior_energy_at_truth:.4f}")

def compute_gradient(x_test, y, A, AT, lam):
    """计算目标函数 J(x) = 0.5||y-A(x)||² + 0.5*λ||x||² 的梯度
    
    ∇J(x) = -A^T(y - A(x)) + λx
    
    这是 ∂E/∂x 在点 x_test 处的值
    """
    return -AT(y - A(x_test)) + lam * x_test

x_random = np.random.randn(n, n) * 0.1
grad_at_random = compute_gradient(x_random, y, A, AT, lambda_Tikh)
grad_at_map = compute_gradient(x_Tikh, y, A, AT, lambda_Tikh)

print(f"\n[梯度验证（关键：梯度是对x求导）]")
print(f"  ∇J(x) = -A^T(y - A(x)) + λx")
print(f"  随机点处梯度范数: {np.linalg.norm(grad_at_random):.4f} (非零)")
print(f"  MAP点处梯度范数: {np.linalg.norm(grad_at_map):.2e} (≈0，验证MAP是驻点)")
print(f"  GD 总迭代: {iter_hist[-1]}, 末目标值: {obj_hist[-1]:.4f}")

DATA_RANGE = 1.0
Orig_MSE = mean_squared_error(x, y)
Tikh_MSE = mean_squared_error(x, x_Tikh)
Orig_PSNR = peak_signal_noise_ratio(x, y, data_range=DATA_RANGE)
Tikh_PSNR = peak_signal_noise_ratio(x, x_Tikh, data_range=DATA_RANGE)

print(f"\n[质量指标] (data_range={DATA_RANGE})")
print(f"  模糊含噪  MSE: {Orig_MSE:.4f}, PSNR: {Orig_PSNR:.2f} dB")
print(f"  Tikhonov  MSE: {Tikh_MSE:.4f}, PSNR: {Tikh_PSNR:.2f} dB")

x_blurred = A(x)

fig, axs = plt.subplots(1, 4, figsize=(16, 4))

axs[0].imshow(x, cmap='gray', vmin=0, vmax=1)
axs[0].set_title('原始图像 x')

axs[1].imshow(x_blurred, cmap='gray', vmin=0, vmax=1)
axs[1].set_title(f'纯模糊 A(x) (无噪声)\nσ_blur={blur_sigma:.1f}')

axs[2].imshow(y, cmap='gray', vmin=0, vmax=1)
axs[2].set_title(f'含噪模糊 y = A(x) + 噪声\nσ_n={noise_lev:.2f}, PSNR={Orig_PSNR:.2f}dB')

axs[3].imshow(x_Tikh, cmap='gray', vmin=0, vmax=1)
axs[3].set_title(f'Tikhonov 重建 (高斯先验)\nλ={lambda_Tikh:.4f}, PSNR={Tikh_PSNR:.2f}dB')

for ax in axs:
    ax.axis('off')

plt.suptitle('先验 = 正则化：非平凡逆问题 (A=模糊) 下的Tikhonov正则化', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_Tikhonov模糊逆问题验证.png'), dpi=150, bbox_inches='tight')
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

lambdas = [0.001, 0.01, 0.1]
n_iters = [1000, 500, 300]
center_row = n // 2
for lam_val, n_it in zip(lambdas, n_iters):
    x_lam, _, _ = solve_tikhonov_gd(y, A, AT, lam_val, n_iter=n_it, lr=lr)
    axes[1].plot(x_lam[center_row, :], linewidth=1.5, label=f'λ={lam_val} ({n_it}步)')
axes[1].plot(x[center_row, :], 'k--', linewidth=1, label='真实')
axes[1].set_title('不同λ下重建中心行剖面\n(小λ条件数大，GD收敛慢，此处为近似结果)')
axes[1].legend()
axes[1].set_xlabel('像素索引')
axes[1].set_ylabel('像素强度')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_正则项形态与λ影响_模糊逆问题.png'), dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(iter_hist, obj_hist, 'o-', markersize=4)
plt.xlabel('迭代次数')
plt.ylabel('Tikhonov 目标函数值')
plt.title(f'梯度下降收敛曲线 (λ={lambda_Tikh:.4f}, lr={lr})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_GD收敛曲线.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n===== 结论 =====")
print("1. -ln p(x|y) = 数据项 + 正则项，验证了先验与正则项的对应关系")
print("2. λ = σ^2/σ_x^2 给出了正则化参数的概率诠释")
print("3. 非平凡前向算子 A ≠ I 下，Tikhonov 进行的是真正的逆问题求解")
print("   (频域恢复 + 噪声抑制)，而非简单的全局像素缩放")
print("4. 不同λ影响重建: λ小→模糊残留(欠正则化)，λ大→过度平滑(过正则化)")
print("5. GD 收敛曲线验证了优化过程的稳定性")
print("6. 自伴性验证: <Ax,z> ≈ <x,Az>，数值确认 A^T = A")
print("7. 学习率选择: lr < 2/L (Lipschitz常数) 保证收敛")
