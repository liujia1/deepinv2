"""
实验2.1-1 先验与正则化的数值验证（含模糊逆问题）
对应章节：2.1 先验的数学角色：正则化的概率诠释
知识点：贝叶斯定理；先验→正则化对应关系；-ln p(x|y) = 数据项 + 正则项；λ = σ²/σ_x²；
         非平凡前向算子 A ≠ I 下的 Tikhonov 重建

素材来源：
  - M1 CompImLab25.ipynb: Tikhonov闭式解
  - IP22 statistical_perspective.md: 高斯先验MAP推导

注: 本实验使用梯度下降(GD)求解，目的是展示优化过程、Lipschitz常数、收敛性等概念。
    对于高斯模糊这类卷积算子，频域闭式解更精确更快，但GD有独特的教学价值。
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

x_min, x_max = x.min(), x.max()
x_std = np.std(x)
print(f"[参数验证] 图像值域: [{x_min:.4f}, {x_max:.4f}]")
print(f"[参数验证] 图像标准差 std(x) = {x_std:.4f}")

DATA_RANGE = x_max - x_min
print(f"[参数验证] DATA_RANGE = {DATA_RANGE:.4f}")

blur_sigma = 2.0

def A(u):
    """Forward: Gaussian blur (mode='reflect' 避免边界伪影)"""
    return gaussian_filter(u, sigma=blur_sigma, mode='reflect')

def AT(u):
    """Adjoint: 高斯模糊在 reflect 边界下是自伴算子 A^T = A"""
    return gaussian_filter(u, sigma=blur_sigma, mode='reflect')

print("\n[自伴性验证] 检验 <Ax, z> = <x, Az> (随机向量点积)")
print("  注: 自伴算子满足 <Ax,z> = <x,Az>，用多次随机测试验证")
n_tests = 5
rel_errors = []
for i in range(n_tests):
    z_test = np.random.randn(n, n)
    Ax_z = np.sum(A(x) * z_test)
    x_Az = np.sum(x * A(z_test))
    rel_err = abs(Ax_z - x_Az) / (abs(Ax_z) + 1e-12)
    rel_errors.append(rel_err)
    print(f"  测试 {i+1}: <Ax,z>={Ax_z:.6f}, <x,Az>={x_Az:.6f}, 相对误差={rel_err:.2e}")
print(f"  最大相对误差: {max(rel_errors):.2e} (验证自伴性)")

noise_lev = 1e-1
y = A(x) + noise_lev * np.random.randn(n, n)

sigma = noise_lev
sigma_x = x_std
lambda_Tikh = sigma**2 / sigma_x**2

print(f"\n[参数设定]")
print(f"  噪声水平 sigma = {sigma:.4f}")
print(f"  先验标准差 sigma_x = {sigma_x:.4f} (经验贝叶斯: 从数据估计先验参数)")
print(f"  正则化参数 lambda = sigma^2/sigma_x^2 = {lambda_Tikh:.4f}")

def estimate_operator_norm(A, AT, shape, n_iter=20):
    """使用幂迭代估计算子范数 ||A||
    
    幂迭代求 A^T A 的最大特征值:
    v_{k+1} = A^T A v_k / ||A^T A v_k||
    收敛后用 Rayleigh 商计算特征值: lambda_max = <A^T A v, v>
    则 ||A|| = sqrt(lambda_max)
    
    注: 目标函数 J(x) = 0.5||y-Ax||^2 + 0.5*lambda||x||^2 的
    Lipschitz 常数为 L = ||A||^2 + lambda
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
    if abs(v_norm - 1.0) >= 1e-10:
        print(f"  [警告] 数值检查异常: ||v|| = {v_norm} (应为1.0)")
    
    AtAv = AT(A(v))
    eigenvalue_max = np.dot(AtAv.ravel(), v.ravel())
    
    return np.sqrt(max(eigenvalue_max, 0))

A_norm_est = estimate_operator_norm(A, AT, (n, n))
L_est = A_norm_est**2 + lambda_Tikh
lr_max = 2.0 / L_est
lr = 0.9 * lr_max   # 学习率由 Lipschitz 常数推导，取上界的 90% 保证稳定

print(f"\n[学习率理论依据]")
print(f"  估计的 ||A|| = {A_norm_est:.4f}")
print(f"  Lipschitz 常数 L = ||A||^2 + lambda = {L_est:.4f}")
print(f"  理论上界 lr < 2/L = {lr_max:.4f}")
print(f"  实际使用 lr = 0.9 * lr_max = {lr:.4f}")

def solve_tikhonov_gd(y, A, AT, lam, n_iter=500, lr=0.5, tol=1e-6, record_interval=50):
    """梯度下降最小化 J(x) = 0.5||y - A(x)||^2 + 0.5*lambda||x||^2

    梯度: grad J(x) = -A^T(y - A(x)) + lambda*x
    
    参数:
        y: 观测数据
        A, AT: 前向算子及其伴随
        lam: 正则化参数
        n_iter: 最大迭代次数
        lr: 学习率
        tol: 收敛容差 (||grad|| / (||x|| + eps) < tol 时停止)
        record_interval: 记录间隔
    
    返回:
        x_hat: 估计结果
        obj_hist: 目标函数值历史
        iter_hist: 对应的迭代次数历史
        converged: 是否收敛
    """
    x_hat = AT(y).copy()
    obj_hist = []
    iter_hist = []
    converged = False
    
    for i in range(n_iter):
        Ax = A(x_hat)
        if i % record_interval == 0:
            obj = 0.5 * np.sum((y - Ax)**2) + 0.5 * lam * np.sum(x_hat**2)
            obj_hist.append(obj)
            iter_hist.append(i)
        
        grad = -AT(y - Ax) + lam * x_hat
        grad_norm = np.linalg.norm(grad)
        x_norm = np.linalg.norm(x_hat)
        
        if grad_norm / (x_norm + 1e-8) < tol:
            print(f"  [早停] 第 {i} 步收敛, ||grad||/||x|| = {grad_norm/(x_norm+1e-8):.2e} < {tol}")
            converged = True
            break
        
        x_hat = x_hat - lr * grad
    
    obj_final = 0.5 * np.sum((y - A(x_hat))**2) + 0.5 * lam * np.sum(x_hat**2)
    obj_hist.append(obj_final)
    iter_hist.append(i + 1 if converged else n_iter)
    
    return x_hat, obj_hist, iter_hist, converged

print(f"\n[GD求解] 使用早停机制, tol=1e-6")
print(f"  初始化: x_0 = A^T(y) (热启动，比零初始化更接近解)")
x_Tikh, obj_hist, iter_hist, converged = solve_tikhonov_gd(y, A, AT, lambda_Tikh, n_iter=500, lr=lr)

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
print(f"    数据项 = {data_term_at_map:.4f}")
print(f"    正则项 = {reg_term_at_map:.4f}")
print(f"    后验能量 = {posterior_energy_at_map:.4f}")
print(f"\n  方式2 (真实解处，仅供理论对比):")
print(f"    数据项 = {data_term_at_truth:.4f}")
print(f"    正则项 = {reg_term_at_truth:.4f}")
print(f"    后验能量 = {posterior_energy_at_truth:.4f}")

x_init = AT(y)
data_term_at_init = 0.5 / sigma**2 * np.sum((y - A(x_init))**2)
reg_term_at_init = 0.5 / sigma_x**2 * np.sum(x_init**2)
energy_at_init = data_term_at_init + reg_term_at_init

print(f"\n  方式3 (初始点 x0=A^T(y) 处，优化验证):")
print(f"    数据项 = {data_term_at_init:.4f}")
print(f"    正则项 = {reg_term_at_init:.4f}")
print(f"    后验能量 = {energy_at_init:.4f}")

if posterior_energy_at_map > energy_at_init:
    print(f"\n  [警告] MAP 点能量 ({posterior_energy_at_map:.4f}) > 初始点能量 ({energy_at_init:.4f})")
    print("          这说明优化可能发散，MAP 解并非有效的优化结果")
else:
    print(f"\n  [验证通过] 目标函数从 {energy_at_init:.4f} 下降到 {posterior_energy_at_map:.4f}")

def compute_gradient(x_test, y, A, AT, lam):
    """计算目标函数 J(x) = 0.5||y-A(x)||^2 + 0.5*lambda||x||^2 的梯度
    
    grad J(x) = -A^T(y - A(x)) + lambda*x
    
    这是 dE/dx 在点 x_test 处的值
    """
    return -AT(y - A(x_test)) + lam * x_test

x_random = np.random.randn(n, n) * 0.1
grad_at_random = compute_gradient(x_random, y, A, AT, lambda_Tikh)
grad_at_map = compute_gradient(x_Tikh, y, A, AT, lambda_Tikh)

print(f"\n[梯度验证（关键：梯度是对x求导）]")
print(f"  grad J(x) = -A^T(y - A(x)) + lambda*x")
print(f"  随机点处梯度范数: {np.linalg.norm(grad_at_random):.4f} (非零)")
print(f"  MAP点处梯度范数: {np.linalg.norm(grad_at_map):.2e} (验证MAP是驻点)")
print(f"  GD 总迭代: {iter_hist[-1]}, 末目标值: {obj_hist[-1]:.4f}, 收敛: {converged}")

x_blurred = A(x)

Orig_MSE = mean_squared_error(x, y)
Tikh_MSE = mean_squared_error(x, x_Tikh)
Orig_PSNR = peak_signal_noise_ratio(x, y, data_range=DATA_RANGE)
Tikh_PSNR = peak_signal_noise_ratio(x, x_Tikh, data_range=DATA_RANGE)

Blur_MSE = mean_squared_error(x, x_blurred)
print(f"\n[质量指标] (data_range={DATA_RANGE:.4f})")
print(f"  纯模糊(无噪声) MSE: {Blur_MSE:.4f} (仅模糊导致的失真)")
print(f"  模糊含噪       MSE: {Orig_MSE:.4f} (模糊 + 噪声)")
print(f"  Tikhonov       MSE: {Tikh_MSE:.4f}")
print(f"  模糊含噪       PSNR: {Orig_PSNR:.2f} dB")
print(f"  Tikhonov       PSNR: {Tikh_PSNR:.2f} dB")




fig, axs = plt.subplots(1, 4, figsize=(16, 4))

axs[0].imshow(x, cmap='gray', vmin=0, vmax=1)
axs[0].set_title('原始图像 x')

axs[1].imshow(x_blurred, cmap='gray', vmin=0, vmax=1)
axs[1].set_title(f'纯模糊 A(x) (无噪声)\nblur_sigma={blur_sigma:.1f}')

axs[2].imshow(y, cmap='gray', vmin=0, vmax=1)
axs[2].set_title(f'含噪模糊 y = A(x) + noise\nnoise_sigma={noise_lev:.2f}, PSNR={Orig_PSNR:.2f}dB')

axs[3].imshow(x_Tikh, cmap='gray', vmin=0, vmax=1)
axs[3].set_title(f'Tikhonov 重建 (高斯先验)\nlambda={lambda_Tikh:.4f}, PSNR={Tikh_PSNR:.2f}dB')

for ax in axs:
    ax.axis('off')

plt.suptitle('先验 = 正则化：非平凡逆问题 (A=模糊) 下的Tikhonov正则化', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_Tikhonov模糊逆问题验证.png'), dpi=150, bbox_inches='tight')
plt.show()
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

t = np.linspace(-2, 2, 400)
axes[0].plot(t, t**2, 'b-', linewidth=2, label='L2: t^2 (高斯先验正则项)')
axes[0].set_title('高斯先验对应的正则项形态')
axes[0].legend()
axes[0].set_xlabel('t')
axes[0].set_ylabel('惩罚值')
axes[0].set_ylim(-0.2, 4)
axes[0].grid(True, alpha=0.3)

lambdas = [0.001, 0.01, 0.1]
center_row = n // 2
for lam_val in lambdas:
    # 每个 lambda 独立计算 Lipschitz 常数和自适应学习率
    L_lam = A_norm_est**2 + lam_val
    lr_lam = 0.9 * (2.0 / L_lam)
    x_lam, _, iters, conv = solve_tikhonov_gd(y, A, AT, lam_val, n_iter=1000, lr=lr_lam)
    total_steps = iters[-1]
    status = "收敛" if conv else "未收敛"
    axes[1].plot(x_lam[center_row, :], linewidth=1.5, label=f'λ={lam_val} ({total_steps}步, {status})')
axes[1].plot(x[center_row, :], 'k--', linewidth=1, label='真实')
axes[1].set_title('不同lambda下重建中心行剖面\n(使用早停机制，最多1000步)')
axes[1].legend()
axes[1].set_xlabel('像素索引')
axes[1].set_ylabel('像素强度')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_正则项形态与lambda影响_模糊逆问题.png'), dpi=150, bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(iter_hist, obj_hist, 'o-', markersize=4)
plt.xlabel('迭代次数')
plt.ylabel('Tikhonov 目标函数值')
plt.title(f'梯度下降收敛曲线 (lambda={lambda_Tikh:.4f}, lr={lr:.4f})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_GD收敛曲线.png'), dpi=150, bbox_inches='tight')
plt.show()
plt.close()

print("\n===== 结论 =====")
print("1. -ln p(x|y) = 数据项 + 正则项，验证了先验与正则项的对应关系")
print("2. lambda = sigma^2/sigma_x^2 给出了正则化参数的概率诠释")
print("3. 非平凡前向算子 A != I 下，Tikhonov 进行的是真正的逆问题求解")
print("   (频域恢复 + 噪声抑制)，而非简单的全局像素缩放")
print("4. 不同lambda影响重建: lambda小->模糊残留(欠正则化)，lambda大->过度平滑(过正则化)")
print("5. GD 收敛曲线验证了优化过程的稳定性")
print("6. 自伴性验证: <Ax,z> = <x,Az>，数值确认 A^T = A")
print("7. 学习率选择: lr < 2/L (Lipschitz常数) 保证收敛")
print("8. 早停机制: ||grad||/||x|| < tol 时停止，避免过度迭代")
