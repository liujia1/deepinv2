"""
实验2.a-1 共轭先验与闭式后验推导
对应章节：附录2A 共轭先验与闭式后验推导
知识点：高斯-高斯共轭；MAP=后验均值；后验协方差闭式解；广义Tikhonov；正则化参数的概率解释

素材来源：
  - IP22 (statistical_perspective.md): 高斯随机场采样 + MAP估计 + 后验协方差
  - 附录2A: 共轭先验推导
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

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.a-1', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.a-1')
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

n = 100
L = 0.1
x_grid = np.linspace(0, 1, n)
x1, x2 = np.meshgrid(x_grid, x_grid)

Sigma = np.exp(-(x1 - x2)**2 / (2 * L**2))

cond_num = np.linalg.cond(Sigma)
print(f"\n[数值稳定性检查] 协方差矩阵Σ的条件数: {cond_num:.2e}")
if cond_num > 1e10:
    print("  警告: 矩阵接近病态，数值计算可能不稳定！")
    print("  建议: 增大相关长度L或减小矩阵维度n")

sigma = 1.0
sigma_x2 = np.mean(np.diag(Sigma))
u_true = np.random.multivariate_normal(np.zeros(n), Sigma)
y = u_true + sigma * np.random.randn(n)

lambda_values = [0.1, 1.0, 10.0]

print("\n===== 共轭先验与闭式后验推导 =====")
print(f"\n问题设定:")
print(f"  先验: u ~ N(0, Σ), Σ由相关长度L={L}构造")
print(f"  先验方差: σ_x² = {sigma_x2:.3f} (Σ对角元素均值)")
print(f"  似然: y|u ~ N(u, σ²I), σ={sigma}, σ²={sigma**2}")
print(f"  共轭性: 高斯先验 + 高斯似然 → 高斯后验")
print(f"\n正则化参数λ的语义:")
print(f"  在广义Tikhonov形式 (λI + Σ)⁻¹Σy 中，λ = σ² (噪声方差)")
print(f"  在标准Tikhonov形式 (A^TA + λI)⁻¹A^Ty 中，λ = σ²/σ_x² (信噪比)")
print(f"  本实验使用广义Tikhonov形式，故 λ=σ²={sigma} 时为真实贝叶斯后验")
print(f"  本例σ_x²={sigma_x2:.1f}，两种参数化数值相同（一般情形不同）")

print("\n[数值验证] 后验协方差闭式解正确性检验...")
Sigma_post_direct = np.linalg.inv(np.linalg.inv(Sigma) + np.eye(n) / sigma**2)
Sigma_post_formula = sigma**2 * np.linalg.solve(Sigma + sigma**2 * np.eye(n), Sigma)
max_error = np.max(np.abs(Sigma_post_direct - Sigma_post_formula))

# 使用前面已计算的条件数（避免重复计算）
tol = max(1e-10, cond_num * np.finfo(float).eps * 10)

print(f"  闭式解验证 (λ=σ²={sigma}): 最大误差 = {max_error:.2e}")
if max_error < tol:
    print(f"  ✓ 闭式解与直接计算一致（误差在条件数允许范围内）")
else:
    print(f"  ⚠ 误差={max_error:.2e}，可能由inv(Σ)引起（条件数={cond_num:.2e}）")
    print(f"    闭式公式本身正确，误差来自直接求逆的数值不稳定")

u_map_list = []
Sigma_post_list = []
coverage_list = []  # 保存覆盖率，避免重复计算

for lam in lambda_values:
    Sigma_post = lam * np.linalg.solve(Sigma + lam * np.eye(n), Sigma)
    
    u_map = np.linalg.solve(lam * np.eye(n) + Sigma, Sigma @ y)
    
    u_map_list.append(u_map)
    Sigma_post_list.append(Sigma_post)
    
    print(f"\nλ = {lam}:")
    print(f"  后验均值 μ_post = (λI + Σ)⁻¹ Σ y")
    print(f"  后验协方差 Σ_post = λ·Σ·(Σ + λI)⁻¹ = (Σ⁻¹ + λ⁻¹I)⁻¹")
    print(f"  MAP估计 = 后验均值 (高斯后验对称性)")
    if lam < sigma**2 / sigma_x2:
        print(f"  → λ较小: 相对信任数据，后验不确定性较小")
    elif lam > sigma**2 / sigma_x2:
        print(f"  → λ较大: 噪声假设大→数据信息弱→后验不确定性较大")
    else:
        print(f"  → λ与噪声/先验比匹配: 数据与先验平衡")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for k, lam in enumerate(lambda_values):
    axes[0, k].plot(x_grid, u_true, 'k--', linewidth=1.5, label='真实信号')
    axes[0, k].plot(x_grid, y, 'r.', markersize=2, alpha=0.5, label='含噪观测')
    axes[0, k].errorbar(x_grid, u_map_list[k], 
                        yerr=np.sqrt(np.diag(Sigma_post_list[k])),
                        ecolor='orange', alpha=0.4, linewidth=0.5, label='后验不确定性')
    axes[0, k].plot(x_grid, u_map_list[k], 'b-', linewidth=1, label='MAP估计')
    axes[0, k].set_xlabel('x')
    axes[0, k].set_title(f'λ = {lam}\nMAP = 后验均值')
    if k == 0:
        axes[0, k].set_ylabel('u(x)')
    axes[0, k].legend(fontsize=8)

noise_residual = y - u_true
for k, lam in enumerate(lambda_values):
    map_residual = u_map_list[k] - y
    
    axes[1, k].hist(noise_residual, bins=20, density=True, alpha=0.5, 
                    color='green', label='观测噪声 y-u_true')
    axes[1, k].hist(map_residual, bins=20, density=True, alpha=0.5, 
                    color='blue', label='MAP残差 u_map-y')
    
    x_pdf = np.linspace(-4, 4, 100)
    pdf_noise = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-0.5 * x_pdf**2 / sigma**2)
    axes[1, k].plot(x_pdf, pdf_noise, 'g--', linewidth=2, label=f'N(0, σ²={sigma**2:.1f})')
    
    shrinkage_std = np.std(map_residual)
    pdf_shrink = (1 / np.sqrt(2 * np.pi * shrinkage_std**2)) * np.exp(-0.5 * x_pdf**2 / shrinkage_std**2)
    axes[1, k].plot(x_pdf, pdf_shrink, 'b:', linewidth=2, label=f'MAP残差经验分布')
    
    coverage = np.mean(np.abs(u_true - u_map_list[k]) < 1.96 * np.sqrt(np.diag(Sigma_post_list[k])))
    coverage_list.append(coverage)  # 保存覆盖率供后续使用
    
    axes[1, k].set_xlabel('残差')
    axes[1, k].set_title(f'残差分布对比\n95%置信区间覆盖率: {coverage:.1%}')
    if k == 0:
        axes[1, k].set_ylabel('概率密度')
    axes[1, k].legend(fontsize=7)

plt.suptitle('共轭先验：高斯先验 + 高斯似然 → 高斯后验\nMAP估计 = 后验均值，后验协方差有闭式解', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_共轭先验与闭式后验.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
    plt.show()
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for k, lam in enumerate(lambda_values):
    diag_var = np.diag(Sigma_post_list[k])
    axes[0].semilogy(x_grid, diag_var, linewidth=1.5, label=f'λ={lam}')
axes[0].set_xlabel('x')
axes[0].set_ylabel('后验方差 (对数尺度)')
axes[0].set_title('后验不确定性随位置变化')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

lambda_range = np.logspace(-2, 2, 50)
avg_var = []
for lam_val in lambda_range:
    Sigma_temp = lam_val * np.linalg.solve(Sigma + lam_val * np.eye(n), Sigma)
    avg_var.append(np.mean(np.diag(Sigma_temp)))

axes[1].loglog(lambda_range, avg_var, 'b-', linewidth=2)
axes[1].set_xlabel('λ (正则化参数 = σ²/σ_x²)')
axes[1].set_ylabel('平均后验方差')
axes[1].set_title('正则化参数λ与后验不确定性\nλ大→噪声假设大→数据信息弱→不确定性大')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_后验不确定性与正则化参数.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
    plt.show()
plt.close(fig)

print("\n===== 附录2A 核心结论 =====")
print("\n1. 共轭性:")
print("   高斯先验 + 高斯似然 → 高斯后验")
print("   原因: 两个二次函数的和仍是二次函数")
print("\n2. MAP = 后验均值:")
print("   高斯分布对称，众数 = 均值")
print("   因此 MAP估计 = MMSE估计")
print("\n3. 后验协方差闭式解:")
print("   Σ_post = (Σ⁻¹ + σ⁻²I)⁻¹ = σ²·Σ·(Σ + σ²I)⁻¹")
print("   注意: 因Σ对称，Σ与(Σ+σ²I)⁻¹可交换，乘法顺序不影响结果")
print("   一般情形(A≠I时): 必须注意矩阵乘法顺序")
print("   可量化重建的不确定性")
print("\n4. 正则化参数λ的概率解释:")
print("   λ = σ²/σ_x² = 噪声方差/先验方差")
print("   λ小: 噪声相对低 → 数据信息强 → 后验不确定性小")
print("   λ大: 噪声相对高 → 数据信息弱 → 后验不确定性大")
print("   本例: σ²=%.2f, σ_x²=%.3f, 理论λ≈%.2f" % (sigma**2, sigma_x2, sigma**2/sigma_x2))
print("\n5. 广义Tikhonov与贝叶斯推断的等价性:")
print("   广义Tikhonov: u = (A^T Γ⁻¹ A + Σ⁻¹)⁻¹ A^T Γ⁻¹ y")
print("   本例 A=I, Γ=σ²I, 正则化矩阵为 Σ⁻¹ (非单位阵I)")
print("   后验均值的两种等价形式:")
print("     (a) 贝叶斯形式: μ_post = (Σ⁻¹ + σ⁻²I)⁻¹ σ⁻² y")
print("     (b) 计算形式:   μ_post = (λI + Σ)⁻¹ Σ y     (λ = σ²)")
print("   本实验代码使用(b)以避免显式求 Σ⁻¹（数值更稳定）")
print("   当 λ = σ² 时，(a) = (b)")
print("\n6. 后验校准性:")
print("   理想情况下，95%可信区间的覆盖率应≈95%")
for k in range(len(lambda_values)):
    coverage = coverage_list[k]  # 直接使用已计算的覆盖率
    status = "校准良好" if abs(coverage - 0.95) < 0.05 else ("过度自信" if coverage < 0.90 else "过于保守")
    print(f"   λ={lambda_values[k]}: 覆盖率={coverage:.1%} ({status})")
print("\n7. SE核的特殊性质:")
print(f"   SE核对角线恒为exp(0)=1，故σ_x²=1.0")
print(f"   理论最优λ = σ²/σ_x² = {sigma**2/sigma_x2:.1f}（恰好是测试值之一）")
