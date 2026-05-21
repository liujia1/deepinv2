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

if SILENT_MODE:
    import matplotlib
    matplotlib.use('Agg')
    warnings.filterwarnings('ignore')
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
else:
    import matplotlib

import matplotlib.pyplot as plt

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.a-1', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.a-1')
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

n = 100
L = 0.1
x_grid = np.linspace(0, 1, n)
x1, x2 = np.meshgrid(x_grid, x_grid)

Sigma = np.exp(-np.abs(x1 - x2)**2 / (2 * L**2))

sigma = 1.0
u_true = np.random.multivariate_normal(np.zeros(n), Sigma)
y = u_true + sigma * np.random.randn(n)

alpha_values = [0.1, 1.0, 10.0]

print("===== 共轭先验与闭式后验推导 =====")
print(f"\n问题设定:")
print(f"  先验: u ~ N(0, Σ), Σ由相关长度L={L}构造")
print(f"  似然: y|u ~ N(u, σ²I), σ={sigma}")
print(f"  共轭性: 高斯先验 + 高斯似然 → 高斯后验")

u_map_list = []
Sigma_post_list = []

for alpha in alpha_values:
    Sigma_post = np.linalg.inv(Sigma + alpha * np.eye(n)) @ Sigma
    Sigma_post = alpha * Sigma_post
    
    u_map = np.linalg.solve(alpha * np.eye(n) + Sigma, Sigma @ y)
    
    u_map_list.append(u_map)
    Sigma_post_list.append(Sigma_post)
    
    print(f"\nα = {alpha}:")
    print(f"  后验均值 μ_post = (αI + Σ)⁻¹ Σ y")
    print(f"  后验协方差 Σ_post = α(Σ + αI)⁻¹ Σ")
    print(f"  MAP估计 = 后验均值 (高斯后验对称性)")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for k, alpha in enumerate(alpha_values):
    axes[0, k].plot(x_grid, u_true, 'k--', linewidth=1.5, label='真实信号')
    axes[0, k].plot(x_grid, y, 'r.', markersize=2, alpha=0.5, label='含噪观测')
    axes[0, k].errorbar(x_grid, u_map_list[k], 
                        yerr=np.sqrt(np.diag(Sigma_post_list[k])),
                        ecolor='orange', alpha=0.4, linewidth=0.5, label='后验不确定性')
    axes[0, k].plot(x_grid, u_map_list[k], 'b-', linewidth=1, label='MAP估计')
    axes[0, k].set_xlabel('x')
    axes[0, k].set_title(f'α = {alpha}\nMAP = 后验均值')
    if k == 0:
        axes[0, k].set_ylabel('u(x)')
    axes[0, k].legend(fontsize=8)

r = np.linspace(-5, 5, 50)
for k, alpha in enumerate(alpha_values):
    residual = u_map_list[k] - y
    axes[1, k].hist(residual, bins=20, density=True, alpha=0.7, label='残差直方图')
    
    x_pdf = np.linspace(-4, 4, 100)
    pdf = (1 / np.sqrt(2 * np.pi * alpha)) * np.exp(-0.5 * x_pdf**2 / alpha)
    axes[1, k].plot(x_pdf, pdf, 'k--', linewidth=2, label=f'N(0, α={alpha})')
    
    axes[1, k].set_xlabel('残差')
    axes[1, k].set_title(f'残差分布 vs 理论预测')
    if k == 0:
        axes[1, k].set_ylabel('概率密度')
    axes[1, k].legend(fontsize=8)

plt.suptitle('共轭先验：高斯先验 + 高斯似然 → 高斯后验\nMAP估计 = 后验均值，后验协方差有闭式解', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_共轭先验与闭式后验.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
    plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for k, alpha in enumerate(alpha_values):
    diag_var = np.diag(Sigma_post_list[k])
    axes[0].semilogy(x_grid, diag_var, linewidth=1.5, label=f'α={alpha}')
axes[0].set_xlabel('x')
axes[0].set_ylabel('后验方差 (对数尺度)')
axes[0].set_title('后验不确定性随位置变化')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

alpha_range = np.logspace(-2, 2, 50)
avg_var = []
for a in alpha_range:
    Sigma_temp = np.linalg.inv(Sigma + a * np.eye(n)) @ Sigma
    Sigma_temp = a * Sigma_temp
    avg_var.append(np.mean(np.diag(Sigma_temp)))

axes[1].loglog(alpha_range, avg_var, 'b-', linewidth=2)
axes[1].set_xlabel('α (正则化参数)')
axes[1].set_ylabel('平均后验方差')
axes[1].set_title('正则化参数α与后验不确定性\nα大→先验强→不确定性小')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_后验不确定性与正则化参数.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
    plt.show()

print("\n===== 附录2A 核心结论 =====")
print("\n1. 共轭性:")
print("   高斯先验 + 高斯似然 → 高斯后验")
print("   原因: 两个二次函数的和仍是二次函数")
print("\n2. MAP = 后验均值:")
print("   高斯分布对称，众数 = 均值")
print("   因此 MAP估计 = MMSE估计")
print("\n3. 后验协方差闭式解:")
print("   Σ_post = (A^T A/σ² + Σ_x⁻¹)⁻¹")
print("   可量化重建的不确定性")
print("\n4. 正则化参数的概率解释:")
print("   α = σ²/σ_x² = 噪声方差/先验方差")
print("   α小: 噪声低/先验弱 → 依赖数据")
print("   α大: 噪声高/先验强 → 依赖先验")
print("\n5. 广义Tikhonov:")
print("   后验均值 = (A^T A + λI)⁻¹ A^T y")
print("   恰好是Tikhonov正则化的解")
