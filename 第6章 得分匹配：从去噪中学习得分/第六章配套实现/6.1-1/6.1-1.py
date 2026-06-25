# -*- coding: utf-8 -*-
"""
实验6.1-1 归一化常数困境与得分函数优势
对应章节：6.1 为什么需要学习得分？
知识点：
  - 归一化常数Z在高维不可算
  - 得分函数"无需Z"的关键优势
  - 得分函数是Langevin动力学的驱动力
  - 先验得分是逆问题中的瓶颈

素材来源：
  - 原参考实验6.1.py 拆分
  - 6.1节"归一化常数Z的不可解性"和"得分函数的关键优势"

运行前提：纯NumPy/Matplotlib，无需GPU
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys
import io

# 设置控制台输出为UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ====== 中文字体配置（兼容本地和Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验6.1-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(_chinese_path, exist_ok=True)

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

# ============================================================
# 引言：从Langevin动力学到得分函数
# ============================================================
# 回顾第4-5章：Langevin动力学 X_{m+1} = X_m + δ∇log p(X_m|y) + √(2δ)Z_{m+1}
# 的驱动力是后验得分 ∇log p(x|y) = ∇log p(y|x) + ∇log p(x)。
# 似然得分可算，但先验得分 ∇log p(x) 是逆问题中的瓶颈。
# 本实验演示：为什么先验得分难算？得分函数如何绕过这一困难？
print("=" * 60)
print("引言：从Langevin动力学到得分函数")
print("=" * 60)
print("\nLangevin动力学递推式：")
print(r"    $X_{m+1} = X_m + \delta \nabla\log p(X_m|y) + \sqrt{2\delta} Z_{m+1}$")
print("\n后验得分分解：$\\nabla\\log p(x|y) = \\nabla\\log p(y|x) + \\nabla\\log p(x)$")
print("  似然得分：通常可算（如高斯似然 $-\\frac{1}{\\sigma^2}A^T(Ax-y)$）")
print("  先验得分：$\\nabla\\log p(x)$ 是瓶颈 ← 本实验的核心问题")


# ============================================================
# 步骤1：归一化常数Z的困境
# ============================================================
print("\n" + "=" * 60)
print("步骤1：归一化常数Z的困境")
print("=" * 60)

from scipy import integrate

# --- 1a. 低维可算：2D高斯混合的数值积分验证 ---
# 定义"未归一化"密度：只保留exp部分，去掉归一化系数
# p̃(x) = 0.5 * exp(-0.5 * (x-μ1)^T Σ1^{-1} (x-μ1))
#       + 0.5 * exp(-0.5 * (x-μ2)^T Σ2^{-1} (x-μ2))
# 注意：这里故意不除 (2π√|Σ|)，使 p̃ 是真正的"未归一化"形式
def gaussian_mixture_2d_unnorm(x, y):
    """未归一化的2D高斯混合密度（故意省略归一化系数）"""
    mu1, mu2 = np.array([-2, -2]), np.array([2, 2])
    Sigma1_inv = np.linalg.inv(np.array([[1, 0.3], [0.3, 1]]))
    Sigma2_inv = np.linalg.inv(np.array([[1, -0.3], [-0.3, 1]]))

    d1 = np.array([x - mu1[0], y - mu1[1]])
    d2 = np.array([x - mu2[0], y - mu2[1]])
    p1 = np.exp(-0.5 * d1 @ Sigma1_inv @ d1)
    p2 = np.exp(-0.5 * d2 @ Sigma2_inv @ d2)

    return 0.5 * p1 + 0.5 * p2

# 数值积分求Z
Z_2d, err_2d = integrate.dblquad(gaussian_mixture_2d_unnorm, -10, 10, -10, 10)
# 理论值：每个高斯分量的积分 = 2π√|Σ| = 2π√0.91
det1 = 1 - 0.09
Z_theory_2d = 2 * np.pi * np.sqrt(det1)  # 两个分量对称，Z = 0.5*c + 0.5*c = c
print(f"2D高斯混合（未归一化形式）:")
print(f"  数值积分: Z = {Z_2d:.6f}, 误差 = {err_2d:.2e}")
print(f"  理论值:   Z = 2π√|Σ| = {Z_theory_2d:.6f}")
print(f"  → 低维时Z可通过数值积分精确计算")

# --- 1b. 高维困境：区分两种情况 ---
print("\n高维Z困境——两种不同层次的困难:")
print()
print("  情况A：有参数化表达式但维数诅咒")
print("    即使每个分量可写，高维数值积分需要 O(2^d) 网格点，不可行")
print("    例：d维标准高斯的 Z = (2π)^{d/2}，虽有闭合形式，")
print("    但若分布稍有扰动（如混合100个分量），闭合形式消失。")
print()
for d in [2, 5, 10, 20, 50]:
    log_Z = d / 2 * np.log(2 * np.pi)
    print(f"    d={d:2d}: log Z = {log_Z:.2f}  (Z = e^{log_Z:.1f})")

print()
print("  情况B：隐式先验——根本没有参数化表达式")
print("    真实图像分布 p(x) 无法写成任何有限参数公式")
print("    Z = ∫p̃(x)dx 甚至无法定义为有限维积分对象")
print("    我们只有来自 p(x) 的样本 {x_i}，无法直接计算Z")
print()
print("  → 无论哪种情况，直接计算Z都不可行")


# ============================================================
# 步骤2：得分函数"无需Z"的优势演示
# ============================================================
print("\n" + "=" * 60)
print("步骤2：得分函数'无需Z'的优势")
print("=" * 60)

# 对比：p(x) = (1/Z)p̃(x) vs p̃(x)（未归一化）
# 使用高斯混合分布：p(x) = 0.5*N(-2,1) + 0.5*N(2,1)
# 未归一化形式：p̃(x) = 0.5*exp(-0.5*(x+2)^2) + 0.5*exp(-0.5*(x-2)^2)
# 未归一化形式的积分 Z = ∫p̃(x)dx = √(2π) ≈ 2.507（每个高斯核积分为√(2π)）

def gm_pdf_normalized(x):
    """归一化的高斯混合密度"""
    return 0.5 * np.exp(-0.5 * (x + 2)**2) / np.sqrt(2*np.pi) + \
           0.5 * np.exp(-0.5 * (x - 2)**2) / np.sqrt(2*np.pi)

def gm_pdf_unnormalized(x):
    """未归一化的高斯混合密度（省略1/√(2π)）"""
    return 0.5 * np.exp(-0.5 * (x + 2)**2) + \
           0.5 * np.exp(-0.5 * (x - 2)**2)

def gm_score(x):
    """精确得分函数 ∇log p(x)，对归一化和未归一化版本相同"""
    p1 = np.exp(-0.5 * (x + 2)**2) / np.sqrt(2*np.pi)
    p2 = np.exp(-0.5 * (x - 2)**2) / np.sqrt(2*np.pi)
    p = 0.5 * p1 + 0.5 * p2
    grad_p1 = -(x + 2) * p1
    grad_p2 = -(x - 2) * p2
    return (0.5 * grad_p1 + 0.5 * grad_p2) / p

x_test = np.array([-3, -2, -1, 0, 1, 2, 3])

# 路径1：从归一化密度 p(x) 出发，用解析公式计算 ∇log p(x)
score_normalized = gm_score(x_test)

# 路径2：从未归一化密度 p̃(x) 出发，用数值微分计算 ∇log p̃(x)
# 这是两条完全独立的计算路径，差异恒为0才有说服力
eps = 1e-5
score_unnorm_numerical = (
    np.log(gm_pdf_unnormalized(x_test + eps)) -
    np.log(gm_pdf_unnormalized(x_test - eps))
) / (2 * eps)

print("得分函数'无需Z'的数值验证（高斯混合分布）:")
print(f"  路径1: 从归一化密度 p(x) 解析计算 ∇log p(x)")
print(f"  路径2: 从未归一化密度 p̃(x) 数值微分计算 ∇log p̃(x)")
print()
print(f"{'x':>5s} | {'∇log p(x) 解析':>18s} | {'∇log p̃(x) 数值':>18s} | {'差异':>12s}")
print("-" * 65)
for i in range(len(x_test)):
    diff = score_normalized[i] - score_unnorm_numerical[i]
    print(f"{x_test[i]:5.1f} | {score_normalized[i]:18.6f} | {score_unnorm_numerical[i]:18.6f} | {diff:12.2e}")

print(f"\n结论：∇log p(x) = ∇log p̃(x)，差异恒为0")
print(f"  对于高斯混合分布，p(x)的形式复杂（两个指数项之和），")
print(f"  但归一化常数Z在求导时仍然完全消去。")
print(f"  → 即使Z不可计算，得分函数仍可学习")


# ============================================================
# 步骤3：经典先验 vs 复杂先验
# ============================================================
print("\n" + "=" * 60)
print("步骤3：经典先验 vs 复杂先验")
print("=" * 60)

print("\n经典先验（第2-3章）：有显式 R(x)，梯度可算")
print()
print("  先验类型         R(x)                    grad R(x)              可计算?")
print("  " + "-" * 60)
print("  高斯(Tikhonov)   (1/2λ)||x||²            x/λ                   是")
print("  Laplace(L1)      (1/λ)||x||₁             (1/λ)sign(x)          是(近端)")
print("  TV               ||∇x||₁                 对偶方法               是")
print()
print("复杂先验（第6章及以后）：R(x) 不可写，梯度不可算")
print("  - 真实图像的统计分布远比任何参数化模型复杂")
print("  - 先验分布 p(x) 的结构完全未知")
print("  - 我们只有来自 p(x) 的数据样本 {x_i}")
print()
print("这正是第1-3章与第6章及以后的根本区别：")
print("  经典方法：有显式 R(x)，但表达能力有限")
print("  复杂先验：表达能力无限，但 R(x) 不可写")


# ============================================================
# 可视化
# ============================================================
print("\n" + "=" * 60)
print("生成可视化图表...")
print("=" * 60)

# 复用步骤2中已定义的 gm_pdf_normalized 和 gm_score
x_grid = np.linspace(-6, 6, 200)
true_score = gm_score(x_grid)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1：数值积分计算量随维度爆炸
dims = np.arange(1, 21)
n_grid = 100  # 每维100个网格点
grid_points = float(n_grid) ** dims  # 避免int64溢出
axes[0, 0].semilogy(dims, grid_points, 'bo-', lw=2, markersize=5)
axes[0, 0].set_xlabel('维度 $d$')
axes[0, 0].set_ylabel('所需网格点数 ($100^d$)')
axes[0, 0].set_title('数值积分计算量随维度爆炸')
axes[0, 0].grid(alpha=0.3, which='both')

# 子图2：密度与得分函数
axes[0, 1].plot(x_grid, gm_pdf_normalized(x_grid), 'b-', lw=2, label='$p(x)$')
ax2 = axes[0, 1].twinx()
ax2.plot(x_grid, true_score, 'r-', lw=2, label='$\\nabla\\log p(x)$')
ax2.set_ylabel('得分 $\\nabla\\log p(x)$', color='red')
ax2.tick_params(axis='y', labelcolor='red')
axes[0, 1].set_xlabel('$x$')
axes[0, 1].set_ylabel('密度 $p(x)$', color='blue')
axes[0, 1].tick_params(axis='y', labelcolor='blue')
axes[0, 1].set_title('密度与得分函数（$Z$ 在得分中消去）')
axes[0, 1].legend(loc='upper left')
ax2.legend(loc='upper right')
axes[0, 1].grid(alpha=0.3)

# 子图3：经典先验的R(x)与∇R(x)
x_prior = np.linspace(-3, 3, 100)
R_gaussian = 0.5 * x_prior**2  # λ=1
grad_R_gaussian = x_prior
R_laplace = np.abs(x_prior)  # λ=1
grad_R_laplace = np.sign(x_prior)

axes[1, 0].plot(x_prior, R_gaussian, 'b-', lw=2, label='$R(x) = \\frac{1}{2}\\|x\\|^2$ (高斯)')
axes[1, 0].plot(x_prior, R_laplace, 'g-', lw=2, label='$R(x) = \\|x\\|_1$ (Laplace)')
axes[1, 0].plot(x_prior, grad_R_gaussian, 'b--', lw=1.5, alpha=0.7, label='$\\nabla R(x)$ (高斯)')
axes[1, 0].plot(x_prior, grad_R_laplace, 'g--', lw=1.5, alpha=0.7, label='$\\nabla R(x)$ (Laplace)')
axes[1, 0].set_xlabel('$x$')
axes[1, 0].set_ylabel('$R(x)$ / $\\nabla R(x)$')
axes[1, 0].set_title('经典先验的能量函数与梯度')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(alpha=0.3)

# 子图4：得分函数指向高密度区域
axes[1, 1].plot(x_grid, gm_pdf_normalized(x_grid), 'b-', lw=2, alpha=0.5, label='$p(x)$')
# 箭头放在密度曲线高度的30%处，与密度曲线视觉关联
arrow_y = 0.3 * gm_pdf_normalized(x_grid[::10])
axes[1, 1].quiver(x_grid[::10], arrow_y,
                  true_score[::10] / np.max(np.abs(true_score)),
                  np.zeros_like(x_grid[::10]),
                  scale=20, width=0.005, alpha=0.7, color='red',
                  label='得分方向 $\\nabla\\log p(x)$')
axes[1, 1].set_xlabel('$x$')
axes[1, 1].set_ylabel('密度/得分方向')
axes[1, 1].set_title('得分函数指向高密度区域')
axes[1, 1].set_xlim(-6, 6)
axes[1, 1].legend(loc='upper right')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_Z困境与得分优势.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"图表已保存: 步骤1_Z困境与得分优势.png")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验6.1-1 总结")
print("=" * 60)
print("1. 得分函数是Langevin动力学的驱动力")
print(r"   $X_{m+1} = X_m + \delta \nabla\log p(X_m|y) + \sqrt{2\delta} Z_{m+1}$")
print("\n2. 先验得分 $\nabla\log p(x)$ 是逆问题中的瓶颈")
print("   - 似然得分可算，先验得分不可算")
print("\n3. 归一化常数Z的困境")
print("   - 低维可算，高维不可算")
print("   - 对于复杂先验，$R(x)$ 本身不可写")
print("\n4. 得分函数的关键优势：无需Z")
print(r"   $\nabla\log p(x) = \nabla\log \tilde{p}(x)$")
print("   Z在求导时完全消去")
print("   即使Z不可计算，得分函数仍可学习")
print("\n5. 从数据中学习得分的可能性")
print("   - 目标：训练网络 $s_\\theta(x) \\approx \\nabla\\log p(x)$")
print("   - 困难：监督信号 $\\nabla\\log p(x)$ 不可得")
print("   - 解决方案：得分匹配（详见6.2节及后续实验）")
print("\n" + "=" * 60)
print("下一步：实验6.2-1 ESM与ISM的验证")
print("=" * 60)
