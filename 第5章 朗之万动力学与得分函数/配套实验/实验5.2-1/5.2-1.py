"""
实验5.2-1 得分函数可视化
对应章节：5.2 得分函数：对数概率的梯度
知识点：
  - 得分函数的定义：s(x) = ∇_x log p(x)
  - 得分函数的几何含义：概率分布的"方向场"
  - 得分函数与Langevin动力学的关系

素材来源：
  - 原创设计：2D高斯混合得分场可视化

修改说明：
  从原参考实验5.1.py拆分，聚焦得分函数的几何含义，
  展示得分函数如何引导Langevin采样。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys

# ====== 中文字体配置（兼容本地和Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验5.2-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)

# ============================================================
# 步骤1：定义2D高斯混合分布及其得分函数
# ============================================================
print("=" * 60)
print("步骤1：定义2D高斯混合分布及其得分函数")
print("=" * 60)

def gaussian_mixture_logprob(x, y):
    """
    2D高斯混合的对数概率
    p(x,y) = 0.5 * N(μ₁, Σ₁) + 0.5 * N(μ₂, Σ₂)
    
    参数:
        x, y: 空间坐标
    
    返回:
        log p(x,y)
    """
    mu1 = np.array([-2, -2])
    mu2 = np.array([2, 2])
    Sigma1 = np.array([[1, 0.3], [0.3, 1]])
    Sigma2 = np.array([[1, -0.3], [-0.3, 1]])

    x_vec = np.array([x, y])

    inv_Sigma1 = np.linalg.inv(Sigma1)
    inv_Sigma2 = np.linalg.inv(Sigma2)
    det_Sigma1 = np.linalg.det(Sigma1)
    det_Sigma2 = np.linalg.det(Sigma2)

    p1 = np.exp(-0.5 * (x_vec - mu1).T @ inv_Sigma1 @ (x_vec - mu1))
    p1 = p1 / (2 * np.pi * np.sqrt(det_Sigma1))

    p2 = np.exp(-0.5 * (x_vec - mu2).T @ inv_Sigma2 @ (x_vec - mu2))
    p2 = p2 / (2 * np.pi * np.sqrt(det_Sigma2))

    p = 0.5 * p1 + 0.5 * p2
    return np.log(p)

def gaussian_mixture_score(x, y):
    """
    计算2D高斯混合的得分函数（对数概率的梯度）
    s(x,y) = ∇ log p(x,y)
    
    使用解析梯度计算：
    ∇log p(x) = Σ_k [w_k p_k(x) / p(x)] · (-Σ_k^{-1}(x - μ_k))
    
    参数:
        x, y: 空间坐标
    
    返回:
        [∂log p/∂x, ∂log p/∂y]
    """
    # 定义两个高斯分量的参数
    mu1 = np.array([-2., -2.])
    mu2 = np.array([2., 2.])
    S1 = np.array([[1., 0.3], [0.3, 1.]])
    S2 = np.array([[1., -0.3], [-0.3, 1.]])
    
    # 预计算逆矩阵和行列式
    inv_S1 = np.linalg.inv(S1)
    inv_S2 = np.linalg.inv(S2)
    det_S1 = np.linalg.det(S1)
    det_S2 = np.linalg.det(S2)
    
    # 输入向量
    xv = np.array([x, y])
    
    # 计算两个高斯分量的密度
    def gauss_density(xv, mu, inv_S, det_S):
        d = xv - mu
        return np.exp(-0.5 * d @ inv_S @ d) / (2 * np.pi * np.sqrt(det_S))
    
    p1 = gauss_density(xv, mu1, inv_S1, det_S1)
    p2 = gauss_density(xv, mu2, inv_S2, det_S2)
    
    # 总密度（等权重混合）
    p = 0.5 * p1 + 0.5 * p2
    
    # 解析梯度：加权平均各分量的梯度
    # ∇log p = Σ_k [w_k p_k / p] · (-Σ_k^{-1}(x - μ_k))
    grad = (0.5 * p1 * (-inv_S1 @ (xv - mu1)) + 
            0.5 * p2 * (-inv_S2 @ (xv - mu2))) / p
    
    return grad

print("得分函数定义：")
print("  s(x) = ∇_x log p(x)")
print("  对于高斯混合分布，得分函数指向高密度区域")
print("  模长表示密度变化的剧烈程度")


# ============================================================
# 步骤2：得分函数向量场可视化
# ============================================================
print("\n" + "=" * 60)
print("步骤2：得分函数向量场可视化")
print("=" * 60)

# 创建网格
x_grid = np.linspace(-5, 5, 20)
y_grid = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x_grid, y_grid)

# 计算得分场
U_grid = np.zeros_like(X)
V_grid = np.zeros_like(Y)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        score = gaussian_mixture_score(X[i,j], Y[i,j])
        U_grid[i,j] = score[0]
        V_grid[i,j] = score[1]

# 可视化
plt.figure(figsize=(15, 5))

# 归一化向量场（仅显示方向，用颜色编码模长）
norm = np.sqrt(U_grid**2 + V_grid**2) + 1e-8
U_normalized = U_grid / norm
V_normalized = V_grid / norm

# 子图1：得分函数向量场（归一化，颜色编码模长）
plt.subplot(1, 3, 1)
plt.quiver(X, Y, U_normalized, V_normalized, np.log1p(norm), 
           cmap='Reds', alpha=0.7, scale=25)
plt.scatter([-2, 2], [-2, 2], c='black', s=100, marker='x', label='众数')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('得分函数向量场 $s(x,y) = \\nabla \\log p(x,y)$\n（箭头归一化，颜色表示模长）')
plt.legend()
plt.grid(alpha=0.3)
plt.colorbar(label='$\\log(1 + |s(x,y)|)$')

# 子图2：概率密度等高线
plt.subplot(1, 3, 2)
x_dense = np.linspace(-5, 5, 100)
y_dense = np.linspace(-5, 5, 100)
X_dense, Y_dense = np.meshgrid(x_dense, y_dense)
Z = np.zeros_like(X_dense)
for i in range(X_dense.shape[0]):
    for j in range(X_dense.shape[1]):
        Z[i,j] = np.exp(gaussian_mixture_logprob(X_dense[i,j], Y_dense[i,j]))

plt.contour(X_dense, Y_dense, Z, levels=20, cmap='viridis')
plt.scatter([-2, 2], [-2, 2], c='red', s=100, marker='x', label='众数')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('概率密度等高线 $p(x,y)$')
plt.legend()
plt.grid(alpha=0.3)

# 子图3：得分场与密度叠加
plt.subplot(1, 3, 3)
plt.contour(X_dense, Y_dense, Z, levels=20, cmap='viridis', alpha=0.5)
plt.quiver(X, Y, U_normalized, V_normalized, np.log1p(norm), 
           cmap='Reds', alpha=0.6, scale=25)
plt.scatter([-2, 2], [-2, 2], c='black', s=100, marker='x', label='众数')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('得分场与密度叠加\n（箭头归一化，颜色表示模长）')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_得分函数向量场.png'), dpi=150)
plt.close()

print("得分函数几何含义：")
print("  方向：指向密度增长最快的方向")
print("  模长：表示密度变化的剧烈程度")
print("  零点：概率密度的驻点（众数、鞍点）")


# ============================================================
# 步骤3：ULA采样轨迹展示
# ============================================================
print("\n" + "=" * 60)
print("步骤3：ULA采样轨迹展示")
print("=" * 60)

def ULA_2d(niter, delta, x0):
    """
    2D ULA采样
    
    参数:
        niter: 迭代次数
        delta: 步长
        x0: 初始点
    
    返回:
        samples: 采样序列 (niter, 2)
    """
    samples = np.zeros((niter, 2))
    X = x0.copy()
    for i in range(niter):
        score = gaussian_mixture_score(X[0], X[1])  # ∇log p(x)
        Z = np.random.randn(2)
        X = X + delta * score + np.sqrt(2 * delta) * Z
        samples[i] = X
    return samples

# 参数设置
niter = 10000
delta = 0.1
x0 = np.array([0.0, 0.0])
burn_in = 2000  # 丢弃初始的burn-in阶段

# 运行ULA
samples = ULA_2d(niter, delta, x0)

# 可视化
plt.figure(figsize=(15, 5))

# 子图1：样本散点图（丢弃burn-in）
plt.subplot(1, 3, 1)
plt.scatter(samples[burn_in:, 0], samples[burn_in:, 1], s=1, alpha=0.3, c='blue')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title(f'ULA采样（丢弃前{burn_in}步burn-in）\n有效样本{niter-burn_in}个')
plt.grid(alpha=0.3)

# 子图2：采样轨迹
plt.subplot(1, 3, 2)
plt.plot(samples[:2000, 0], samples[:2000, 1], 'g-', lw=0.8, alpha=0.7)
plt.scatter(samples[0, 0], samples[0, 1], c='green', s=50, label='起点')
plt.scatter(samples[-1, 0], samples[-1, 1], c='red', s=50, label='终点')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('采样轨迹（前2000步）')
plt.legend()
plt.grid(alpha=0.3)

# 子图3：轨迹与得分场叠加
plt.subplot(1, 3, 3)
plt.contour(X_dense, Y_dense, Z, levels=20, cmap='viridis', alpha=0.3)
plt.quiver(X, Y, U_normalized, V_normalized, np.log1p(norm), 
           cmap='Reds', alpha=0.3, scale=25)
plt.plot(samples[:2000, 0], samples[:2000, 1], 'g-', lw=0.8, alpha=0.7)
plt.scatter(samples[0, 0], samples[0, 1], c='green', s=50, label='起点')
plt.scatter(samples[-1, 0], samples[-1, 1], c='red', s=50, label='终点')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('轨迹与得分场叠加\n（箭头归一化，颜色表示模长）')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_ULA采样轨迹.png'), dpi=150)
plt.close()

print("Langevin动力学机制：")
print("  漂移项：沿得分方向'爬坡'（推向高密度区域）")
print("  扩散项：随机'游走'（防止坍缩到众数）")
print("  平衡：漂移力与扩散力平衡产生目标分布")
print(f"  burn-in：丢弃前{burn_in}步，确保样本来自平稳分布")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验5.2-1 总结")
print("=" * 60)
print("1. 得分函数定义：s(x) = ∇_x log p(x)，是对数概率的梯度")
print("2. 解析梯度计算：")
print("   对于高斯混合 p(x) = Σ_k w_k N(μ_k, Σ_k)")
print("   ∇log p(x) = Σ_k [w_k p_k(x) / p(x)] · (-Σ_k^{-1}(x - μ_k))")
print("   比数值梯度更精确，且避免了浮点精度问题")
print("3. 几何含义：")
print("   - 方向：指向密度增长最快的方向")
print("   - 模长：表示密度变化的剧烈程度")
print("   - 零点：概率密度的驻点")
print("4. 与Langevin动力学的关系：")
print("   - 漂移项：s(x)dt 推向高密度区域")
print("   - 扩散项：√2 dW_t 防止坍缩")
print("   - 两者平衡产生目标分布 p(x)")
print("5. 得分函数天然消去归一化常数：")
print("   s(x) = ∇ log p(x) = ∇ log p̃(x)")
print("   这对贝叶斯推断至关重要")
