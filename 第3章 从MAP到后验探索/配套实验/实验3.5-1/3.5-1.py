"""
2x2像素TV最小化：暴力搜索 vs 优化求解
对应章节：3.5 TV正则化与原始-对偶算法（开篇：TV优化的几何直觉）
知识点：TV正则化的优化地形；暴力搜索 vs 网格搜索 vs 优化求解的效率对比；
         各向异性TV的离散梯度算子；维度诅咒与迭代优化算法的必要性
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import time
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
    SAVE_DIR = os.path.join(_gdrive, 'TV最小化暴力搜索')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)
else:
    _chinese_path = '.chinese'
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")

np.random.seed(42)

# ══════════════════════════════════════════════════════════
# 1. 构造2x2像素CT问题
# ══════════════════════════════════════════════════════════

A = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
], dtype=float)

target = np.array([2.0, 2.0, 6.0, 7.0])
rhs = A @ target
alpha = 1.0

LH = np.array([[1, 0, -1, 0],
               [0, 1, 0, -1]])
LV = np.array([[1, -1, 0, 0],
               [0, 0, 1, -1]])

def tv_cost(x_vec):
    data_fit = 0.5 * np.sum((A @ x_vec - rhs) ** 2)
    tv = np.sum(np.abs(LH @ x_vec)) + np.sum(np.abs(LV @ x_vec))
    return data_fit + alpha * tv

# ══════════════════════════════════════════════════════════
# 2. 暴力搜索与随机采样
# ══════════════════════════════════════════════════════════

print("=" * 60)
print("暴力搜索 vs 优化求解：2x2 TV最小化")
print("=" * 60)
print(f"目标图像: {target.reshape(2, 2).tolist()}")
print(f"观测数据: rhs = {rhs}")
print(f"TV参数: alpha = {alpha}")

print("\n--- 方法A: 粗网格搜索 (10^4 = 1 万点) ---")
t0 = time.time()
n_grid = 10
x_range = np.linspace(1, 7, n_grid)
best_cost_grid = np.inf
best_x_grid = None

for x1 in x_range:
    for x2 in x_range:
        for x3 in x_range:
            for x4 in x_range:
                x_vec = np.array([x1, x2, x3, x4])
                cost = tv_cost(x_vec)
                if cost < best_cost_grid:
                    best_cost_grid = cost
                    best_x_grid = x_vec.copy()

t_grid = time.time() - t0
print(f"粗网格解: x = [{best_x_grid[0]:.2f}, {best_x_grid[1]:.2f}, "
      f"{best_x_grid[2]:.2f}, {best_x_grid[3]:.2f}]")
print(f"目标函数 F = {best_cost_grid:.4f}, 耗时 = {t_grid:.3f} 秒")

print("\n--- 方法B: 随机采样 (1 万点) ---")
t0 = time.time()
N_rand = 10000
best_cost_rand = np.inf
best_x_rand = None

for _ in range(N_rand):
    x_vec = 1 + 6 * np.random.rand(4)
    cost = tv_cost(x_vec)
    if cost < best_cost_rand:
        best_cost_rand = cost
        best_x_rand = x_vec.copy()

t_rand = time.time() - t0
print(f"随机采样解: x = [{best_x_rand[0]:.2f}, {best_x_rand[1]:.2f}, "
      f"{best_x_rand[2]:.2f}, {best_x_rand[3]:.2f}]")
print(f"目标函数 F = {best_cost_rand:.4f}, 耗时 = {t_rand:.3f} 秒")

# ══════════════════════════════════════════════════════════
# 3. 优化求解 (scipy.optimize.minimize, L-BFGS-B)
# ══════════════════════════════════════════════════════════

print("\n--- 方法C: 优化求解 (scipy L-BFGS-B) ---")
t0 = time.time()

result = minimize(tv_cost, x0=[4.0, 4.0, 4.0, 4.0], method='L-BFGS-B',
                  bounds=[(0, None)] * 4)
x_opt = result.x
t_opt = time.time() - t0
print(f"优化解: x = [{x_opt[0]:.4f}, {x_opt[1]:.4f}, "
      f"{x_opt[2]:.4f}, {x_opt[3]:.4f}]")
print(f"目标函数 F = {result.fun:.4f}, 耗时 = {t_opt:.6f} 秒")

print("\n" + "=" * 60)
print("对比总结:")
print(f"  粗网格搜索:  F = {best_cost_grid:.4f}, 耗时 = {t_grid:.3f} 秒")
print(f"  随机采样:    F = {best_cost_rand:.4f}, 耗时 = {t_rand:.3f} 秒")
print(f"  优化求解:    F = {result.fun:.4f}, 耗时 = {t_opt:.6f} 秒")
print(f"  优化优于粗网格: {best_cost_grid - result.fun:.4f}")
print(f"  优化优于随机采样: {best_cost_rand - result.fun:.4f}")

# ══════════════════════════════════════════════════════════
# 4. 可视化
# ══════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

im0 = axes[0].imshow(target.reshape(2, 2), cmap='gray', vmin=0, vmax=7)
axes[0].set_title(r'真解 $[2, 2; 6, 7]$')
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(best_x_grid.reshape(2, 2), cmap='gray', vmin=0, vmax=7)
axes[1].set_title(
    r'粗网格搜索' + '\n' +
    rf'$F={best_cost_grid:.2f}$ $({t_grid:.2f}s)$'
)
plt.colorbar(im1, ax=axes[1], fraction=0.046)

im2 = axes[2].imshow(best_x_rand.reshape(2, 2), cmap='gray', vmin=0, vmax=7)
axes[2].set_title(
    r'随机采样(1万)' + '\n' +
    rf'$F={best_cost_rand:.2f}$ $({t_rand:.2f}s)$'
)
plt.colorbar(im2, ax=axes[2], fraction=0.046)

im3 = axes[3].imshow(x_opt.reshape(2, 2), cmap='gray', vmin=0, vmax=7)
axes[3].set_title(
    r'优化求解' + '\n' +
    rf'$F={result.fun:.2f}$ $({t_opt:.4f}s)$'
)
plt.colorbar(im3, ax=axes[3], fraction=0.046)

plt.suptitle(
    r'2$\times$2像素TV最小化 —— 暴力搜索 vs 优化求解',
    fontsize=12
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(SAVE_DIR, '步骤1_2x2_TV搜索对比.png'), dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════
# 5. 结果输出与教学说明
# ══════════════════════════════════════════════════════════

print(f"\n[核心发现]")
print(f"  1. 2x2 像素 TV 优化是一个 4 维连续优化问题，")
print(f"     目标函数 F(x) = 0.5*||Ax - b||^2 + alpha * ||Dx||_1")
print(f"  2. 粗网格搜索 (10^4 点): 精度受网格分辨率限制，")
print(f"     加密一倍网格 → 搜索量增至 16 倍 (维度诅咒)")
print(f"  3. 随机采样 (10^4 点): 无系统性覆盖，精度有限")
print(f"  4. 优化求解 (L-BFGS-B): 高效精确，")
print(f"     但在大规模问题中 TV 的不可微性使一阶方法失效，")
print(f"     需使用 Chambolle-Pock 或 ADMM 等保留非光滑结构的专用算法")

print(f"\n[联系后续实验]")
print(f"  本实验展示了 TV 目标函数的几何地形与暴力搜索的局限。")
print(f"  3.5 节正文从 Fenchel 对偶出发，推导 Chambolle-Pock 原始-对偶算法，")
print(f"  在后续实验中将演示该算法在大规模图像上的保边去噪效果。")

print(f"\n{'=' * 60}")
print(f"实验完成")
print(f"输出图片: {os.path.join(SAVE_DIR, '步骤1_2x2_TV搜索对比.png')}")
print(f"{'=' * 60}")