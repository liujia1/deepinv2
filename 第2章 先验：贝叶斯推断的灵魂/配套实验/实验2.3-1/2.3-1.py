"""
实验2.3-1 贝叶斯去噪器：不同先验下去噪器的表现
对应章节：2.3 先验的质量：MMSE vs MAP估计器
知识点：MMSE估计器 = 贝叶斯去噪器；先验质量决定去噪器质量；非高斯先验的挑战

素材来源：
  - Ratti Pre_course_II_Ratti.md P9-P13: 十字形去噪回归实验
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.3-1', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.3-1')
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

N = 1000
sigma_eps = 0.1

x = np.zeros((N, 2))
for i in range(N):
    if np.random.rand() < 0.5:
        x[i, 0] = 0.0
        x[i, 1] = np.random.uniform(-1, 1)
    else:
        x[i, 0] = np.random.uniform(-1, 1)
        x[i, 1] = 0.0

y = x + sigma_eps * np.random.randn(N, 2)

x_identity = y
err_identity = np.sqrt(np.mean(np.sum((x_identity - x)**2, axis=1)) /
                       np.mean(np.sum(x**2, axis=1)))

lambda_tikh = 0.1
x_tikh = y / (1 + lambda_tikh)
err_tikh = np.sqrt(np.mean(np.sum((x_tikh - x)**2, axis=1)) /
                   np.mean(np.sum(x**2, axis=1)))

def soft_thresh(x, l):
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - l, np.zeros_like(x)))

lambda_lasso = 0.1
x_lasso = soft_thresh(y, lambda_lasso)
err_lasso = np.sqrt(np.mean(np.sum((x_lasso - x)**2, axis=1)) /
                    np.mean(np.sum(x**2, axis=1)))

N_mc = 1000
x_mc = np.zeros((N_mc, 2))
for i in range(N_mc):
    if np.random.rand() < 0.5:
        x_mc[i] = [0.0, np.random.uniform(-1, 1)]
    else:
        x_mc[i] = [np.random.uniform(-1, 1), 0.0]

N_test = min(500, N)
x_bayes = np.zeros((N_test, 2))
for i in range(N_test):
    diff = x_mc - y[i]
    log_w = -np.sum(diff**2, axis=1) / (2 * sigma_eps**2)
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    w /= np.sum(w)
    x_bayes[i] = np.sum(w[:, np.newaxis] * x_mc, axis=0)

err_bayes = np.sqrt(np.mean(np.sum((x_bayes - x[:N_test])**2, axis=1)) /
                    np.mean(np.sum(x[:N_test]**2, axis=1)))

print("===== 贝叶斯去噪器：不同先验下去噪器的表现 =====")
print(f"\n实验设定:")
print(f"  先验: 十字形分布 (非高斯)")
print(f"  噪声: y = x + ε, ε ~ N(0, {sigma_eps}²I)")
print(f"\n去噪器相对误差对比:")
print(f"  恒等映射 (无先验):     {err_identity*100:.2f}%")
print(f"  Tikhonov (高斯先验):   {err_tikh*100:.2f}%")
print(f"  LASSO (Laplace先验):   {err_lasso*100:.2f}%")
print(f"  贝叶斯去噪器 (真实先验): {err_bayes*100:.2f}% ← MMSE最优")
print(f"\n结论:")
print(f"  贝叶斯去噪器 = E[x|y] = MMSE估计器")
print(f"  先验越准确，去噪器性能越好")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

idx = np.random.choice(N, 200, replace=False)

axes[0, 0].scatter(x[idx, 0], x[idx, 1], s=1, alpha=0.5, c='blue')
axes[0, 0].set_xlim(-1.5, 1.5)
axes[0, 0].set_ylim(-1.5, 1.5)
axes[0, 0].set_aspect('equal')
axes[0, 0].set_title('真实信号 x\n(十字形分布)')

axes[0, 1].scatter(y[idx, 0], y[idx, 1], s=1, alpha=0.5, c='red')
axes[0, 1].set_xlim(-1.5, 1.5)
axes[0, 1].set_ylim(-1.5, 1.5)
axes[0, 1].set_aspect('equal')
axes[0, 1].set_title(f'含噪观测 y\n恒等映射 err={err_identity*100:.1f}%')

axes[0, 2].scatter(x_tikh[idx, 0], x_tikh[idx, 1], s=1, alpha=0.5, c='green')
axes[0, 2].set_xlim(-1.5, 1.5)
axes[0, 2].set_ylim(-1.5, 1.5)
axes[0, 2].set_aspect('equal')
axes[0, 2].set_title(f'Tikhonov去噪\n(高斯先验) err={err_tikh*100:.1f}%')

axes[1, 0].scatter(x_lasso[idx, 0], x_lasso[idx, 1], s=1, alpha=0.5, c='orange')
axes[1, 0].set_xlim(-1.5, 1.5)
axes[1, 0].set_ylim(-1.5, 1.5)
axes[1, 0].set_aspect('equal')
axes[1, 0].set_title(f'LASSO去噪\n(Laplace先验) err={err_lasso*100:.1f}%')

axes[1, 1].scatter(x_bayes[:200, 0], x_bayes[:200, 1], s=1, alpha=0.5, c='purple')
axes[1, 1].set_xlim(-1.5, 1.5)
axes[1, 1].set_ylim(-1.5, 1.5)
axes[1, 1].set_aspect('equal')
axes[1, 1].set_title(f'贝叶斯去噪器\n(真实先验) err={err_bayes*100:.1f}%\n= MMSE估计器')

methods = ['恒等映射\n(无先验)', 'Tikhonov\n(高斯先验)', 'LASSO\n(Laplace先验)', '贝叶斯去噪器\n(真实先验)']
errors = [err_identity, err_tikh, err_lasso, err_bayes]
colors = ['gray', 'green', 'orange', 'purple']

axes[1, 2].bar(methods, [e*100 for e in errors], color=colors, alpha=0.7)
axes[1, 2].set_ylabel('相对误差 (%)')
axes[1, 2].set_title('去噪器性能对比')
for i, (m, e) in enumerate(zip(methods, errors)):
    axes[1, 2].text(i, e*100 + 0.5, f'{e*100:.2f}%', ha='center', fontsize=9)

plt.suptitle('贝叶斯去噪器：先验质量决定去噪器质量\nD(y) = E[x|y] 是给定先验下的最优去噪器', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_去噪器对比.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n===== 2.3章节核心结论 =====")
print("\n1. 贝叶斯去噪器定义:")
print("   D_σ(y) = E[x|y] = ∫ x p(x|y) dx")
print("   是给定先验 p(x) 和噪声水平 σ 下的 MMSE 最优去噪器")
print("\n2. 先验与去噪器的双向映射:")
print("   先验 p(x) → 去噪器 D_σ(y)")
print("   去噪器残差 D_σ(y) - y ∝ ∇_y ln p_σ(y) (Tweedie等式, 第5章)")
print("\n3. 实验结论:")
print("   真实先验 > Laplace先验 > 高斯先验 > 无先验")
print("   先验假设越准确，去噪器性能越好")
print("\n4. 非高斯先验的挑战:")
print("   十字形分布是非高斯的，E[x|y] 无闭式解")
print("   需要用蒙特卡洛方法近似 (详见第4章 MCMC)")
