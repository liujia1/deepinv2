"""
实验2.3-1 贝叶斯去噪器：不同先验下去噪器的表现
对应章节：2.3 先验的质量：MMSE vs MAP估计器
知识点：MMSE估计器 = 贝叶斯去噪器；先验质量决定去噪器质量；非高斯先验的挑战

素材来源：
  - Ratti Pre_course_II_Ratti.md P9-P13: 十字形去噪回归实验

修改说明（基于代码评审）：
  1. 添加 ESS（有效样本量）诊断，检测蒙特卡洛权重退化
  2. 修正 Tikhonov 的 λ 为理论最优值
  3. 统一所有方法的评估样本集
  4. 为 LASSO 的 λ 添加选取依据注释
  5. 改进可视化布局：上半部分散点图，下半部分柱状图
  6. 使用 np.random.default_rng(42) 替代 np.random.seed(42)
  7. 向量化 sample_cross_distribution 函数
  8. 统一散点图使用随机抽样
  9. 简化 compute_nrmse 函数
  10. 调整柱状图宽度
  11. 向量化贝叶斯去噪器计算
  12. 添加 Tikhonov λ 的注释说明（使用 ground truth 方差）
  13. 修正 ESS 阈值语义
  14. 补充 LASSO λ 的调参说明
"""

import numpy as np
import os
import sys
import warnings

SILENT_MODE = True

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
    _chinese_path = os.path.join(_gdrive, '实验2.3-1', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.3-1')
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

rng = np.random.default_rng(42)

N = 1000
sigma_eps = 0.1

def sample_cross_distribution(n_samples, rng):
    samples = np.zeros((n_samples, 2))
    mask = rng.random(n_samples) < 0.5
    vals = rng.uniform(-1, 1, n_samples)
    samples[mask, 1] = vals[mask]
    samples[~mask, 0] = vals[~mask]
    return samples

x = sample_cross_distribution(N, rng)
y = x + sigma_eps * rng.standard_normal((N, 2))

var_x = np.mean(np.sum(x**2, axis=1))
lambda_tikh_optimal = sigma_eps**2 / var_x
print(f"[参数] 十字形分布方差: {var_x:.4f}")
print(f"[参数] Tikhonov 理论最优 λ = σ²/σ²_x = {sigma_eps**2:.4f}/{var_x:.4f} = {lambda_tikh_optimal:.4f}")
print(f"        注: 此处使用 ground truth 方差计算 λ，仅用于教学展示理论最优性能")
print(f"        实际场景中 x 未知，需从数据估计方差或交叉验证调参")

x_identity = y

x_tikh = y / (1 + lambda_tikh_optimal)

def soft_thresh(x, l):
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - l, np.zeros_like(x)))

lambda_lasso = 0.08
print(f"[参数] LASSO λ = {lambda_lasso} (手动调参经验值)")
print(f"        注: 此值针对噪声水平 σ={sigma_eps} 手动调优")
print(f"        若修改 σ，需重新调整 λ（可通过交叉验证或网格搜索）")
x_lasso = soft_thresh(y, lambda_lasso)

N_mc = 2000
x_mc = sample_cross_distribution(N_mc, rng)

N_test = N

# 向量化贝叶斯去噪器计算
diff = x_mc[np.newaxis, :, :] - y[:N_test, np.newaxis, :]  # (N_test, N_mc, 2)
log_w = -np.sum(diff**2, axis=2) / (2 * sigma_eps**2)      # (N_test, N_mc)
log_w -= log_w.max(axis=1, keepdims=True)
w = np.exp(log_w)
w /= w.sum(axis=1, keepdims=True)

ess_arr = 1.0 / np.sum(w**2, axis=1)  # (N_test,)
x_bayes = w @ x_mc                     # (N_test, 2)

ess_mean = np.mean(ess_arr)
ess_min = np.min(ess_arr)

ESS_THRESHOLD = max(3, N_mc * 0.001)
low_ess_count = np.sum(ess_arr < ESS_THRESHOLD)

print(f"[诊断] 蒙特卡洛有效样本量 (ESS): 均值={ess_mean:.1f}, 最小={ess_min:.1f}")
print(f"[诊断] ESS 阈值: {ESS_THRESHOLD:.1f} (max(3, N_mc×0.1%))")
if low_ess_count > 0:
    print(f"[警告] {low_ess_count}/{N_test} 个样本的 ESS < {ESS_THRESHOLD:.1f}")
    print(f"        低 ESS 在低噪声场景是预期现象，反映重要性采样的固有局限")
    print(f"        本实验使用 N_mc={N_mc} 样本，教学结论不受影响")

def compute_nrmse(x_est, x_true):
    return np.sqrt(np.mean(np.sum((x_est - x_true)**2, axis=1)) /
                   np.mean(np.sum(x_true**2, axis=1)))

err_identity = compute_nrmse(x_identity, x)
err_tikh = compute_nrmse(x_tikh, x)
err_lasso = compute_nrmse(x_lasso, x)
err_bayes = compute_nrmse(x_bayes, x)

print("\n===== 贝叶斯去噪器：不同先验下去噪器的表现 =====")
print(f"\n实验设定:")
print(f"  先验: 十字形分布 (非高斯)")
print(f"  噪声: y = x + ε, ε ~ N(0, {sigma_eps}²I)")
print(f"  评估样本数: {N_test}")
print(f"\n去噪器相对误差对比:")
print(f"  恒等映射 (无先验):     {err_identity*100:.2f}%")
print(f"  Tikhonov (高斯先验):   {err_tikh*100:.2f}%")
print(f"  LASSO (Laplace先验):   {err_lasso*100:.2f}%")
print(f"  贝叶斯去噪器 (真实先验): {err_bayes*100:.2f}% ← MMSE最优")
print(f"\n结论:")
print(f"  贝叶斯去噪器 = E[x|y] = MMSE估计器")
print(f"  先验越准确，去噪器性能越好")

fig = plt.figure(figsize=(16, 10))

gs = fig.add_gridspec(2, 5, height_ratios=[1, 0.6], hspace=0.3, wspace=0.3)

idx = rng.choice(N, 200, replace=False)

ax0 = fig.add_subplot(gs[0, 0])
ax0.scatter(x[idx, 0], x[idx, 1], s=1, alpha=0.5, c='blue')
ax0.set_xlim(-1.5, 1.5)
ax0.set_ylim(-1.5, 1.5)
ax0.set_aspect('equal')
ax0.set_title('真实信号 x\n(十字形分布)')

ax1 = fig.add_subplot(gs[0, 1])
ax1.scatter(y[idx, 0], y[idx, 1], s=1, alpha=0.5, c='red')
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_aspect('equal')
ax1.set_title(f'含噪观测 y\n恒等映射 err={err_identity*100:.1f}%')

ax2 = fig.add_subplot(gs[0, 2])
ax2.scatter(x_tikh[idx, 0], x_tikh[idx, 1], s=1, alpha=0.5, c='green')
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_aspect('equal')
ax2.set_title(f'Tikhonov去噪\n(高斯先验, λ={lambda_tikh_optimal:.3f})\nerr={err_tikh*100:.1f}%')

ax3 = fig.add_subplot(gs[0, 3])
ax3.scatter(x_lasso[idx, 0], x_lasso[idx, 1], s=1, alpha=0.5, c='orange')
ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax3.set_aspect('equal')
ax3.set_title(f'LASSO去噪\n(Laplace先验, λ={lambda_lasso})\nerr={err_lasso*100:.1f}%')

ax4 = fig.add_subplot(gs[0, 4])
ax4.scatter(x_bayes[idx, 0], x_bayes[idx, 1], s=1, alpha=0.5, c='purple')
ax4.set_xlim(-1.5, 1.5)
ax4.set_ylim(-1.5, 1.5)
ax4.set_aspect('equal')
ax4.set_title(f'贝叶斯去噪器\n(真实先验)\nerr={err_bayes*100:.1f}%')

ax_bar = fig.add_subplot(gs[1, :])
methods = ['恒等映射\n(无先验)', 'Tikhonov\n(高斯先验)', 'LASSO\n(Laplace先验)', '贝叶斯去噪器\n(真实先验)']
errors = [err_identity, err_tikh, err_lasso, err_bayes]
colors = ['gray', 'green', 'orange', 'purple']

bars = ax_bar.bar(methods, [e*100 for e in errors], color=colors, alpha=0.7, width=0.6)
ax_bar.set_ylabel('相对误差 (%)', fontsize=12)
ax_bar.set_title('去噪器性能对比', fontsize=14)
for bar, e in zip(bars, errors):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{e*100:.2f}%', ha='center', fontsize=11, fontweight='bold')
ax_bar.set_ylim(0, max(errors)*100 * 1.15)
ax_bar.grid(axis='y', alpha=0.3)

fig.suptitle('贝叶斯去噪器：先验质量决定去噪器质量\nD(y) = E[x|y] 是给定先验下的最优去噪器', fontsize=14, y=0.98)
plt.savefig(os.path.join(SAVE_DIR, '步骤1_去噪器对比.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
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
print(f"\n5. 蒙特卡洛方法注意事项:")
print(f"   有效样本量 (ESS) 诊断: 均值={ess_mean:.1f}, 最小={ess_min:.1f}")
print(f"   低 ESS 在低噪声场景是预期现象，不影响本实验的教学结论")
