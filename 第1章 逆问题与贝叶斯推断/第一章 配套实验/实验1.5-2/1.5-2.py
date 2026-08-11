# -*- coding: utf-8 -*-
"""
实验1.5-2 高斯似然 = 最小二乘：负对数似然与数据项的代数等价
对应章节：1.5 贝叶斯框架：从似然到后验

知识点：
  - 高斯噪声假设下，负对数似然 -ln p(y|x) 正比于最小二乘数据项 ||y - Ax||^2
  - 后验能量 = 数据项 + 正则项 中"数据项"的来源
  - 似然只关心数据拟合，与先验无关

实验内容：
  - 构造线性正向模型 y = A x + noise（一维信号，频域对角算子，无外部文件依赖）
  - 在多个候选 x 上同时计算最小二乘目标 ||y - Ax||^2 与高斯负对数似然
  - 验证二者仅差常数因子 1/(2 sigma^2)，曲线完全重合 -> 代数等价成立

素材来源：
  - 1.5 节"动手感受一下"代码段（仅抽取"似然=最小二乘"部分，去掉与 1.3-1 重复的噪声放大部分）
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 静默模式，非交互式后端，不弹出 GUI 窗口
import matplotlib.pyplot as plt
import os
import sys

# ====== 中文字体配置（兼容本地和 Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验1.5-2')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(SAVE_DIR, exist_ok=True)
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

sys.path.insert(0, _chinese_path)
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=_chinese_path)
# ========================================================

np.random.seed(42)

print("=" * 60)
print("实验1.5-2：高斯似然 = 最小二乘")
print("=" * 60)
print("验证：高斯噪声假设下 -ln p(y|x) = (1/(2 sigma^2)) * ||y - A x||^2 + const")

# ---- 1. 构造自包含的一维线性逆问题 ----
# 用频域对角算子 A（高斯低通）：A 在 DFT 基下对角，对角元 = 频域传递函数 H
# 这样无需外部图片，即可演示 y = A x + noise
n = 256
t = np.linspace(0, 1, n)
x_true = np.sin(2 * np.pi * 3 * t) + 0.5 * np.cos(2 * np.pi * 7 * t)

sigma_psf = 4.0               # 频域高斯核标准差（小 => 严重低通 => 病态）
k = np.fft.fftfreq(n) * n     # 频率坐标
H = np.exp(-0.5 * (k / sigma_psf) ** 2)  # 频域传递函数，高频 -> 0
H = H.astype(complex)

# 正向生成含噪观测 y = A x + noise
X = np.fft.fft(x_true)
y_clean = np.real(np.fft.ifft(H * X))

sigma = 1.0                   # 噪声标准差
noise = sigma * np.random.randn(n)
y = y_clean + noise

print(f"\n[问题设置]")
print(f"  信号维度 n = {n}")
print(f"  噪声标准差 sigma = {sigma}")
print(f"  正向算子 A 类型：频域对角（高斯低通），高频被压制")

# ---- 2. 构造一组候选解 x，对比两种目标 ----
# 候选解：真实解 + 一系列扰动，覆盖"拟合好"到"拟合差"
n_candidates = 200
least_squares_list = []
neg_log_lik_list = []

for i in range(n_candidates):
    # 扰动幅度从 0（完美拟合）到较大（拟合差），均匀采样
    scale = (i / (n_candidates - 1)) * 2.0
    x_cand = x_true + scale * np.random.randn(n)
    residual = y - np.real(np.fft.ifft(H * np.fft.fft(x_cand)))
    ls = np.sum(residual ** 2)                       # ||y - A x||^2
    nll = 0.5 / sigma ** 2 * np.sum(residual ** 2)   # -ln p(y|x) 不计常数
    least_squares_list.append(ls)
    neg_log_lik_list.append(nll)

least_squares_list = np.array(least_squares_list)
neg_log_lik_list = np.array(neg_log_lik_list)

# 验证等价性：nll 应约等于 (1/(2 sigma^2)) * ls
factor = 1.0 / (2.0 * sigma ** 2)
predicted_nll = factor * least_squares_list
max_rel_err = float(np.max(np.abs(neg_log_lik_list - predicted_nll) /
                           (np.abs(predicted_nll) + 1e-12)))

print(f"\n[等价性验证]")
print(f"  理论比例因子 1/(2 sigma^2) = {factor:.4f}")
print(f"  最大相对偏差 = {max_rel_err:.3e}  （应接近 0）")
if max_rel_err < 1e-10:
    print(f"  [验证通过] 负对数似然与最小二乘仅差常数因子，二者代数等价")
else:
    print(f"  [警告] 偏差偏大，请检查实现")

# 在"完美候选"处（residual = noise）单独打印一次直观对比
residual_ideal = y - y_clean
ls_ideal = float(np.sum(residual_ideal ** 2))
nll_ideal = float(0.5 / sigma ** 2 * np.sum(residual_ideal ** 2))
print(f"\n[单点直观对比]（候选 = 真实 x，residual 应≈纯噪声）")
print(f"  最小二乘 ||y - A x||^2          = {ls_ideal:.4f}")
print(f"  高斯负对数似然 (不计常数)       = {nll_ideal:.4f}")
print(f"  比值 nll / ls                   = {nll_ideal / ls_ideal:.4f}  (= 1/(2 sigma^2) = {factor:.4f} [OK])")

# ---- 3. 可视化：两条曲线完全重合 ----
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# 左：最小二乘 vs 负对数似然（按候选排序，两条线应完全重合）
ax[0].plot(least_squares_list, label=r'$\|y - A x\|^2$（最小二乘）', linewidth=2)
ax[0].plot(neg_log_lik_list, '--', label=r'$-\ln p(y|x)$（负对数似然）', linewidth=2)
ax[0].set_xlabel('候选解索引')
ax[0].set_ylabel('目标值')
ax[0].set_title(r'最小二乘 vs 负对数似然（完全重合 = 代数等价）')
ax[0].legend(fontsize=9)
ax[0].grid(True, alpha=0.3)

# 右：负对数似然 对 最小二乘 的散点，应落在过原点的直线 y = (1/(2 sigma^2)) x 上
ax[1].scatter(least_squares_list, neg_log_lik_list, s=10, alpha=0.5, label=r'样本点')
ls_grid = np.linspace(least_squares_list.min(), least_squares_list.max(), 100)
ax[1].plot(ls_grid, factor * ls_grid, 'r-', linewidth=2, label=r'$y = \frac{1}{2\sigma^2} x$')
ax[1].set_xlabel(r'$\|y - A x\|^2$（最小二乘）')
ax[1].set_ylabel(r'$-\ln p(y|x)$（负对数似然）')
ax[1].set_title(r'负对数似然 正比于 最小二乘（斜率 = $1/(2\sigma^2)$）')
ax[1].legend(fontsize=9)
ax[1].grid(True, alpha=0.3)

plt.suptitle(r'实验1.5-2：高斯似然 = 最小二乘（$y = A x + \varepsilon,\ \varepsilon\sim\mathcal{N}(0,\sigma^2 I)$）',
             fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_5_2_似然等于最小二乘.png'), dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# ---- 4. 结论 ----
print("\n" + "=" * 60)
print("结论")
print("=" * 60)
print("1. 高斯噪声假设下，负对数似然 -ln p(y|x) 正比于最小二乘数据项 ||y - Ax||^2")
print("2. 比例因子 = 1/(2 sigma^2)，与候选 x 无关 -> 仅缩放不影响最优解位置")
print("3. 因此最大化似然 == 最小化 ||y - Ax||^2，MAP 退化为经典最小二乘（无先验时）")
print("4. 这正是 1.5 节'后验能量 = 数据项 + 正则项'中数据项的来源")
print("5. 若加入先验 p(x)（如高斯），则后验能量 = ||y - Ax||^2 + (sigma^2/sigma_x^2)||x||^2 = Tikhonov")

# ===== 保存数值结果 =====
import json
results_summary = {
    'signal_dimension': int(n),
    'noise_sigma': float(sigma),
    'proportionality_factor': float(round(factor, 6)),
    'max_relative_deviation': float(f"{max_rel_err:.3e}"),
    'verification_passed': bool(max_rel_err < 1e-10),
    'least_squares_at_true': float(round(ls_ideal, 4)),
    'neg_log_likelihood_at_true': float(round(nll_ideal, 4)),
    'ratio_nll_over_ls_at_true': float(round(nll_ideal / ls_ideal, 4)),
    'n_candidates': int(n_candidates),
}

def _to_native(obj):
    import numpy as np
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return _to_native(obj.tolist())
    try:
        import torch
        if isinstance(obj, torch.Tensor): return _to_native(obj.detach().cpu().tolist())
    except: pass
    return obj

results_summary = {k: _to_native(v) for k, v in results_summary.items()}
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"\n数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
