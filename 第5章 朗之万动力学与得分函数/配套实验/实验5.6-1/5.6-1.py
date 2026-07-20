# -*- coding: utf-8 -*-
"""
实验5.6-1 后验标准差与不确定性图
对应章节：5.6 近似理论与收敛保证 -> 5.6.5 像素级不确定性：后验方差
知识点：
  - 后验方差估计：Var(x_i) = 1/(M-B) Σ(X_m^(i) - x_MMSE^(i))^2
  - 高方差区域 = 信息丢失区域
  - 低方差区域 = 数据约束强

前置实验：
  - 需要先运行实验5.5-1，生成sampling_results.npz

素材来源：
  - Mathematics.../Teaching Unit 2/labs/lab2_PnP_sol.ipynb
    - Cell 18-20: 结果评估与可视化
  - 实验5.3.py 不确定性量化部分
"""

import torch
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
    SAVE_DIR = os.path.join(_gdrive, '实验5.6-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)
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
    print("警告: chinese_font 模块未找到")
# ========================================================

np.random.seed(42)


# ============================================================
# 加载采样结果
# ============================================================
print("=" * 60)
print("步骤1：加载采样结果")
print("=" * 60)

# 尝试从多个位置加载
_possible_paths = [
    os.path.join(SAVE_DIR, 'sampling_results.npz'),
    os.path.join(os.path.dirname(SAVE_DIR), '实验5.5-1', 'sampling_results.npz'),
]

if _IN_COLAB:
    _possible_paths.insert(0, os.path.join(_gdrive, '实验5.5-1', 'sampling_results.npz'))

_data_loaded = False
for _path in _possible_paths:
    if os.path.exists(_path):
        print(f"[Data] 从 {_path} 加载采样结果")
        data = np.load(_path)
        post_mean = data['post_mean']
        post_var = data['post_var']
        mc_samples = data['mc_samples']
        x_true = data['x_true']
        y_obs = data['y_obs']
        _data_loaded = True
        break

if not _data_loaded:
    print("[Error] 未找到采样结果文件 sampling_results.npz")
    print("[Error] 请先运行实验5.5-1生成采样结果")
    print("\n[Info] 将使用合成数据进行演示...")

    # 创建合成数据
    x_true = np.zeros((64, 64))
    x_true[20:40, 20:40] = 1.0
    x_true[10:20, 40:50] = 0.8

    # 模拟后验统计
    post_mean = x_true + 0.05 * np.random.randn(*x_true.shape)
    post_var = 0.01 * np.ones_like(x_true)
    post_var[20:40, 20:40] = 0.02  # 目标区域内部不确定性更高

    # 模拟样本
    mc_samples = np.array([post_mean + np.sqrt(post_var) * np.random.randn(*post_mean.shape)
                          for _ in range(10)])

    y_obs = x_true + 0.1 * np.random.randn(*x_true.shape)

print(f"后验均值形状: {post_mean.shape}")
print(f"后验方差形状: {post_var.shape}")
print(f"样本数量: {len(mc_samples)}")

# 从样本重新估计统计量（验证采样结果的核心操作）
post_mean_recomputed = np.mean(mc_samples, axis=0)
post_var_recomputed = np.var(mc_samples, axis=0, ddof=1)

print(f"\n[验证] 从样本重新估计的统计量:")
print(f"  均值差异 (L2范数): {np.linalg.norm(post_mean - post_mean_recomputed):.6f}")
print(f"  方差差异 (L2范数): {np.linalg.norm(post_var - post_var_recomputed):.6f}")

# 使用重新估计的方差（教学意义：展示如何从样本计算后验统计量）
post_var = post_var_recomputed
print(f"\n[Info] 使用从样本重新估计的后验方差进行后续分析")


# ============================================================
# 步骤2：计算后验标准差
# ============================================================
print("\n" + "=" * 60)
print("步骤2：计算后验标准差")
print("=" * 60)

post_std = np.sqrt(post_var)

print(f"后验标准差统计:")
print(f"  平均值: {np.mean(post_std):.4f}")
print(f"  最大值: {np.max(post_std):.4f}")
print(f"  最小值: {np.min(post_std):.4f}")
print(f"  中位数: {np.median(post_std):.4f}")


# ============================================================
# 步骤3：可视化不确定性
# ============================================================
print("\n" + "=" * 60)
print("步骤3：可视化不确定性")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 第1行：原始、观测、后验均值
axes[0, 0].imshow(x_true, cmap='gray')
axes[0, 0].set_title('原始图像 $x$')
axes[0, 0].axis('off')

axes[0, 1].imshow(y_obs, cmap='gray')
axes[0, 1].set_title('含噪观测 $y$')
axes[0, 1].axis('off')

axes[0, 2].imshow(post_mean, cmap='gray', vmin=0, vmax=1)
axes[0, 2].set_title('后验均值 $\\hat{x}_{MMSE}$')
axes[0, 2].axis('off')

# 第2行：后验标准差、直方图、高不确定性区域
im_std = axes[1, 0].imshow(post_std, cmap='hot')
axes[1, 0].set_title('后验标准差 $\\sqrt{\\mathrm{Var}(x|y)}$')
axes[1, 0].axis('off')
plt.colorbar(im_std, ax=axes[1, 0], fraction=0.046, pad=0.04)

# 标准差直方图
axes[1, 1].hist(post_std.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='white')
axes[1, 1].axvline(np.mean(post_std), color='red', linestyle='--', label=f'均值: {np.mean(post_std):.4f}')
axes[1, 1].set_xlabel('后验标准差')
axes[1, 1].set_ylabel('像素数量')
axes[1, 1].set_title('不确定性分布')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# 高不确定性区域标记（超过均值+1倍标准差）
threshold = np.mean(post_std) + np.std(post_std)
high_uncertainty = post_std > threshold
axes[1, 2].imshow(high_uncertainty, cmap='Reds')
axes[1, 2].set_title(f'高不确定性区域 (>{threshold:.4f})')
axes[1, 2].axis('off')

fig.suptitle('实验5.6-1 后验标准差与不确定性图', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_后验标准差与不确定性.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n高不确定性像素占比: {np.sum(high_uncertainty) / high_uncertainty.size * 100:.1f}%")


# ============================================================
# 步骤4：不确定性与重建误差的关系
# ============================================================
print("\n" + "=" * 60)
print("步骤4：不确定性与重建误差的关系")
print("=" * 60)

# 计算重建误差
error = np.abs(x_true - post_mean)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 误差图
im_err = axes[0].imshow(error, cmap='hot')
axes[0].set_title('重建误差 $|x - \\hat{x}_{MMSE}|$')
axes[0].axis('off')
plt.colorbar(im_err, ax=axes[0], fraction=0.046, pad=0.04)

# 散点图：不确定性 vs 误差（随机采样避免空间相关性偏差）
post_std_flat = post_std.flatten()
error_flat = error.flatten()
n_samples_scatter = min(500, len(post_std_flat))  # 采样点数
idx = np.random.choice(len(post_std_flat), size=n_samples_scatter, replace=False)
axes[1].scatter(post_std_flat[idx], error_flat[idx], alpha=0.3, s=1)
axes[1].set_xlabel('不确定性（标准差）')
axes[1].set_ylabel('重建误差')
axes[1].set_title('不确定性 vs 误差')
corr = np.corrcoef(post_std.flatten(), error.flatten())[0, 1]
axes[1].annotate(f'相关系数: $r$ = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1].grid(alpha=0.3)

# 分区域统计
# 按不确定性分位数分组
n_bins = 5
uncertainty_bins = np.percentile(post_std.flatten(), np.linspace(0, 100, n_bins+1))
bin_means = []
bin_labels = []
for i in range(n_bins):
    mask = (post_std >= uncertainty_bins[i]) & (post_std < uncertainty_bins[i+1])
    if np.sum(mask) > 0:
        bin_means.append(np.mean(error[mask]))
        bin_labels.append(f'Q{i+1}')

if len(bin_means) > 0:
    axes[2].bar(range(len(bin_means)), bin_means, color='steelblue', alpha=0.7)
    axes[2].set_xlabel('不确定性分位数')
    axes[2].set_ylabel('平均重建误差')
    axes[2].set_title('分位数统计')
    axes[2].set_xticks(range(len(bin_means)))
    axes[2].set_xticklabels(bin_labels)
    axes[2].grid(alpha=0.3, axis='y')
else:
    axes[2].text(0.5, 0.5, '数据不足', ha='center', va='center', transform=axes[2].transAxes)
    axes[2].set_title('分位数统计')
    axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤4_不确定性与误差关系.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"不确定性-误差相关系数: {corr:.3f}")
print("解读: 相关系数越高，说明后验方差是可靠的误差代理指标")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验5.6-1 总结")
print("=" * 60)
print("1. 后验标准差提供像素级的不确定性度量")
print("2. 高方差区域对应信息丢失区域（模糊核导致的频率缺失）")
print("3. 低方差区域对应数据约束强的区域")
print("4. 不确定性与重建误差正相关，是可靠的误差代理指标")


# ===== 保存数值结果 =====
import json

def _to_native(obj):
    """递归转换numpy/torch类型为Python原生类型"""
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

results_summary = {
    "实验名称": "实验5.6-1 后验标准差与不确定性图",
    "数据信息": {
        "后验均值形状": list(post_mean.shape),
        "样本数量": int(len(mc_samples)),
        "均值差异_L2范数": round(float(np.linalg.norm(post_mean - post_mean_recomputed)), 8),
        "方差差异_L2范数": round(float(np.linalg.norm(post_var - post_var_recomputed)), 8),
    },
    "步骤2_后验标准差统计": {
        "平均值": round(float(np.mean(post_std)), 6),
        "最大值": round(float(np.max(post_std)), 6),
        "最小值": round(float(np.min(post_std)), 6),
        "中位数": round(float(np.median(post_std)), 6),
    },
    "步骤3_不确定性": {
        "高不确定性阈值": round(float(threshold), 6),
        "高不确定性像素占比_百分比": round(float(np.sum(high_uncertainty) / high_uncertainty.size * 100), 4),
    },
    "步骤4_不确定性与误差关系": {
        "相关系数": round(float(corr), 6),
    },
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
