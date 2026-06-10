# -*- coding: utf-8 -*-
"""
实验5.6-2 置信区间与多样性分析
对应章节：5.6 近似理论与收敛保证 -> 5.6.5 置信区间 + 后验样本多样性与多峰性
知识点：
  - 95%置信区间：x_MMSE ± 1.96√Var(x|y)
  - 置信区间宽度反映局部不确定性
  - 后验样本展示分布的多峰性
  - 样本差异大的区域 = 高后验方差区域

前置实验：
  - 需要先运行实验5.5-1，生成sampling_results.npz

素材来源：
  - 实验5.3.py 不确定性量化部分（置信区间、多样本展示）
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
    SAVE_DIR = os.path.join(_gdrive, '实验5.6-2')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)

    # 在Colab中自动创建chinese_font.py
    _chinese_font_path = os.path.join(_chinese_path, 'chinese_font.py')
    if not os.path.exists(_chinese_font_path):
        print("正在创建中文字体配置模块...")
        _chinese_font_code = '''# -*- coding: utf-8 -*-
import os
import sys
import platform
import warnings
import logging
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

def _find_chinese_font():
    candidates = []
    if platform.system() == 'Windows':
        candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
    else:
        candidates = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans CJK', 'Source Han Sans SC', 'AR PL UMing CN', 'SimHei']
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available:
            return font
    import re
    cjk_patterns = ['cjk', 'wqy', 'noto.*cjk', 'wenquan', 'chinese', 'simhei']
    for f in fm.ttflist:
        name_lower = f.name.lower()
        fname_lower = (os.path.basename(f.fname) if hasattr(f, 'fname') else '').lower()
        for pat in cjk_patterns:
            if re.search(pat, name_lower) or re.search(pat, fname_lower):
                return f.name
    return None

def setup_chinese_font(save_dir=None):
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    _cn_font = _find_chinese_font()
    if _cn_font:
        plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
        plt.rcParams['font.family'] = 'sans-serif'
        print(f"[Font] 已检测到中文字体: {_cn_font}")
        return _cn_font
    if platform.system() != 'Windows':
        _font_url = 'https://github.com/jsntn/webfonts/raw/master/NotoSansSC-Regular.ttf'
        _font_file = os.path.join(save_dir, 'NotoSansSC-Regular.ttf')
        if os.path.exists(_font_file):
            from matplotlib.font_manager import fontManager
            fontManager.addfont(_font_file)
            plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] + plt.rcParams.get('font.sans-serif', [])
            plt.rcParams['font.family'] = 'sans-serif'
            return 'Noto Sans SC'
        else:
            try:
                import urllib.request
                urllib.request.urlretrieve(_font_url, _font_file)
                from matplotlib.font_manager import fontManager
                fontManager.addfont(_font_file)
                plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] + plt.rcParams.get('font.sans-serif', [])
                return 'Noto Sans SC'
            except: pass
    return None

__all__ = ['setup_chinese_font']
'''
        with open(_chinese_font_path, 'w', encoding='utf-8') as f:
            f.write(_chinese_font_code)
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
    print("警告: chinese_font 模块未找到")
# ========================================================

np.random.seed(42)


# ============================================================
# 加载采样结果
# ============================================================
print("=" * 60)
print("步骤1：加载采样结果")
print("=" * 60)

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
        _data_loaded = True
        break

if not _data_loaded:
    print("[Error] 未找到采样结果文件")
    print("[Info] 使用合成数据进行演示...")
    x_true = np.zeros((64, 64))
    x_true[20:40, 20:40] = 1.0
    post_mean = x_true + 0.05 * np.random.randn(*x_true.shape)
    post_var = 0.01 * np.ones_like(x_true)
    mc_samples = np.array([post_mean + np.sqrt(post_var) * np.random.randn(*post_mean.shape)
                          for _ in range(100)])  # 增加到100样本以提高统计可靠性

print(f"后验均值形状: {post_mean.shape}")
print(f"样本数量: {len(mc_samples)}")

# 样本数警告
if len(mc_samples) < 50:
    print(f"\n[警告] 样本数量 {len(mc_samples)} 过少，统计估计不可靠（建议≥100）")

# 从样本重新估计统计量（与5.6-1保持一致）
post_mean_recomputed = np.mean(mc_samples, axis=0)
post_var_recomputed = np.var(mc_samples, axis=0, ddof=1)

print(f"\n[验证] 从样本重新估计的统计量:")
print(f"  均值差异 (L2范数): {np.linalg.norm(post_mean - post_mean_recomputed):.6f}")
print(f"  方差差异 (L2范数): {np.linalg.norm(post_var - post_var_recomputed):.6f}")

# 使用重新估计的方差
post_var = post_var_recomputed
print(f"\n[Info] 使用从样本重新估计的后验方差进行后续分析")


# ============================================================
# 步骤2：计算置信区间
# ============================================================
print("\n" + "=" * 60)
print("步骤2：计算95%置信区间")
print("=" * 60)

post_std = np.sqrt(post_var)

# 方法1：正态近似（假设后验为正态分布）
z_95 = 1.96
q_low = post_mean - z_95 * post_std
q_high = post_mean + z_95 * post_std
ci_width = q_high - q_low  # 置信区间宽度

# 方法2：基于样本分位数（非参数方法，不假设正态）
q_low_empirical = np.percentile(mc_samples, 2.5, axis=0)
q_high_empirical = np.percentile(mc_samples, 97.5, axis=0)
ci_width_empirical = q_high_empirical - q_low_empirical

print(f"95%置信区间公式（正态近似）: $\\hat{{x}}_{{MMSE}} \\pm 1.96\\sqrt{{\\mathrm{{Var}}(x|y)}}$")
print(f"\n[注意] ±1.96σ 假设后验为正态分布，是近似结果")
print(f"       若后验非正态（如TV先验），应使用样本分位数方法")
print(f"\n正态近似置信区间宽度统计:")
print(f"  平均宽度: {np.mean(ci_width):.4f}")
print(f"  最大宽度: {np.max(ci_width):.4f}")
print(f"  最小宽度: {np.min(ci_width):.4f}")
print(f"\n样本分位数置信区间宽度统计（非参数）:")
print(f"  平均宽度: {np.mean(ci_width_empirical):.4f}")
print(f"  最大宽度: {np.max(ci_width_empirical):.4f}")
print(f"  最小宽度: {np.min(ci_width_empirical):.4f}")


# ============================================================
# 步骤3：可视化置信区间
# ============================================================
print("\n" + "=" * 60)
print("步骤3：可视化置信区间")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 第1行：后验均值、下界、上界
axes[0, 0].imshow(post_mean, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('后验均值 $\\hat{x}_{MMSE}$')
axes[0, 0].axis('off')

axes[0, 1].imshow(np.clip(q_low, 0, 1), cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title('95%置信区间下界')
axes[0, 1].axis('off')

axes[0, 2].imshow(np.clip(q_high, 0, 1), cmap='gray', vmin=0, vmax=1)
axes[0, 2].set_title('95%置信区间上界')
axes[0, 2].axis('off')

# 第2行：区间宽度、宽度直方图、高宽度区域
im_width = axes[1, 0].imshow(ci_width, cmap='hot')
axes[1, 0].set_title('置信区间宽度 $w = 3.92\\sqrt{\\mathrm{Var}}$')
axes[1, 0].axis('off')
plt.colorbar(im_width, ax=axes[1, 0], fraction=0.046, pad=0.04)

# 宽度直方图
unique_widths = np.unique(ci_width)
if len(unique_widths) > 10:
    n_bins = 50
else:
    n_bins = min(10, len(unique_widths))

if n_bins > 1:
    axes[1, 1].hist(ci_width.flatten(), bins=n_bins, color='coral', alpha=0.7, edgecolor='white')
    axes[1, 1].axvline(np.mean(ci_width), color='red', linestyle='--',
                       label=f'均值：{np.mean(ci_width):.4f}')
    axes[1, 1].set_xlabel('置信区间宽度')
    axes[1, 1].set_ylabel('像素数量')
    axes[1, 1].set_title('宽度分布')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
else:
    axes[1, 1].text(0.5, 0.5, f'宽度恒定：{ci_width[0,0]:.4f}', ha='center', va='center',
                    transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_title('宽度分布（常数）')
    axes[1, 1].axis('off')

# 高宽度区域
threshold = np.percentile(ci_width, 90)
high_width = ci_width > threshold
axes[1, 2].imshow(high_width, cmap='Reds')
axes[1, 2].set_title(f'高宽度区域 (>P90={threshold:.4f})')
axes[1, 2].axis('off')

fig.suptitle('实验5.6-2 置信区间分析', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_置信区间分析.png'), dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 步骤4：后验样本多样性展示
# ============================================================
print("\n" + "=" * 60)
print("步骤4：后验样本多样性展示")
print("=" * 60)

num_show = min(16, len(mc_samples))
step = max(1, len(mc_samples) // num_show)
selected = mc_samples[::step][:num_show]

ncols = 4
nrows = (num_show + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4*nrows))

for i in range(nrows * ncols):
    ax = axes[i // ncols][i % ncols] if nrows > 1 else axes[i]
    if i < len(selected):
        ax.imshow(selected[i], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'样本 {i+1}')
    ax.axis('off')

fig.suptitle('PnP-ULA后验样本（展示后验分布的多样性）', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤4_后验样本多样性.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"展示了 {len(selected)} 个后验样本")
print("\n观察要点:")
print("  - 共享相似的全局结构（先验约束）")
print("  - 细节存在差异（数据一致性与先验的权衡）")
print("  - 差异区域 = 高不确定性区域")


# ============================================================
# 步骤5：样本差异分析
# ============================================================
print("\n" + "=" * 60)
print("步骤5：样本差异分析")
print("=" * 60)

if len(mc_samples) >= 2:
    # 计算样本间差异
    sample_std = np.std(mc_samples, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 样本标准差
    im_sample_std = axes[0].imshow(sample_std, cmap='hot')
    axes[0].set_title('样本标准差（多样性度量）')
    axes[0].axis('off')
    plt.colorbar(im_sample_std, ax=axes[0], fraction=0.046, pad=0.04)

    # 与后验标准差对比（随机采样避免空间相关性偏差）
    post_std_flat = post_std.flatten()
    sample_std_flat = sample_std.flatten()
    n_samples_scatter = min(500, len(post_std_flat))
    idx = np.random.choice(len(post_std_flat), size=n_samples_scatter, replace=False)
    axes[1].scatter(post_std_flat[idx], sample_std_flat[idx], alpha=0.3, s=1)
    axes[1].plot([0, max(post_std.max(), sample_std.max())],
                [0, max(post_std.max(), sample_std.max())], 'r--', label='y=x')
    axes[1].set_xlabel('后验标准差')
    axes[1].set_ylabel('样本标准差')
    axes[1].set_title('后验标准差 vs 样本标准差')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 差异图（相邻样本）
    diff = np.abs(mc_samples[0] - mc_samples[1])
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('样本1与样本2的差异')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤5_样本差异分析.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 相关性
    corr = np.corrcoef(post_std.flatten(), sample_std.flatten())[0, 1]
    print(f"后验标准差与样本标准差相关系数: {corr:.3f}")
    print("解读: 相关系数接近1说明样本差异与不确定性一致")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验5.6-2 总结")
print("=" * 60)
print("1. 95%置信区间: $\\hat{x}_{MMSE} \\pm 1.96\\sqrt{\\mathrm{Var}(x|y)}$")
print("2. 置信区间宽度直观反映局部不确定性")
print("3. 后验样本展示分布的多峰性:")
print("   - 共享全局结构（先验约束）")
print("   - 细节存在差异（多峰后验）")
print("4. 样本差异大的区域 = 高后验方差区域")
print("5. 置信区间可用于下游决策（如医学影像中的专家复核区域）")
