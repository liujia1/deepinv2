# -*- coding: utf-8 -*-
"""
实验5.6-3 MAP vs MMSE综合对比
对应章节：5.6 近似理论与收敛保证 -> 5.6.5 MAP vs MMSE + 不确定性量化的实践指导意义
知识点：
  - MAP估计：后验众数，"最可能"的重建
  - MMSE估计：后验均值，"平均意义下最好"的重建
  - MMSE更平滑，MAP保留更多细节
  - 不确定性量化的4个应用场景

前置实验：
  - 需要先运行实验5.5-1，生成sampling_results.npz

素材来源：
  - 实验5.3.py 实验总结部分
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
    SAVE_DIR = os.path.join(_gdrive, '实验5.6-3')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)

    # 在Colab中自动创建chinese_font.py
    _chinese_font_path = os.path.join(_chinese_path, 'chinese_font.py')
    if not os.path.exists(_chinese_font_path):
        print("正在创建中文字体配置模块...")
        _chinese_font_code = '''# -*- coding: utf-8 -*-
import os, sys, platform, warnings, logging
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

def _find_chinese_font():
    candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong'] if platform.system() == 'Windows' else ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans CJK', 'Source Han Sans SC', 'AR PL UMing CN', 'SimHei']
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available: return font
    import re
    for f in fm.ttflist:
        for pat in ['cjk', 'wqy', 'noto.*cjk', 'wenquan', 'chinese', 'simhei']:
            if re.search(pat, f.name.lower()): return f.name
    return None

def setup_chinese_font(save_dir=None):
    if save_dir is None: save_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
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
            return 'Noto Sans SC'
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
    os.makedirs(SAVE_DIR, exist_ok=True)
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
# 辅助函数：图像质量指标
# 注意：这是简化版实现，用于相对比较
# 与sampling_tools中的实现可能有数值差异
# ============================================================
def compute_psnr(x, y):
    """计算PSNR"""
    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return float('inf')
    max_val = max(x.max(), 1.0)
    return 10 * np.log10(max_val ** 2 / mse)

def compute_global_similarity(x, y):
    """
    全局相似度指标（简化版，用于相对比较）
    
    注意：这不是标准的局部窗口SSIM，仅用于对比不同方法的相对性能。
    标准SSIM需要使用skimage.metrics.structural_similarity。
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return ssim

def compute_nrmse(x, y):
    """计算NRMSE"""
    std_x = np.std(x)
    if std_x == 0:
        return float('inf')
    return np.sqrt(np.mean((x - y) ** 2)) / std_x


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
        y_obs = data['y_obs']
        _data_loaded = True
        break

if not _data_loaded:
    print("[Error] 未找到采样结果文件")
    print("[Info] 使用合成数据进行演示...")
    x_true = np.zeros((64, 64))
    x_true[20:40, 20:40] = 1.0
    x_true[10:20, 40:50] = 0.8
    y_obs = x_true + 0.1 * np.random.randn(*x_true.shape)
    post_mean = x_true + 0.05 * np.random.randn(*x_true.shape)
    post_var = 0.01 * np.ones_like(x_true)
    mc_samples = np.array([post_mean + np.sqrt(post_var) * np.random.randn(*post_mean.shape)
                          for _ in range(100)])  # 100个样本用于统计

# 从样本重新估计方差（无论真实数据还是合成数据，统一重估）
post_var = np.var(mc_samples, axis=0, ddof=1)

print(f"后验均值形状: {post_mean.shape}")
print(f"样本数量: {len(mc_samples)}")


# ============================================================
# 步骤2：MAP vs MMSE 对比
# ============================================================
print("\n" + "=" * 60)
print("步骤2：MAP vs MMSE 对比")
print("=" * 60)

# MMSE估计 = 后验均值
x_mmse = post_mean

# MAP估计说明：
# ULA是朗之万采样算法，其样本是从后验分布中随机抽取的，
# 每个样本都是后验的一个代表，而不是后验众数。
# 从ULA样本中无法得到真正的MAP估计（后验众数），
# 因为距离后验均值最近的样本本质上也趋向于后验均值本身，
# 导致MMSE与MAP的对比失去教学意义。
# 在实际PnP-ULA中，MAP通常通过PnP-ADMM获得。
# 这里用含噪观测作为对比基准，展示贝叶斯推断（MMSE）相对于无正则化重建的价值。
x_map = y_obs
print("[注意] 此处用含噪观测代替MAP作为对比基准，展示正则化的效果")
print("       在实际PnP-ULA中，MAP通常通过PnP-ADMM获得")

# 计算质量指标
psnr_mmse = compute_psnr(x_true, x_mmse)
psnr_map = compute_psnr(x_true, x_map)
sim_mmse = compute_global_similarity(x_true, x_mmse)
sim_map = compute_global_similarity(x_true, x_map)
nrmse_mmse = compute_nrmse(x_true, x_mmse)
nrmse_map = compute_nrmse(x_true, x_map)

print(f"\nMMSE估计（后验均值）:")
print(f"  PSNR: {psnr_mmse:.2f} dB")
print(f"  全局相似度: {sim_mmse:.4f} (全局近似，非标准SSIM)")
print(f"  NRMSE: {nrmse_mmse:.4f}")

print(f"\n含噪观测（作为对比基准）:")
print(f"  PSNR: {psnr_map:.2f} dB")
print(f"  全局相似度: {sim_map:.4f} (全局近似，非标准SSIM)")
print(f"  NRMSE: {nrmse_map:.4f}")


# ============================================================
# 步骤3：可视化对比
# ============================================================
print("\n" + "=" * 60)
print("步骤3：可视化对比")
print("=" * 60)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# 第1行：原始、观测、MMSE、MAP
axes[0, 0].imshow(x_true, cmap='gray')
axes[0, 0].set_title('原始图像 $x$')
axes[0, 0].axis('off')

axes[0, 1].imshow(y_obs, cmap='gray')
axes[0, 1].set_title('含噪观测 $y$')
axes[0, 1].axis('off')

axes[0, 2].imshow(x_mmse, cmap='gray', vmin=0, vmax=1)
axes[0, 2].set_title(f'MMSE估计 (PSNR: {psnr_mmse:.2f}dB)')
axes[0, 2].axis('off')

axes[0, 3].imshow(x_map, cmap='gray', vmin=0, vmax=1)
axes[0, 3].set_title(f'含噪观测 (PSNR: {psnr_map:.2f}dB)')
axes[0, 3].axis('off')

# 第2行：误差图、差异图、平滑度对比
error_mmse = np.abs(x_true - x_mmse)
error_map = np.abs(x_true - x_map)
diff = np.abs(x_mmse - x_map)

axes[1, 0].imshow(error_mmse, cmap='hot')
axes[1, 0].set_title('MMSE误差')
axes[1, 0].axis('off')

axes[1, 1].imshow(error_map, cmap='hot')
axes[1, 1].set_title('含噪观测误差')
axes[1, 1].axis('off')

axes[1, 2].imshow(diff, cmap='hot')
axes[1, 2].set_title('MMSE与含噪观测的差异')
axes[1, 2].axis('off')

# 平滑度对比（局部放大：MMSE vs MAP）
h, w = x_true.shape
zoom_size = min(32, h//2, w//2)
zoom_region = (h//2-zoom_size//2, h//2+zoom_size//2, w//2-zoom_size//2, w//2+zoom_size//2)

# 在axes[1, 3]中创建两个子图展示MMSE和MAP的局部放大
axes[1, 3].axis('off')  # 先关闭主轴
# 创建两个子图
ax_mmse_zoom = axes[1, 3].inset_axes([0, 0, 0.48, 1])
ax_map_zoom = axes[1, 3].inset_axes([0.52, 0, 0.48, 1])

ax_mmse_zoom.imshow(x_mmse[zoom_region[0]:zoom_region[1], zoom_region[2]:zoom_region[3]], cmap='gray')
ax_mmse_zoom.set_title('MMSE', fontsize=10)
ax_mmse_zoom.axis('off')

ax_map_zoom.imshow(x_map[zoom_region[0]:zoom_region[1], zoom_region[2]:zoom_region[3]], cmap='gray')
ax_map_zoom.set_title('含噪观测', fontsize=10)
ax_map_zoom.axis('off')

axes[1, 3].set_title('局部放大对比', fontsize=11)

fig.suptitle('实验5.6-3 MMSE vs 含噪观测对比', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_MAP_vs_MMSE对比.png'), dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 步骤4：两种估计哲学的对比
# ============================================================
print("\n" + "=" * 60)
print("步骤4：两种估计哲学的对比")
print("=" * 60)

print("\nMAP估计（第3-5章优化路径）:")
print("  $\\hat{x}_{MAP} = \\arg\\max_x \\log p(x|y) = \\arg\\min_x \\{f(x) + g(x)\\}$")
print("  - 给出'最可能的'重建——后验分布的众数")
print("  - 是点估计：只返回一个值，不提供置信信息")
print("  - PnP-ADMM、近端梯度下降都属于这一路径")

print("\nMMSE估计（第5章采样路径）:")
print("  $\\hat{x}_{MMSE} = E[x|y] = \\int x\\,p(x|y)\\,dx$")
print("  - 给出'平均意义下最好'的重建——后验分布的均值")
print("  - 是估计+不确定性：均值是点估计，方差是置信度")
print("  - PnP-ULA、MALA等采样算法都属于这一路径")

print("\n关键洞察:")
print("  后验均值天然平滑多峰不确定性")
print("  当后验是多峰的，MAP只取最高峰，MMSE平均所有峰")
print("  这就是为什么MMSE通常视觉上更平滑")


# ============================================================
# 步骤5：不确定性量化的实践指导意义
# ============================================================
print("\n" + "=" * 60)
print("步骤5：不确定性量化的实践指导意义")
print("=" * 60)

# 计算后验标准差
post_std = np.sqrt(post_var)

# 创建不确定性叠加可视化（模拟医学影像应用场景）
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 左图：MMSE重建结果
axes[0].imshow(x_mmse, cmap='gray', vmin=0, vmax=1)
axes[0].set_title('MMSE重建结果')
axes[0].axis('off')

# 中图：不确定性热图
im1 = axes[1].imshow(post_std, cmap='hot')
axes[1].set_title('不确定性（后验标准差）')
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# 右图：不确定性叠加 + 高不确定区域标注
# 将不确定性归一化到[0,1]用于叠加
uncertainty_norm = (post_std - post_std.min()) / (post_std.max() - post_std.min() + 1e-10)
# 创建RGB叠加图
overlay = np.zeros((*x_mmse.shape, 3))
overlay[:, :, 0] = x_mmse  # 灰度底图
overlay[:, :, 1] = x_mmse
overlay[:, :, 2] = x_mmse
# 在高不确定性区域叠加红色
threshold = np.percentile(uncertainty_norm, 90)  # 前10%高不确定性
high_uncertainty_mask = uncertainty_norm > threshold
overlay[high_uncertainty_mask, 0] = 1.0  # 红色
overlay[high_uncertainty_mask, 1] = 0.3
overlay[high_uncertainty_mask, 2] = 0.3

axes[2].imshow(overlay)
axes[2].set_title(f'不确定性叠加（红色=高不确定区域，占{np.mean(high_uncertainty_mask)*100:.1f}%）')
axes[2].axis('off')

fig.suptitle('不确定性量化在医学影像中的应用示意', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤5_实践指导意义.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n应用场景说明:")
print("  医学影像：高不确定性区域（红色）应标记为'需医生复核'")
print("  遥感图像：不确定性图指导采样策略——高不确定区域增加采样")
print("  科学计算：不确定性量化是实验结果可信度的核心指标")
print("  主动学习：选择高不确定性样本进行主动标注或重新测量")


# ============================================================
# 步骤6：参数敏感性分析
# ============================================================
print("\n" + "=" * 60)
print("步骤6：参数敏感性分析")
print("=" * 60)

# 使用合成数据演示参数对不确定性估计的影响
# 模拟不同噪声水平ε对后验方差的影响
print("\n模拟参数敏感性实验...")

# 基准参数
base_var = np.mean(post_var)  # 使用实际数据的平均方差作为基准

# 实验1：噪声水平ε的影响（ε越大，后验方差越大）
epsilon_values = np.array([0.01, 0.02, 0.05, 0.1, 0.2])
# 后验方差与噪声水平的关系：var ∝ ε²（简化模型）
var_vs_epsilon = base_var * (epsilon_values / 0.05) ** 2  # 以0.05为基准

# 实验2：样本数M的影响（M越大，方差估计越稳定）
M_values = np.array([10, 20, 50, 100, 200, 500])
# 方差估计的标准误差 ∝ 1/√M
std_error_vs_M = base_var * 0.5 / np.sqrt(M_values)

# 实验3：步长δ的影响（δ过小导致有效样本减少）
delta_values = np.array([0.001, 0.005, 0.01, 0.05, 0.1])
# 有效样本数与步长的关系（简化模型）
effective_samples = 100 * delta_values / 0.01  # 以0.01为基准
effective_samples = np.clip(effective_samples, 5, 200)  # 限制范围
var_stability = base_var * (1 + 0.5 / np.sqrt(effective_samples))

# 创建可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 子图1：噪声水平ε的影响
axes[0].plot(epsilon_values, var_vs_epsilon, 'o-', color='#2E86AB', linewidth=2, markersize=8)
axes[0].set_xlabel('噪声水平 ε')
axes[0].set_ylabel('平均后验方差')
axes[0].set_title('ε对后验方差的影响（理论示意）')
axes[0].grid(True, alpha=0.3)
axes[0].annotate('ε↑ → 先验约束弱 → 方差↑', xy=(0.7, 0.7), xycoords='axes fraction',
                fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 子图2：样本数M的影响
axes[1].plot(M_values, std_error_vs_M, 's-', color='#A23B72', linewidth=2, markersize=8)
axes[1].set_xlabel('样本数 M')
axes[1].set_ylabel('方差估计标准误差')
axes[1].set_title('M对估计稳定性的影响（理论示意）')
axes[1].grid(True, alpha=0.3)
axes[1].annotate('M↑ → 估计更稳定', xy=(0.7, 0.7), xycoords='axes fraction',
                fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 子图3：步长δ的影响
axes[2].plot(delta_values, var_stability, '^-', color='#F18F01', linewidth=2, markersize=8)
axes[2].set_xlabel('步长 δ')
axes[2].set_ylabel('方差估计波动')
axes[2].set_title('δ对估计稳定性的影响（理论示意）')
axes[2].grid(True, alpha=0.3)
axes[2].annotate('δ过小 → 有效样本少 → 不稳定', xy=(0.25, 0.8), xycoords='axes fraction',
                fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle('关键参数对不确定性估计的影响（理论示意）', fontsize=14, y=1.02)
fig.text(0.5, 0.01, '注：以上曲线为理论模型示意，非真实实验数据', ha='center', fontsize=9, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤6_参数敏感性.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n参数敏感性结论:")
print("  1. ε（去噪器噪声水平）: ε越大 → 先验约束越弱 → 后验方差越大")
print("  2. M（迭代次数/样本数）: M越大 → 方差估计越稳定（标准误差 ∝ 1/√M）")
print("  3. δ（步长）: δ过小 → burn-in长 → 有效样本少 → 估计不稳定")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验5.6-3 总结")
print("=" * 60)
print("1. MAP估计: 后验众数，'最可能'的重建，保留更多细节")
print("2. MMSE估计: 后验均值，'平均意义下最好'的重建，更平滑")
print("3. 两种估计的PSNR/SSIM差异通常不大，但视觉特征不同")
print("4. 不确定性量化在医学、遥感、科学计算、主动学习中有重要应用")
print("5. 参数选择（ε、δ、M）影响不确定性估计的质量")
print("\n来源: 实验5.3.py; Terris et al. (2022); Laumont et al. (2021)")
