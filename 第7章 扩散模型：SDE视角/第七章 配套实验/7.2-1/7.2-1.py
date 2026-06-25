# -*- coding: utf-8 -*-
"""
实验7.2-1 正向SDE——VE-SDE与VP-SDE的加噪过程
对应章节: 7.2（正向SDE：从数据到噪声的连续过程）
素材来源:
  - 02-ddpm.ipynb的DDPM加噪代码
  - 04-sde.ipynb的VP-SDE连续加噪代码
  - ★ 原创设计：VE-SDE vs VP-SDE对比可视化

实验内容:
  步骤1: VP-SDE正向加噪——从数据到噪声的演化
  步骤2: VE-SDE正向加噪——方差爆炸 vs 方差保持
  步骤3: 两种SDE的SNR曲线与方差演化对比

运行前提: 纯NumPy/PyTorch CPU即可
"""

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys
import io
import warnings
import logging

# 设置控制台输出为 UTF-8 (Windows 下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# 静默 matplotlib 相关警告
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验7.2-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(_chinese_path, exist_ok=True)

# 在Colab或本地首次运行时自动创建chinese_font.py
_chinese_font_path = os.path.join(_chinese_path, 'chinese_font.py')
if not os.path.exists(_chinese_font_path):
    print("正在创建中文字体配置模块...")
    _chinese_font_code = '''# -*- coding: utf-8 -*-
"""
中文显示支持模块 - 兼容 Windows / Linux / Colab
"""
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
            print(f"[Font] 已加载缓存字体: Noto Sans SC")
            return 'Noto Sans SC'
        else:
            try:
                import urllib.request
                print(f"[Font] 正在下载中文字体 NotoSansSC...")
                urllib.request.urlretrieve(_font_url, _font_file)
                from matplotlib.font_manager import fontManager
                fontManager.addfont(_font_file)
                plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] + plt.rcParams.get('font.sans-serif', [])
                plt.rcParams['font.family'] = 'sans-serif'
                print(f"[Font] 已下载并注册中文字体: Noto Sans SC")
                return 'Noto Sans SC'
            except Exception as e:
                print(f"[Font] 字体下载失败: {e}")
    else:
        print("[Font] 未找到中文字体")
    return None

__all__ = ['setup_chinese_font']
'''
    with open(_chinese_font_path, 'w', encoding='utf-8') as f:
        f.write(_chinese_font_code)
    print(f"[Font] 已创建字体配置模块: {_chinese_font_path}")

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)
import torch
torch.manual_seed(42)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()


# ============================================================
# 步骤1：VP-SDE正向加噪
# 参考：02-ddpm.ipynb, 04-sde.ipynb
# ============================================================
print("=" * 60)
print("步骤1：VP-SDE正向加噪——从数据到噪声")
print("=" * 60)

# VP-SDE参数
beta_min, beta_max = 0.1, 20.0

def vp_sde_marginal_params(t, beta_min=0.1, beta_max=20.0):
    """VP-SDE的边际分布参数
    参考04-sde.ipynb Line 1772-1773
    x_t = mean_t * x_0 + std_t * epsilon
    """
    log_mean = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    mean_t = np.exp(log_mean)
    std_t = np.sqrt(1 - np.exp(2 * log_mean))
    return mean_t, std_t

# 在1D高斯混合上演示VP-SDE正向过程
def gm1d_sample(n):
    k = np.random.choice(2, p=[0.3, 0.7], size=n)
    means = np.array([-2, 1])
    return np.where(k == 0, np.random.randn(n) + means[0], np.random.randn(n) + means[1])

N = 5000
x0 = gm1d_sample(N)

# 不同时间步的加噪结果
t_values = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
vp_trajectories = []
for t in t_values:
    mean_t, std_t = vp_sde_marginal_params(t)
    epsilon = np.random.randn(N)
    x_t = mean_t * x0 + std_t * epsilon
    vp_trajectories.append(x_t)
    var_x = np.var(x_t)
    print(f"  t={t:.1f}: mean_t={mean_t:.4f}, std_t={std_t:.4f}, Var(x_t)={var_x:.4f}")

print(f"\nVP-SDE关键性质：Var(x_t) ≤ 1（方差保持）")


# ============================================================
# 步骤2：VE-SDE正向加噪
# 参考：03-smld.ipynb, 7.2节VE-SDE定义
# ============================================================
print("\n" + "=" * 60)
print("步骤2：VE-SDE正向加噪——方差爆炸")
print("=" * 60)

# VE-SDE参数
sigma_min, sigma_max = 0.01, 50.0

def ve_sde_sigma(t, sigma_min=0.01, sigma_max=50.0):
    """VE-SDE的σ(t)：几何插值"""
    return sigma_max * (sigma_min / sigma_max) ** t

# VE-SDE正向过程：x_t = x_0 + σ(t) * epsilon
ve_trajectories = []
for t in t_values:
    sigma_t = ve_sde_sigma(t)
    epsilon = np.random.randn(N)
    x_t = x0 + sigma_t * epsilon
    ve_trajectories.append(x_t)
    var_x = np.var(x_t)
    print(f"  t={t:.1f}: σ(t)={sigma_t:.4f}, Var(x_t)={var_x:.4f}")

print(f"\nVE-SDE关键性质：Var(x_t) → ∞（方差爆炸，当t→0时σ(t)→{ve_sde_sigma(0.0):.1f}）")


# ============================================================
# 步骤3：VE-SDE vs VP-SDE对比
# ★ 原创设计：系统对比两种SDE的性质差异
# ============================================================
print("\n" + "=" * 60)
print("步骤3：VE-SDE vs VP-SDE系统对比")
print("=" * 60)

t_grid = np.linspace(0, 1, 200)

# VP-SDE演化
vp_means = []
vp_stds = []
vp_vars = []
vp_snrs = []
for t in t_grid:
    m, s = vp_sde_marginal_params(t)
    vp_means.append(m)
    vp_stds.append(s)
    vp_vars.append(m**2 + s**2)  # Var(x_t) = m²Var(x_0) + s²
    vp_snrs.append(m**2 / s**2 if s > 1e-10 else float('inf'))

# VE-SDE演化
ve_sigmas = [ve_sde_sigma(t) for t in t_grid]
ve_vars = [1 + s**2 for s in ve_sigmas]  # Var(x_t) = Var(x_0) + σ(t)²
ve_snrs = [1.0 / s**2 if s > 1e-10 else float('inf') for s in ve_sigmas]

print(f"{'属性':<15s} | {'VE-SDE':<25s} | {'VP-SDE':<25s}")
print("-" * 70)
print(f"{'drift f(x,t)':<15s} | {'0':<25s} | {'-β(t)/2·x':<25s}")
print(f"{'diffusion g(t)':<15s} | {'√(d[σ²]/dt)':<25s} | {'√β(t)':<25s}")
print(f"{'discrete':<15s} | {'SMLD/NCSN':<25s} | {'DDPM':<25s}")
print(f"{'Var(x_t)':<15s} | {'→∞ (explode)':<25s} | {'≤1 (preserve)':<25s}")
print(f"{'signal scale':<15s} | {'no scaling':<25s} | {'√ᾱ_t decay':<25s}")
print(f"{'terminal':<15s} | {'large-var Gaussian':<25s} | {'N(0,I)':<25s}")

# Karras et al. (2022) 统一框架
print(f"\nKarras et al. (2022) 统一框架（7.2节）：")
print(f"  两种SDE可通过信号缩放s(t)统一")
print(f"  VE-SDE: s(t)=1, σ(t)从σ_min增长到σ_max")
print(f"  VP-SDE: s(t)=√ᾱ_t, σ(t)=(1-ᾱ_t)/ᾱ_t")


# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# VP-SDE加噪轨迹
for i, (t, xt) in enumerate(zip(t_values, vp_trajectories)):
    row, col = i // 3, i % 3
    axes[row, col].hist(xt, bins=50, density=True, alpha=0.6, color='blue')
    axes[row, col].set_title(f'VP-SDE t={t:.1f}')
    axes[row, col].set_xlim(-8, 8)
    axes[row, col].grid(alpha=0.3)

# VE-SDE加噪轨迹（单独图）
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
for i, (t, xt) in enumerate(zip(t_values, ve_trajectories)):
    row, col = i // 3, i % 3
    axes2[row, col].hist(xt, bins=50, density=True, alpha=0.6, color='red')
    axes2[row, col].set_title(f'VE-SDE t={t:.1f}')
    xlim = max(8, np.percentile(np.abs(xt), 99) * 1.2)
    axes2[row, col].set_xlim(-xlim, xlim)
    axes2[row, col].grid(alpha=0.3)

fig2.suptitle('VE-SDE正向加噪过程（注意方差爆炸）', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_VE-SDE加噪.png'), dpi=150, bbox_inches='tight')
plt.close()

# VP-SDE加噪图
fig.suptitle('VP-SDE正向加噪过程（方差保持≤1）', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_VP-SDE加噪.png'), dpi=150, bbox_inches='tight')
plt.close()

# 对比图
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))

# SNR对比
axes3[0].semilogy(t_grid, vp_snrs, 'b-', lw=2, label='VP-SDE SNR')
axes3[1].semilogy(t_grid, ve_snrs, 'r-', lw=2, label='VE-SDE SNR')
axes3[0].set_xlabel('t'); axes3[0].set_ylabel('SNR')
axes3[0].set_title('VP-SDE SNR随时间变化'); axes3[0].legend(); axes3[0].grid(alpha=0.3)
axes3[1].set_xlabel('t'); axes3[1].set_ylabel('SNR')
axes3[1].set_title('VE-SDE SNR随时间变化'); axes3[1].legend(); axes3[1].grid(alpha=0.3)

# 方差对比
axes3[2].plot(t_grid, vp_vars, 'b-', lw=2, label='VP-SDE Var(x_t)')
axes3[2].plot(t_grid, ve_vars, 'r-', lw=2, label='VE-SDE Var(x_t)')
axes3[2].axhline(y=1, color='k', linestyle='--', alpha=0.3)
axes3[2].set_xlabel('t'); axes3[2].set_ylabel('Var(x_t)')
axes3[2].set_title('方差演化：保持 vs 爆炸'); axes3[2].legend(); axes3[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_VE_vs_VP对比.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n图表已保存:")
print(f"  - 步骤1_VP-SDE加噪.png")
print(f"  - 步骤2_VE-SDE加噪.png")
print(f"  - 步骤3_VE_vs_VP对比.png")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验7.2-1 总结")
print("=" * 60)
print("1. VP-SDE正向加噪：x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε")
print("   - 信号逐渐衰减（√ᾱ_t→0），噪声逐渐增强")
print("   - Var(x_t) ≤ 1（方差保持），终态≈N(0,I)")
print("2. VE-SDE正向加噪：x_t = x_0 + σ(t)·ε")
print("   - 信号不缩放，纯噪声叠加")
print("   - Var(x_t) → ∞（方差爆炸），终态≈大方差高斯")
print("3. 两种SDE的SNR都单调递减——这是扩散模型的基本特性")
print("4. Karras统一框架：VE-SDE和VP-SDE通过信号缩放s(t)统一")