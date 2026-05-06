import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import logging

# ====== 解决中文乱码的核心代码（Windows + Linux 自动适配）======
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

import platform
from matplotlib.font_manager import FontManager, FontProperties

def _find_chinese_font():
    """自动检测系统中可用的中文字体，兼容 Windows / Linux"""
    candidates = []
    if platform.system() == 'Windows':
        candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
    else:
        candidates = [
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
            'Noto Sans CJK SC', 'Noto Sans CJK',
            'Source Han Sans SC', 'AR PL UMing CN',
            'SimHei',
        ]
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available:
            return font
    import os, re
    cjk_patterns = ['cjk', 'wqy', 'noto.*cjk', 'wenquan', 'chinese', 'simhei']
    for f in fm.ttflist:
        name_lower = f.name.lower()
        fname_lower = (os.path.basename(f.fname) if hasattr(f, 'fname') else '').lower()
        for pat in cjk_patterns:
            if re.search(pat, name_lower) or re.search(pat, fname_lower):
                return f.name
    return None

_cn_font = _find_chinese_font()
if _cn_font:
    plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['font.family'] = 'sans-serif'
    print(f"[Font] 已检测到中文字体: {_cn_font}")
else:
    print("[Font] 未找到中文字体，中文可能显示为方框")
# ========================================================

np.random.seed(42)

# ---- 1. 构造不同条件数的对角矩阵 ----
n = 100
x_true = np.random.randn(n)

# 对角矩阵：最大奇异值=1，最小奇异值=1/κ
# 条件数从 1 到 10^12
kappas = [1, 10, 1e2, 1e3, 1e4, 1e6, 1e8, 1e10, 1e12]
noise_level = 1e-6  # 数据中的噪声水平

rel_errors = []

for kappa in kappas:
    # 构造对角矩阵 A：σ_1=1, σ_n=1/κ, 中间线性插值
    singular_values = np.linspace(1, 1.0 / kappa, n)

    # 正问题：y = Ax
    y = singular_values * x_true

    # 加入噪声
    y_noisy = y + noise_level * np.random.randn(n)

    # 逆问题：x = A^{-1} y_noisy
    x_recon = y_noisy / singular_values

    # 相对误差
    rel_err = np.linalg.norm(x_recon - x_true) / np.linalg.norm(x_true)
    rel_errors.append(rel_err)

# ---- 2. 可视化 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 重建误差 vs 条件数
axes[0].loglog(kappas, rel_errors, 'o-', linewidth=2, markersize=8)
axes[0].loglog(kappas, [kappa * noise_level / np.linalg.norm(x_true) * np.linalg.norm(singular_values * x_true) / np.linalg.norm(x_true)
                         for kappa in kappas], '--', alpha=0.7, label='理论上界 κ·(δy/‖y‖)')
axes[0].set_xlabel('条件数 κ(A)')
axes[0].set_ylabel('重建相对误差 ‖x-x̂‖/‖x‖')
axes[0].set_title('条件数与误差放大\nκ 越大→噪声放大越严重')
axes[0].legend()
axes[0].grid(True, which='both')

# 奇异值衰减示意
for kappa in [1, 1e3, 1e6, 1e10]:
    sv = np.linspace(1, 1.0 / kappa, n)
    axes[1].semilogy(np.arange(1, n + 1), sv, label=f'κ={kappa:.0e}')
axes[1].set_xlabel('奇异值索引 i')
axes[1].set_ylabel('奇异值 σ_i')
axes[1].set_title('不同条件数的奇异值分布\n小 σ_i → 1/σ_i 爆炸 → 噪声放大')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('实验1_6_条件数.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"{'κ(A)':>12s}  {'相对误差':>12s}")
print("-" * 28)
for kappa, err in zip(kappas, rel_errors):
    print(f"{kappa:12.0e}  {err:12.4e}")