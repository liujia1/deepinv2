"""
实验2.4 Perona-Malik非线性扩散：保边去噪
对应章节：2.2 TV先验保边思想
知识点：非线性扩散；边缘保持；扩散系数在边缘处趋零

素材来源：
  - IP25 (variational_formulations.md): 线性扩散(热方程)去噪
  - IP26 (variational_formulations.md): Perona-Malik非线性扩散去噪
  代码逐行取自 IP25 + IP26 的 code-cell
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import random_noise
from skimage.transform import resize
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
# ========================================================

# ====== 来源: IP25 + IP26 (variational_formulations.md) ======

# 参数 (IP25/IP26)
sigma = 0.1
alpha = 1
dt = 1e-6
niter = 1001
n = 200

# 扩散算子 (IP25/IP26: 支持 c(s) = 1 或 c(s) = 1/(1+1e6*s))
def L(u, coeff=lambda s: 1):
    """扩散算子 (来源: IP25/IP26 variational_formulations.md)"""
    ue = np.pad(u, 1, mode='edge')

    # 扩散系数 (中心差分计算梯度模)
    grad_norm = ((ue[2:, 1:-1] - ue[:-2, 1:-1]) / (2 / n))**2 + \
                ((ue[1:-1, 2:] - ue[1:-1, :-2]) / (2 / n))**2
    c = np.pad(coeff(grad_norm), 1, mode='edge')

    # 扩散项 (前向+后向差分组合)
    uxx = ((c[1:-1, 1:-1] + c[2:, 1:-1]) * (ue[2:, 1:-1] - ue[1:-1, 1:-1]) -
           (c[:-2, 1:-1] + c[1:-1, 1:-1]) * (ue[1:-1, 1:-1] - ue[:-2, 1:-1])) / (2 / n**2)
    uyy = ((c[1:-1, 1:-1] + c[1:-1, 2:]) * (ue[1:-1, 2:] - ue[1:-1, 1:-1]) -
           (c[1:-1, :-2] + c[1:-1, 1:-1]) * (ue[1:-1, 1:-1] - ue[1:-1, :-2])) / (2 / n**2)

    return uxx + uyy

# 含噪图像 (IP25/IP26)
f = resize(data.camera(), (n, n))
f_delta = random_noise(f, var=sigma**2)

# 线性扩散 (IP25: coeff = lambda s: 1 + 0*s)
u_linear = np.zeros((n, n))
for k in range(niter - 1):
    u_linear = u_linear - dt * (u_linear - alpha * L(u_linear, lambda s: 1 + 0 * s)) + dt * f_delta

# Perona-Malik 扩散 (IP26: coeff = lambda s: 1/(1+1e6*s))
u_pm = np.zeros((n, n))
for k in range(niter - 1):
    u_pm = u_pm - dt * (u_pm - alpha * L(u_pm, lambda s: 1 / (1 + 1e6 * s))) + dt * f_delta

# ====== 可视化 (IP25/IP26 风格) ======
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(f_delta, cmap='gray')
ax[0].set_title('含噪图像')
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[1].imshow(u_linear, cmap='gray')
ax[1].set_title('线性扩散 (Tikhonov)')
ax[1].set_xticks([])
ax[1].set_yticks([])

ax[2].imshow(u_pm, cmap='gray')
ax[2].set_title('Perona-Malik (保边)')
ax[2].set_xticks([])
ax[2].set_yticks([])

plt.tight_layout()
plt.savefig('2_4_result.png', dpi=150, bbox_inches='tight')
plt.show()
