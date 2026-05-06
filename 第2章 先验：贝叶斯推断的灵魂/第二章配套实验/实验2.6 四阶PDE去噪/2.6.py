"""
实验2.6 四阶PDE去噪：避免阶梯效应
对应章节：附录2B TGV：TV的改进
知识点：四阶PDE→分段线性解；避免TV阶梯效应；TGV先验的直觉

素材来源：
  - IP27 (variational_formulations.md): 四阶PDE(双调和)去噪
  代码逐行取自 IP27 的 code-cell
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dia_matrix
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

# ====== 来源: IP27 (variational_formulations.md) ======

def getD(n, h):
    """四阶差分算子矩阵 (来源: IP27 variational_formulations.md)"""
    e = np.ones(n)
    D = (1 / h**4) * dia_matrix(
        (np.array([e, -4*e, 6*e, -4*e, e]),
         np.array([-2, -1, 0, 1, 2])),
        shape=(n, n)).toarray()

    # 边界修正 (IP27)
    D[0, :3] = [6/h**4, -8/h**4, 2/h**4]
    D[1, :4] = [-4/h**4, 7/h**4, -4/h**4, 1/h**4]
    D[-2, -4:] = [1/h**4, -4/h**4, 7/h**4, -4/h**4]
    D[-1, -3:] = [2/h**4, -8/h**4, 6/h**4]
    return D

# 参数 (IP27)
nx = 101
dx = 1 / (nx - 1)
dt = 1e-8
nt = 10001
sigma = 1e-1
alpha = 1e-2

# 算子 (IP27)
D = getD(nx, dx)

# ground truth 和含噪数据 (IP27)
x = np.linspace(0, 1, nx)
u_true = np.exp(-1e2 * (x - 0.5)**2)
f_delta = u_true + sigma * np.random.randn(nx)

# 四阶 PDE 前向 Euler 求解 (IP27)
u = np.zeros((nt, nx))
u[0] = f_delta

for k in range(nt - 1):
    u[k+1] = u[k] - dt * (u[k] + alpha * D @ u[k] - f_delta)

# ====== 可视化 (IP27 风格) ======
index = [0, nt-1]
fig, ax = plt.subplots(1, len(index), sharey=True, figsize=(12, 4))
for k in range(len(index)):
    ax[k].plot(x, u_true, 'k--', label='真实信号')
    ax[k].plot(x, u[index[k]], label=f'去噪结果')
    ax[k].set_title(f't = {index[k]*dt:.6f}')
    ax[k].set_xlabel('x')
    ax[k].set_aspect(1)
ax[0].set_ylabel('u(x)')
ax[0].legend()

fig.suptitle('四阶PDE去噪: 避免TV阶梯效应', fontsize=14)
plt.tight_layout()
plt.savefig('2_6_result.png', dpi=150, bbox_inches='tight')
plt.show()
