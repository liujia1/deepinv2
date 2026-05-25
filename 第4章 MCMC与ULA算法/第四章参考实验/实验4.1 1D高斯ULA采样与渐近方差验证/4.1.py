"""
实验4.1 1D高斯ULA采样与渐近方差验证
对应章节：4.3（ULA递推式、渐近方差、偏差）
素材来源：Mathematics.../Teaching Unit 2/labs/lab1_ULA_sol.ipynb
  - ULA_gauss() 函数
  - var_compare() 函数
  - 1D实验代码（ULA直方图 vs 真实密度）
"""

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

# ============================================================
# ULA_gauss 函数（取自 lab1_ULA_sol.ipynb）
# 从1D标准高斯 N(0,1) 中用ULA采样
# 势能 U(x)=x²/2, ∇U(x)=x
# ULA: X_{m+1} = X_m - δ·∇U(X_m) + √(2δ)·Z_{m+1}
#                  = (1-δ)·X_m + √(2δ)·Z_{m+1}
# ============================================================
def ULA_gauss(niter, delta, x0=0):
    Y = np.zeros(niter,)
    X = x0
    for i in range(niter):
        Z = np.random.randn()
        grad = X
        X = X - delta*grad + np.sqrt(2*delta)*Z
        Y[i] = X
    return Y, np.var(Y)


# ============================================================
# 实验1：ULA直方图 vs 真实密度（取自 lab1_ULA_sol 的1D实验代码）
# ============================================================
x0 = 0
niter = int(1e5)
delta = 0.1

Y, _ = ULA_gauss(niter, delta, x0)

# 显示高斯分布（蓝色曲线）
x = np.linspace(-3, 3, 100)
y = 1/(np.sqrt(2*np.pi))*np.exp(-x**2 / 2)
plt.plot(x, y, 'b-', linewidth=2, label='真实密度 $\\mathcal{N}(0,1)$')

# 显示ULA采样直方图
plt.hist(Y, bins=100, range=[-3, 3], density=1, alpha=0.6, label='ULA采样')
plt.xlabel('$x$')
plt.ylabel('密度')
plt.title(f'ULA采样 vs 真实密度（δ={delta}）')
plt.legend()
plt.savefig('实验4_1_ULA直方图.png', dpi=150)
plt.show()


# ============================================================
# 实验2：渐近方差验证（取自 lab1_ULA_sol 的 var_compare 函数）
# 理论渐近方差 Var_ULA = 2/(2-δ)
# ============================================================
def var_compare(delta):
    _, var1 = ULA_gauss(int(1e5), delta, x0=0)
    var2 = 2/(2-delta)
    print(f"δ={delta}: 经验方差={var1:.6f}, 理论方差={var2:.6f}")

var_compare(0.1)
print('#####')
var_compare(1)

print("\n渐近方差公式验证：Var_ULA = 2/(2-delta)")
print("-" * 60)
for d in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    var_compare(d)
