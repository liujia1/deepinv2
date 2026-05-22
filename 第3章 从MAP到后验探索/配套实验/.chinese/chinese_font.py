"""
中文字体配置模块
为 matplotlib 提供跨平台（Windows / Linux / Google Colab）的中文字体支持。
使用时在实验代码中 import 并调用 setup_chinese_font() 即可。
"""

import warnings
import logging
import platform
import os
import re

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")


def setup_chinese_font(save_dir='.'):
    """
    检测并设置 matplotlib 中文字体。
    
    参数:
        save_dir: 字体缓存文件保存路径（默认当前目录）
    
    返回:
        找到的字体名称，或 None
    """
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontManager

    plt.rcParams['axes.unicode_minus'] = False

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
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams.get('font.sans-serif', [])
            plt.rcParams['font.family'] = 'sans-serif'
            print(f"[Font] 已检测到中文字体: {font}")
            return font

    # 备选：模糊匹配含 CJK 关键字的字体
    cjk_patterns = ['cjk', 'wqy', 'noto.*cjk', 'wenquan', 'chinese', 'simhei']
    for f in fm.ttflist:
        name_lower = f.name.lower()
        fname_lower = (os.path.basename(f.fname) if hasattr(f, 'fname') else '').lower()
        for pat in cjk_patterns:
            if re.search(pat, name_lower) or re.search(pat, fname_lower):
                plt.rcParams['font.sans-serif'] = [f.name] + plt.rcParams.get('font.sans-serif', [])
                plt.rcParams['font.family'] = 'sans-serif'
                print(f"[Font] 已检测到中文字体 (模糊匹配): {f.name}")
                return f.name

    print("[Font] 未找到中文字体，中文可能显示为方框")
    return None