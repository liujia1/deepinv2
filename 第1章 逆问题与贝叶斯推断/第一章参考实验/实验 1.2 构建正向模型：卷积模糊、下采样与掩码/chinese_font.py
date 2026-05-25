"""
中文字体配置模块
自动检测系统可用的中文字体，并配置 matplotlib 显示
"""
import logging
import warnings
import platform
from matplotlib.font_manager import FontManager


def setup_chinese_font():
    """
    自动检测系统中可用的中文字体，并配置 matplotlib
    
    兼容 Windows / Linux 系统
    """
    # 关闭 matplotlib 的字体警告
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*U\\+2212.*")
    warnings.filterwarnings("ignore", message=".*glyph.*")
    
    import matplotlib.pyplot as plt
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 候选字体列表
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
    
    # 查找可用字体
    for font in candidates:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams.get('font.sans-serif', [])
            plt.rcParams['font.family'] = 'sans-serif'
            print(f"[Font] 已检测到中文字体：{font}")
            return font
    
    # 如果候选列表中没有，尝试正则匹配
    import os
    import re
    cjk_patterns = ['cjk', 'wqy', 'noto.*cjk', 'wenquan', 'chinese', 'simhei']
    for f in fm.ttflist:
        name_lower = f.name.lower()
        fname_lower = (os.path.basename(f.fname) if hasattr(f, 'fname') else '').lower()
        for pat in cjk_patterns:
            if re.search(pat, name_lower) or re.search(pat, fname_lower):
                plt.rcParams['font.sans-serif'] = [f.name] + plt.rcParams.get('font.sans-serif', [])
                plt.rcParams['font.family'] = 'sans-serif'
                print(f"[Font] 已检测到中文字体：{f.name}")
                return f.name
    
    # 未找到中文字体
    print("[Font] 未找到中文字体，中文可能显示为方框")
    return None


# 自动执行配置
_chinese_font = setup_chinese_font()
