# -*- coding: utf-8 -*-
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
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False


def _fix_mathtext_log_minus():
    """修复对数轴刻度负号豆腐块问题。

    根因：semilogy 主刻度默认标签为 ``$\\mathdefault{10^{-1}}$``，``\\mathdefault``
    强制走 rcParams 指定的 sans-serif 中文字体（如 SimHei），而 SimHei 字形表
    不含 U+2212（数学减号），负号被替换成 dummy 方块；axes.unicode_minus=False
    对 mathtext 内部的 U+2212 不起作用。

    方案：把 10 的整数次幂主刻度标签改为标准 mathtext ``$10^{n}$``（走
    mathtext.fontset=dejavusans，含完整 U+2212 字形），样式与原来一致；
    非整数次幂（对数轴次刻度，默认无标签）仍交还原格式化器。
    """
    try:
        import numpy as np
        from matplotlib.ticker import LogFormatterSciNotation
        if getattr(LogFormatterSciNotation, '_cn_minus_patched', False):
            return
        _orig_call = LogFormatterSciNotation.__call__

        def __call__(self, x, pos=None):
            try:
                if np.isfinite(x) and x > 0:
                    e = int(round(np.log10(x)))
                    if abs(x / 10.0 ** e - 1.0) < 1e-8:
                        return r'$10^{%d}$' % e
            except Exception:
                pass
            return _orig_call(self, x, pos)

        LogFormatterSciNotation.__call__ = __call__
        LogFormatterSciNotation._cn_minus_patched = True
    except Exception as e:
        print(f"[Font] 对数轴负号修复未生效: {e}")

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
    _fix_mathtext_log_minus()  # 先修复对数轴负号豆腐块，再配置中文字体
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
