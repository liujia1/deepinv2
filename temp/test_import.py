# -*- coding: utf-8 -*-
"""
模拟 15.4-1.py 的中文字体加载逻辑
"""
import os
import sys

print("=" * 60)
print("模拟中文字体加载测试")
print("=" * 60)

# 模拟 __file__ 未定义的情况
print("\n--- 场景1: __file__ 未定义 ---")
SAVE_DIR = os.getcwd()  # 模拟 NameError 异常处理
print(f"SAVE_DIR = {SAVE_DIR}")
_chinese_path = os.path.join(SAVE_DIR, '.chinese')
print(f"_chinese_path = {_chinese_path}")
print(f".chinese 目录是否存在: {os.path.exists(_chinese_path)}")

if os.path.exists(_chinese_path):
    sys.path.insert(0, _chinese_path)
    print(f"已将 {_chinese_path} 添加到 sys.path")
    
    try:
        from chinese_font import setup_chinese_font
        print("✓ chinese_font 导入成功!")
        font = setup_chinese_font(save_dir=_chinese_path)
        if font:
            print(f"✓ 中文字体设置成功: {font}")
        else:
            print("⚠ 未找到可用中文字体")
    except ImportError as e:
        print(f"✗ ImportError: {e}")
        print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")

# 模拟 __file__ 定义的情况
print("\n--- 场景2: __file__ 已定义 (正常运行) ---")
# 假设脚本在 d:\works\deepinv\第15章...\15.4-1\ 目录下
test_script_dir = r"d:\works\deepinv\第15章 扩散模型的架构实践：UNet → DiT\第十五章 配套实验\15.4-1"
print(f"假设脚本目录: {test_script_dir}")
print(f"该目录是否存在: {os.path.exists(test_script_dir)}")

_chinese_path2 = os.path.join(test_script_dir, '.chinese')
print(f"_chinese_path = {_chinese_path2}")
print(f".chinese 目录是否存在: {os.path.exists(_chinese_path2)}")

if os.path.exists(_chinese_path2):
    # 检查是否有 chinese_font.py
    chinese_font_path = os.path.join(_chinese_path2, 'chinese_font.py')
    print(f"chinese_font.py 是否存在: {os.path.exists(chinese_font_path)}")

print("\n" + "=" * 60)