# -*- coding: utf-8 -*-
import os
import sys

print("=" * 60)
print("测试 __file__ 和路径计算")
print("=" * 60)

print(f"当前工作目录 (os.getcwd()): {os.getcwd()}")
print(f"__file__ 是否定义:", end=" ")
try:
    print(f"是 -> {__file__}")
    print(f"os.path.dirname(os.path.abspath(__file__)): {os.path.dirname(os.path.abspath(__file__))}")
except NameError as e:
    print(f"否 -> NameError: {e}")
    print(f"模拟脚本行为: SAVE_DIR = os.getcwd() -> {os.getcwd()}")
    _chinese_path = os.path.join(os.getcwd(), '.chinese')
    print(f"模拟 _chinese_path: {_chinese_path}")
    print(f"该路径是否存在: {os.path.exists(_chinese_path)}")
    
    # 检查项目根目录的 .chinese
    root_chinese = os.path.join(os.getcwd(), '.chinese')
    print(f"项目根目录 .chinese: {root_chinese}")
    print(f"项目根目录 .chinese 是否存在: {os.path.exists(root_chinese)}")

# 检查 sys.path
print(f"\nsys.path 前5项:")
for p in sys.path[:5]:
    print(f"  {p}")