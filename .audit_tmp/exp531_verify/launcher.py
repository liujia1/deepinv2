"""复核用启动器：避免 PowerShell 传参时中文路径被编码破坏"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
target = r"d:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.3-1\5.3-1.py"
sys.argv = [target]
runpy.run_path(target, run_name="__main__")
