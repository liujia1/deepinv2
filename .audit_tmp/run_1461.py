"""在实验目录内以 __main__ 方式运行 14.6-1.py。"""
import os
import sys
import runpy

EXP_DIR = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.6-1'
os.chdir(EXP_DIR)
sys.path.insert(0, EXP_DIR)
sys.argv = ['14.6-1.py']
runpy.run_path(os.path.join(EXP_DIR, '14.6-1.py'), run_name='__main__')
