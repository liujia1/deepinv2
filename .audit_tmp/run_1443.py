# -*- coding: utf-8 -*-
"""以 runpy 运行 14.4-3（避免 cwd/中文路径问题，且不写 __pycache__）。"""
import os
import runpy
import sys

sys.dont_write_bytecode = True
p = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.4-3\14.4-3.py'
os.chdir(os.path.dirname(p))
runpy.run_path(p, run_name='__main__')
