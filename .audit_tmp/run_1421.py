# -*- coding: utf-8 -*-
"""以 runpy 运行 14.2-1；强制 CPU，避免与后台 14.4-1 的 GPU 训练抢资源。"""
import os
import runpy
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
p = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.2-1\14.2-1.py'
os.chdir(os.path.dirname(p))
runpy.run_path(p, run_name='__main__')
