# -*- coding: utf-8 -*-
"""清理 14.4-1 重跑产生的运行时文件（.pth 未被 git 跟踪；reflow_pairs 达 376MB）。"""
import os, glob, shutil
D = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.4-1'
for p in glob.glob(os.path.join(D, '*.pth')) + glob.glob(os.path.join(D, '*partial*')):
    print('del', os.path.basename(p), '%.1f MB' % (os.path.getsize(p) / 1e6))
    os.remove(p)
pc = os.path.join(D, '__pycache__')
if os.path.isdir(pc):
    print('del __pycache__')
    shutil.rmtree(pc)
print('left:', sorted(os.listdir(D)))
