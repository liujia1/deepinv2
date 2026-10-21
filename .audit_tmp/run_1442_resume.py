"""以 EXP_RESUME=1 运行 14.4-2.py，用于验证 resume 分支与出图。"""
import os
import runpy

os.environ['EXP_RESUME'] = '1'
target = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..',
    '第14章 Flow Matching与最优传输', '第十四章 配套实验', '实验14.4-2', '14.4-2.py'))
print('运行:', target)
runpy.run_path(target, run_name='__main__')
