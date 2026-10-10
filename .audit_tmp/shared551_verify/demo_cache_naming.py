# -*- coding: utf-8 -*-
"""演示：缓存文件名不是随机的，而是 参数+版本 的确定性 MD5 指纹。"""
import hashlib
import json
import os

import numpy as np

CD = r'd:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.6-3\步骤6_扫描链缓存'
VER = '0bdaff85d6'   # Colab 这轮的版本键

# ε×0.5 扫描链（seed=100）的真实参数（与 5.6-3.py 中 _sparams 构造完全一致）
params = {'eps': 0.00019223375624759708, 'delta_frac': 0.99, 'maxit': 2000,
          'burnin': 500, 'n_samples': 60, 'seed': 100,
          'burn_frac': 0.25, 'trace_every': 25}

# 与 _cache_fingerprint 完全相同的公式（这就是"命名规则"）
src = json.dumps(params, sort_keys=True, ensure_ascii=False)
h = hashlib.md5(('scan|' + src + '|' + VER).encode('utf-8')).hexdigest()[:16]
path = os.path.join(CD, h + '.npz')

print('参与命名的参数原文（sort_keys 保证键顺序固定）:')
print(' ', src)
print()
print('版本键:', VER)
print('确定性算出的文件名:', h + '.npz')
print('文件是否存在:', os.path.exists(path))
with np.load(path) as z:
    print('文件内 _meta 与当前参数逐字一致:', json.loads(str(z['_meta'])) == params)

# 改任意一个参数 -> 文件名立刻不同（这就是"改参数自动重算"的原理）
params2 = dict(params, seed=101)
h2 = hashlib.md5(('scan|' + json.dumps(params2, sort_keys=True, ensure_ascii=False)
                  + '|' + VER).encode('utf-8')).hexdigest()[:16]
print()
print('只把 seed 100 改成 101 -> 文件名变为:', h2 + '.npz',
      ' 存在:', os.path.exists(os.path.join(CD, h2 + '.npz')))

# 同参数重复算两次 -> 必然相同（MD5 无随机性；注意与 Python 内置 hash() 的区别）
h_again = hashlib.md5(('scan|' + src + '|' + VER).encode('utf-8')).hexdigest()[:16]
print('同参数再算一次:', h_again, ' 与首次相同:', h_again == h)
