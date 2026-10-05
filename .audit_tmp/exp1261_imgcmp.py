# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from PIL import Image

orig = r'D:\works\deepinv\.audit_tmp\exp1261_verify\步骤2_采样结果对比.png'
new = r'd:\works\deepinv\第12章 Score ≡ ELBO：殊途同归\第十二章 配套实验\实验12.6-1\步骤2_采样结果对比.png'

a = np.asarray(Image.open(orig).convert('RGB')).astype(int)
b = np.asarray(Image.open(new).convert('RGB')).astype(int)
print('orig shape', a.shape, 'new shape', b.shape)
if a.shape == b.shape:
    d = np.abs(a - b)
    print('max abs diff', d.max(), 'mean abs diff', round(d.mean(), 4))
    print('identical?', bool((d == 0).all()))
else:
    print('shape mismatch -> not directly comparable')

# also training curve
orig2 = r'D:\works\deepinv\.audit_tmp\exp1261_verify\步骤1_训练曲线对比.png'
new2 = r'd:\works\deepinv\第12章 Score ≡ ELBO：殊途同归\第十二章 配套实验\实验12.6-1\步骤1_训练曲线对比.png'
c = np.asarray(Image.open(orig2).convert('RGB')).astype(int)
e = np.asarray(Image.open(new2).convert('RGB')).astype(int)
print('\n[train curve] orig', c.shape, 'new', e.shape)
if c.shape == e.shape:
    d2 = np.abs(c - e)
    print('[train curve] max', d2.max(), 'mean', round(d2.mean(), 4), 'identical?', bool((d2==0).all()))
