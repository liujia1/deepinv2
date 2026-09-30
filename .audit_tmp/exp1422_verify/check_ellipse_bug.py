# -*- coding: utf-8 -*-
"""定位 14.2-2.py 步骤4 椭圆画不出来的根因：矩阵乘法写法错误"""
import numpy as np

cov = np.array([[0.0588, 0.0], [0.0, 1.0]])
eigv, eigvec = np.linalg.eigh(cov)
theta = np.linspace(0, 2 * np.pi, 100)
circle = np.stack([np.cos(theta), np.sin(theta)])  # (2,100)

# 原代码写法
ell_bad = 2 * np.sqrt(eigv) @ (eigvec @ circle)
print("原代码 ell.shape =", ell_bad.shape, " ell[0]=", ell_bad[0], " ell[1]=", ell_bad[1])
print("→ ell 是一维数组，ell[0]/ell[1] 是标量，ax.plot 只画一个点（不可见）")

# 正确写法：先按主轴缩放，再旋转
ell_ok = eigvec @ (2 * np.sqrt(eigv)[:, None] * circle)
print("修正后 ell.shape =", ell_ok.shape, " x范围=[%.2f,%.2f] y范围=[%.2f,%.2f]"
      % (ell_ok[0].min(), ell_ok[0].max(), ell_ok[1].min(), ell_ok[1].max()))
