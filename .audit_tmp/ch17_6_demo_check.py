# -*- coding: utf-8 -*-
"""复现 17.6 动手 demo，检查三件事：
   (1) x_ei 与 x_true 是"接近"还是"完全相等"？
   (2) x_true 是否恰好关于中心对称（人为构造的精确对称）？
   (3) 换一个不对称信号做反例，看翻转约束会怎样
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

n, m = 16, 8
A = np.zeros((m, n)); A[:, :m] = np.eye(m)

x_true = np.sin(np.linspace(0, 3 * np.pi, n))
y = A @ x_true

x_mc = np.zeros(n); x_mc[:m] = y
x_ei = np.zeros(n); x_ei[:m] = y; x_ei[m:] = y[::-1]

print('[1] ||x_ei - x_true||_inf =', np.abs(x_ei - x_true).max())
print('    ||x_ei - x_true||_2   =', np.linalg.norm(x_ei - x_true))
print('    x_true 关于中心对称? max|x[k]-x[n-1-k]| =',
      np.abs(x_true - x_true[::-1]).max())
print('    x_ei 关于中心对称?   max|x[k]-x[n-1-k]| =',
      np.abs(x_ei - x_ei[::-1]).max())
print('    后8维 真值 :', np.round(x_true[m:], 3))
print('    后8维 MC+EI:', np.round(x_ei[m:], 3))

# 反例：指数衰减信号，明显不具备翻转对称性
x2 = np.exp(-np.linspace(0, 2, n))
y2 = A @ x2
x2_ei = np.zeros(n); x2_ei[:m] = y2; x2_ei[m:] = y2[::-1]
print('\n[2] 非对称信号反例：max|x[k]-x[n-1-k]| =',
      np.abs(x2 - x2[::-1]).max())
print('    翻转约束恢复误差 ||x2_ei-x2||_2 =', np.linalg.norm(x2_ei - x2))
x2_mc = np.concatenate([y2, np.zeros(n - m)])
print('    零填充基线误差   ||x2_mc-x2||_2 =', np.linalg.norm(x2_mc - x2))
print('    → 先验与真值不一致时，EI 约束反而把解拉偏（不保证更接近真值）')
