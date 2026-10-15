# -*- coding: utf-8 -*-
"""细查方差单调性：逐步(stepwise)与检查点(checkpoint)两个层面"""
import numpy as np
th = 0.7
Ns = np.array([0, 2, 10, 50, 500])

def var_of(h, n):
    a = 1.0 + h; b = 1.0 + n - h
    return a * b / ((a + b) ** 2 * (a + b + 1))

rng = np.random.default_rng(2026)
M = 40000
allf = (rng.random((M, 500)) < th)
hs = np.stack([allf[:, :n].sum(1) for n in Ns], 1).astype(float)
vs = var_of(hs, Ns[None, :])
d = np.diff(vs, axis=1)
print('各步 increase(>0) 的比例:', [f'{Ns[i]}->{Ns[i+1]}: {np.mean(d[:, i] > 0):.5f}' for i in range(d.shape[1])])
print('任一步上升比例:', np.mean(np.any(d > 0, axis=1)))
print('路径0:', vs[0], ' 路径-1:', vs[-1])
print('seed42 路径:', [round(var_of(int((np.random.default_rng(42).random(500) < th).astype(int)[:n].sum()), n), 6) for n in [0,2,10,50,500]])

# 逐抛（每 1 次）单调性
fl = allf.astype(int)
cum = np.cumsum(fl, axis=1)
v_all = var_of(cum, np.arange(1, 501)[None, :])
v_step = np.concatenate([np.full((M, 1), 1/12), v_all], axis=1)
inc = np.any(np.diff(v_step, axis=1) > 1e-15, axis=1)
print('逐抛层面：至少一次方差上升的路径比例 =', np.mean(inc))
# 期望：Beta(1,3)->Beta(2,3) 型上升在前几次很常见
print('前 10 抛内就上升的比例 =', np.mean(np.any(np.diff(v_step[:, :11], axis=1) > 1e-15, axis=1)))
