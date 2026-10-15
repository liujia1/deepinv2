# -*- coding: utf-8 -*-
"""变异测试：界定 1.5-3 现有"等价性验证"能捕获什么 bug、捕获不了什么 bug。"""
import io, os
import numpy as np

out = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out_mutation.txt'), 'w', encoding='utf-8')


def w(s):
    out.write(s + '\n')


n = 256
sigma = 1.0
rng = np.random.default_rng(7)
res = rng.standard_normal(n) * 1.3
ls = float(np.sum(res ** 2))
factor = 1.0 / (2.0 * sigma ** 2)          # 代码 L103 的定义


def check(nll_val):
    pred = factor * ls
    return abs(nll_val - pred) / (abs(pred) + 1e-12)


w('=== 现有检验 max_rel_err 对各类"写错"的反应 (sigma=1, ls=%.4f) ===' % ls)
cases = [
    ('正确: 0.5/sigma^2 * ls', 0.5 / sigma ** 2 * ls),
    ('系数写成 1/sigma^2 (漏 1/2)', 1.0 / sigma ** 2 * ls),
    ('系数写成 1/(4 sigma^2)', 0.25 / sigma ** 2 * ls),
    ('用了 sigma 而非 sigma^2', 0.5 / sigma * ls),
    ('residual 少平方 (L1 型)', 0.5 / sigma ** 2 * float(np.sum(np.abs(res)) ** 2 / n)),
    ('完全漏掉常数项 n/2 ln(2pi sigma^2)', 0.5 / sigma ** 2 * ls),
    ('错误地加上常数项', 0.5 / sigma ** 2 * ls + 0.5 * n * np.log(2 * np.pi * sigma ** 2)),
]
for name, v in cases:
    d = check(v)
    verdict = '通过(<1e-10)' if d < 1e-10 else '报警(偏差 %.2e)' % d
    w('  %-42s  nll=%12.4f  偏差=%.3e  -> %s' % (name, v, d, verdict))

w('')
w('=== 关键：residual 本身算错时, 检验是否还能通过? ===')
# 模拟"正向算子用错"：residual 少减了 A x（只用 noise），两边仍用同一个 residual
wrong_res = res + 0.7 * rng.standard_normal(n)      # 与真实 y-Ax 无关的向量
ls_wrong = float(np.sum(wrong_res ** 2))
nll_wrong = 0.5 / sigma ** 2 * ls_wrong
w('  把 residual 换成任意向量后: 偏差 = %.3e -> %s'
  % (abs(nll_wrong - factor * ls_wrong) / (abs(factor * ls_wrong) + 1e-12),
     '仍然通过（检验无法发现 residual 计算错误）'))

w('')
w('=== 独立实现能发现的量：常数项 ===')
const = 0.5 * n * np.log(2 * np.pi * sigma ** 2)
w('  n/2 * ln(2 pi sigma^2) = %.6f   (占 nll 绝对值 %.1f%%)' % (const, 100 * const / (const + ls / 2)))
try:
    from scipy.stats import norm
    a = float(-norm.logpdf(res, 0, sigma).sum())
    b = const + ls / (2 * sigma ** 2)
    w('  scipy -logpdf 求和 = %.12f' % a)
    w('  闭式 常数项+ls/(2sigma^2) = %.12f' % b)
    w('  相对偏差 = %.3e  (非零, 是真实浮点误差, 这才叫验证)' % (abs(a - b) / abs(b)))
except ImportError:
    w('  scipy 不可用, 跳过')
out.close()
print('ok')
