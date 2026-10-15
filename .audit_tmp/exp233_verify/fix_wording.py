# -*- coding: utf-8 -*-
import io, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

# --- 1) 核对 "最大偏差 0.25 出现在 |y|≈0.5" ---
sigma, b = 0.5, 0.5
xg = np.linspace(-40, 40, 8001)
lp = lambda x: -np.abs(x) / b
ys = np.linspace(0.001, 3.0, 300)
mm = np.array([np.sum(xg * np.exp(lp(xg) - (y - xg) ** 2 / (2 * sigma ** 2))) /
               np.sum(np.exp(lp(xg) - (y - xg) ** 2 / (2 * sigma ** 2))) for y in ys])
mp = np.maximum(ys - sigma ** 2 / b, 0.0)
d = mm - mp
k = int(np.argmax(np.abs(d)))
print('|MMSE-MAP| 最大 = %.4f 出现在 y=%.3f ; y=0.5 处 = %.4f ; y=2.0 处 = %.4f'
      % (abs(d[k]), ys[k], np.interp(0.5, ys, np.abs(d)), np.interp(2.0, ys, np.abs(d))))

# --- 2) 修正正文措辞 ---
P = r'd:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\2.3 先验的质量：MMSE vs MAP估计器.md'
s = io.open(P, encoding='utf-8').read()
old = '近零处只做温和抑制（比 Laplace 的 MMSE 温和，但略弱于高斯的线性收缩）'
new = '近零处只做温和抑制（比 Laplace 的 MMSE 温和，但收缩仍略强于高斯的线性缩放：压制 23% vs 20%）'
assert s.count(old) == 1, s.count(old)
io.open(P, 'w', encoding='utf-8', newline='\n').write(s.replace(old, new))
print('措辞已修正')
