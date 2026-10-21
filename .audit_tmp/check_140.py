# -*- coding: utf-8 -*-
"""抽取 14.0 导读中的代码块并实跑，验证欧拉修正：步数=50、t∈[0,0.98]、终点恰为 t=1"""
import re, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

md = r"h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\14.0 本章导读.md"
src = open(md, encoding='utf-8').read()
block = re.search(r"```python\n(.*?)```", src, re.S).group(1)
block = block.replace("plt.show()", "plt.savefig('_140_smoke.png', dpi=80); plt.close('all')")

ns = {}
exec(compile(block, '14.0-block', 'exec'), ns)
print("---- 数值校验 ----")
print("生成样本均值:", ns['x_gen'].mean(0).round(3), "（目标≈[0,0]）")
print("生成样本 std :", ns['x_gen'].std(0).round(3), "（目标≈[2.0,2.0]）")

# 单独抽出 euler_flow 定义，注入探针统计实际用到的 t
fn_src = re.search(r"def euler_flow.*?return x\n", block, re.S).group(0)
ts = []
def traced(x, t):
    ts.append(t)
    return ns['net_forward'](x, t)
ns2 = {'np': ns['np'], 'net_forward': traced}
exec(fn_src, ns2)
ns2['euler_flow'](ns['z_test'])
print("---- 步长校验 ----")
print("步数:", len(ts), " t 首项:", ts[0], " t 末项:", round(ts[-1], 4))
print("累计积分长度:", round(len(ts) * (1.0 / 50), 6), "（应=1.0）")
print("图片:", os.path.exists('_140_smoke.png'))
