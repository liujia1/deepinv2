# -*- coding: utf-8 -*-
"""实验1.5-1 复核：(A) 复跑代码比对 json/png; (B) 解析验证; (C) 蒙特卡洛验证 md 中的"免责"论断"""
import json, os, sys, shutil
import numpy as np
from scipy.stats import beta, binom

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = r"d:/works/deepinv/第1章 逆问题与贝叶斯推断/第一章 配套实验/实验1.5-1"
OUT = os.path.join(HERE, 'rerun')
os.makedirs(OUT, exist_ok=True)

# ---------- A. 复跑原脚本到临时目录 ----------
src = os.path.join(EXP, '1.5-1.py')
code = open(src, encoding='utf-8').read()
# 强制 SAVE_DIR 指向 OUT
code = code.replace("SAVE_DIR = os.path.dirname(os.path.abspath(__file__))", f"SAVE_DIR = r'{OUT}'")
code = code.replace("_chinese_path = os.path.join(SAVE_DIR, '.chinese')",
                    f"_chinese_path = os.path.join(r'{EXP}', '.chinese')")
open(os.path.join(HERE, 'rerun_script.py'), 'w', encoding='utf-8').write(code)
print('=== A. 复跑 ===')
import subprocess
r = subprocess.run([sys.executable, os.path.join(HERE, 'rerun_script.py')], capture_output=True, text=True, encoding='utf-8', errors='replace')
print(r.stdout, r.stderr)

new_json = json.load(open(os.path.join(OUT, 'results_summary.json'), encoding='utf-8'))
old_json = json.load(open(os.path.join(EXP, 'results_summary.json'), encoding='utf-8'))
print('json 完全一致:', json.dumps(new_json, sort_keys=True) == json.dumps(old_json, sort_keys=True))

p_new = os.path.join(OUT, '实验1_5_1_硬币贝叶斯更新.png')
p_old = os.path.join(EXP, '实验1_5_1_硬币贝叶斯更新.png')
b_new = open(p_new, 'rb').read(); b_old = open(p_old, 'rb').read()
print('png 字节一致:', b_new == b_old, len(b_new), len(b_old))
try:
    import matplotlib.image as mpimg
    a1 = mpimg.imread(p_new).astype(np.float32); a2 = mpimg.imread(p_old).astype(np.float32)
    if a1.shape != a2.shape:
        print('png 形状不同', a1.shape, a2.shape)
    else:
        print('png 像素 maxdiff:', np.abs(a1 - a2).max(), ' mean:', np.abs(a1 - a2).mean())
except Exception as e:
    print('pixel cmp fail', e)

# ---------- B. 解析验证 ----------
print('\n=== B. 解析验证 ===')
th = 0.7; a0 = b0 = 1.0
flips = (np.random.default_rng(42).random(500) < th).astype(int)
print('总正面数(500) =', flips.sum(), ' 前50正面 =', flips[:50].sum(), ' 前10 =', flips[:10].sum(), ' 前2 =', flips[:2].sum())
for n in [0, 2, 10, 50, 500]:
    h = int(flips[:n].sum()); ap, bp = a0 + h, b0 + n - h
    m = ap / (ap + bp)
    v = ap * bp / ((ap + bp) ** 2 * (ap + bp + 1))
    # 用 scipy 独立核对
    m2, v2 = beta.mean(ap, bp), beta.var(ap, bp)
    print(f'n={n:>3} heads={h:>3} tails={n-h:>3} Beta({ap},{bp}) mean={m:.6f}/{m2:.6f} var={v:.8f}/{v2:.8f} mode={max(min((ap-1)/(ap+bp-2),1),0) if ap>1 and bp>1 else "n/a"}')

print('\n-- md 中的概率论断 --')
p1 = binom.pmf(1, 2, th)
p2 = binom.pmf(4, 8, th)
p3 = binom.pmf(5, 10, th)
print(f'P(前2次恰1正) = {p1:.4f}   (md: 0.42)')
print(f'P(后8次恰4正 | 前2次各半) = {p2:.4f} (md: 0.136)')
print(f'联合 = {p1*p2:.4f}              (md: ~0.057)')
print(f'P(n=10 后验均值恰=0.5) = P(H=5) = {p3:.5f} (md: 10.4%)')
med = np.median(binom.ppf(0.5, 10, th))
print(f'Bin(10,0.7) 中位数 = {binom.ppf(0.5,10,th)}, 后验均值中位数 = {(1+binom.ppf(0.5,10,th))/12:.4f} (md: 0.667)')
print('先验权重 2/(2+n):', [round(2/(2+n)*100, 2) for n in [2, 10, 50, 500]], '%')

# ---------- C. 蒙特卡洛：4万次重演 ----------
print('\n=== C. 蒙特卡洛 (40000 次重演) ===')
rng = np.random.default_rng(2026)
M = 40000
allf = (rng.random((M, 500)) < th)
for n in [2, 10, 50, 500]:
    h = allf[:, :n].sum(1)
    m = (1 + h) / (2 + n)
    print(f'  n={n:>3}: 均值 median={np.median(m):.4f} mean={m.mean():.4f}  P(mean=0.5)={np.mean(np.isclose(m,0.5)):.4f}  P(>0.7)={np.mean(m>0.7):.4f}')
# 均值路径非单调的比例
hs = np.stack([allf[:, :n].sum(1) for n in [0, 2, 10, 50, 500]], 1)
ms = (1 + hs) / (2 + np.array([0, 2, 10, 50, 500]))
nonmono_m = np.mean(np.any(np.diff(ms) < 0, axis=1))
print(f'  后验均值序列在 5 个检查点上非单调的比例 = {nonmono_m:.3f}')
vs = (1 + hs) * (1 + np.array([0,2,10,50,500]) - hs) / ((2+np.array([0,2,10,50,500]))**2 * (3+np.array([0,2,10,50,500])))
nonmono_v = np.mean(np.any(np.diff(vs) <= 0, axis=1))
print(f'  后验方差序列在 5 个检查点上非严格下降的比例 = {nonmono_v:.5f}')
# 逐步（每抛一次）方差是否单调？ 反例搜索
print('\n  方差单调性反例（一般性检验）:')
for (aa, bb) in [(2, 5), (3, 5), (1, 3), (5, 2)]:
    v1 = aa*bb/((aa+bb)**2*(aa+bb+1)); v2 = (aa+1)*bb/(((aa+1+bb)**2)*(aa+1+bb+1))
    print(f'    Beta({aa},{bb}) var={v1:.5f} -> 观测一次正面后 Beta({aa+1},{bb}) var={v2:.5f}  {"增加!" if v2>v1 else "减少"}')
# 逐步单调：从 Beta(1,1) 出发，任意路径
print('\n  从 Beta(1,1) 起，逐次更新方差是否必然下降? 上面反例 Beta(2,5)->Beta(3,5) 说明不必然。')
print('  本次种子路径方差:', [round(v, 6) for v in vs[-1]])
