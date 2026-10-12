# -*- coding: utf-8 -*-
"""实验3.7-1 复核脚本：
1) 用代码声明的参数(种子42, dims=[2,10,100,1000], 500样本)复现 results_summary.json；
2) 用讲义声明的参数(种子20240717, 10个维度, 2000样本)检验讲义表格能否复现；
3) 独立理论值(chi分布均值/中位数)对照，检查讲义"最大偏差仅1.17%"断言。
"""
import numpy as np
import json, math, io, sys
from scipy import stats

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

base = r'd:\works\deepinv\第3章 从MAP到后验探索\第三章 配套实验\实验3.7-1'
with open(base + r'\results_summary.json', encoding='utf-8') as f:
    js = json.load(f)

# ---------- 1) 复现 JSON（种子42，与 3.7-1.py 完全一致的调用顺序） ----------
np.random.seed(42)
dimensions = [2, 10, 100, 1000]
n_samples = 500
samples = {d: np.random.randn(n_samples, d) for d in dimensions}

print('=== 1) 种子42复现 vs results_summary.json ===')
ok_all = True
for d in dimensions:
    norms = np.linalg.norm(samples[d], axis=1)
    med, mean = np.median(norms), np.mean(norms)
    jmed = js['scene_a_typical_set']['typical_norm_median'][str(d)]
    jmean = js['scene_a_typical_set']['mean_norm'][str(d)]
    match = (abs(med - jmed) < 1e-9) and (abs(mean - jmean) < 1e-9)
    ok_all &= match
    print(f'  n={d:<5} 复现median={med:.6f} json={jmed:.6f} | 复现mean={mean:.6f} json={jmean:.6f} | 逐位一致={match}')
print('  JSON可复现性:', 'PASS（逐位一致）' if ok_all else 'FAIL')

# ---------- 2) 讲义参数（种子20240717, dims 10个, 2000样本）能否复现讲义表 ----------
print('\n=== 2) 讲义声明参数：seed=20240717, dims=[2,5,10,20,50,100,200,500,1000,2000], 2000样本 ===')
doc_table = {2:1.392, 5:2.193, 10:3.013, 20:4.223, 50:7.030,
             100:9.857, 200:14.188, 500:22.587, 1000:31.557, 2000:44.623}
np.random.seed(20240717)
dims2 = [2,5,10,20,50,100,200,500,1000,2000]
res2 = {}
for d in dims2:
    s = np.random.randn(2000, d)
    nrm = np.linalg.norm(s, axis=1)
    res2[d] = dict(mean=nrm.mean(), median=np.median(nrm))

print(f"  {'n':>5} {'讲义MC':>8} {'复现mean':>10} {'复现median':>10} {'理论E||x||':>10} {'理论median':>10} {'sqrt(n)':>8}")
for d in dims2:
    tmean = stats.chi.mean(d)                                          # E||x|| chi(d)
    tmed = stats.chi.median(d)                                         # median||x||
    dev_m = abs(res2[d]['mean']-doc_table[d])/doc_table[d]*100
    dev_md = abs(res2[d]['median']-doc_table[d])/doc_table[d]*100
    print(f'  {d:>5} {doc_table[d]:>8.3f} {res2[d]["mean"]:>10.3f} {res2[d]["median"]:>10.3f} '
          f'{tmean:>10.3f} {tmed:>10.3f} {math.sqrt(d):>8.3f}   |mean偏差{dev_m:5.2f}% |median偏差{dev_md:5.2f}%')

# ---------- 3) 讲义"最大偏差仅1.17%"断言检验（讲义表 vs sqrt(n)；vs 理论均值） ----------
print('\n=== 3) 讲义表内部自洽性："典型范数贴近sqrt(n)，最大偏差仅1.17%" ===')
mx_sqrt = mx_mean = 0
for d in dims2:
    tmean = stats.chi.mean(d)
    ds = abs(doc_table[d]-math.sqrt(d))/math.sqrt(d)*100
    dm = abs(doc_table[d]-tmean)/tmean*100
    mx_sqrt = max(mx_sqrt, ds); mx_mean = max(mx_mean, dm)
    print(f'  n={d:<5} 讲义值 vs sqrt(n) 偏差 {ds:5.2f}%   vs 理论E||x|| 偏差 {dm:5.2f}%')
print(f'  对sqrt(n)最大偏差={mx_sqrt:.2f}%  对理论均值最大偏差={mx_mean:.2f}%  （讲义宣称1.17%）')

# ---------- 4) MC噪声量级：讲义n=2值 1.392 距理论多远（几个标准误） ----------
print('\n=== 4) n=2 处讲义值 1.392 的统计不可能性检验 ===')
# Rayleigh(sigma=1): mean=sqrt(pi/2), median=sqrt(2 ln2)
m2, med2 = math.sqrt(math.pi/2), math.sqrt(2*math.log(2))
sd2 = math.sqrt((4-math.pi)/2)
se_mean = sd2/math.sqrt(2000)
dens_at_med = med2*math.exp(-med2**2/2)
se_med = 1/(2*dens_at_med*math.sqrt(2000))
print(f'  理论mean={m2:.4f} (SE={se_mean:.4f}) -> 1.392 距 mean {abs(1.392-m2)/se_mean:.1f} 个SE')
print(f'  理论median={med2:.4f} (SE={se_med:.4f}) -> 1.392 距 median {abs(1.392-med2)/se_med:.1f} 个SE')

# ---------- 5) 右图标准化断言: (||x||-sqrt(n))/(1/sqrt(2)) ~ N(0,1)? ----------
print('\n=== 5) n=1000 标准化范数是否 ~ N(0,1)（右图断言） ===')
nrm1000 = np.linalg.norm(samples[1000], axis=1)
z = (nrm1000 - math.sqrt(1000))/(1/math.sqrt(2))
print(f'  z mean={z.mean():.3f}, std={z.std(ddof=1):.3f}, KS p={stats.kstest(z, "norm").pvalue:.3f}')
print('  （注：delta method 给 Var(||x||)≈1/2 仅在 n 大时成立，检验经验分布是否与N(0,1)目测一致）')
