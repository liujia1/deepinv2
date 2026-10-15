# -*- coding: utf-8 -*-
"""取证1.5-3: (a)重跑一致性 (b)等价性验证是否为同义反复 (c)独立路径含常数项核对 (d)随机流耦合敏感度"""
import io, os, re, sys, json, hashlib, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = r'd:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.5-3'
PYEXE = r'd:\works\deepinv\.audit_tmp\venv042\Scripts\python.exe'
import numpy as np

def md5(p):
    return hashlib.md5(io.open(p, 'rb').read()).hexdigest()[:12]

print('=== (a) 归档现状 ===')
pj = os.path.join(EXP, 'results_summary.json')
png = os.path.join(EXP, '实验1_5_3_似然等于最小二乘.png')
j0 = json.load(io.open(pj, encoding='utf-8'))
print('  JSON:', j0)
print('  PNG size=%d md5=%s' % (os.path.getsize(png), md5(png)))

print('\n=== (b) 原脚本内的等价性检查是否同义反复 ===')
# 逐字复现代码里的两步: nll = 0.5/sigma**2 * sum(r**2); predicted = factor*ls, factor=1/(2*sigma**2)
sigma = 1.0
rng = np.random.default_rng(0)
for trial in range(3):
    r = rng.standard_normal(256)
    ls = float(np.sum(r ** 2))
    nll = 0.5 / sigma ** 2 * ls
    factor = 1.0 / (2.0 * sigma ** 2)
    print('  residual^2=%.6f  nll=%.17g  factor*ls=%.17g  完全相等? %s'
          % (ls, nll, factor * ls, nll == factor * ls))
print('  -> 二者由同一 sum(residual**2) 乘同一标量得到, 差恒为浮点 0, 与数据无关')

print('\n=== (c) 独立路径: 真正计算高斯对数似然(含常数项) ===')
try:
    from scipy.stats import norm
    have_scipy = True
except ImportError:
    have_scipy = False
print('  scipy available:', have_scipy)

n = 256
t = np.linspace(0, 1, n)
x_true = np.sin(2 * np.pi * 3 * t) + 0.5 * np.cos(2 * np.pi * 7 * t)
sigma_psf = 4.0
k = np.fft.fftfreq(n) * n
H = np.exp(-0.5 * (k / sigma_psf) ** 2).astype(complex)
np.random.seed(42)
y_clean = np.real(np.fft.ifft(H * np.fft.fft(x_true)))
noise = sigma * np.random.randn(n)
y = y_clean + noise
print('  重跑前 4 个 noise 值:', np.round(noise[:4], 8))
print('  sum(noise^2)=%.6f  (JSON least_squares_at_true=%.4f)' % (np.sum(noise**2), j0['least_squares_at_true']))

np.random.seed(12345)
worst = []
for trial in range(200):
    scale = rng.uniform(0, 2)
    x_cand = x_true + scale * rng.standard_normal(n)
    res = y - np.real(np.fft.ifft(H * np.fft.fft(x_cand)))
    ls = float(np.sum(res ** 2))
    nll_indep = float(0.5 * n * np.log(2 * np.pi * sigma ** 2) + ls / (2 * sigma ** 2))
    if have_scipy:
        nll_sp = float(-norm.logpdf(res, loc=0, scale=sigma).sum())
        d1 = abs(nll_sp - nll_indep) / max(abs(nll_indep), 1e-12)
    else:
        d1 = float('nan')
    d2 = abs((-np.log(1.0) - 0) ) if False else abs(nll_indep - (ls/(2*sigma**2) + 0.5*n*np.log(2*np.pi)))
    worst.append((d1, ls, nll_indep))
w = max(x[0] for x in worst if x[0] == x[0]) if worst else float('nan')
print('  独立实现(scipy求和 vs 闭式公式)最大相对偏差 = %.3e' % w)
print('  示例: ls=%.4f  -ln p(含常数)=%.4f  常数项 n/2*ln(2pi sigma^2)=%.4f'
      % (worst[5][1], worst[5][2], 0.5 * n * np.log(2 * np.pi * sigma ** 2)))
print('  -> 去掉常数项后两者差为 %.3e (纯截断误差量级), 这才是"非平凡"的验证'
      % max(abs(x[2] - (x[1] / (2 * sigma ** 2) + 0.5 * n * np.log(2 * np.pi))) for x in worst))

print('\n=== (d) 循环内 randn 的耦合敏感度: 换种子看图数据是否变 ===')
for seed in (42, 7, 2026):
    np.random.seed(seed)
    y2 = y_clean + sigma * np.random.randn(n)
    lsv, nllv = [], []
    for i in range(200):
        scale = (i / 199.0) * 2.0
        xc = x_true + scale * np.random.randn(n)
        r = y2 - np.real(np.fft.ifft(H * np.fft.fft(xc)))
        lsv.append(np.sum(r ** 2)); nllv.append(0.5 / sigma ** 2 * np.sum(r ** 2))
    lsv = np.array(lsv); nllv = np.array(nllv)
    print('  seed=%d: ls范围[%.2f, %.2f]  max|nll-0.5ls|/0.5ls = %.1e'
          % (seed, lsv.min(), lsv.max(), np.max(np.abs(nllv - 0.5 * lsv) / (0.5 * lsv))))
print('  -> ls/nll 逐点比值恒为0.5, 与种子无关; 但曲线形状(散点)依赖随机流')

print('\n=== (e) 重跑原脚本对比归档 ===')
tmp = os.path.join(HERE, 'exp153_rerun')
os.makedirs(tmp, exist_ok=True)
src = io.open(os.path.join(EXP, '1.5-3.py'), encoding='utf-8').read()
io.open(os.path.join(tmp, '1.5-3.py'), 'w', encoding='utf-8').write(src)
import shutil
if os.path.isdir(os.path.join(EXP, '.chinese')) and not os.path.isdir(os.path.join(tmp, '.chinese')):
    shutil.copytree(os.path.join(EXP, '.chinese'), os.path.join(tmp, '.chinese'))
shutil.copy(pj, os.path.join(tmp, 'json_before.json'))
shutil.copy(png, os.path.join(tmp, 'png_before.png'))
r = subprocess.run([PYEXE, os.path.join(tmp, '1.5-3.py')], capture_output=True, cwd=tmp)
print('  exit=', r.returncode)
tail = r.stdout.decode('utf-8', 'ignore').strip().splitlines()[-12:]
print('\n'.join('    ' + x for x in tail))
if r.returncode:
    print('  STDERR:', r.stderr.decode('utf-8', 'ignore')[-1500:])
else:
    pj2 = os.path.join(tmp, 'results_summary.json')
    png2 = os.path.join(tmp, '实验1_5_3_似然等于最小二乘.png')
    if os.path.exists(pj2):
        same = json.load(io.open(pj2, encoding='utf-8')) == json.load(io.open(os.path.join(tmp, 'json_before.json'), encoding='utf-8'))
        print('  JSON 与归档逐位一致:', same)
    if os.path.exists(png2):
        print('  PNG md5 新=%s 旧=%s 一致:%s' % (md5(png2), md5(os.path.join(tmp, 'png_before.png')), md5(png2) == md5(os.path.join(tmp, 'png_before.png'))))
