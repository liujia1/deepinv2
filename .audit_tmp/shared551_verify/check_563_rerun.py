# -*- coding: utf-8 -*-
"""5.6-3 Colab 重跑产物闭环复核（2026-10-10 第二轮，新IAT口径+缓存迁移）。
校验：
  1. 闸门为新口径（IAT下界/ESS上界/truncated 警告2条）；M注记用新IAT 1101
  2. 版本键与迁移记录：JSON 声明键能由 _meta+ver 重算命中 11 个新缓存文件
  3. 迁移=纯重命名：迁移记录 old->new 两文件原始字节 MD5 完全相同（证明零重算）
  4. 新缓存数值（gain/ε方差/δ ESS/SE）与 JSON 逐点勾稽；MAP PSNR 重算
  5. 新旧两轮 JSON 数值逐位一致（MMSE/MAP/扫描/M实验）
"""
import os
import json
import glob
import math
import hashlib

import numpy as np

D = r'd:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.6-3'
CD = os.path.join(D, '步骤6_扫描链缓存')
j = json.load(open(os.path.join(D, 'results_summary.json'), encoding='utf-8'))
s6 = j['步骤6_参数敏感性']
cache_info = s6['结果缓存']
ver = cache_info['采样内核版本']
print(f'[信息] JSON 声明缓存版本键 = {ver}（诊断指纹={cache_info["共享模块诊断指纹"]}）')

# --- 1. 闸门新口径 ---
g = j['上游链状态_5.5-1']
assert isinstance(g['全链IAT_步'], str) and '下界' in g['全链IAT_步'] and '1100.65' in g['全链IAT_步']
assert isinstance(g['全链ESS'], str) and '上界' in g['全链ESS'] and '8.6' in g['全链ESS']
assert g['抽稀间隔_ge_IAT'] is False
assert len(g['警告']) == 2 and '3166' in g['警告'][0] and '1100.7' in g['警告'][1]
assert '全链IAT 1101 步' in s6['M实验说明'] and 'ESS仅9' in s6['M实验说明']
print('[OK] 闸门为新口径：IAT>=1100.65(下界)/ESS<=8.6(上界)/2条警告；M注记 IAT1101/ESS9')

# --- 2. 版本键重算定位当前代缓存 ---
def fp(kind, params, ver):
    src = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.md5((kind + '|' + src + '|' + str(ver)).encode('utf-8')).hexdigest()[:16]

all_scan = glob.glob(os.path.join(CD, '*.npz'))
current = {}
for f in all_scan:
    with np.load(f) as z:
        params = json.loads(str(z['_meta']))
    if fp('scan', params, ver) == os.path.basename(f)[:-4]:
        current[os.path.basename(f)] = f
assert len(current) == 10, f'当前代扫描缓存应10个，实际{len(current)}'
map_params = {'delta_frac': 0.99, 'maxit': 3000, 'momentum': 0.9, 'tol': 1e-06}
map_name = 'cache_map_' + fp('map', map_params, ver) + '.npz'
assert os.path.exists(os.path.join(D, map_name)), f'MAP当前代缓存缺失: {map_name}'
print(f'[OK] 版本键 {ver} 重算命中 10 条扫描链 + MAP 缓存 {map_name}')

# --- 3. 迁移记录：old->new 字节级相同 ---
recs = cache_info['旧别名缓存迁移记录']
assert len(recs) == 11, f'迁移记录应11条，实际{len(recs)}'
for r in recs:
    old, new = [x.strip() for x in r.split('->')]
    po = os.path.join(D if old.startswith('cache_') else CD, old)
    pn = os.path.join(D if new.startswith('cache_') else CD, new)
    assert os.path.exists(po) and os.path.exists(pn), f'迁移文件缺失: {r}'
    ho = hashlib.md5(open(po, 'rb').read()).hexdigest()
    hn = hashlib.md5(open(pn, 'rb').read()).hexdigest()
    assert ho == hn, f'迁移前后字节不同(疑似重跑): {r}'
    if new.startswith('cache_'):
        assert new == map_name
print('[OK] 11 条迁移记录 old->new 原始字节 MD5 全部相同（纯重命名，Colab 零重算）')

# --- 4. 当前代缓存数值与 JSON 勾稽 ---
rows = []
for name, f in current.items():
    with np.load(f) as z:
        rows.append((json.loads(str(z['_meta'])), z['samples'].shape,
                     len(z['trace']), float(z['gain']), f))
eps_rows = sorted([t for t in rows if 100 <= t[0]['seed'] <= 104], key=lambda t: t[0]['eps'])
del_rows = sorted([t for t in rows if 200 <= t[0]['seed'] <= 204], key=lambda t: t[0]['delta_frac'])
assert len(eps_rows) == 5 and len(del_rows) == 5

for (_, _, _, gv, f), gj in zip(eps_rows, s6['epsilon扫描链末1/4段PSNR提升_dB']):
    assert abs(gv - gj) < 5e-4
for (_, _, _, gv, f), gj in zip(del_rows, s6['步长扫描链末1/4段PSNR提升_dB']):
    assert abs(gv - gj) < 5e-4
print('[OK] 10 链 gain 与 JSON 逐点一致')

for (_, _, _, _, f), vj in zip(eps_rows, s6['epsilon对应平均后验方差']):
    with np.load(f) as z:
        v = float(np.mean(np.var(z['samples'], axis=0, ddof=1)))
    assert abs(v - vj) / abs(vj) < 1e-9
print('[OK] ε链平均后验方差重算与 JSON 一致（<1e-9）')

for (_, _, _, _, f), sej, ej in zip(del_rows, s6['步长对应均值估计标准误差'],
                                    s6['步长对应ESS_lag1近似']):
    with np.load(f) as z:
        tr = np.asarray(z['trace'], dtype=np.float64)
    lag1 = np.corrcoef(tr[:-1], tr[1:])[0, 1]
    ess = len(tr) * (1 - lag1) / (1 + lag1)
    se = float(np.std(tr, ddof=1) / math.sqrt(ess))
    assert abs(ess - ej) / ej < 1e-9 and abs(se - sej) / sej < 1e-9
print('[OK] δ链 lag-1 ESS 与均值SE重算与 JSON 一致')

with np.load(os.path.join(D, map_name)) as z:
    x_map = np.asarray(z['x_map'])
with np.load(r'd:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.5-1\sampling_results.npz') as z5:
    x_true = np.asarray(z5['x_true'], dtype=np.float64)
mse = float(np.mean((np.clip(x_map, 0, 1) - x_true) ** 2))
psnr = 10 * math.log10(1.0 / mse)
assert abs(psnr - j['步骤2_MAP_vs_MMSE对比']['MAP估计_确定性PnP']['PSNR_dB']) < 1e-3
print(f'[OK] MAP缓存重算 PSNR={psnr:.4f} 与 JSON 一致')

# M 实验缺失点
se_m = s6['样本数对应方差估计标准误差']
assert se_m[:3] == [1.5170721742894024e-05, 1.0103178529947664e-05, 3.98477243765379e-06]
assert se_m[3] is None and s6['样本数测试值_实际使用样本数'][3] == 94
assert len(s6['M实验有限总体警告']) == 1
print('[OK] M实验：前3点SE一致，M=100=null（有限总体效应）')

# --- 5. 与上轮（旧口径 47bc93b584）数值逐位一致 ---
expect = {
    'MMSE': (33.1169, 0.996634, 0.082131, 0.043454),
    'MAP':  (32.5683, 0.996177, 0.087486, 0.041809),
}
mm = j['步骤2_MAP_vs_MMSE对比']['MMSE估计_后验均值']
mp = j['步骤2_MAP_vs_MMSE对比']['MAP估计_确定性PnP']
assert (mm['PSNR_dB'], mm['全局相似度'], mm['NRMSE'], mm['高频能量']) == expect['MMSE']
assert (mp['PSNR_dB'], mp['全局相似度'], mp['NRMSE'], mp['高频能量']) == expect['MAP']
assert s6['epsilon对应平均后验方差'] == [0.00021456604008562863, 0.000450222403742373,
    0.0011725517688319087, 0.0013785955961793661, 0.0014441106468439102]
assert s6['步长对应ESS_lag1近似'] == [66.96579855021403, 279.42214411200445,
    468.24430389415306, 763.6479654660014, 1390.9658700663583]
print('[OK] MMSE/MAP/ε方差/δESS 与上轮旧口径产物逐位一致（诊断升级未扰动任何数值）')

# --- 6. 本地共享模块 vs Colab 实际版本 ---
local_common = r'd:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.5-1\pnp_ula_common.py'
local_ver = hashlib.md5(open(local_common, 'rb').read()).hexdigest()[:10]
has_const = b'SAMPLING_KERNEL_VERSION' in open(local_common, 'rb').read()
if ver == local_ver:
    print(f'[OK] Colab 与本地共享模块同为 {ver}（语义内核常量已生效）')
else:
    print(f'[信息] Colab 键={ver}，本地共享模块={local_ver}（含SAMPLING_KERNEL_VERSION={has_const}）：'
          f'Colab 上传的是无内核常量版本，5.6-3.py 已安全回退全文MD5键，迁移与数值均正确；'
          f'若要规范为语义键，需重传含常量版并将 {ver} 加入别名列表后重跑')

print('\n563_RERUN_AUDIT_OK')
