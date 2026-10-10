# -*- coding: utf-8 -*-
"""复核 5.6-3 Colab 新产物：扫描链缓存身份/内容、MAP缓存、JSON勾稽。"""
import os, json, math, glob, hashlib
import numpy as np

D = r'd:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.6-3'
CD = os.path.join(D, '步骤6_扫描链缓存')
j = json.load(open(os.path.join(D, 'results_summary.json'), encoding='utf-8'))
s6 = j['步骤6_参数敏感性']

files = sorted(glob.glob(os.path.join(CD, '*.npz')))
print(f'扫描缓存文件数: {len(files)}')
assert len(files) == 10, '应有10条链缓存'

metas = []
for f in files:
    z = np.load(f)
    assert set(['samples', 'trace', 'gain', '_meta']) <= set(z.files), f'{f} 键不全'
    meta = json.loads(str(z['_meta']))
    size = os.path.getsize(f)
    metas.append((os.path.basename(f), meta, z['samples'].shape, len(z['trace']),
                  float(z['gain']), size))
    z.close()

# 身份唯一性：10 组参数互不相同
keys = [(m['eps'], m['delta_frac'], m['seed']) for _, m, *_ in metas]
assert len(set(keys)) == 10, '缓存参数指纹有重复'

eps_rows = [t for t in metas if abs(t[1]['delta_frac'] - 0.99) < 1e-12]
delta_rows = [t for t in metas if abs(t[1]['eps'] - metas[0][1]['eps']) > 0
              and any(abs(t[1]['delta_frac'] - x) < 1e-12 for x in [0.1,0.3,0.5,0.7,0.99])]
# 更稳的划分：seed 100..104 = eps 链，200..204 = delta 链
eps_rows = sorted([t for t in metas if 100 <= t[1]['seed'] <= 104], key=lambda t: t[1]['eps'])
delta_rows = sorted([t for t in metas if 200 <= t[1]['seed'] <= 204],
                    key=lambda t: t[1]['delta_frac'])
assert len(eps_rows) == 5 and len(delta_rows) == 5

print('\nε链(seed100-104): eps / sample形状 / trace长 / gain / 文件MB')
for name, m, shp, nt, g, size in eps_rows:
    print(f"  {m['eps']:.3e} {shp} trace={nt} gain={g:+.4f} {size/1e6:.1f}MB")

print('δ链(seed200-204): delta_frac / sample形状 / trace长 / gain / 文件MB')
for name, m, shp, nt, g, size in delta_rows:
    print(f"  {m['delta_frac']:g} {shp} trace={nt} gain={g:+.4f} {size/1e6:.1f}MB")

# 1) gain 与 JSON 逐点勾稽
for (_, m, _, _, g, _), gj in zip(eps_rows, s6['epsilon扫描链末1/4段PSNR提升_dB']):
    assert abs(g - gj) < 5e-4, (g, gj)
for (_, m, _, _, g, _), gj in zip(delta_rows, s6['步长扫描链末1/4段PSNR提升_dB']):
    assert abs(g - gj) < 5e-4, (g, gj)
print('\n[OK] 10个缓存 gain 与 JSON 逐点一致')

# 2) ε链平均后验方差（ddof=1，全像素均值）重算勾稽
for (name, m, shp, nt, g, size), vj in zip(eps_rows, s6['epsilon对应平均后验方差']):
    z = np.load(os.path.join(CD, name))
    v = float(np.mean(np.var(z['samples'], axis=0, ddof=1)))
    z.close()
    assert abs(v - vj) / abs(vj) < 1e-9, (v, vj)
print('[OK] ε链平均后验方差由缓存重算与 JSON 一致（<1e-9相对差）')

# 3) δ链 lag-1 ESS / 均值SE 重算勾稽
for (name, m, shp, nt, g, size), ej, sj in zip(
        delta_rows, s6['步长对应ESS_lag1近似'], s6['步长对应均值估计标准误差']):
    z = np.load(os.path.join(CD, name))
    tr = np.asarray(z['trace'], dtype=np.float64)
    z.close()
    tr = tr - tr.mean()
    rho = float(np.corrcoef(tr[:-1], tr[1:])[0, 1])
    rho = min(max(rho, 0.0), 0.999)
    ess = len(tr) * (1 - rho) / (1 + rho)
    se = float(np.std(tr, ddof=1) / (math.sqrt(max(ess, 1.0)) + 1e-12))
    assert abs(ess - ej) / ej < 1e-9, (ess, ej)
    assert abs(se - sj) / sj < 1e-9, (se, sj)
print('[OK] δ链 lag-1 ESS 与均值SE由缓存trace重算与 JSON 一致')

# 4) 采样配置：trace 烧入后逐步记录 maxit-burnin-1=1499 点；
#    mc_samples 粗化 thinning=int(maxit/n_samples)：int(2000/60)=33→1499//33=45；
#    int(2000/120)=16→1499//16=93
for _, m, shp, nt, _, _ in eps_rows:
    assert m['maxit'] == 2000 and m['burnin'] == 500 and m['trace_every'] == 25
    assert shp[0] == 1499 // int(2000 / 60) and nt == 1499, (shp, nt)
for _, m, shp, nt, _, _ in delta_rows:
    assert shp[0] == 1499 // int(2000 / 120) and nt == 1499, (shp, nt)
print(f'[OK] 配置一致；ε链粗化45样本、δ链粗化93样本，trace长度={nt}（均与共享模块逻辑吻合）')

# 5) MAP 缓存复核
mf = glob.glob(os.path.join(D, 'cache_map_*.npz'))
assert len(mf) == 1
mz = np.load(mf[0])
xmap = mz['x_map']
meta = json.loads(str(mz['_meta']))
mz.close()
print(f"\nMAP缓存: {os.path.basename(mf[0])}, shape={xmap.shape}, 参数={meta}")
assert xmap.shape == (256, 256) and xmap.min() >= 0 and xmap.max() <= 1
npz551 = np.load(r'd:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.5-1\sampling_results.npz')
xt = npz551['x_true']; npz551.close()
mse = float(np.mean((xmap - xt) ** 2))
psnr = 10 * math.log10(1.0 / mse)
assert abs(psnr - j['步骤2_MAP_vs_MMSE对比']['MAP估计_确定性PnP']['PSNR_dB']) < 1e-3
print(f'[OK] MAP缓存重算 PSNR={psnr:.4f} 与 JSON {32.5683} 一致')

# 6) 指纹：记录 Colab 产出版本与本地当前共享模块版本（本地已升级 IAT 诊断时预期不同）
colab_cache = s6['结果缓存']
local_ver = hashlib.md5(open(
    r'd:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.5-1\pnp_ula_common.py',
    'rb').read()).hexdigest()[:10]
if '采样内核版本' in colab_cache:
    # 新机制：内核版本决定缓存身份，诊断指纹仅展示
    import sys as _sys
    _sys.path.insert(0, r'd:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.5-1')
    import pnp_ula_common as _m
    _k = getattr(_m, 'SAMPLING_KERNEL_VERSION', None)
    assert colab_cache['采样内核版本'] == _k, (colab_cache['采样内核版本'], _k)
    print(f"[OK] Colab采样内核版本={_k} 与本地一致；"
          f"Colab诊断指纹={colab_cache.get('共享模块诊断指纹')}，本地={local_ver}")
    print(f"     旧别名迁移记录 {len(colab_cache.get('旧别名缓存迁移记录', []))} 条"
          f"（MAP+10链应共11条）")
    assert len(colab_cache.get('旧别名缓存迁移记录', [])) == 11
else:
    colab_ver = colab_cache['共享模块版本指纹']  # 旧版JSON
    print(f'[信息] Colab产出指纹={colab_ver}；本地共享模块指纹={local_ver}；'
          + ('一致' if local_ver == colab_ver else '不一致（本地已升级，Colab为旧版，诊断口径需同步）'))

# 7) 用本地新版 iat_geyer 对同一份 npz 重算，记录口径差异
import sys
sys.path.insert(0, r'd:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.5-1')
from pnp_ula_common import iat_geyer
npz551b = np.load(r'd:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.5-1\sampling_results.npz')
iat_n, ess_n, diag_n = iat_geyer(npz551b['trace_fid'].astype(np.float64))
npz551b.close()
print(f'[信息] 同一份npz：旧版IAT=788.7/ESS=12.0(被lag1000截断) → '
      f'新版IAT>={iat_n:.1f}(下界)/ESS<={ess_n:.1f}(上界), truncated={diag_n["truncated"]}')
assert diag_n['truncated'] is True
# 定性结论不变：抽稀100仍远小于新IAT
assert 100 < iat_n

print('\n563_CACHE_AUDIT_OK')
