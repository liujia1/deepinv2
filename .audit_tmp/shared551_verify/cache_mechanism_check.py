# -*- coding: utf-8 -*-
"""独立验证 5.6-3 新增的缓存机制（不导入 torch/整个脚本，复刻缓存函数）。"""
import os, json, hashlib, shutil
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(ROOT, '_cache_probe_tmp')
SCAN_CACHE_DIR = os.path.join(SAVE_DIR, '步骤6_扫描链缓存')
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)

REUSE_CACHE = True
_COMMON_VER = 'abcdef1234'

def _fp(kind, params):
    src = json.dumps(params, sort_keys=True, ensure_ascii=False)
    h = hashlib.md5((kind + '|' + src + '|' + _COMMON_VER).encode('utf-8')).hexdigest()[:16]
    return h, src

def cache_load(kind, params):
    if not REUSE_CACHE:
        return None
    h, _ = _fp(kind, params)
    path = (os.path.join(SCAN_CACHE_DIR, h + '.npz') if kind == 'scan'
            else os.path.join(SAVE_DIR, f'cache_{kind}_{h}.npz'))
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    except Exception as e:
        return None

def cache_save(kind, params, **arrays):
    h, src = _fp(kind, params)
    if kind == 'scan':
        os.makedirs(SCAN_CACHE_DIR, exist_ok=True)
        path = os.path.join(SCAN_CACHE_DIR, h + '.npz')
    else:
        path = os.path.join(SAVE_DIR, f'cache_{kind}_{h}.npz')
    arrays['_meta'] = np.array(src)
    np.savez(path, **arrays)
    return path

p = dict(eps=0.001, delta_frac=0.99, maxit=2000, burnin=500,
         n_samples=60, seed=100, burn_frac=0.25, trace_every=25)

# 1) 首次 miss
assert cache_load('scan', p) is None, '首次应为 miss'
s = np.random.rand(60, 256, 256).astype(np.float32)
tr = np.arange(60.0)
g = 0.18
fn = cache_save('scan', p, samples=s, trace=tr, gain=np.array(g))
assert os.path.exists(fn) and '步骤6_扫描链缓存' in fn

# 2) 同参数 hit，数据逐元素一致
z = cache_load('scan', p)
assert z is not None, '保存后应 hit'
assert np.array_equal(z['samples'], s)
assert np.allclose(z['trace'], tr) and abs(float(z['gain']) - g) < 1e-12
assert json.loads(str(z['_meta']))['eps'] == 0.001

# 3) 改任一参数 -> miss（独立文件）
p2 = dict(p); p2['seed'] = 101
assert cache_load('scan', p2) is None, '参数变应 miss'
cache_save('scan', p2, samples=s, trace=tr, gain=np.array(g))
assert len(os.listdir(SCAN_CACHE_DIR)) == 2, '两条不同参数链=两个缓存文件'

# 4) 改共享模块版本 -> 指纹变化 -> miss
_COMMON_VER_OLD = _COMMON_VER
_COMMON_VER = '0000000000'
assert cache_load('scan', p) is None, '版本变应 miss'
_COMMON_VER = _COMMON_VER_OLD

# 5) kind 隔离：map 缓存不在扫描目录
mp = dict(maxit=3000, delta_frac=0.99, momentum=0.9, tol=1e-6)
assert cache_load('map', mp) is None
xm = np.random.rand(256, 256).astype(np.float32)
mfn = cache_save('map', mp, x_map=xm)
assert os.path.dirname(mfn) == SAVE_DIR and cache_load('map', mp) is not None
assert np.array_equal(cache_load('map', mp)['x_map'], xm)

# 6) REUSE_CACHE=False -> 一律 miss
REUSE_CACHE = False
assert cache_load('scan', p) is None and cache_load('map', mp) is None
REUSE_CACHE = True

# 7) 损坏文件不崩，返回 None
bad = fn
with open(bad, 'wb') as f:
    f.write(b'not a real npz')
assert cache_load('scan', p) is None, '损坏缓存应优雅降级'

shutil.rmtree(SAVE_DIR)
print('CACHE_MECHANISM_OK')
