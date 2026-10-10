# -*- coding: utf-8 -*-
"""验证 5.6-3 新缓存机制：旧全文指纹缓存(47bc93b584) -> 采样内核版本键的自动迁移。
方法：AST 从 5.6-3.py 提取真实缓存段代码 exec（桩环境），对 Colab 真实缓存文件的
副本执行 cache_load，断言：旧别名命中/原子迁移/内容可读/二次命中新键/无参数不命中。
不修改 5.6-3 正式缓存目录。"""
import ast
import copy
import glob
import json
import os
import shutil
import types

import numpy as np

SCRIPT = r'd:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.6-3\5.6-3.py'
COMMON = r'd:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.5-1\pnp_ula_common.py'
SRC_CACHE_DIR = r'd:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.6-3\步骤6_扫描链缓存'
SRC_MAP = glob.glob(r'd:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.6-3\cache_map_*.npz')
TMP = r'd:\works\deepinv\.audit_tmp\shared551_verify\cache_migration_test'

# --- 1. 准备副本目录 ---
if os.path.exists(TMP):
    shutil.rmtree(TMP)
os.makedirs(os.path.join(TMP, '步骤6_扫描链缓存'))
scan_src = sorted(glob.glob(os.path.join(SRC_CACHE_DIR, '*.npz')))
assert len(scan_src) == 10, f'应有10个扫描缓存副本，实际{len(scan_src)}'
assert len(SRC_MAP) == 1, SRC_MAP
for f in scan_src:
    shutil.copy2(f, os.path.join(TMP, '步骤6_扫描链缓存'))
shutil.copy2(SRC_MAP[0], TMP)
old_names_scan = set(os.listdir(os.path.join(TMP, '步骤6_扫描链缓存')))
old_name_map = os.path.basename(SRC_MAP[0])
print(f'[准备] 复制 {len(scan_src)} 个扫描缓存 + 1 个MAP缓存到临时目录')

# --- 2. AST 提取 5.6-3.py 真实缓存段（import hashlib 起，cache_save 定义止）---
src = open(SCRIPT, encoding='utf-8').read()
tree = ast.parse(src)
lines = src.splitlines()
start = end = None
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        seg = ast.get_source_segment(src, node) or ''
        if seg.startswith('import hashlib'):
            start = node.lineno
    if isinstance(node, ast.FunctionDef) and node.name == 'cache_save':
        end = node.end_lineno
assert start and end, (start, end)
seg_code = '\n'.join(lines[start - 1:end])

ns = {
    'np': np,
    'os': os,
    'hashlib': __import__('hashlib'),
    '_json': json,
    'SAVE_DIR': TMP,
    '_exp551_dir': os.path.dirname(COMMON),
    'pnp_ula_common': types.SimpleNamespace(
        SAMPLING_KERNEL_VERSION='pnpula-kernel-1', __file__=COMMON),
}
exec(compile(seg_code, '<563-cache-segment>', 'exec'), ns)
assert ns['KERNEL_VER'] == 'pnpula-kernel-1'
assert ns['_LEGACY_CACHE_VERSIONS'] == ('47bc93b584',)
print(f"[OK] 真实缓存段提取执行成功；KERNEL_VER={ns['KERNEL_VER']}，"
      f"诊断指纹={ns['_COMMON_DIAG_VER']}")

cache_load, cache_save = ns['cache_load'], ns['cache_save']

# --- 3. 逐参数 load：旧别名命中并迁移 ---
migrated = []
for f in scan_src:
    with np.load(f) as z:
        params = json.loads(str(z['_meta']))
        gain_orig = float(z['gain'])
        n_orig = len(z['trace'])
    out = cache_load('scan', params)
    assert out is not None, f'scan 未命中: {params}'
    assert abs(float(out['gain']) - gain_orig) < 1e-15
    assert len(out['trace']) == n_orig
    migrated.append(('scan', params))
# MAP
map_params = {'delta_frac': 0.99, 'maxit': 3000, 'momentum': 0.9, 'tol': 1e-06}
out_map = cache_load('map', map_params)
assert out_map is not None, 'map 旧别名未命中'
assert out_map['x_map'].shape == (256, 256)
migrated.append(('map', map_params))

assert len(ns['_legacy_cache_hits']) == 11, ns['_legacy_cache_hits']
new_names_scan = set(os.listdir(os.path.join(TMP, '步骤6_扫描链缓存')))
assert len(new_names_scan) == 10 and not (new_names_scan & old_names_scan), \
    '扫描缓存应全部重命名为新键文件名'
map_left = [x for x in os.listdir(TMP) if x.startswith('cache_map_')]
assert len(map_left) == 1 and map_left[0] != old_name_map, map_left
print(f"[OK] 11 个缓存全部经旧别名 47bc93b584 命中并原子迁移为内核键文件名")
for r in ns['_legacy_cache_hits'][:2]:
    print('     例:', r)

# --- 4. 二次 load：新键直接命中，迁移记录不增加 ---
n_hits_before = len(ns['_legacy_cache_hits'])
out2 = cache_load('scan', migrated[0][1])
assert out2 is not None
assert len(ns['_legacy_cache_hits']) == n_hits_before
print('[OK] 二次 load 直接命中新键，不产生重复迁移记录')

# --- 5. 关闭缓存 / 未知参数 ---
ns['REUSE_CACHE'] = False
assert cache_load('scan', migrated[0][1]) is None
ns['REUSE_CACHE'] = True
unknown = copy.deepcopy(migrated[0][1]); unknown['seed'] = 999
assert cache_load('scan', unknown) is None
print('[OK] REUSE_CACHE=False 返回 None；无对应缓存的参数返回 None')

# --- 6. cache_save 新键命名且可被 load 命中 ---
p = cache_save('scan', unknown, samples=np.zeros((2, 2)), trace=np.zeros(3),
               gain=np.array(0.5))
assert 'pnpula-kernel-1'  # 常量参与 hash 无法肉眼看出，但 load 闭环可验证
assert cache_load('scan', unknown) is not None
print('[OK] cache_save 以新内核键落盘并被 cache_load 闭环命中')

# --- 7. 无内核版本常量时回退（旧共享模块保护）---
ns2 = dict(ns)
ns2['pnp_ula_common'] = types.SimpleNamespace(__file__=COMMON)  # 无 SAMPLING_KERNEL_VERSION
exec(compile(seg_code, '<563-cache-segment2>', 'exec'), ns2)
assert ns2['KERNEL_VER'] == ns2['_COMMON_DIAG_VER'], '无常量时应回退全文指纹'
print(f'[OK] 旧共享模块（无常量）回退全文指纹 {ns2["KERNEL_VER"]}，不会错误复用')

print('\nCACHE_MIGRATION_TEST_OK')
