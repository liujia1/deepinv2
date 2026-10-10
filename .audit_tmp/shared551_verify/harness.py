# -*- coding: utf-8 -*-
"""5.6-1 / 5.6-2 加载器与上游闸门的端到端 harness（合成 npz，不跑采样链）。

每个 case 是独立目录（复制真实脚本 + .chinese），脚本 SAVE_DIR 即 case 目录；
在其兄弟目录放一份 pnp_ula_common.py（模拟“实验5.5-1”目录），使
“找不到共享模块”与“找不到 npz”两条失败路径可以分别验证。

场景：
  new_plateau   : 完整新键 + 末1/4段增益0 dB + 近iid轨迹 -> 无警告
  new_noplateau : 完整新键 + 末1/4段增益>0.2 dB -> 平台警告
  old_keys      : 仅5个必需键 -> 2条“无法核查”警告 + 5.6-2劈半降级
  missing       : 无 npz -> 硬失败，stderr 必须含“运行实验5.5-1”
产物全部留在 .audit_tmp/shared551_verify 下。
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHAPTER = os.path.abspath(os.path.join(
    HERE, '..', '..',
    '第5章 朗之万动力学与得分函数', '第五章配套实验'))
EXP551 = os.path.join(CHAPTER, '实验5.5-1')
PY = sys.executable

# 模拟兄弟“实验5.5-1”目录：只需 pnp_ula_common.py（加载/闸门不依赖 sampling_tools）
FAKE551 = os.path.join(HERE, '实验5.5-1')
os.makedirs(FAKE551, exist_ok=True)
shutil.copy2(os.path.join(EXP551, 'pnp_ula_common.py'),
             os.path.join(FAKE551, 'pnp_ula_common.py'))

rng = np.random.default_rng(0)
H, W = 32, 32
M = 60


def make_npz(path, kind):
    x_true = rng.uniform(0, 1, (H, W)).astype(np.float64)
    y_obs = np.clip(x_true + rng.normal(0, 0.05, (H, W)), 0, 1)
    mc = rng.normal(0.5, 0.1, (M, H, W)).clip(0, 1).astype(np.float64)
    post_mean = mc.mean(axis=0)
    post_var = mc.var(axis=0, ddof=0)
    kw = dict(post_mean=post_mean, post_var=post_var, mc_samples=mc,
              x_true=x_true, y_obs=y_obs)
    if kind in ('new_plateau', 'new_noplateau'):
        if kind == 'new_plateau':
            tp = np.concatenate([np.linspace(20.0, 30.0, 25),
                                 np.full(75, 30.05)])
        else:
            tp = np.linspace(20.0, 30.5, 100)
        kw.update(
            trace_psnr=tp.astype(np.float64),
            # 近 iid 轨迹：IAT≈1，抽稀间隔100充分，不产生抽稀警告
            trace_fid=rng.normal(0, 1, 100).astype(np.float64),
            chain_config=np.array([10000, 500, 5250, 100], dtype=np.int64),
            post_mean_h1=post_mean, post_var_h1=post_var,
            post_mean_h2=post_mean, post_var_h2=post_var,
        )
    np.savez(path, **kw)


def setup_case(exp_name, case, with_npz, kind=None):
    src_script = os.path.join(CHAPTER, f'实验{exp_name}', f'{exp_name}.py')
    case_dir = os.path.join(HERE, f'case_{exp_name}_{case}')
    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(os.path.join(case_dir, '.chinese'))
    shutil.copy2(src_script, os.path.join(case_dir, f'{exp_name}.py'))
    shutil.copy2(os.path.join(CHAPTER, f'实验{exp_name}', '.chinese', 'chinese_font.py'),
                 os.path.join(case_dir, '.chinese', 'chinese_font.py'))
    if with_npz:
        make_npz(os.path.join(case_dir, 'sampling_results.npz'), kind)
    return case_dir


def run_case(exp_name, case, kind, expect_ok):
    case_dir = setup_case(exp_name, case, with_npz=(kind is not None), kind=kind)
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONIOENCODING='utf-8')
    proc = subprocess.run(
        [PY, os.path.join(case_dir, f'{exp_name}.py')],
        capture_output=True, text=True, env=env, encoding='utf-8', errors='replace')
    tag = f'{exp_name}/{case}'
    if expect_ok:
        if proc.returncode != 0:
            print(f'[FAIL] {tag} 应成功但退出码={proc.returncode}')
            print(proc.stdout[-3000:])
            print(proc.stderr[-3000:])
            return False
        js_path = os.path.join(case_dir, 'results_summary.json')
        if not os.path.exists(js_path):
            print(f'[FAIL] {tag} 缺 results_summary.json')
            return False
        with open(js_path, encoding='utf-8') as f:
            js = json.load(f)
        gate = js.get('上游链状态_5.5-1', {})
        warns = gate.get('警告', [])
        print(f'[ok]   {tag}: 退出0, 警告{len(warns)}条')
        return js, warns
    else:
        if proc.returncode == 0:
            print(f'[FAIL] {tag} 应硬失败但退出码=0')
            return False
        blob = proc.stdout + proc.stderr
        if '运行实验5.5-1' not in blob:
            print(f'[FAIL] {tag} 失败消息未提示“运行实验5.5-1”')
            print(blob[-2000:])
            return False
        print(f'[ok]   {tag}: 硬失败且提示先运行5.5-1')
        return True


results = []

# ---- 5.6-1 ----
js, w = run_case('5.6-1', 'new_plateau', 'new_plateau', True)
results.append(len(w) == 0 and js['上游链状态_5.5-1'].get('maxit') == 10000)
js, w = run_case('5.6-1', 'new_noplateau', 'new_noplateau', True)
results.append(any('未确认到平台' in x for x in w))
js, w = run_case('5.6-1', 'old_keys', 'old_keys', True)
# 两条警告措辞有意区分：trace_psnr 缺失=“无法核查”平台，trace_fid 缺失=“无法复核”IAT/ESS
results.append(len(w) == 2
               and any('无法核查' in x for x in w)
               and any('无法复核' in x for x in w))
results.append(run_case('5.6-1', 'missing', None, False))

# ---- 5.6-2 ----
js, w = run_case('5.6-2', 'new_plateau', 'new_plateau', True)
results.append(len(w) == 0 and js['上游链状态_5.5-1'].get('抽稀间隔') == 100)
js, w = run_case('5.6-2', 'new_noplateau', 'new_noplateau', True)
results.append(any('未确认到平台' in x for x in w))
js, w = run_case('5.6-2', 'old_keys', 'old_keys', True)
results.append(len(w) == 2
               and any('无法核查' in x for x in w)
               and any('无法复核' in x for x in w))
results.append(run_case('5.6-2', 'missing', None, False))

if all(results):
    print(f'\nHARNESS_OK ({len(results)}/{len(results)} assertions)')
else:
    print(f'\nHARNESS_FAIL ({sum(results)}/{len(results)})')
    sys.exit(1)
