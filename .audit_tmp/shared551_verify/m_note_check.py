# -*- coding: utf-8 -*-
"""验证5.6-3 M实验独立性注记三分支（用真实npz=强自相关分支；另两个模拟）。"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP551 = os.path.abspath(os.path.join(
    HERE, '..', '..',
    '第5章 朗之万动力学与得分函数', '第五章配套实验', '实验5.5-1'))
sys.path.insert(0, EXP551)
from pnp_ula_common import load_sampling_result, upstream_chain_gate


def branch(iat, thin, ess, n_mc):
    if iat is None:
        independent = None
        note = '独立性无法核查'
    elif thin is not None and thin >= iat:
        independent = True
        note = '近似独立'
    else:
        independent = False
        note = (f"抽稀间隔 {thin} 步 < 全链IAT {iat:.0f} 步：{n_mc}个样本仍强自相关"
                f"（全链ESS仅{ess:.0f}）")
    return independent, note


# 真实 npz（应为 False 分支）
data, path = load_sampling_result(EXP551)
_, _, extras = upstream_chain_gate(data, path)
ind, note = branch(extras['iat'], extras['thinning'], extras['ess'],
                   len(data['mc_samples']))
print('真实npz ->', ind, '|', note)
assert ind is False and '789' in note and '94' in note

# 模拟另两分支
ind2, note2 = branch(1.31, 100, 76.6, 94)
assert ind2 is True
ind3, note3 = branch(None, None, None, 94)
assert ind3 is None
print('plateau模拟 ->', ind2, '|', note2)
print('oldkeys模拟 ->', ind3, '|', note3)
print('M_NOTE_OK')
