# -*- coding: utf-8 -*-
"""修复后确认：用真实数值执行 17.5-1.py 中新增的 G=1 基线动态文案段，
   确认输出与 JSON 数值自洽、且不与既有判定冲突。"""
import sys, os, json, re
sys.stdout.reconfigure(encoding='utf-8')

BASE = r'h:\deepinv\deepinv2\第17章 自监督学习与等变架构\第十七章 配套实验\实验17.5-1'
src = open(os.path.join(BASE, '17.5-1.py'), encoding='utf-8').read()
js = json.load(open(os.path.join(BASE, 'results_summary.json'), encoding='utf-8'))
g1 = json.load(open(os.path.join(BASE, '_g1_result.json'), encoding='utf-8'))
reg = js['分区域PSNR结果']['各方法分区域PSNR']

# 构造与主脚本同名的最小环境
ns = {
    'region_results': {**reg, 'MOI G=1': {'total': g1['psnr'], 'obs': g1['obs'],
                                          'miss': g1['miss'], 'gap': g1['gap']}},
    'moi_results': {1: {'psnr': g1['psnr'], 'miss': g1['miss']},
                    2: {'psnr': reg['MOI G=2']['total']},
                    4: {'psnr': reg['MOI G=4']['total']},
                    8: {'psnr': reg['MOI G=8']['total']}},
    'psnr_sup': reg['监督']['total'],
    'moi_ei_desc': '(此处为 moi_ei_desc，实际由主脚本动态生成)',
}
seg = src[src.index("moi_g8_gap = psnr_sup"): src.index("# 详细版（含数值")]
seg2 = src[src.index("# ★ G=1 纯MC基线"): src.index("# ★ 动态校验")]
print('=== 执行 G=1 基线文案段 ===')
exec(compile(seg2, 'seg_g1desc', 'exec'), ns)
print('mc_baseline_desc =', ns['mc_baseline_desc'])
print()
print('=== 校验 ===')
assert '14.15' in ns['mc_baseline_desc'] or '14.1' in ns['mc_baseline_desc'], '应含真实G=1数值'
assert ns['g1_psnr'] == g1['psnr'] and ns['g1_miss'] == g1['miss']
# 与既有判定一致性：G=1 必须低于 G=2/4/8 与 EI
ei = reg['单算子+EI']['total']
assert g1['psnr'] < min(reg['MOI G=2']['total'], reg['MOI G=4']['total'],
                        reg['MOI G=8']['total'], ei), 'G=1 应低于所有带零空间约束的方法'
print(f"G=1 ({g1['psnr']:.2f}) < EI ({ei:.2f}) 且 < 所有 MOI(G>=2) → 文案成立 ✓")
print('已确认：动态文案使用真实数值、结论方向正确')
