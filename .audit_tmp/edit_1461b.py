# -*- coding: utf-8 -*-
"""14.6-1 第二版：改写率主口径 = 条件于"更新前为实词"（两版同一条件语义），
另附"每步每位置"口径作对照；面板/JSON/总结同步。"""
P = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.6-1\14.6-1.py'
s = open(P, encoding='utf-8').read()
subs = []

# 1) 打印段
subs.append((
    '''n_slots = N_STEPS_MAIN * N_SAMPLES_EVAL * SEQ_LEN
rewrite_mlm_all = rw_mlm / n_slots
rewrite_unif_all = rw_unif / n_slots
rewrite_mlm = rw_mlm / max(exp_mlm, 1)      # 旧口径: 仅已揭示步·位（保留作参考）
rewrite_unif = rw_unif / max(exp_unif, 1)   # 旧口径
print(f"  MDLM   版: 模板命中率 {hit_mlm:.3f}, token改写率(同口径) {rewrite_mlm_all:.4f}"
      f"（若只按已揭示步·位计 {rewrite_mlm:.4f}）")
print(f"  均匀噪声版: 模板命中率 {hit_unif:.3f}, token改写率(同口径) {rewrite_unif_all:.4f}")
print(f"  (改写 = 上一步为实词、这一步被换成另一个实词; 分母 = {N_STEPS_MAIN}步×{N_SAMPLES_EVAL}句×{SEQ_LEN}位置 = {n_slots})")''',
    '''# 两个口径并列（分子相同: "实词 -> 另一个实词"的改写次数, MDLM 的 MASK->实词 属揭示不计）:
#   主口径 = 分子 / "更新前处于实词状态的步·位"——条件语义两版一致（给定当前是实词, 被改写的概率）;
#   附口径 = 分子 / "全部步·位"——MDLM 的 MASK 位置不可能改写, 会把率稀释近一半。
n_slots = N_STEPS_MAIN * N_SAMPLES_EVAL * SEQ_LEN
rewrite_mlm_all = rw_mlm / n_slots
rewrite_unif_all = rw_unif / n_slots
rewrite_mlm = rw_mlm / max(exp_mlm, 1)      # 主口径: 条件于更新前为实词
rewrite_unif = rw_unif / max(exp_unif, 1)   # 主口径（uniform 无 MASK, 分母即全部步·位）
print(f"  MDLM   版: 模板命中率 {hit_mlm:.3f}, token改写率 {rewrite_mlm:.4f}（每步每位置计 {rewrite_mlm_all:.4f}）")
print(f"  均匀噪声版: 模板命中率 {hit_unif:.3f}, token改写率 {rewrite_unif:.4f}（每步每位置计 {rewrite_unif_all:.4f}）")
print(f"  (改写 = 更新前为实词、这一步被换成另一个实词; 主口径分母 = 更新前实词状态的步·位, "
      f"MDLM 共 {exp_mlm}, 均匀版共 {exp_unif}（= {N_STEPS_MAIN}步×{N_SAMPLES_EVAL}句×{SEQ_LEN}位置）)")''',
))

# 2) 面板 (d) 用主口径
subs.append((
    '''              [rewrite_mlm_all, rewrite_unif_all], color=['#2a7de1', '#e1812c'], alpha=0.8, width=0.5)
ax.grid(alpha=0.3, axis='y')
ax.set_ylabel('token 改写率（每步每位置, $N=32$）', fontsize=11)
ax.set_title('(d) 轨迹形态: 揭示式 vs 改写式', fontsize=12)
for b, v in zip(bars, [rewrite_mlm_all, rewrite_unif_all]):
    ax.text(b.get_x() + b.get_width()/2, v + max(rewrite_mlm_all, rewrite_unif_all)*0.02 + 0.001,''',
    '''              [rewrite_mlm, rewrite_unif], color=['#2a7de1', '#e1812c'], alpha=0.8, width=0.5)
ax.grid(alpha=0.3, axis='y')
ax.set_ylabel('token 改写率（更新前为实词的位置）', fontsize=11)
ax.set_title('(d) 轨迹形态: 揭示式 vs 改写式', fontsize=12)
for b, v in zip(bars, [rewrite_mlm, rewrite_unif]):
    ax.text(b.get_x() + b.get_width()/2, v + max(rewrite_mlm, rewrite_unif)*0.02 + 0.001,''',
))

# 3) json
subs.append((
    '''        'rewrite_rate_mlm': round(float(rewrite_mlm_all), 6),
        'rewrite_rate_unif': round(float(rewrite_unif_all), 6),
        'rewrite_rate_mlm_given_exposed': round(float(rewrite_mlm), 6),
        'rewrite_rate_unif_given_exposed': round(float(rewrite_unif), 6),
        'rewrite_rate_note': '同口径 = 改写次数 / (步数×句数×位置数); given_exposed 为旧口径',''',
    '''        'rewrite_rate_mlm': round(float(rewrite_mlm), 6),
        'rewrite_rate_unif': round(float(rewrite_unif), 6),
        'rewrite_rate_mlm_per_slot': round(float(rewrite_mlm_all), 6),
        'rewrite_rate_unif_per_slot': round(float(rewrite_unif_all), 6),
        'rewrite_rate_note': '主口径 = 改写次数 / 更新前为实词的步·位（两版同一条件语义）; per_slot = 改写次数 / (步数×句数×位置数)',''',
))

# 4) 总结段
subs.append((
    '''   - token 改写率(同口径, 每步每位置): MDLM {rewrite_mlm_all:.4f} vs 均匀 {rewrite_unif_all:.4f}
     (MDLM 若只按已揭示步·位计为 {rewrite_mlm:.4f})''',
    '''   - token 改写率(更新前为实词的位置): MDLM {rewrite_mlm:.4f} vs 均匀 {rewrite_unif:.4f}
     (若按每步每位置计: MDLM {rewrite_mlm_all:.4f} vs 均匀 {rewrite_unif_all:.4f})''',
))

for old, new in subs:
    c = s.count(old)
    print('count', c, '|', old.strip().splitlines()[0][:44])
    assert c >= 1, 'NOT FOUND'
    s = s.replace(old, new, 1)

open(P, 'w', encoding='utf-8').write(s)
print('OK')
