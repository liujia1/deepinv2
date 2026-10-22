# -*- coding: utf-8 -*-
"""14.6-1：改写率同口径 + conf 越界防护 + decode 注释。"""
P = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.6-1\14.6-1.py'
s = open(P, encoding='utf-8').read()
subs = []

# 1) decode 注释与实现对齐
subs.append((
    '''def decode(ids):
    """token id 序列 -> 可读字符串（[MASK] 显示为 ____）"""
    toks = [MASK_TOKEN if i == MASK_ID else id2word[int(i)] for i in ids]''',
    '''def decode(ids):
    """token id 序列 -> 可读字符串。

    注意: [MASK] 不在 id2word 里（id2word 只含 14 个实词，MASK_ID=14 会越过表尾），
    故必须先按 MASK_ID 单独映射为字面量 '[MASK]'，再查实词表——顺序不能反。
    """
    toks = [MASK_TOKEN if i == MASK_ID else id2word[int(i)] for i in ids]''',
))

# 2) conf 越界防护
subs.append((
    '''conf = probs_rec.gather(2, final_ids[None, :, None].expand(probs_rec.shape[0], -1, -1)).squeeze(-1)  # (32, d)''',
    '''conf = probs_rec.gather(2, final_ids[None, :, None].expand(probs_rec.shape[0], -1, -1)).squeeze(-1)  # (32, d)
# 越界防护: softmax 的 float32 归一化误差可能让单路 gather 出 1+ε，而正文/图把 conf
# 当作概率引用（如"0.98 以上"），故显式裁剪到 [0, 1]，避免出现 >1 的"置信度"。
conf = conf.clamp(0.0, 1.0)''',
))

# 3) 改写率同口径
subs.append((
    '''rewrite_mlm = rw_mlm / max(exp_mlm, 1)
rewrite_unif = rw_unif / max(exp_unif, 1)
print(f"  MDLM   版: 模板命中率 {hit_mlm:.3f}, 已揭示token改写率 {rewrite_mlm:.4f}")
print(f"  均匀噪声版: 模板命中率 {hit_unif:.3f}, 已揭示token改写率 {rewrite_unif:.4f}")
print("  (均匀版每位置都可被反复改写; MDLM 版已揭示位置的后验集中在自身, 改写率应显著更低)")''',
    '''# 改写率口径统一: 分子都是"实词 -> 另一个实词"的改写次数（MDLM 的 MASK->实词 属揭示, 不计）；
# 分母统一为"步数×句数×位置数"。旧写法对 MDLM 只除以"已揭示步·位"、对 uniform 除以
# "全部步·位"（uniform 无 MASK, 任何位置都算），两版分母语义不同, 倍数会被口径混合。
n_slots = N_STEPS_MAIN * N_SAMPLES_EVAL * SEQ_LEN
rewrite_mlm_all = rw_mlm / n_slots
rewrite_unif_all = rw_unif / n_slots
rewrite_mlm = rw_mlm / max(exp_mlm, 1)      # 旧口径: 仅已揭示步·位（保留作参考）
rewrite_unif = rw_unif / max(exp_unif, 1)   # 旧口径
print(f"  MDLM   版: 模板命中率 {hit_mlm:.3f}, token改写率(同口径) {rewrite_mlm_all:.4f}"
      f"（若只按已揭示步·位计 {rewrite_mlm:.4f}）")
print(f"  均匀噪声版: 模板命中率 {hit_unif:.3f}, token改写率(同口径) {rewrite_unif_all:.4f}")
print(f"  (改写 = 上一步为实词、这一步被换成另一个实词; 分母 = {N_STEPS_MAIN}步×{N_SAMPLES_EVAL}句×{SEQ_LEN}位置 = {n_slots})")''',
))

# 4) 面板 (d) 改用同口径
subs.append((
    '''              [rewrite_mlm, rewrite_unif], color=['#2a7de1', '#e1812c'], alpha=0.8, width=0.5)
ax.grid(alpha=0.3, axis='y')
ax.set_ylabel('已揭示 token 改写率（$N=32$）', fontsize=11)
ax.set_title('(d) 轨迹形态: 揭示式 vs 改写式', fontsize=12)
for b, v in zip(bars, [rewrite_mlm, rewrite_unif]):
    ax.text(b.get_x() + b.get_width()/2, v + max(rewrite_mlm, rewrite_unif)*0.02 + 0.001,''',
    '''              [rewrite_mlm_all, rewrite_unif_all], color=['#2a7de1', '#e1812c'], alpha=0.8, width=0.5)
ax.grid(alpha=0.3, axis='y')
ax.set_ylabel('token 改写率（每步每位置, $N=32$）', fontsize=11)
ax.set_title('(d) 轨迹形态: 揭示式 vs 改写式', fontsize=12)
for b, v in zip(bars, [rewrite_mlm_all, rewrite_unif_all]):
    ax.text(b.get_x() + b.get_width()/2, v + max(rewrite_mlm_all, rewrite_unif_all)*0.02 + 0.001,''',
))

# 5) json
subs.append((
    '''        'rewrite_rate_mlm': round(float(rewrite_mlm), 6),
        'rewrite_rate_unif': round(float(rewrite_unif), 6),''',
    '''        'rewrite_rate_mlm': round(float(rewrite_mlm_all), 6),
        'rewrite_rate_unif': round(float(rewrite_unif_all), 6),
        'rewrite_rate_mlm_given_exposed': round(float(rewrite_mlm), 6),
        'rewrite_rate_unif_given_exposed': round(float(rewrite_unif), 6),
        'rewrite_rate_note': '同口径 = 改写次数 / (步数×句数×位置数); given_exposed 为旧口径',
''',
))

# 6) 总结段
subs.append((
    '''   - 已揭示 token 改写率: MDLM {rewrite_mlm:.4f} vs 均匀 {rewrite_unif:.4f}''',
    '''   - token 改写率(同口径, 每步每位置): MDLM {rewrite_mlm_all:.4f} vs 均匀 {rewrite_unif_all:.4f}
     (MDLM 若只按已揭示步·位计为 {rewrite_mlm:.4f})''',
))

for old, new in subs:
    c = s.count(old)
    print('count', c, '|', old.strip().splitlines()[0][:46])
    assert c >= 1, 'NOT FOUND'
    s = s.replace(old, new, 1)

open(P, 'w', encoding='utf-8').write(s)
print('OK')
