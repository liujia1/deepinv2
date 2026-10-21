# -*- coding: utf-8 -*-
"""14.4-3：固定 CPU（跨机器可复现）+ 训练 loss 存档进 json。"""
P = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.4-3\14.4-3.py'
s = open(P, encoding='utf-8').read()
subs = []

# 1) 固定 CPU
subs.append((
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    "device = torch.device('cpu')  # 固定 CPU：2D 点云无需 GPU，也避免有卡的机器悄悄换设备导致数值不可复现",
))

# 2) 收集各轮 loss 曲线
subs.append((
    """                             final_checkpoint_path=final_paths[1],
                             model_label='1-RF')""",
    """                             final_checkpoint_path=final_paths[1],
                             model_label='1-RF')
losses_by_round = {1: losses_1rf}  # 各轮训练 loss 曲线，供 json 存档""",
))
subs.append((
    """    # 计算曲率（使用固定z_curv，CRN原则确保跨轮比较不受采样噪声干扰）""",
    """    losses_by_round[k] = losses_k  # 存档本轮 loss 曲线

    # 计算曲率（使用固定z_curv，CRN原则确保跨轮比较不受采样噪声干扰）""",
))

# 3) json 增 loss 字段
subs.append((
    """results_summary = {
    'curvature_1rf': round(float(curvatures[1]), 6),
    'curvature_2rf': round(float(curvatures[2]), 6),
    'curvature_3rf': round(float(curvatures[3]), 6),
    'monotonic': bool(monotonic),
}""",
    """def _loss_tail(losses):
    \"\"\"末值 + 每 50 epoch 抽一点的曲线（含首末），空列表返回 None/[]。\"\"\"
    if not losses:
        return None, []
    curve = [round(float(v), 6) for v in losses[:-1:50]] + [round(float(losses[-1]), 6)]
    return round(float(losses[-1]), 6), curve

_l1, _c1 = _loss_tail(losses_by_round[1])
_l2, _c2 = _loss_tail(losses_by_round[2])
_l3, _c3 = _loss_tail(losses_by_round[3])
results_summary = {
    'curvature_1rf': round(float(curvatures[1]), 6),
    'curvature_2rf': round(float(curvatures[2]), 6),
    'curvature_3rf': round(float(curvatures[3]), 6),
    'monotonic': bool(monotonic),
    # 训练 loss 存档：正文「1.865 -> 0.00098 断崖」必须可核对
    'final_loss_1rf': _l1,
    'final_loss_2rf': _l2,
    'final_loss_3rf': _l3,
    'loss_curve_1rf': _c1,
    'loss_curve_2rf': _c2,
    'loss_curve_3rf': _c3,
    'loss_note': 'final_loss_* = 最后一个 epoch 的样本加权 MSE；loss_curve_* = 每 50 epoch 抽 1 点（含首末）；device=cpu',
}""",
))

for old, new in subs:
    c = s.count(old)
    print('count', c, '|', old.strip().splitlines()[0][:52])
    assert c == 1, 'COUNT NOT 1'
    s = s.replace(old, new)

open(P, 'w', encoding='utf-8').write(s)
print('OK')
