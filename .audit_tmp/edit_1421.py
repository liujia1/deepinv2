# -*- coding: utf-8 -*-
"""14.2-1：桥接点占比 / NLL 20·100 步重估 / 理论熵 入脚本与 json，去掉总结里的硬编码。"""
P = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.2-1\14.2-1.py'
s = open(P, encoding='utf-8').read()
subs = []

# 1) 计算段：插在 MMD 之后
subs.append((
    """mmd_cnll = compute_mmd(samples_cnll, test_data)
mmd_fm = compute_mmd(samples_fm, test_data)""",
    """mmd_cnll = compute_mmd(samples_cnll, test_data)
mmd_fm = compute_mmd(samples_fm, test_data)

# ---- 桥接点占比：样本到最近高斯中心的距离 > 0.5 的比例（正文引用的"跨众数弧线"指标）----
def bridge_fraction(x, thresh=0.5, radius=2.0, n_modes=8):
    ang = torch.arange(n_modes, device=x.device) * 2 * np.pi / n_modes
    centers = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=1)
    d_near = torch.cdist(x, centers).min(dim=1)[0]
    return (d_near > thresh).float().mean().item()

bridge_cnll = bridge_fraction(samples_cnll)
bridge_fm = bridge_fraction(samples_fm)
bridge_real = bridge_fraction(test_data)

# ---- 同一 CNLL 模型换 20/100 步重估 NLL：诊断离散化偏差（Hutchinson 探针固定种子）----
model_cnll.eval()
eval_data = train_data[:512]
with torch.no_grad():
    torch.manual_seed(7)
    nll_20_eval = -compute_log_likelihood(model_cnll, eval_data, n_steps=20).mean().item()
    torch.manual_seed(7)
    nll_100_eval = -compute_log_likelihood(model_cnll, eval_data, n_steps=100).mean().item()
# 目标分布微分熵（等权 8 高斯、分量几乎不重叠）：H ≈ log 8 + log(2πeσ²)，σ=0.15，单位 nats
entropy_theory = float(np.log(8) + np.log(2 * np.pi * np.e * 0.15 ** 2))

print(f"\\n桥接点占比（到最近中心距离 > 0.5）: CNLL {bridge_cnll*100:.2f}% | FM {bridge_fm*100:.2f}% | 真实 {bridge_real*100:.2f}%")
print(f"NLL 重估（同一 CNLL 模型，{len(eval_data)} 个训练点）: 20步 {nll_20_eval:.4f} | 100步 {nll_100_eval:.4f} | 目标分布微分熵 {entropy_theory:.4f}")""",
))

# 2) 总结去硬编码
subs.append((
    """   - 统计采样点到最近高斯中心的距离，FM的"桥接点"（距最近中心>0.5）占比约4%，
     高于Neural ODE的约1%。这说明FM在离散众数意义上更容易产生跨众数弧线。""",
    """   - 统计采样点到最近高斯中心的距离，FM的"桥接点"（距最近中心>0.5）占比见
     results_summary.json 的 bridge_fm_pct，高于Neural ODE的 bridge_cnll_pct。
     这说明FM在离散众数意义上更容易产生跨众数弧线。""",
))

# 3) json 字段
subs.append((
    """    'final_loss_fm': round(float(losses_fm[-1]), 6) if losses_fm else None,
}""",
    """    'final_loss_fm': round(float(losses_fm[-1]), 6) if losses_fm else None,
    'bridge_cnll_pct': round(float(bridge_cnll) * 100, 2),
    'bridge_fm_pct': round(float(bridge_fm) * 100, 2),
    'bridge_real_pct': round(float(bridge_real) * 100, 2),
    'bridge_note': '到最近高斯中心距离 > 0.5 的样本占比（n=2000）',
    'nll_recheck_20step': round(float(nll_20_eval), 4),
    'nll_recheck_100step': round(float(nll_100_eval), 4),
    'entropy_theory': round(float(entropy_theory), 4),
    'entropy_note': 'log 8 + log(2*pi*e*sigma^2), sigma=0.15, 单位 nats',
}""",
))

for old, new in subs:
    c = s.count(old)
    print('count', c, '|', old.strip().splitlines()[0][:52])
    assert c == 1, 'COUNT NOT 1'
    s = s.replace(old, new)

open(P, 'w', encoding='utf-8').write(s)
print('OK')
