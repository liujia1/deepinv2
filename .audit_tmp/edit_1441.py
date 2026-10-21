# -*- coding: utf-8 -*-
"""14.4-1：四组采样共用同一批噪声（CRN），消除采样噪声对 1-RF/2-RF 对比的干扰。"""
P = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.4-1\14.4-1.py'
s = open(P, encoding='utf-8').read()
subs = []

# 1) 采样函数接受外部固定噪声
subs.append((
    '''def flow_ode_sample(model, shape, n_steps=50):
    """从Flow ODE采样: dx/dt = v_θ(x, t), t: 0→1

    14.3.1节: Flow ODE直接参数化向量场
    对比DDPM需要200步，FM理论上可以1步
    """
    model.eval()
    x = torch.randn(shape, device=device)  # z ~ N(0,I)''',
    '''def flow_ode_sample(model, shape, n_steps=50, z=None):
    """从Flow ODE采样: dx/dt = v_θ(x, t), t: 0→1

    14.3.1节: Flow ODE直接参数化向量场
    对比DDPM需要200步，FM理论上可以1步

    z: 可选，外部传入的固定噪声（CRN原则：多组采样共用同一批 z，
       组间差异只来自模型与步数，而不是各抽各的初始噪声）
    """
    model.eval()
    x = z.clone() if z is not None else torch.randn(shape, device=device)  # z ~ N(0,I)''',
))

# 2) 主对比共享噪声
subs.append((
    '''n_samples = 8
sample_shape = (n_samples, 1, 28, 28)

rf_1step_1rf = flow_ode_sample(rf_model, sample_shape, n_steps=1)
rf_1step_2rf = flow_ode_sample(model_2rf, sample_shape, n_steps=1)
rf_50step_1rf = flow_ode_sample(rf_model, sample_shape, n_steps=50)
rf_10step_1rf = flow_ode_sample(rf_model, sample_shape, n_steps=10)''',
    '''n_samples = 8
sample_shape = (n_samples, 1, 28, 28)

# CRN原则：四组采样共用同一批噪声，差异只来自"模型 × 步数"。
# 旧实现各自 torch.randn，8 个样本的采样噪声会盖过 1-RF/2-RF 的真实差异。
_SAMPLING_SEED = 2025
torch.manual_seed(_SAMPLING_SEED)
z_eval = torch.randn(sample_shape).to(device)

rf_1step_1rf = flow_ode_sample(rf_model, sample_shape, n_steps=1, z=z_eval)
rf_1step_2rf = flow_ode_sample(model_2rf, sample_shape, n_steps=1, z=z_eval)
rf_50step_1rf = flow_ode_sample(rf_model, sample_shape, n_steps=50, z=z_eval)
rf_10step_1rf = flow_ode_sample(rf_model, sample_shape, n_steps=10, z=z_eval)''',
))

# 3) json 记录采样口径
subs.append((
    """    'diversity_real': round(float(div_real), 4),
}""",
    """    'diversity_real': round(float(div_real), 4),
    'sampling_seed': int(_SAMPLING_SEED),
    'sampling_note': 'CRN：四组采样共用 seed=2025 的同一批 8 个噪声，组间差异只来自模型与步数',
}""",
))

for old, new in subs:
    c = s.count(old)
    print('count', c, '|', old.strip().splitlines()[0][:52])
    assert c == 1, 'COUNT NOT 1'
    s = s.replace(old, new)

open(P, 'w', encoding='utf-8').write(s)
print('OK')
