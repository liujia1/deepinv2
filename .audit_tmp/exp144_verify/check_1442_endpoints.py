# -*- coding: utf-8 -*-
"""核查 14.4-2：Reflow 训练端点对的方差收缩与模式外泄漏（正文声称 8.18→7.94→7.54, 泄漏 0→2.5%→4.5%）"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, '实验14.4-2')


class VectorFieldNet(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x_t, t):
        return self.net(torch.cat([x_t, t], dim=-1))


def sample_source(n):
    mode = np.random.rand(n) < 0.5
    x = np.where(mode, -2, 2) + 0.3 * np.random.randn(n)
    y = np.where(mode, 2, -2) + 0.3 * np.random.randn(n)
    return np.stack([x, y], axis=1)


@torch.no_grad()
def ode_endpoint(model, z, n_steps=100):
    model.eval()
    x = z.clone()
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full((x.shape[0], 1), step / n_steps)
        x = x + model(x, t) * dt
    return x


def load(name):
    ck = torch.load(os.path.join(EXP, name), map_location='cpu', weights_only=False)
    m = VectorFieldNet()
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m


def stats(x, tag):
    x = x.numpy() if torch.is_tensor(x) else x
    tot_var = float(np.var(x, axis=0).sum())
    # 目标两模式中心
    centers = np.array([[2.0, 2.0], [-2.0, -2.0]])
    d = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=-1).min(1)
    leak09 = float((d > 0.9).mean())
    leak12 = float((d > 1.2).mean())
    print(f"  {tag:<22s} n={len(x):4d}  总方差={tot_var:6.3f}  泄漏(>0.9)={leak09*100:5.2f}%  泄漏(>1.2)={leak12*100:5.2f}%")
    return dict(tag=tag, n=int(len(x)), total_var=tot_var, leak_09=leak09, leak_12=leak12)


print("真实目标（同分布 512+512 作参照）:")
np.random.seed(7777)
res = [stats(sample_source(512), '真实目标#1'), stats(sample_source(512), '真实目标#2')]

m1 = load('model_1rf_final.pth')
m2 = load('model_2rf_final.pth')
m3 = load('model_3rf_final.pth')

# 复刻脚本里的端点对生成（reflow_round=1 -> seed 1042; round=2 -> seed 2042）
np.random.seed(1 * 1000 + 42)
z1 = torch.FloatTensor(sample_source(512))
x1 = ode_endpoint(m1, z1)
np.random.seed(2 * 1000 + 42)
z2 = torch.FloatTensor(sample_source(512))
x2 = ode_endpoint(m2, z2)
np.random.seed(3 * 1000 + 42)
z3 = torch.FloatTensor(sample_source(512))
x3 = ode_endpoint(m3, z3)

print("\n各轮 Reflow 的训练端点 x_endpoint（512 对，正文称 8.18→7.94→7.54 / 0→2.5%→4.5%）:")
res.append(stats(x1, '1-RF 端点 (2-RF用)'))
res.append(stats(x2, '2-RF 端点 (3-RF用)'))
res.append(stats(x3, '3-RF 端点'))

# 2-RF 对训练端点的复现精度（正文称同批 512 个 z 上 W1≈0.12）
from scipy.stats import wasserstein_distance_nd
pred2 = ode_endpoint(m2, z1)
print(f"\n2-RF 在同批512 z 上重跑 ODE 与训练端点的 W1 = {wasserstein_distance_nd(pred2.numpy(), x1.numpy()):.4f}")

# 采样端点（用测试噪声）
np.random.seed(42)
zt = torch.FloatTensor(sample_source(200))
np.random.seed(4243)
print("\n训练端点 vs 独立同分布真实样本 的 W1（跨批次，正文称约0.35）:")
np.random.seed(4243)
tgt = sample_source(512)
print(f"  W1(x1_endpoint, 真实512) = {wasserstein_distance_nd(x1.numpy(), tgt):.4f}")

with open(os.path.join(HERE, 'check_1442_endpoints.json'), 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("\n已保存 check_1442_endpoints.json")
