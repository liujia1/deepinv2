# -*- coding: utf-8 -*-
"""对照实验：验证14.4-2中W1随Reflow轮数上升的机理
假设：reflow的训练目标端点是上一轮模型自己的输出（非真实目标），
每轮误差累积 → W1上升是reflow的固有属性（分布漂移），而非代码bug。
检验：
 A) 2-RF训练端点对(x_endpoint来自1-RF) vs 真实目标 的W1 ≈ 0.36（=1-RF的W1）
 B) 3-RF训练端点对(来自2-RF) vs 真实目标 的W1 ≈ 0.48（=2-RF的W1）
 C) 2-RF自己100步输出 vs 其训练端点对分布 的W1（若很小→2-RF忠实拟合了端点分布，
    说明W1上升源于端点对本身漂移，不是2-RF训练失败）
 D) 1步 vs 100步位移对比：验证1-RF单步塌缩、reflow后单步即近终点（直线化的实用价值）
"""
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import os
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance_nd

SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else r'd:\works\deepinv\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.4-2'
SAVE_DIR = r'd:\works\deepinv\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.4-2'

def sample_source(n):
    mode = np.random.rand(n) < 0.5
    x = np.where(mode, -2, 2) + 0.3 * np.random.randn(n)
    y = np.where(mode, 2, -2) + 0.3 * np.random.randn(n)
    return np.stack([x, y], axis=1)

def sample_target(n):
    mode = np.random.rand(n) < 0.5
    x = np.where(mode, 2, -2) + 0.3 * np.random.randn(n)
    y = np.where(mode, 2, -2) + 0.3 * np.random.randn(n)
    return np.stack([x, y], axis=1)

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

def load(name):
    m = VectorFieldNet()
    ck = torch.load(os.path.join(SAVE_DIR, name), map_location='cpu', weights_only=False)
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m, ck

@torch.no_grad()
def integrate(model, z, n_steps):
    x = z.clone()
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full((x.shape[0], 1), step / n_steps)
        x = x + model(x, t) * dt
    return x.numpy()

np.random.seed(42)
torch.manual_seed(42)

m1, _ = load('model_1rf_final.pth')
m2, _ = load('model_2rf_final.pth')
m3, _ = load('model_3rf_final.pth')
mo, _ = load('model_ot_final.pth')

# 复现 reflow 轮次的训练端点对（与主脚本相同的种子与步数）
np.random.seed(1 * 1000 + 42)
z_r1 = torch.FloatTensor(sample_source(512))
ep1 = integrate(m1, z_r1, 50)          # 2-RF 的训练终点（1-RF 的输出）
np.random.seed(2 * 1000 + 42)
z_r2 = torch.FloatTensor(sample_source(512))
ep2 = integrate(m2, z_r2, 50)          # 3-RF 的训练终点（2-RF 的输出）

# 真实留出目标（与主脚本测试集一致）
np.random.seed(42)
z_test = torch.FloatTensor(sample_source(200))
target_test = sample_target(200)

out_1 = integrate(m1, z_test, 100)
out_2 = integrate(m2, z_test, 100)
out_3 = integrate(m3, z_test, 100)
out_ot = integrate(mo, z_test, 100)

print('== 主脚本复现检查（应与json一致: 0.3598/0.4776/0.5543/0.4059）==')
print(f'  W1(1-RF, target) ={wasserstein_distance_nd(out_1, target_test):.4f}')
print(f'  W1(2-RF, target) ={wasserstein_distance_nd(out_2, target_test):.4f}')
print(f'  W1(3-RF, target) ={wasserstein_distance_nd(out_3, target_test):.4f}')
print(f'  W1(OT , target) ={wasserstein_distance_nd(out_ot, target_test):.4f}')

print('\n== A/B: reflow训练端点对 vs 真实目标（误差累积假说）==')
print(f'  W1(1-RF端点对[2-RF训练数据], target) = {wasserstein_distance_nd(ep1, target_test):.4f}')
print(f'  W1(2-RF端点对[3-RF训练数据], target) = {wasserstein_distance_nd(ep2, target_test):.4f}')

print('\n== C: 各轮模型是否忠实拟合自己的训练端点分布 ==')
print(f'  W1(2-RF输出, 其训练端点ep1) = {wasserstein_distance_nd(out_2, ep1):.4f}')
print(f'  W1(3-RF输出, 其训练端点ep2) = {wasserstein_distance_nd(out_3, ep2):.4f}')

print('\n== D: 单步Euler位移 vs 100步总位移（少步可用性的机理）==')
for name, m in [('1-RF', m1), ('2-RF', m2), ('3-RF', m3), ('OT-CFM', mo)]:
    one = integrate(m, z_test, 1)
    ratio = np.linalg.norm(one - z_test.numpy(), axis=1).mean() / \
            np.linalg.norm(out_1 * 0 + integrate(m, z_test, 100) - z_test.numpy(), axis=1).mean()
    w1_one = wasserstein_distance_nd(one, target_test)
    print(f'  {name:7s} 1步W1={w1_one:.4f}  (100步W1={wasserstein_distance_nd(integrate(m, z_test, 100), target_test):.4f})')

print('\n== E: 端点对“塌缩度”：reflow后终点方差是否收缩（分布漂移的具体形式）==')
for name, arr in [('真实target', target_test), ('ep1(1-RF输出)', ep1), ('ep2(2-RF输出)', ep2),
                  ('1-RF 100步', out_1), ('2-RF 100步', out_2), ('3-RF 100步', out_3)]:
    var = arr.var(axis=0).sum()
    # 每个高斯模式到最近模式中心的距离>1的点占比（质量泄漏/拉扁程度）
    centers = np.array([[2, 2], [-2, -2]])
    dmin = np.linalg.norm(arr[:, None] - centers[None], axis=2).min(axis=1)
    leak = (dmin > 1.0).mean()
    print(f'  {name:14s} 总方差={var:.3f}  模式外泄漏比例={leak:.3f}')
