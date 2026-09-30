# -*- coding: utf-8 -*-
"""实验14.4-3 复核脚本：
1) 加载三个 final checkpoint，用固定 z (CRN) 重算曲率，验证单调性
2) 读取 checkpoint 中 train_losses，验证"2-RF loss 断崖式下降"
3) 量化采样质量：W1(模型样本 vs 真实目标)、总方差、模式外泄漏比例、零基线 W1
"""
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import os
import numpy as np
import torch

EXP_DIR = r"d:\works\deepinv\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.4-3"
sys.path.insert(0, EXP_DIR)

import importlib.util
spec = importlib.util.spec_from_file_location("exp1443", os.path.join(EXP_DIR, "14.4-3.py"))

# 不执行整个脚本（会重跑），手动复制关键函数
class VectorFieldMLP(torch.nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, hidden_dim), torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 2),
        )
    def forward(self, x, t):
        inp = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        return self.net(inp)

@torch.no_grad()
def ode_sample(model, z, n_steps=100):
    model.eval()
    x = z.clone()
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t_val = step / n_steps
        t_tensor = torch.full((x.shape[0],), t_val, device=x.device)
        x = x + model(x, t_tensor) * dt
    return x

def curvature_from_traj(traj):
    ks = []
    for i in range(traj.shape[1]):
        p = traj[:, i, :]
        sd = np.linalg.norm(p[-1] - p[0])
        pl = np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
        k = max(1.0 - sd / pl, 0.0) if pl > 1e-8 else 0.0
        ks.append(k)
    return float(np.mean(ks))

def sample_crossed_gaussians(n, seed):
    g = torch.Generator().manual_seed(seed)
    mode = torch.rand(n, generator=g) < 0.5
    x = torch.where(mode, -2.0, 2.0) + 0.3 * torch.randn(n, generator=g)
    y = torch.where(mode, 2.0, -2.0) + 0.3 * torch.randn(n, generator=g)
    return torch.stack([x, y], dim=1)

def w1_2d(a, b, sub=500, seed=0):
    """Sinkhorn-free: 用 scipy linear_sum_assignment 的经验 W1 (L2 cost) 子采样"""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(seed)
    A = a[rng.choice(len(a), sub, replace=False)]
    B = b[rng.choice(len(b), sub, replace=False)]
    C = cdist(A, B)
    r, c = linear_sum_assignment(C)
    return float(C[r, c].mean())

torch.manual_seed(12345)
models = {}
losses = {}
for k in (1, 2, 3):
    path = os.path.join(EXP_DIR, f'{k}rf_final.pth')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    m = VectorFieldMLP(128)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    models[k] = m
    losses[k] = ckpt.get('train_losses', [])
    print(f"{k}rf_final.pth: epochs记录={len(losses[k])}, 最终loss={losses[k][-1]:.6f}, "
          f"前50均值={np.mean(losses[k][:50]):.6f}, 后50均值={np.mean(losses[k][-50:]):.6f}")

# 1) 曲率重算（固定 CRN z）
z_curv = torch.randn(500, 2)
print("\n--- 曲率重算（固定z, CRN）---")
kappas = {}
for k in (1, 2, 3):
    x = z_curv.clone()
    traj = [x.numpy().copy()]
    with torch.no_grad():
        for step in range(100):
            t_tensor = torch.full((500,), step / 100)
            x = x + models[k](x, t_tensor) * 0.01
            traj.append(x.numpy().copy())
    kappas[k] = curvature_from_traj(np.array(traj))
    print(f"kappa_{k}RF = {kappas[k]:.6f}")
mono = all(kappas[k+1] <= kappas[k] + 1e-6 for k in (1, 2))
print(f"单调性: {mono}")

# 2) 采样质量量化
print("\n--- 采样质量量化 (n=1000) ---")
target = sample_crossed_gaussians(5000, seed=777).numpy()
# 零基线：两个独立真实样本集之间的 W1
null_w1 = w1_2d(sample_crossed_gaussians(2000, 11).numpy(),
                sample_crossed_gaussians(2000, 22).numpy(), sub=500)
print(f"零基线 W1(真实 vs 真实) ≈ {null_w1:.4f}")
g = torch.Generator().manual_seed(2024)
for k in (1, 2, 3):
    z = torch.randn(1000, 2, generator=g)
    with torch.no_grad():
        s = ode_sample(models[k], z).numpy()
    d_target = w1_2d(s, target, sub=500, seed=k)
    var = s.var(axis=0).sum()
    # 模式外泄漏：离两个模式中心 (-2,2),(2,-2) 都超过 1.5 的样本
    dd = np.minimum(np.linalg.norm(s - np.array([-2, 2]), axis=1),
                    np.linalg.norm(s - np.array([2, -2]), axis=1))
    leak = float((dd > 1.5).mean() * 100)
    print(f"{k}-RF: W1(vs目标)={d_target:.4f}  总方差={var:.4f}  模式外泄漏={leak:.1f}%")

# 真实数据总方差基线
print(f"真实目标分布总方差基线 = {target.var(axis=0).sum():.4f}")
