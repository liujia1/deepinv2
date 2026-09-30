# -*- coding: utf-8 -*-
"""审计脚本：核查实验14.2-1讲义中的定量声明
1) 桥接点占比（讲义称 FM~4% vs CNLL~1%）
2) loss曲线平滑度（讲义称 Neural ODE 更毛躁、FM 更平滑——用数据验证方向）
3) CNLL最终NLL vs 真实分布理论熵（交叉熵下界）
"""
import sys, io, os
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import numpy as np
import torch
import torch.nn as nn

EXP = r"d:\works\deepinv\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.2-1"
sys.path.insert(0, EXP)

np.random.seed(42)
torch.manual_seed(42)

# ---- 复刻 14.2-1.py 的模型定义 ----
class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        self.register_buffer('freqs', torch.exp(
            -torch.log(torch.tensor(10000.0)) * torch.arange(0, half_dim) / half_dim))
    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        args = t * self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class VectorFieldMLP(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbed(dim=64)
        self.net = nn.Sequential(
            nn.Linear(2 + 64, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, x, t):
        return self.net(torch.cat([x, self.time_embed(t)], dim=-1))

def sample_8gaussian(n, device='cpu'):
    n_modes, radius = 8, 2.0
    angles = torch.arange(n_modes, device=device) * 2 * np.pi / n_modes
    centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
    mode_idx = torch.randint(0, n_modes, (n,), device=device)
    noise = 0.15 * torch.randn(n, 2, device=device)
    return centers[mode_idx] + noise

def ode_solve_euler(vf_net, z0, n_steps, t_start=0.0, t_end=1.0):
    dt = (t_end - t_start) / n_steps
    x = z0
    for step in range(n_steps):
        t_val = t_start + step * dt
        t_tensor = torch.full((x.shape[0],), t_val, device=x.device)
        x = x + vf_net(x, t_tensor) * dt
    return x

# ---- 复刻主脚本的 RNG 消耗顺序（跳过训练路径）----
train_data = sample_8gaussian(2000)          # 消耗 RNG
model_cnll = VectorFieldMLP(); model_fm = VectorFieldMLP()  # 初始化消耗 RNG
ck = torch.load(os.path.join(EXP, 'cnll_final.pth'), map_location='cpu', weights_only=False)
model_cnll.load_state_dict(ck['model_state_dict'])
ck2 = torch.load(os.path.join(EXP, 'fm_final.pth'), map_location='cpu', weights_only=False)
model_fm.load_state_dict(ck2['model_state_dict'])

torch.manual_seed(42)
with torch.no_grad():
    z = torch.randn(2000, 2)
    s_cnll = ode_solve_euler(model_cnll, z, 100)
torch.manual_seed(42)
with torch.no_grad():
    z = torch.randn(2000, 2)
    s_fm = ode_solve_euler(model_fm, z, 100)
test_data = sample_8gaussian(2000)

# 与图2一致性抽查：验证采样点与主脚本一致（MMD 应等于 json 记录值）
def compute_mmd(x, y, sigma=1.0):
    xx = torch.cdist(x, x)**2; yy = torch.cdist(y, y)**2; xy = torch.cdist(x, y)**2
    c = 2*sigma**2
    return (torch.mean(torch.exp(-xx/c)) + torch.mean(torch.exp(-yy/c)) - 2*torch.mean(torch.exp(-xy/c))).item()
print(f"[复现校验] MMD cnll={compute_mmd(s_cnll, test_data):.6f}  fm={compute_mmd(s_fm, test_data):.6f}  (期望 0.001453 / 0.000992)")

# ---- 1) 桥接点占比 ----
angles = torch.arange(8) * 2*np.pi/8
centers = torch.stack([2.0*torch.cos(angles), 2.0*torch.sin(angles)], dim=1)
def bridge_ratio(samples, thr=0.5):
    d = torch.cdist(samples, centers).min(dim=1).values
    return (d > thr).float().mean().item() * 100, d.mean().item()
bc, mc = bridge_ratio(s_cnll); bf, mf = bridge_ratio(s_fm)
print(f"[桥接点占比 >0.5] Neural ODE: {bc:.2f}%   FM: {bf:.2f}%   (讲义声称 ~1% vs ~4%)")
print(f"  平均最近中心距离: CNLL={mc:.3f}  FM={mf:.3f}")
bt, _ = bridge_ratio(test_data)
print(f"  参照: 真实数据自身桥接占比 = {bt:.2f}%")

# ---- 2) loss 曲线平滑度 ----
L_cnll = np.array(ck['train_losses']); L_fm = np.array(ck2['train_losses'])
def noise_metric(L, win=21):
    """去趋势后的残差标准差（滚动均值滤波）"""
    pad = win//2
    P = np.pad(L, pad, mode='edge')
    ma = np.convolve(P, np.ones(win)/win, mode='valid')[:len(L)]
    return np.std(L - ma)
tail = slice(200, None)
print(f"[平滑度] 后300epoch 去趋势残差std: Neural ODE={noise_metric(L_cnll[tail]):.5f}  FM={noise_metric(L_fm[tail]):.5f}")
print(f"[平滑度] 相邻epoch差分std:        Neural ODE={np.std(np.diff(L_cnll[tail])):.5f}  FM={np.std(np.diff(L_fm[tail])):.5f}")
print(f"  epochs: CNLL={len(L_cnll)}, FM={len(L_fm)}; final: CNLL={L_cnll[-1]:.4f}, FM={L_fm[-1]:.4f}")

# ---- 3) 理论熵 vs CNLL最终NLL ----
# 8-Gaussian(等权, 分离良好) 微分熵 ≈ ln8 + ln(2πe σ²)
H = np.log(8) + np.log(2*np.pi*np.e*0.15**2)
print(f"[似然下界] 目标分布理论熵 H ≈ {H:.4f}（交叉熵不可能低于它）")
print(f"  CNLL 最终'NLL' = {L_cnll[-1]:.4f}  → {'低于熵，说明估计有偏!' if L_cnll[-1] < H - 0.05 else '不低于熵，合理'}")
# 用真实数据算 CNLL 模型的经验交叉熵（用同一 compute_log_likelihood 函数）
sys.path.insert(0, EXP)
def compute_log_likelihood(vf_net, x_data, n_steps):
    batch = x_data.shape[0]; dt = -1.0/n_steps
    x = x_data.clone(); log_det_sum = 0.0
    for step in range(n_steps):
        t_val = 1.0 - step/n_steps
        t_tensor = torch.full((batch,), t_val, device=x.device)
        v = vf_net(x, t_tensor)
        eps = torch.randn_like(x)
        v_p = vf_net(x + eps*1e-5, t_tensor)
        trace_est = (eps*(v_p - v)/1e-5).sum(dim=-1)
        log_det_sum = log_det_sum + trace_est*abs(dt)
        x = x + v*dt
    log_p0 = -0.5*(x**2).sum(dim=-1) - np.log(2*np.pi)
    return log_p0 - log_det_sum
torch.manual_seed(0)
with torch.no_grad():
    lp_true = compute_log_likelihood(model_cnll, test_data, 20).mean().item()
    lp_true100 = compute_log_likelihood(model_cnll, test_data, 100).mean().item()
print(f"  同一模型在独立测试集上的NLL: 20步={lp_true:.4f}, 100步={lp_true100:.4f}（步数改变→数值变化反映离散化偏差）")
