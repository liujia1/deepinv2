# -*- coding: utf-8 -*-
"""诊断6.3-1: (A) C(σ) MC-理论差异是否为有限样本噪声(固有属性裁决)
            (B) 讲义对 σ=0.1 得分场"峰间几乎为零"的定性说法核查"""
import numpy as np, torch, sys
sys.stdout.reconfigure(encoding='utf-8')

np.random.seed(42); torch.manual_seed(42)

def gm1d_score_noisy(x, sigma):
    v = 1 + sigma**2
    p1 = np.exp(-0.5*(x+2)**2/v)/np.sqrt(2*np.pi*v)
    p2 = np.exp(-0.5*(x-2)**2/v)/np.sqrt(2*np.pi*v)
    p = 0.5*p1 + 0.5*p2
    return (0.5*-(x+2)/v*p1 + 0.5*-(x-2)/v*p2) / p

class ScoreNet1D(torch.nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x.unsqueeze(-1)).squeeze(-1)

sigma = 1.0
# --- 训练与主脚本完全一致(种子42) ---
x_train = np.concatenate([np.random.randn(5000)-2, np.random.randn(5000)+2])
x_train_t = torch.tensor(x_train, dtype=torch.float32)
model = ScoreNet1D(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(500):
    z = torch.randn_like(x_train_t); x_noisy = x_train_t + sigma*z
    loss = torch.mean((model(x_noisy) - (-z/sigma))**2)
    opt.zero_grad(); loss.backward(); opt.step()

# 理论 C(σ)
x_grid = np.linspace(-8, 8, 2000)
sg = gm1d_score_noisy(x_grid, sigma)
pdf = 0.5*np.exp(-0.5*(x_grid+2)**2/(1+sigma**2))/np.sqrt(2*np.pi*(1+sigma**2)) + \
      0.5*np.exp(-0.5*(x_grid-2)**2/(1+sigma**2))/np.sqrt(2*np.pi*(1+sigma**2))
c_theory = 0.5/sigma**2 - 0.5*np.trapezoid(sg**2*pdf, x_grid)
print(f"c_theory = {c_theory:.6f}")

# (A) 多 seed / 多 N_mc 下 C(σ)=J_DSM-J_ESM 的收敛性
def c_mc(seed, N):
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.standard_normal(N//2)-2, rng.standard_normal(N//2)+2])
    z = rng.standard_normal(N)
    xn = x + sigma*z
    with torch.no_grad():
        s = model(torch.tensor(xn, dtype=torch.float32)).numpy()
    j_dsm = 0.5*np.mean((s + z/sigma)**2)
    j_esm = 0.5*np.mean((s - gm1d_score_noisy(xn, sigma))**2)
    return j_dsm - j_esm

for N in [5000, 20000, 100000, 500000]:
    vals = [c_mc(1000+i, N) for i in range(8)]
    vals = np.array(vals)
    print(f"N={N:>7d}: 均值={vals.mean():.6f}  标准差={vals.std():.6f}  "
          f"与理论差={vals.mean()-c_theory:+.6f}  单seed范围=[{vals.min():.4f},{vals.max():.4f}]")

# 主脚本原始口径(seed42续流, N=5000)对照: 期望≈0.3631
# (B) 随机初始化模型 C(σ) 与训练后一致性(多seed)
c_r = []
for k in range(5):
    torch.manual_seed(k)
    mrand = ScoreNet1D()
    rng = np.random.default_rng(2000+k)
    x = np.concatenate([rng.standard_normal(2500)-2, rng.standard_normal(2500)+2])
    z = rng.standard_normal(5000); xn = x+sigma*z
    with torch.no_grad():
        s = mrand(torch.tensor(xn, dtype=torch.float32)).numpy()
    c_r.append(0.5*np.mean((s+z/sigma)**2) - 0.5*np.mean((s-gm1d_score_noisy(xn,sigma))**2))
print(f"随机初始化 C(σ) 5个seed: {np.round(c_r,4)}  均值={np.mean(c_r):.6f}")

# (B) σ=0.1 得分场在峰间区域的量级
print("\n得分 |s| 量级对照 (峰间区域 x∈[-1.5,1.5]):")
for s_lev in [0.1, 0.5, 1.0, 2.0]:
    xs = np.array([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])
    v = gm1d_score_noisy(xs, s_lev)
    print(f"  σ={s_lev}: " + "  ".join(f"x={x:+.1f}:{abs(sc):.2f}" for x, sc in zip(xs, v)))
